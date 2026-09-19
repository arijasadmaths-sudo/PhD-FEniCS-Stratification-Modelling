"""Isolated diagnostic continuation and an independent residual correction.

The correction is measured and then discarded. Neither the production solver
nor the baseline fields accepted by the diagnostic are modified.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from rect3d_bounds_report import BoundsReport, summarize


FOLLOWUP_REVISION = 'residual-correction-2026-09-18'


def prepare_diagnostic_checkpoint(source, checkpoint_dir, read_metadata,
                                  validate_metadata):
    """Copy the completed step-50 diagnostic state into a new case directory."""
    source, directory = Path(source), Path(checkpoint_dir)
    if not source.is_file():
        raise RuntimeError('Required diagnostic checkpoint is missing: '+str(source))
    metadata = read_metadata(str(source), validate_payload=True)
    validate_metadata(metadata, check_mesh=False)
    if (metadata.get('solver_schema') != 'rect3d_bounds_diagnostic_only_20260917a'
            or int(metadata.get('step', -1)) != 50
            or int(metadata.get('mesh_cells', -1)) != 608706
            or not math.isclose(float(metadata.get('time', -1)), .05, rel_tol=0., abs_tol=1e-14)
            or not math.isclose(float(metadata.get('dt', -1)), .001, rel_tol=0., abs_tol=1e-15)):
        raise RuntimeError('Continuation requires the fine diagnostic checkpoint at step 50, t=0.050 s.')
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise RuntimeError('Diagnostic destination is not empty; refusing to overwrite a checkpoint.')
    destination = directory/'state_step_000000050.npz'
    temporary = destination.with_suffix('.npz.tmp')
    digest = hashlib.sha256()
    try:
        with source.open('rb') as incoming, temporary.open('xb') as outgoing:
            while True:
                block = incoming.read(8*1024*1024)
                if not block:
                    break
                digest.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    provenance = dict(source_checkpoint=str(source.resolve()),
                      copied_checkpoint=str(destination.resolve()),
                      source_sha256=digest.hexdigest(), source_step=50,
                      source_time_s=.05, production_eligible=False)
    (directory.parent/'diagnostic_restart_origin.json').write_text(
        json.dumps(provenance, indent=2, sort_keys=True))
    return provenance


class FollowupBoundsReport(BoundsReport):
    def __init__(self, ns):
        super().__init__(ns)
        self.report.update(
            scope='Isolated continuation from 0.050 to 0.100 s; not production results',
            followup_revision=FOLLOWUP_REVISION,
            start_time_s=float(ns['start_time']), start_step=int(ns['start_step']),
            diagnostic_restart_origin=ns['diagnostic_restart_origin'],
            accuracy_probe='Solve A*delta=b-A*x from zero, inspect x+delta, then restore x',
        )
        self._save_json()

    def _tight_check(self, transport, baseline):
        """Solve the original residual equation, rather than reuse a warm start.

        The convergence scale is the correction RHS, not the full tank RHS.
        An independent residual check rejects an ineffective correction. The
        candidate concentration and its changes are diagnostics only.
        """
        from dolfin import Function, as_backend_type

        state = transport.state.vector()
        saved_state = state.copy()
        statistic_names = ('last_iterations', 'last_true_residual',
                           'last_relative_residual', 'last_recovery_used',
                           'recovery_count', 'solve_calls')
        statistics = {name: getattr(transport, name) for name in statistic_names}
        parameters = transport.solver.parameters
        parameter_names = ('relative_tolerance', 'absolute_tolerance', 'nonzero_initial_guess')
        saved_parameters = {name: parameters[name] for name in parameter_names}
        saved_ksp_tolerances = tuple(transport._ksp.getTolerances())
        out = Function(transport.C)
        result = dict(method='zero_initial_guess_residual_correction', completed=False)
        error = None

        def scaled_norm(vector):
            scaled = vector.copy()
            as_backend_type(scaled).vec().pointwiseMult(
                as_backend_type(transport._scale).vec(), as_backend_type(vector).vec())
            return float(scaled.norm('l2'))

        try:
            # A and b are the already assembled, unscaled baseline system.
            residual = transport.b.copy()
            transport.A.mult(saved_state, residual)
            residual *= -1.
            residual.axpy(1., transport.b)
            original_norm = float(residual.norm('l2'))
            rhs_norm = scaled_norm(residual)
            result.update(baseline_original_residual=original_norm,
                          correction_rhs_scaled_norm=rhs_norm)
            if not math.isfinite(rhs_norm) or rhs_norm <= 0.:
                raise RuntimeError('Residual correction has a zero or nonfinite RHS; no independent solve was performed.')
            correction = residual.copy()
            correction.zero()
            correction.apply('insert')
            # Tolerances now refer to the small correction equation. A zero
            # initial correction cannot pass by matching the old full RHS.
            requested_rtol = min(float(saved_parameters['relative_tolerance']), 1e-10)
            requested_atol = min(float(saved_parameters['absolute_tolerance']), 1e-24, 1e-12*rhs_norm)
            parameters['relative_tolerance'] = requested_rtol
            parameters['absolute_tolerance'] = requested_atol
            parameters['nonzero_initial_guess'] = False
            iterations = int(transport._solve_equilibrated(correction, residual))
            effective = tuple(transport._ksp.getTolerances())
            result.update(iterations=iterations, requested_rtol=requested_rtol,
                          requested_atol=requested_atol,
                          effective_ksp_rtol=float(effective[0]), effective_ksp_atol=float(effective[1]),
                          ksp_converged_reason=int(transport._ksp.getConvergedReason()))
            if iterations < 1:
                raise RuntimeError('Residual-correction probe took zero iterations; its accuracy comparison is inconclusive.')
            remainder = residual.copy()
            transport.A.mult(correction, remainder)
            remainder *= -1.
            remainder.axpy(1., residual)
            remainder_norm = scaled_norm(remainder)
            relative_remainder = remainder_norm/rhs_norm
            result.update(correction_remainder_scaled_norm=remainder_norm,
                          correction_relative_residual=relative_remainder)
            if not math.isfinite(relative_remainder) or relative_remainder > 1e-7:
                raise RuntimeError('Independent residual-correction accuracy check failed: relative residual {}'.format(relative_remainder))
            state.axpy(1., correction)
            state.apply('insert')
            candidate_residual = transport.b.copy()
            transport.A.mult(state, candidate_residual)
            candidate_residual.axpy(-1., transport.b)
            result['true_residual'] = float(candidate_residual.norm('l2'))
            transport._copy_scalar.assign(out, transport.state.sub(1))
            out.vector().apply('insert')
        except RuntimeError as exc:
            error = str(exc)
        finally:
            for name, value in saved_parameters.items():
                parameters[name] = value
            transport._ksp.setTolerances(rtol=saved_ksp_tolerances[0],
                                         atol=saved_ksp_tolerances[1],
                                         divtol=saved_ksp_tolerances[2],
                                         max_it=saved_ksp_tolerances[3])
            state.zero()
            state.axpy(1., saved_state)
            state.apply('insert')
            for name, value in statistics.items():
                setattr(transport, name, value)
        errors = self.comm.allgather(error)
        error = next((item for item in errors if item is not None), None)
        if error is not None:
            result['error'] = error
            self.ns['root_print']('RESIDUAL-CORRECTION PROBE INCONCLUSIVE:', error)
            return result, None
        values = np.asarray(out.vector().get_local(), dtype=float)
        result.update(completed=True,
                      concentration=summarize(values, self.volumes, self.comm, self.mpi),
                      max_abs_change_from_baseline=self.comm.allreduce(
                          float(np.max(np.abs(values-baseline))) if len(values) else 0., op=self.mpi.MAX))
        self.ns['root_print'](
            'RESIDUAL-CORRECTION PROBE: D={:.6e}; iterations={}; relative correction residual={:.3e}; '
            'maximum concentration change={:.3e}; corrected range=[{:.9f},{:.9f}]'.format(
                transport.D, result['iterations'], result['correction_relative_residual'],
                result['max_abs_change_from_baseline'], result['concentration']['minimum'],
                result['concentration']['maximum']))
        return result, values

    def observe(self):
        super().observe()
        # Still make an accuracy measurement if the restarted fields happen
        # to return inside the production limit immediately.
        if len(self.report['steps']) == 1 and not self.report['tighter_solve_checks']:
            snapshot = self.snapshots['last'].copy()
            for field, transport in (('density', self.ns['scalar_transport']),
                                     ('dye', self.ns['dye_transport'])):
                result, values = self._tight_check(transport, snapshot[field])
                self.report['tighter_solve_checks'][field] = result
                if values is not None:
                    snapshot[field+'_tight'] = values
            self.snapshots['accuracy_probe'] = snapshot
            self._save_json()

    def finish(self, reason):
        rows = self.report['steps']
        for field in ('density', 'dye'):
            within = [row[field]['maximum_excursion'] <= self.report['production_margin'] for row in rows]
            trailing = 0
            for value in reversed(within):
                if not value:
                    break
                trailing += 1
            self.report[field+'_final_consecutive_steps_within_production_limit'] = trailing
        self.report['completed_requested_interval'] = bool(
            rows and rows[-1]['step'] == 100 and len(rows) == 50)
        super().finish(reason)
