"""Reports for the isolated 50-step rectangular scalar-bounds diagnostic.

Candidate fields and tighter re-solves are diagnostic data, not accepted
production results. No clipping, redistribution or field correction is used.
"""

import csv
import json
import math
import os
from pathlib import Path

import numpy as np


def local_integrals(values, volumes, margin=0.001):
    """Volume-weighted quantities for owned DG0 cells only."""
    values, volumes = np.asarray(values), np.asarray(volumes)
    if values.shape != volumes.shape or np.any(volumes <= 0):
        raise ValueError("Invalid owned concentration/volume arrays.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(volumes)):
        raise ValueError("Nonfinite concentration or cell volume.")
    beyond = (values < -margin) | (values > 1.0+margin)
    measurable = (values < -1e-10) | (values > 1.0+1e-10)
    return np.array([
        len(values), np.sum(volumes), np.sum(beyond), np.sum(volumes[beyond]),
        np.sum(measurable), np.sum(volumes[measurable]),
        np.dot(volumes, np.maximum(values-1.0, 0.0)),
        np.dot(volumes, np.maximum(-values, 0.0)),
        np.dot(volumes, 1.0-values),
    ], dtype=float)


def summarize(values, volumes, pycomm, mpi, margin=0.001):
    totals = pycomm.allreduce(local_integrals(values, volumes, margin), op=mpi.SUM)
    lo = pycomm.allreduce(float(np.min(values)) if len(values) else math.inf, op=mpi.MIN)
    hi = pycomm.allreduce(float(np.max(values)) if len(values) else -math.inf, op=mpi.MAX)
    excess = float(totals[6]+totals[7])
    return dict(
        minimum=lo, maximum=hi,
        maximum_excursion=max(0.0, -lo, hi-1.0),
        owned_cells_total=int(round(totals[0])), volume_m3=float(totals[1]),
        cells_beyond_production_limit=int(round(totals[2])),
        volume_beyond_production_limit_m3=float(totals[3]),
        cells_outside_0_1_by_more_than_1e_10=int(round(totals[4])),
        volume_outside_0_1_by_more_than_1e_10_m3=float(totals[5]),
        excess_above_one_m3=float(totals[6]), deficit_below_zero_m3=float(totals[7]),
        equivalent_source_volume_m3=float(totals[8]),
        out_of_bounds_amount_over_abs_net_source_volume=(
            excess/abs(totals[8]) if abs(totals[8]) > 1e-30 else None),
    )


class BoundsReport:
    def __init__(self, ns):
        from dolfin import TestFunction, assemble
        self.ns = ns
        self.comm, self.mpi, self.rank = ns['pycomm'], ns['PY_MPI'], ns['rank']
        self.coords = ns['owned_dof_coordinates'](ns['C'])
        self.volumes = assemble(TestFunction(ns['C'])*ns['dx_measure']).get_local()
        if len(self.volumes) != len(self.coords):
            raise RuntimeError("Diagnostic coordinates and DG0 volumes do not align.")
        volume = self.comm.allreduce(float(np.sum(self.volumes)), op=self.mpi.SUM)
        if not math.isclose(volume, ns['volume_mesh'], rel_tol=1e-10, abs_tol=1e-14):
            raise RuntimeError("Diagnostic cell volumes do not sum to the mesh volume.")
        self.directory = Path(ns['script_directory'])/('RETURN_rect3d_bounds_'+ns['diagnostic_job_tag'])
        self.rows, self.snapshots = [], {}
        self.peak_excursion = -1.0
        self.report = dict(
            scope='Isolated 50-step startup diagnostic; not production results',
            production_eligible=False, solver_schema=ns['solver_schema'],
            job_id=ns['diagnostic_job_tag'], mesh_level=ns['mesh_level'],
            mesh_cells=int(ns['mesh_cells_global']), dt_s=ns['dt'],
            target_time_s=ns['T_global'], density_diffusivity_m2_s=ns['D'],
            dye_diffusivity_m2_s=ns['D_dye'], production_margin=0.001,
            diagnostic_emergency_margin=ns['scalar_abort_margin'],
            first_production_limit_crossing=None, tighter_solve_checks={},
            steps=[], diagnostic_output_root=ns['OUTPUT_ROOT'],
        )
        self._root_io(lambda: self.directory.mkdir(parents=True, exist_ok=True))
        self._save_json()

    def _root_io(self, operation):
        error = None
        if self.rank == 0:
            try:
                operation()
            except Exception as exc:
                error = '{}: {}'.format(type(exc).__name__, exc)
        error = self.comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError('Bounds diagnostic write failed: '+error)

    def _save_json(self):
        def write():
            path = self.directory/'report.json'
            temp = path.with_suffix('.json.tmp')
            with temp.open('w') as stream:
                json.dump(self.report, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(temp), str(path))
        self._root_io(write)

    def _tight_check(self, transport, baseline):
        """Re-solve the SAME timestep, then restore all state used by the run."""
        from dolfin import Function
        saved_state = transport.state.vector().copy()
        statistic_names = (
            'last_iterations', 'last_true_residual', 'last_relative_residual',
            'last_recovery_used', 'recovery_count', 'solve_calls',
        )
        statistics = {name: getattr(transport, name) for name in statistic_names}
        p = transport.solver.parameters
        original_rtol, original_atol = float(p['relative_tolerance']), float(p['absolute_tolerance'])
        out = Function(transport.C)
        error, result = None, None
        try:
            p['relative_tolerance'] = min(original_rtol, 1e-12)
            p['absolute_tolerance'] = min(original_atol, 1e-16)
            iterations = transport.solve(out)
            result = dict(iterations=int(iterations), true_residual=float(transport.last_true_residual))
        except RuntimeError as exc:
            error = str(exc)
        finally:
            p['relative_tolerance'], p['absolute_tolerance'] = original_rtol, original_atol
            transport.state.vector().zero()
            transport.state.vector().axpy(1.0, saved_state)
            transport.state.vector().apply('insert')
            for name, value in statistics.items():
                setattr(transport, name, value)
        errors = self.comm.allgather(error)
        error = next((item for item in errors if item is not None), None)
        if error is not None:
            return dict(completed=False, error=error), None
        values = np.asarray(out.vector().get_local(), dtype=float)
        result.update(completed=True, rtol=min(original_rtol, 1e-12),
                      atol=min(original_atol, 1e-16),
                      concentration=summarize(values, self.volumes, self.comm, self.mpi),
                      max_abs_change_from_baseline=self.comm.allreduce(
                          float(np.max(np.abs(values-baseline))) if len(values) else 0.0,
                          op=self.mpi.MAX))
        return result, values

    def observe(self):
        ns = self.ns
        fields = dict(density=ns['c_new'].vector().get_local(), dye=ns['c_dye_new'].vector().get_local())
        row = dict(step=int(ns['step']), time_s=float(ns['t_now']))
        for name, values in fields.items():
            row[name] = summarize(values, self.volumes, self.comm, self.mpi)
        row['density_budget_relative_error'] = float(ns['budget_relative_error'])
        row['dye_budget_relative_error'] = float(ns['dye_budget_relative_error'])
        row['density_true_residual'] = float(ns['scalar_transport'].last_true_residual)
        row['dye_true_residual'] = float(ns['dye_transport'].last_true_residual)
        snapshot = dict(step=row['step'], time_s=row['time_s'],
                        density=np.array(fields['density'], copy=True), dye=np.array(fields['dye'], copy=True))
        crossing = max(row['density']['maximum_excursion'], row['dye']['maximum_excursion']) > .001
        if crossing and self.report['first_production_limit_crossing'] is None:
            self.report['first_production_limit_crossing'] = dict(step=row['step'], time_s=row['time_s'])
            self.snapshots['first_crossing'] = snapshot.copy()
            for name, transport in (('density', ns['scalar_transport']), ('dye', ns['dye_transport'])):
                result, tight_values = self._tight_check(transport, fields[name])
                self.report['tighter_solve_checks'][name] = result
                if tight_values is not None:
                    self.snapshots['first_crossing'][name+'_tight'] = tight_values
        excursion = max(row['density']['maximum_excursion'], row['dye']['maximum_excursion'])
        if excursion > self.peak_excursion:
            self.peak_excursion = excursion
            self.snapshots['peak'] = snapshot.copy()
            self.report['peak'] = row
        self.snapshots['last'] = snapshot
        self.report['steps'].append(row)
        self.report['last'] = row
        self._save_json()
        ns['root_print']('BOUNDS DIAGNOSTIC: step={}; t={:.6f}; density=[{:.9f},{:.9f}]; '
                         'dye=[{:.9f},{:.9f}]; cells beyond production limit={}/{}'.format(
                             row['step'], row['time_s'], row['density']['minimum'], row['density']['maximum'],
                             row['dye']['minimum'], row['dye']['maximum'],
                             row['density']['cells_beyond_production_limit'], row['dye']['cells_beyond_production_limit']))

    def finish(self, reason):
        """Collect diagnostic snapshots; these cannot serve as checkpoints."""
        self.report['stop_reason'] = str(reason)
        self.report['completed_50_steps'] = len(self.report['steps']) == 50
        self.report['production_eligible'] = False
        self._save_json()
        arrays = dict(coordinates_m=self.coords, cell_volumes_m3=self.volumes)
        for name, snapshot in self.snapshots.items():
            for field in ('density', 'dye', 'density_tight', 'dye_tight'):
                if field in snapshot:
                    arrays[name+'_'+field] = snapshot[field]
        parts = self.comm.gather(arrays, root=0)
        def write():
            merged = {name: np.concatenate([part[name] for part in parts], axis=0) for name in arrays}
            xyz = merged['coordinates_m']
            order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
            merged = {name: value[order] for name, value in merged.items()}
            if len(order) != self.report['mesh_cells']:
                raise RuntimeError('Diagnostic snapshot has missing or duplicated DG0 cells.')
            for name, snapshot in self.snapshots.items():
                merged[name+'_step'] = np.array(snapshot['step'])
                merged[name+'_time_s'] = np.array(snapshot['time_s'])
            np.savez_compressed(str(self.directory/'scalar_bounds_snapshots.npz'), **merged)
            with (self.directory/'bounds_history.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                keys = ('minimum', 'maximum', 'maximum_excursion', 'cells_beyond_production_limit',
                        'volume_beyond_production_limit_m3', 'excess_above_one_m3',
                        'deficit_below_zero_m3', 'equivalent_source_volume_m3')
                writer.writerow(['step', 'time_s']+[scalar+'_'+key for scalar in ('density', 'dye') for key in keys])
                for row in self.report['steps']:
                    writer.writerow([row['step'], row['time_s']]+[row[s][key] for s in ('density', 'dye') for key in keys])
            with (self.directory/'largest_excursions.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['snapshot', 'step', 'time_s', 'scalar', 'x_m', 'y_m', 'z_m', 'cell_volume_m3', 'concentration'])
                for name, snapshot in self.snapshots.items():
                    for scalar in ('density', 'dye'):
                        values = merged[name+'_'+scalar]
                        excursion = np.maximum(np.maximum(values-1., -values), 0.)
                        candidates = np.flatnonzero(excursion > 1e-10)
                        candidates = candidates[np.argsort(excursion[candidates])[::-1][:500]]
                        for i in candidates:
                            writer.writerow([name, snapshot['step'], snapshot['time_s'], scalar,
                                             *merged['coordinates_m'][i], merged['cell_volumes_m3'][i], values[i]])
            (self.directory/'README.txt').write_text(
                'Diagnostic data only. Production limit: 0.001 outside [0,1].\n'
                'This isolated run used an emergency margin of 0.02 for at most 50 steps.\n'
                'Its fields are not approved thesis results or restart checkpoints.\n'
                'report.json includes the peak, duration, affected volume and tighter-solve comparison.\n'
                'scalar_bounds_snapshots.npz contains full owned-cell snapshots in coordinate order.\n'
                'largest_excursions.csv lists at most 500 cells per field and snapshot; NPZ data are complete.\n'
            )
        self._root_io(write)
        self.ns['root_print']('BOUNDS DIAGNOSTIC REPORT SAVED:', str(self.directory))
