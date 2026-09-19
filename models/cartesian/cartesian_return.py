#!/usr/bin/env python3
"""Checkpoint preflight and flat, cumulative RETURN packaging.

Packaging uses the Python standard library even if numerical imports failed.
"""
import argparse
import ast
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import zipfile

DT = 0.0005
STEPS = 10000
SOURCE = 'cartesian_half_step_5s.py'
SOURCES = [SOURCE, 'cartesian_return.py', 'cartesian_resume_driver.py',
           'run_cartesian_half_step_5s.sh', 'README.md']
TARGETS = {1000: '0.5', 2000: '1', 4000: '2', 6000: '3', 8000: '4', 10000: '5'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def npz_scalar(path, name):
    """Read numeric NPZ metadata without importing NumPy or unpickling."""
    with zipfile.ZipFile(str(path)) as archive:
        with archive.open(name+'.npy') as handle:
            if handle.read(6) != b'\x93NUMPY':
                raise ValueError('Invalid NPY magic')
            version = tuple(handle.read(2))
            if version == (1, 0):
                size = struct.unpack('<H', handle.read(2))[0]
            elif version in ((2, 0), (3, 0)):
                size = struct.unpack('<I', handle.read(4))[0]
            else:
                raise ValueError('Unsupported NPY version')
            header = ast.literal_eval(handle.read(size).decode('utf8' if version == (3, 0) else 'latin1'))
            if header['shape'] not in ((), (1,)):
                raise ValueError('Expected scalar: '+name)
            match = re.fullmatch(r'([<>|=])([fiu])(\d+)', header['descr'])
            if not match:
                raise ValueError('Non-numeric scalar: '+name)
            order, kind, width = match.groups()
            width = int(width)
            codes = {('f', 4): 'f', ('f', 8): 'd', ('i', 1): 'b', ('i', 2): 'h',
                     ('i', 4): 'i', ('i', 8): 'q', ('u', 1): 'B', ('u', 2): 'H',
                     ('u', 4): 'I', ('u', 8): 'Q'}
            value = struct.unpack(('=' if order == '|' else order)+codes[(kind, width)], handle.read(width))[0]
            if not math.isfinite(value):
                raise ValueError('Non-finite metadata')
            return value


def saved_step(path):
    step, time = npz_scalar(path, 'step'), npz_scalar(path, 'time_s')
    if int(step) != step or step < 0 or step > STEPS or abs(time-step*DT) > 1e-12:
        raise ValueError('Invalid half-step snapshot/checkpoint time: '+str(path))
    return int(step)


def prepare(args):
    """Refuse incompatible checkpoints before the expensive FE setup."""
    import importlib.util
    import numpy as np
    root, checkpoint = Path(args.root), Path(args.checkpoint).resolve()
    source = root/SOURCE
    spec = importlib.util.spec_from_file_location('half_step_solver', str(source))
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    with np.load(str(checkpoint), allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    meta = json.loads(str(state['metadata_json']))
    sha = digest(source)
    fingerprint = hashlib.sha256((sha+'|'+args.case+'|dt=.0005|T=5').encode()).hexdigest()
    if meta.get('source_sha256') != sha or meta.get('dt_s') != DT:
        raise ValueError('Checkpoint source or dt differs. Do not use the dt=0.001 checkpoint.')
    grid = solver.BalancedFluxGrid(*solver.make_graded_grid(solver.CASE_H[args.case]))
    nx, ny = grid.nx, grid.ny
    if state['mesh_cells'].shape != (2*nx*ny, 3):
        raise ValueError('Incorrect triangle count')
    if state['u_dofs'].size != 2*(2*nx+1)*(2*ny+1) or state['p_dofs'].size != (nx+1)*(ny+1):
        raise ValueError('Incorrect FE DOF counts')
    for key in ('velocity_dof_xy_m', 'pressure_dof_xy_m', 'mesh_coordinates_m'):
        a = state[key]
        if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
            raise ValueError('Invalid coordinate array: '+key)
    step, cumulative = solver.validate_restart(state, args.case, fingerprint, grid,
        state['velocity_dof_xy_m'], state['pressure_dof_xy_m'],
        state['mesh_coordinates_m'], state['mesh_cells'])
    wall = float(state['cumulative_wall_s'])
    if not np.isfinite(wall) or wall < 0:
        raise ValueError('Invalid cumulative wall time')
    # Resume from this launcher's return folder so earlier accepted CSV rows
    # and physical-time snapshots are carried forward without nested archives.
    prior = checkpoint.parent
    info = json.loads((prior/'launcher_status.json').read_text())
    if info.get('case') != args.case or info.get('dt_s') != DT or info.get('source_sha256', {}).get(SOURCE) != sha:
        raise ValueError('Return folder provenance does not match checkpoint')
    if info.get('checkpoint_step') != step or info.get('checkpoint_sha256') != digest(checkpoint):
        raise ValueError('Checkpoint differs from this return folder manifest')
    if not (prior/'history').is_dir():
        raise ValueError('Return folder is missing its CSV history; preserve the whole return folder')
    shutil.copytree(str(prior/'history'), str(root/'history_input'))
    if (prior/'snapshots').is_dir():
        shutil.copytree(str(prior/'snapshots'), str(root/'snapshots_input'))
    shutil.copy2(str(checkpoint), str(root/'checkpoint_input.npz'))
    write_json(root/'checkpoint_preflight.json', {
        'check': 'passed', 'case': args.case, 'dt_s': DT, 'source_sha256': sha,
        'checkpoint_sha256': digest(checkpoint), 'checkpoint_step': step,
        'checkpoint_time_s': step*DT, 'remaining_steps': STEPS-step,
        'input_return_folder': str(prior), 'history_segments': len(list((root/'history_input').iterdir())),
        'limitation': 'The production solver also regenerates and checks exact DOLFIN mesh and DOF ordering.'})
    print('Checkpoint preflight passed:', args.case, 'step', step, 'time', step*DT)


def pack(args):
    root, submit = Path(args.root), Path(args.submit)
    sim = root/'simulation'
    identity = args.array+'_'+args.task
    name = 'cartesian_'+args.case+'_half_RETURN_'+identity
    out, zip_path = submit/name, submit/(name+'.zip')
    if out.exists() or zip_path.exists():
        raise ValueError('Refusing to overwrite RETURN files')
    out.mkdir()
    missing, errors = [], []
    def copy(source, dest):
        if source.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(dest))
            return True
        missing.append(str(source.relative_to(root)))
        return False
    for source in SOURCES:
        copy(root/source, out/source)
    if (root/'reference_dt001').is_dir():
        shutil.copytree(str(root/'reference_dt001'), str(out/'reference_dt001'))
    for filename in ('run.log', 'checkpoint_preflight.json'):
        if (root/filename).is_file():
            copy(root/filename, out/filename)
    for filename in ('budget_configuration.json', 'run_status.json', 'failed_candidate.npz'):
        if filename == 'failed_candidate.npz' and not (sim/filename).is_file():
            continue
        copy(sim/filename, out/filename)
    status = None
    try:
        status = json.loads((out/'run_status.json').read_text())
    except (OSError, ValueError) as exc:
        errors.append('Could not read solver status: '+str(exc))
    history = out/'history'
    if (root/'history_input').is_dir():
        shutil.copytree(str(root/'history_input'), str(history))
    else:
        history.mkdir()
    segment = history/identity
    segment.mkdir()
    for filename in ('budget_configuration.json', 'run_status.json', 'scalar_budget.csv'):
        copy(sim/filename, segment/filename)
    copy(root/'run.log', segment/'run.log')
    if (root/'checkpoint_preflight.json').is_file():
        copy(root/'checkpoint_preflight.json', segment/'checkpoint_preflight.json')
    write_json(segment/'allocation.json', {'array_id': args.array, 'task_id': args.task, 'job_id': args.job,
        'workload_exit_code': args.rc, 'stage': args.stage, 'resume_input': args.resume or None,
        'source_sha256': {p: digest(root/p) for p in SOURCES if (root/p).is_file()}})
    checkpoint_step = None
    checkpoint = sim/'checkpoint_latest.npz'
    if checkpoint.is_file():
        try:
            checkpoint_step = saved_step(checkpoint)
            copy(checkpoint, out/checkpoint.name)
        except Exception as exc:
            errors.append('Checkpoint metadata check failed: '+str(exc))
    else:
        missing.append('simulation/checkpoint_latest.npz')
    accepted_step = status.get('last_completed_step') if status else None
    accepted_time = status.get('last_completed_time_s') if status else None
    if accepted_step is not None and abs(float(accepted_time)-DT*int(accepted_step)) > 1e-12:
        errors.append('Solver status step/time mismatch')
    # Complete segment CSVs remain immutable; the combined file contains only
    # accepted rows through the newest accepted checkpoint. Never fill gaps.
    rows, columns = {}, None
    for path in sorted(history.glob('*/scalar_budget.csv')):
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            if columns is None:
                columns = reader.fieldnames
            elif columns != reader.fieldnames:
                raise ValueError('CSV schema changed across allocations')
            for row in reader:
                step = int(row['step'])
                if row.get('accepted') != '1' or checkpoint_step is None or step > checkpoint_step:
                    continue
                if abs(float(row['time_s'])-step*DT) > 1e-12:
                    raise ValueError('CSV row has an incompatible time step')
                if step in rows:
                    # Recomputed overlap can differ in timing. Later segment
                    # wins; original rows remain in the immutable segment files.
                    pass
                rows[step] = row
    if columns:
        with (out/'scalar_budget.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows[s] for s in sorted(rows))
    gaps = []
    previous = 0
    for step in sorted(rows):
        if step > previous+1:
            gaps.append([previous+1, step-1])
        previous = step
    if checkpoint_step and previous < checkpoint_step:
        gaps.append([previous+1, checkpoint_step])
    snapshots = out/'snapshots'
    snapshots.mkdir()
    candidates = []
    if (root/'snapshots_input').is_dir():
        candidates += sorted((root/'snapshots_input').glob('*.npz'))
    if sim.is_dir():
        candidates += sorted(sim.glob('fv_step_*.npz'))
        if (sim/'fv_final.npz').is_file():
            candidates.append(sim/'fv_final.npz')
    chosen = {}
    for path in candidates:
        try:
            step = saved_step(path)
        except Exception as exc:
            errors.append('Snapshot metadata rejected: '+path.name+': '+str(exc))
            continue
        if checkpoint_step is not None and step <= checkpoint_step and (step in TARGETS or step == accepted_step):
            chosen[step] = path
    target_map = {}
    for step, path in sorted(chosen.items()):
        dest = snapshots/('fv_step_%07d.npz' % step)
        shutil.copy2(str(path), str(dest))
        if step in TARGETS:
            target_map[TARGETS[step]] = str(dest.relative_to(out))
    selected_snapshot = None
    if accepted_step in chosen:
        selected_snapshot = 'fv_final.npz' if accepted_step == STEPS else 'fv_accepted.npz'
        shutil.copy2(str(chosen[accepted_step]), str(out/selected_snapshot))
    else:
        missing.append('FV snapshot matching the last accepted status')
    missing_times = [text for step, text in TARGETS.items() if checkpoint_step is not None and step <= checkpoint_step and step not in chosen]
    if checkpoint_step != accepted_step:
        errors.append('Checkpoint step differs from last accepted status; use its own metadata for restart')
    if args.rc == 0 and (not status or status.get('status') != 'completed' or checkpoint_step != STEPS):
        errors.append('Exit zero did not produce a complete 5 s checkpoint')
    metadata = {'case': args.case, 'dt_s': DT, 'target_time_s': 5., 'target_steps': STEPS,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'slurm_job_id': args.job, 'slurm_array_job_id': args.array, 'slurm_array_task_id': args.task,
        'workload_exit_code': args.rc, 'log_capture_exit_code': args.log_rc, 'last_workload_stage': args.stage,
        'work_directory': str(root), 'simulation_status': status, 'resume_input': args.resume or None,
        'checkpoint_step': checkpoint_step, 'checkpoint_sha256': digest(out/checkpoint.name) if (out/checkpoint.name).is_file() else None,
        'source_sha256': {p: digest(out/p) for p in SOURCES if (out/p).is_file()},
        'accepted_csv_rows': len(rows), 'accepted_csv_missing_step_ranges': gaps,
        'history_segment_count': len(list(history.iterdir())), 'selected_snapshot': selected_snapshot,
        'physical_time_snapshots_s': target_map, 'missing_elapsed_snapshots_s': missing_times,
        'missing_files': missing, 'errors': errors}
    write_json(out/'launcher_status.json', metadata)
    (out/'RETURN_README.txt').write_text(
        'Inspect launcher_status.json and run_status.json. Exit 75 means an expected accepted-step pause.\n'
        'scalar_budget.csv combines available accepted rows; missing ranges are reported, never invented.\n'
        'history/ preserves original CSV segments and allocation records without nested checkpoints.\n'
        'Keep this whole folder for restart. Only checkpoint_latest.npz is a restart state.\n'
        'Fresh and resume commands are in README.md. Full FE outputs remain at '+str(root)+'\n')
    fd, temporary = tempfile.mkstemp(prefix='.'+name+'_', suffix='.zip.tmp', dir=str(submit))
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in sorted(out.rglob('*')):
                if path.is_file():
                    archive.write(str(path), name+'/'+str(path.relative_to(out)))
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError('ZIP verification failed: '+bad)
        os.link(temporary, str(zip_path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print('Return folder:', out)
    print('Return ZIP:', zip_path)
    print('Accepted checkpoint step:', checkpoint_step, 'CSV rows:', len(rows), 'gaps:', gaps)
    if errors or missing:
        print('Packaging observations:', errors, missing)
    if args.rc == 0 and (errors or missing or gaps or missing_times):
        raise RuntimeError('Completed workload has missing or inconsistent return data; ZIP saved for diagnosis')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action')
    pre = sub.add_parser('prepare')
    pre.add_argument('--checkpoint', required=True)
    pre.add_argument('--root', required=True)
    pre.add_argument('--case', choices=['compact312p5', 'compact156p25'], required=True)
    pkg = sub.add_parser('pack')
    for name in ('root', 'submit', 'case', 'job', 'array', 'task', 'stage'):
        pkg.add_argument('--'+name, required=True)
    pkg.add_argument('--rc', type=int, required=True)
    pkg.add_argument('--log-rc', type=int, required=True)
    pkg.add_argument('--resume', default='')
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'pack':
        pack(args)
    else:
        p.error('Select prepare or pack')


if __name__ == '__main__':
    main()
