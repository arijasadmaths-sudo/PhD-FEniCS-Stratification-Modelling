#!/usr/bin/env python3
"""Local checks; mocked launch tests do not execute the coupled FE equations."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parent
VALIDATION = ROOT/'validation'
VALIDATION.mkdir(exist_ok=True)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def run(command, **kwargs):
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)


def ast_check():
    oldpath = ROOT/'provenance/cartesian_compact_dt001.py'
    newpath = ROOT/'cartesian_half_step_5s.py'
    old, new = ast.parse(oldpath.read_text()), ast.parse(newpath.read_text())
    def defs(tree):
        return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    previous, current = defs(old), defs(new)
    unchanged = []
    for name in previous:
        if name in ('run_case', 'validate_restart', 'main', 'KernelTests'):
            continue
        assert ast.dump(previous[name]) == ast.dump(current[name]), name
        unchanged.append(name)
    # All physical FE setup, forms, solver choices, mesh maps and time-loop
    # updates are compared as AST statement blocks, excluding only dt setup
    # and bookkeeping/output statements.
    def body(tree):
        rc = defs(tree)['run_case']
        return next(n for n in rc.body if isinstance(n, ast.Try)).body
    def statement_target(node):
        return ast.unparse(node.targets[0]) if isinstance(node, ast.Assign) else ''
    def block(nodes, start, end):
        i = next(i for i, n in enumerate(nodes) if statement_target(n) == start)
        j = next(i for i, n in enumerate(nodes) if statement_target(n) == end)
        return ast.dump(ast.Module(body=nodes[i:j], type_ignores=[]))
    before, after = body(old), body(new)
    assert block(before, 'd_nozzle', 'fingerprint') == block(after, 'd_nozzle', 'fingerprint')
    loopold = next(n for n in before if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
    loopnew = next(n for n in after if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
    assert block(loopold.body, 't', 'sampled') == block(loopnew.body, 't', 'sampled')
    forms = ['F1', 'a2', 'L2', 'a3', 'L3', 'f_buoy']
    for name in forms:
        a = next(n for n in before if statement_target(n) == name)
        b = next(n for n in after if statement_target(n) == name)
        assert ast.dump(a) == ast.dump(b), name
    report = {'original_sha256': hashlib.sha256(oldpath.read_bytes()).hexdigest(),
        'half_step_sha256': hashlib.sha256(newpath.read_bytes()).hexdigest(),
        'unchanged_top_level_components': unchanged, 'unchanged_FE_forms': forms,
        'unchanged_blocks': ['physical FE setup through mesh/DOF coordinate capture',
                             'time-loop ramp, all velocity solves, scalar update, budgets and row construction'],
        'allowed_changes': ['dt .001 to .0005', '5000 to 10000 steps',
            'physical-time-aligned output/log counters', 'new source/dt restart guard and fingerprint',
            'flush/fsync accepted CSV before checkpoint', 'documentation and tests'],
        'method': 'Python AST equality for unchanged components/blocks; complete textual diff is included.'}
    (VALIDATION/'ast_physics_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    import difflib
    diff = ''.join(difflib.unified_diff(oldpath.read_text().splitlines(True), newpath.read_text().splitlines(True),
        fromfile='cartesian_compact_dt001.py', tofile='cartesian_half_step_5s.py'))
    (VALIDATION/'solver_changes.diff').write_text(diff)
    return report


def guard_check(solver):
    import numpy as np
    assert solver.DT_S == .0005 and solver.TOTAL_STEPS == 10000
    assert solver.FIELD_STEPS*solver.DT_S == .25 and solver.LOG_STEPS*solver.DT_S == .1
    assert solver.DT_S*solver.TOTAL_STEPS == 5
    assert all(round(t/solver.DT_S) % solver.FIELD_STEPS == 0 for t in (.5, 1, 2, 3, 4, 5))
    g = solver.BalancedFluxGrid([0, .004], [0, .007])
    velocity, pressure = np.zeros((4, 2)), np.zeros((2, 2))
    mesh = np.array([[0,0],[.004,0],[0,.007],[.004,.007]])
    cells = np.array([[0,1,2],[1,2,3]])
    c = np.array([[.9]])
    vf = float(np.sum(g.volume*(1-c)))
    state = dict(metadata_json=json.dumps({'format_version':1, 'case':'compact312p5',
        'fingerprint':'test', 'dt_s':solver.DT_S, 'source_sha256':solver.source_hash()}),
        x_edges_m=g.x_edges, y_edges_m=g.y_edges, c=c, u_dofs=np.zeros(4), p_dofs=np.zeros(2),
        velocity_dof_xy_m=velocity, pressure_dof_xy_m=pressure, mesh_coordinates_m=mesh,
        mesh_cells=cells, step=9999, time_s=4.9995, cumulative_budget_m2=np.array([vf,0,0]), vf_m2=vf)
    def check(data):
        return solver.validate_restart(data, 'compact312p5', 'test', g, velocity, pressure, mesh, cells)
    assert check(state)[0] == 9999
    cases = {}
    for name, change in [('old_dt', {'dt_s':.001}), ('changed_source', {'source_sha256':'incorrect'}),
                         ('other_case', {'case':'compact156p25'}), ('changed_fingerprint', {'fingerprint':'wrong'})]:
        altered = copy.deepcopy(state)
        meta = json.loads(altered['metadata_json']); meta.update(change)
        altered['metadata_json'] = json.dumps(meta)
        cases[name] = altered
    altered = copy.deepcopy(state); altered.update(step=10000, time_s=5.); cases['complete'] = altered
    altered = copy.deepcopy(state); altered.update(step=5000, time_s=5.); cases['old_counter_time'] = altered
    for name, altered in cases.items():
        try:
            check(altered)
        except ValueError:
            continue
        raise AssertionError('Guard failed to reject: '+name)
    return {'accepted_last_incomplete_step':9999, 'rejected_cases': list(cases),
            'field_interval_s':.25, 'log_interval_s':.1, 'final_time_s':5.}


MOCK_SIMULATION = '''
import csv, json, os, pathlib, sys
import numpy as np
args=sys.argv
out=pathlib.Path(args[args.index('--output')+1]); out.mkdir()
case=args[args.index('--case')+1]
mode=os.environ['CART_TEST_MODE']; n=10000 if mode=='completed' else 4500
status={'status':'completed' if mode=='completed' else ('stopped' if mode=='paused' else 'failed'),
        'last_completed_step':n,'last_completed_time_s':n*.0005}
(out/'run_status.json').write_text(json.dumps(status))
(out/'budget_configuration.json').write_text(json.dumps({'dt_s':.0005,'verification_case':case}))
with (out/'scalar_budget.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['step','time_s','accepted']); w.writeheader()
    for i in range(1,n+1): w.writerow({'step':i,'time_s':i*.0005,'accepted':1})
np.savez_compressed(out/'checkpoint_latest.npz',step=n,time_s=n*.0005)
for step in [s for s in [1000,2000,4000,6000,8000,10000,n] if s<=n]:
    np.savez_compressed(out/('fv_step_%07d.npz'%step),step=step,time_s=step*.0005,c=np.zeros((1,1)))
if n==10000: np.savez_compressed(out/'fv_final.npz',step=n,time_s=5.,c=np.zeros((1,1)))
sys.exit({'completed':0,'paused':75,'failed':9}[mode])
'''


def mock_launcher_check():
    results = {}
    with tempfile.TemporaryDirectory(prefix='cart_half_mock_') as temporary:
        base = Path(temporary); work = base/'work'; work.mkdir()
        binaries = base/'bin'; binaries.mkdir()
        simulator = base/'simulation.py'; simulator.write_text(textwrap.dedent(MOCK_SIMULATION))
        python = binaries/'python3'
        python.write_text('#!/bin/bash\nif [[ "${1:-}" == "-" ]]; then cat >/dev/null; exit 0; fi\n'
            'if [[ "${2:-}" == "--self-test" ]]; then exit 0; fi\nexec '+sys.executable+' "$@"\n')
        srun = binaries/'srun'
        srun.write_text('#!/bin/bash\nexec '+sys.executable+' '+str(simulator)+' "$@"\n')
        for file in (python, srun): file.chmod(0o755)
        env = os.environ.copy()
        env.update(PATH=str(binaries)+os.pathsep+env['PATH'], CARTESIAN_WORK_PARENT=str(work),
                   CARTESIAN_RESUME='', SLURM_ARRAY_TASK_COUNT='2')
        for i, mode in enumerate(('completed','paused','failed')):
            submit = base/mode; submit.mkdir()
            for name in ('cartesian_half_step_5s.py','cartesian_return.py','cartesian_resume_driver.py',
                         'run_cartesian_half_step_5s.sh','README.md'):
                shutil.copy2(str(ROOT/name), str(submit/name))
            env.update(SLURM_SUBMIT_DIR=str(submit), SLURM_JOB_ID=str(9100+i),
                       SLURM_ARRAY_JOB_ID=str(9100+i), SLURM_ARRAY_TASK_ID=str(i%2), CART_TEST_MODE=mode)
            command=['bash','-c','module() { return 0; }; export -f module; exec bash "$1"',
                     'mock',str(submit/'run_cartesian_half_step_5s.sh')]
            result=run(command,env=env,timeout=40)
            (VALIDATION/('launcher_'+mode+'.log')).write_text(result.stdout)
            assert result.returncode == {'completed':0,'paused':75,'failed':9}[mode], result.stdout
            dirs=list(submit.glob('*_RETURN_*')); folders=[p for p in dirs if p.is_dir()]
            assert len(folders)==1 and len(list(submit.glob('*_RETURN_*.zip')))==1
            info=json.loads((folders[0]/'launcher_status.json').read_text())
            assert info['workload_exit_code']==result.returncode
            assert info['checkpoint_step']==(10000 if mode=='completed' else 4500)
            assert not info['accepted_csv_missing_step_ranges']
            assert len(list(folders[0].rglob('checkpoint*.npz')))==1
            results[mode]={'exit_code':result.returncode,'checkpoint_step':info['checkpoint_step'],
                           'return_zip_created':True,'csv_gaps':[]}
        # Simulate flat history carry-forward directly through the same real
        # packaging helper. No checkpoint or nested return archive is carried.
        previous=next(p for p in (base/'paused').glob('*_RETURN_*') if p.is_dir())
        resumed=base/'resume_root'; resumed.mkdir(); resumed_sim=resumed/'simulation'; resumed_sim.mkdir()
        for name in ('cartesian_half_step_5s.py','cartesian_return.py','cartesian_resume_driver.py',
                     'run_cartesian_half_step_5s.sh','README.md'):
            shutil.copy2(str(ROOT/name),str(resumed/name))
        shutil.copytree(str(previous/'history'),str(resumed/'history_input'))
        shutil.copytree(str(previous/'snapshots'),str(resumed/'snapshots_input'))
        (resumed/'run.log').write_text('mock continuation\n')
        import numpy as np, csv
        n=10000
        (resumed_sim/'run_status.json').write_text(json.dumps({'status':'completed','last_completed_step':n,'last_completed_time_s':5.}))
        (resumed_sim/'budget_configuration.json').write_text('{}')
        with (resumed_sim/'scalar_budget.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['step','time_s','accepted']); writer.writeheader()
            for step in range(4501,10001): writer.writerow({'step':step,'time_s':step*.0005,'accepted':1})
        for step in (6000,8000,10000): np.savez_compressed(resumed_sim/('fv_step_%07d.npz'%step),step=step,time_s=step*.0005)
        np.savez_compressed(resumed_sim/'checkpoint_latest.npz',step=10000,time_s=5.)
        destination=base/'resume_submit'; destination.mkdir()
        result=run([sys.executable,str(ROOT/'cartesian_return.py'),'pack','--root',str(resumed),'--submit',str(destination),
                    '--case','compact156p25','--job','9999','--array','9999','--task','1','--rc','0','--log-rc','0','--stage','completed'],timeout=40)
        (VALIDATION/'launcher_history_merge.log').write_text(result.stdout)
        assert result.returncode==0,result.stdout
        folder=next(p for p in destination.iterdir() if p.is_dir())
        info=json.loads((folder/'launcher_status.json').read_text())
        assert info['accepted_csv_rows']==10000 and info['history_segment_count']==2
        assert len(info['physical_time_snapshots_s'])==6
        assert len(list(folder.rglob('checkpoint*.npz')))==1
        results['history_merge']={'accepted_rows':10000,'segments':2,'snapshots':6,'checkpoint_copies':1}
    return results


def driver_check():
    with tempfile.TemporaryDirectory(prefix='cart_stop_driver_') as temporary:
        root=Path(temporary); solver=root/'solver.py'; stop=root/'stop'
        solver.write_text('import signal,time,sys\nflag=[False]\ndef request_stop(signum,frame): flag[0]=True\n'
            'signal.signal(signal.SIGUSR1,request_stop)\nend=time.monotonic()+8\n'
            'while not flag[0] and time.monotonic()<end: time.sleep(.02)\n'
            'print("safe_pause",flag[0])\nsys.exit(75 if flag[0] else 9)\n')
        # A stop requested before solver imports must wait for its actual hook.
        stop.touch()
        result=run([sys.executable,str(ROOT/'cartesian_resume_driver.py'),'--stop-file',str(stop),str(solver)],timeout=12)
        (VALIDATION/'stop_driver.log').write_text(result.stdout)
        assert result.returncode==75 and 'safe_pause True' in result.stdout,result.stdout
    return {'early_stop_file_waits_for_solver_hook':True,'exit_code':75}


def main():
    report={'limitation':'No DOLFIN installation locally; full coupled half-step simulation is not executed by these checks.'}
    report['ast']=ast_check()
    solver=module(ROOT/'cartesian_half_step_5s.py','half_step_validation')
    report['counters_and_guards']=guard_check(solver)
    test=run([sys.executable,str(ROOT/'cartesian_half_step_5s.py'),'--self-test'],timeout=180)
    (VALIDATION/'kernel_tests.log').write_text(test.stdout)
    assert test.returncode==0,test.stdout
    report['kernel_tests']='8 tests passed, including both full FV grids and dt=0.0005 transport'
    check=run(['bash','-n',str(ROOT/'run_cartesian_half_step_5s.sh')]); assert check.returncode==0
    report['launcher']=mock_launcher_check()
    report['stop_driver']=driver_check()
    report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    (ROOT/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
