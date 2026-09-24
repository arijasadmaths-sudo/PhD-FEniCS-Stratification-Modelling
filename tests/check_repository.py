#!/usr/bin/env python3
"""Local repository checks. No Slurm jobs or coupled FEniCS runs are started."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CART = ROOT / 'models/cartesian'
AXI = ROOT / 'models/axisymmetric/current'
RECT = ROOT / 'models/rectangular_3d_quantitative'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(args, cwd=None, env=None, expected=0):
    result = subprocess.run(args, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=120)
    require(result.returncode == expected,
            '{} returned {}:\n{}'.format(args, result.returncode, result.stdout))
    return result.stdout


def sources():
    return [p for p in ROOT.rglob('*') if p.is_file()
            and not any(x in p.parts for x in ('.git', '__pycache__', 'validation'))]


def layout():
    py = [p for p in sources() if p.suffix == '.py']
    shell = [p for p in sources() if p.suffix in ('.sh', '.slurm')]
    for p in py:
        compile(p.read_bytes(), str(p), 'exec')
    for p in shell:
        run(['bash', '-n', str(p)])
    launcher = (AXI/'run_axisymmetric_extra_fine_quarter_50s.sh').read_text()
    names = re.search(r'axisym_files=\(([^)]+)\)', launcher).group(1).split()
    for name in names:
        require((AXI/name).is_file(), 'Missing axisymmetric launcher input: '+name)
    require((AXI/'provenance/study_checkpoint.py').is_file(), 'Missing provenance helper')
    meta = json.loads((AXI/'SOURCE_PROVENANCE.json').read_text())
    require(meta['repository_solver_sha256'] == hashlib.sha256(
        (AXI/'axisymmetric_70_study_50s.py').read_bytes()).hexdigest(), 'Wrong current source hash')
    for folder in (RECT, ROOT/'archive/rectangular_3d_reference'):
        for name in ('rectangular_3d_h30_quantitative.py', 'mixed_scalar_transport.py',
                     'verify_mixed_scalar_transport.py', 'submit_rectangular_3d_quantitative.sh'):
            require((folder/name).is_file(), 'Missing 3D input: '+str(folder/name))
    rectangular_source = (RECT/'rectangular_3d_h30_quantitative.py').read_text()
    require('RECT3D_MONO_BLOCK_PC", "bjacobi"' in rectangular_source,
            'Later 3D source does not default to the validated block-Jacobi flow preconditioner')
    require('mono_fieldsplit_0_sub_pc_type"] = "ilu"' in rectangular_source
            and 'mono_fieldsplit_1_sub_pc_type"] = "ilu"' in rectangular_source,
            'Later 3D source is missing the local ILU block solves')
    require((ROOT/'archive/early_fenics_ipcs/IPSC_0.001_budget_diagnostic.py').is_file(),
            'Missing Cartesian budget solver')
    check = CART/'verification_and_development/verify_finish_input.py'
    tree = ast.parse(check.read_text())
    expected = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == 'EXPECTED_SOURCE')
    require(hashlib.sha256((CART/'provenance/cartesian_compact_dt001.py').read_bytes()).hexdigest()
            == expected, 'Historical Cartesian restart source changed')
    print('PASS: {} Python files, {} shell files and required bundle dependencies'.format(len(py), len(shell)))


def kernels():
    paths = [
        'models/cartesian/provenance/cartesian_compact_dt001.py',
        'models/cartesian/cartesian_half_step_5s.py',
        'models/cartesian/verification_and_development/cartesian_ipcs_conservative_5s.py',
        'models/cartesian/verification_and_development/cartesian_ipcs_verification_5s.py',
        'models/cartesian/verification_and_development/cartesian_ipcs_mesh_480x240_5s.py',
        'models/cartesian/verification_and_development/cartesian_focused_5s.py',
        'models/cartesian/verification_and_development/cartesian_focused_next_5s.py',
        'models/axisymmetric/current/axisymmetric_70_study_50s.py',
        'models/axisymmetric/current/provenance/axisymmetric_70_study_50s.py',
        'models/axisymmetric/verification/axisymmetric_70_flux_2s.py',
        'models/axisymmetric/verification/axisymmetric_70_fine_flux_2s.py',
        'models/axisymmetric/verification/axisymmetric_70_verify_50s.py',
        'models/rectangular_3d_quantitative/analyse_rectangular_3d_quantitative.py',
    ]
    for rel in paths:
        run([sys.executable, str(ROOT/rel), '--self-test'])
        print('PASS: '+rel, flush=True)
    with tempfile.TemporaryDirectory() as d:
        report = Path(d)/'underflow.json'
        run([sys.executable, str(AXI/'check_underflow.py'), '--output', str(report)])
        require(json.loads(report.read_text())['status'] == 'passed', 'Underflow regression failed')
    print('PASS: axisymmetric underflow regression')


def launchers():
    with tempfile.TemporaryDirectory() as d:
        temp = Path(d)
        cart = temp/'cartesian'
        shutil.copytree(CART, cart, ignore=shutil.ignore_patterns('__pycache__', 'validation', 'validation_report.json'))
        run([sys.executable, str(cart/'validate_bundle.py')], cwd=cart)
        print('PASS: Cartesian solver identity, restart and mocked return packaging', flush=True)
        fake = temp/'bin'
        fake.mkdir()
        script = fake/'sbatch'
        script.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$MOCK_SBATCH_LOG"\necho 100001\n')
        script.chmod(0o755)
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(('SLURM_', 'RECT3D_', 'BASH_FUNC_')) or key in (
                    'SEED_CHECKPOINT', 'MESH_LEVEL', 'DT', 'T_GLOBAL', 'PIN_CHECKPOINT_TIME',
                    'FLOW_RATE_CC_MIN', 'INLET_PERTURBATION', 'SCALAR_DIFFUSIVITY', 'DYE_DIFFUSIVITY', 'BASH_ENV'):
                env.pop(key)
        env['PATH'] = str(fake)+os.pathsep+env.get('PATH', '')
        calls = temp/'submissions.txt'
        env['MOCK_SBATCH_LOG'] = str(calls)
        launcher = RECT/'submit_rectangular_3d_quantitative.sh'
        run(['bash', str(launcher), '--chain', '1'], env=env)
        require(len(calls.read_text().splitlines()) == 1, 'Wrong single-chain submission count')
        calls.unlink()
        run(['bash', str(launcher), '--study', '1'], env=env)
        require(len(calls.read_text().splitlines()) == 4, 'Wrong bounded-study submission count')
        calls.unlink()
        env['T_GLOBAL'] = '20'
        run(['bash', str(launcher), '--study', '1'], env=env, expected=1)
        require(not calls.exists(), 'Invalid study reached sbatch')
        print('PASS: mocked 3D chain, study and short-target rejection', flush=True)
        module = fake/'module'
        module.write_text('#!/bin/sh\nexit 0\n')
        module.chmod(0o755)
        srun = fake/'srun'
        srun.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$MOCK_SRUN_LOG"\n')
        srun.chmod(0o755)
        srun_log = temp/'srun.txt'
        env.update(SLURM_JOB_ID='990000', SLURM_SUBMIT_DIR=str(CART/'verification_and_development'),
                   CARTESIAN_WORK_PARENT=str(temp), MOCK_SRUN_LOG=str(srun_log))
        run(['bash', str(CART/'verification_and_development/run_cartesian_budget.sh')], env=env)
        solver = Path(srun_log.read_text().splitlines()[-1]).resolve()
        require(solver == ROOT/'archive/early_fenics_ipcs/IPSC_0.001_budget_diagnostic.py',
                'Budget launcher selected the wrong solver')
        print('PASS: Cartesian budget launcher resolves the archived solver')
        # Stop at environment setup, after all real bundle files are staged.
        axi = temp/'axisymmetric'
        shutil.copytree(AXI, axi, ignore=shutil.ignore_patterns('__pycache__'))
        work = temp/'work'
        work.mkdir()
        module = fake/'module'
        module.write_text('#!/bin/sh\nexit 17\n')
        module.chmod(0o755)
        python3 = fake/'python3'
        python3.symlink_to(sys.executable)
        env.update(SLURM_JOB_ID='990001', SLURM_SUBMIT_DIR=str(axi), AXISYM_WORK_PARENT=str(work))
        output = run(['bash', str(axi/'run_axisymmetric_extra_fine_quarter_50s.sh')],
                     cwd=axi, env=env, expected=17)
        require('stage: module_setup' in output, 'Axisymmetric launcher failed before environment setup')
        staged = work/'axisymmetric50_extra_fine_quarter_990001'
        require((staged/'README.txt').is_file() and (staged/'SOURCE_PROVENANCE.json').is_file(),
                'Axisymmetric sources were not staged')
        print('PASS: axisymmetric source staging and environment-failure return path')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('all', 'layout', 'kernels', 'launchers'), default='all')
    args = parser.parse_args()
    for name, fn in [('layout', layout), ('kernels', kernels), ('launchers', launchers)]:
        if args.group in ('all', name):
            fn()
    print('Checks passed. Coupled FEniCS/MPI and MATLAB execution are not covered.')


if __name__ == '__main__':
    main()
