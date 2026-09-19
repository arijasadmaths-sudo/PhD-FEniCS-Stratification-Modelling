#!/bin/bash

#SBATCH --job-name=fenics3D001_monitored
#SBATCH --account=YOUR_ACCOUNT

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1

#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G

#SBATCH --chdir=/path/to/home
#SBATCH --output=/path/to/home/fenics3D001_monitored_%j.out
#SBATCH --error=/path/to/home/fenics3D001_monitored_%j.err

# Production directory fixed to the location used by the successful
# restart diagnostic (job 18897197): /path/to/home
#
# Put this launcher there alongside the existing:
#   run3D001.py
#   make_run3D001_diagnostic.py
#   results_3D_rectangular_hdiv_restart_v3/checkpoint/
#
# Submit from ANY directory using its absolute filename:
#   sbatch /path/to/home/run3D001_monitored_fixed.sh
#
# Do not move/copy checkpoints into the verification-test folder.
# The original solver still selects the latest compatible checkpoint.
#
# This retains the diagnostic traceback/step logging but replaces the
# ten-step test cap with a maximum of 3000 NEW steps (= 3 s at dt=0.001).
# The source solver's own segment/global limits still apply if smaller.
# It does not alter the equations, mesh, dt, tolerances or preconditioner.
# It writes through the existing PRODUCTION checkpoint/output machinery.
# Do NOT run another writer to that production directory simultaneously.
# Separate verification tests must use separate output/checkpoint roots.
#
# This is a monitored continuation, not a repair for error 76. A longer
# allocation does not prevent the solver from failing before walltime.

set -eo pipefail

# Do NOT cd to SLURM_SUBMIT_DIR: that may be a verification-test folder.
# Use the known production working directory for ALL relative paths.
PRODUCTION_DIR="/path/to/home"
cd "$PRODUCTION_DIR" || {
    printf 'ERROR: Cannot access production directory: %s\n' "$PRODUCTION_DIR" >&2
    exit 1
}

# Preserve the environment setup and MPI command of the successful test.
# No new thread, MPI-backend or PETSc solver-option overrides are added.
module purge
module add languages/python/fenics-2019.1.0

# Explicitly restore the production directory after loading the module.
cd "$PRODUCTION_DIR"

MAX_NEW_STEPS=3000
MONITORED_SCRIPT="run3D001_monitored_job_${SLURM_JOB_ID}.py"

printf '%s\n' '========================================'
printf '%s\n' '3D production continuation -- detailed PETSc error reporting retained'
printf 'Job ID: %s\n' "$SLURM_JOB_ID"
printf 'Nodes: %s\n' "$SLURM_JOB_NUM_NODES"
printf 'Tasks: %s\n' "$SLURM_NTASKS"
printf 'Submission directory: %s\n' "${SLURM_SUBMIT_DIR:-not reported}"
printf 'Production working directory: %s\n' "$PWD"
printf 'Python: %s\n' "$(command -v python3)"
printf 'Maximum new steps: %s (3 physical seconds at dt=0.001)\n' "$MAX_NEW_STEPS"
printf 'Start: %s\n' "$(date)"
printf '%s\n' '========================================'
module list

# Build a NEW job-specific monitored copy from the actual production file.
# Never reuse or edit the old ten-step run3D001_diagnostic.py, and never
# overwrite run3D001.py. This is a serial text patch, not a FEniCS solve.
python3 - "$MAX_NEW_STEPS" "$MONITORED_SCRIPT" <<'PY'
import ast
import hashlib
import runpy
import sys
from pathlib import Path

try:
    max_steps = int(sys.argv[1])
    destination = Path(sys.argv[2])
    if max_steps < 1:
        raise ValueError("MAX_NEW_STEPS must be positive.")

    production_dir = Path.cwd().resolve()
    source = production_dir / "run3D001.py"
    patcher = production_dir / "make_run3D001_diagnostic.py"
    print("Checking production source:", source, flush=True)
    print("Checking diagnostic patcher:", patcher, flush=True)
    if not source.is_file() or not patcher.is_file():
        raise FileNotFoundError(
            "Required source or patcher missing in {}. Keep run3D001.py "
            "and make_run3D001_diagnostic.py in the production directory."
            .format(production_dir)
        )

    raw = source.read_bytes()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    tree = ast.parse(text)
    output_nodes = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "OUTPUT_ROOT"
                for t in node.targets)
    ]
    expected_root = "results_3D_rectangular_hdiv_restart_v3"
    if (len(output_nodes) != 1 or
            ast.literal_eval(output_nodes[0].value) != expected_root):
        raise ValueError(
            "Expected OUTPUT_ROOT = {!r}; refusing to launch a different "
            "or ambiguously configured branch.".format(expected_root)
        )

    checkpoint_dir = production_dir / expected_root / "checkpoint"
    print("Checking checkpoint directory:", checkpoint_dir, flush=True)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            "Production checkpoint directory is missing or inaccessible: {}. "
            "No solver was launched. Do not create an empty replacement."
            .format(checkpoint_dir)
        )
    checkpoints = sorted(
        p for p in checkpoint_dir.glob("state_step_*.npz") if p.is_file()
    )
    if not checkpoints:
        entries = sorted(p.name for p in checkpoint_dir.iterdir())[:20]
        raise FileNotFoundError(
            "No state_step_*.npz files found in {}. Directory entries "
            "(first 20): {}. Refusing a fresh run. Verify where the "
            "production checkpoints are stored before resubmitting."
            .format(checkpoint_dir, entries)
        )
    print("Checkpoint files found:", len(checkpoints), flush=True)
    print("Highest numbered checkpoint file:", checkpoints[-1], flush=True)
    # Presence is not a validity test. The original solver still selects
    # and validates the actual restart, including its state fingerprint.

    helpers = runpy.run_path(str(patcher))
    build = helpers.get("build_diagnostic")
    if not callable(build):
        raise ValueError("The diagnostic patcher does not contain build_diagnostic.")
    revised = build(text, max_steps)  # Also checks dt=0.001 and expected structure.
    compile(revised, str(destination), "exec")

    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(revised)
    print("Created monitored copy:", destination.resolve(), flush=True)
    print("Original unchanged:", source.resolve(), flush=True)
    print("Original SHA256:", hashlib.sha256(raw).hexdigest(), flush=True)
    print("Maximum new steps:", max_steps, "(dt remains 0.001)", flush=True)
    print("Production checkpoint directory:", checkpoint_dir.resolve(), flush=True)
except Exception as exc:
    print("Preparation failed; solver NOT launched:", exc, file=sys.stderr, flush=True)
    sys.exit(1)
PY

printf '\nLaunching %s using the production MPI command.\n' "$MONITORED_SCRIPT"

# Preserve a nonzero solver/launcher status; a final printf must not turn
# a failed simulation into an apparently successful Slurm batch job.
set +e
srun --mpi=pmi2 python3 -u "$MONITORED_SCRIPT"
run_status=$?
set -e

printf '\n%s\n' '========================================'
printf 'Finished: %s\n' "$(date)"
printf 'srun exit code: %s\n' "$run_status"
if [[ "$run_status" -eq 0 ]]; then
    printf '%s\n' 'Monitored segment exited normally; inspect the final checkpoint time.'
else
    printf '%s\n' 'Monitored run failed; retain BOTH the .out and .err files.'
    printf '%s\n' 'Do not infer the cause from error code 76 alone.'
fi
printf '%s\n' '========================================'
exit "$run_status"
