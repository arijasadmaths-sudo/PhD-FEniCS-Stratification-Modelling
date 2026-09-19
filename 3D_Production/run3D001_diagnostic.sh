#!/bin/bash

#SBATCH --job-name=fenics3D001_diagnostic
#SBATCH --account=SEMT035584

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1

#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G

#SBATCH --output=fenics3D001_diagnostic_%j.out
#SBATCH --error=fenics3D001_diagnostic_%j.err

# Submit from the directory containing run3D001.py and either:
#   run3D001_diagnostic.py (already created), or
#   make_run3D001_diagnostic.py (to create it automatically).
#
# This is a short diagnostic run, NOT a replacement production solver.
# It uses the production checkpoint directory. Do not run it alongside
# another job writing to that same directory.

set -eo pipefail

# Keep the same FEniCS module and MPI launcher as the production job.
module purge
module add languages/python/fenics-2019.1.0

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from your production directory.}"

printf '%s\n' '========================================'
printf '%s\n' '3D production restart -- PETSc error-76 diagnostic'
printf 'Job ID: %s\n' "$SLURM_JOB_ID"
printf 'Nodes: %s\n' "$SLURM_JOB_NUM_NODES"
printf 'Tasks: %s\n' "$SLURM_NTASKS"
printf 'Working directory: %s\n' "$PWD"
printf 'Python: %s\n' "$(command -v python3)"
printf 'Start: %s\n' "$(date)"
printf '%s\n' '========================================'

module list

# Do not overwrite a diagnostic script that has already been created.
if [[ ! -f run3D001_diagnostic.py ]]; then
    if [[ ! -f run3D001.py || ! -f make_run3D001_diagnostic.py ]]; then
        printf '%s\n' \
            'ERROR: run3D001_diagnostic.py is missing.' \
            'Place run3D001.py and make_run3D001_diagnostic.py in this directory,' \
            'or create run3D001_diagnostic.py with the patcher before submitting.' >&2
        exit 1
    fi
    python3 make_run3D001_diagnostic.py run3D001.py --max-steps 10
fi

# Syntax check only: this does not import DOLFIN or run the simulation.
python3 -m py_compile run3D001_diagnostic.py

printf '\n%s\n' 'Launching run3D001_diagnostic.py using the production MPI command.'

# Capture srun's exit status so a solver failure is not hidden by the
# final echo/printf. -u flushes Python output for readable diagnostics.
set +e
srun --mpi=pmi2 python3 -u run3D001_diagnostic.py
run_status=$?
set -e

printf '\n%s\n' '========================================'
printf 'Finished: %s\n' "$(date)"
printf 'srun exit code: %s\n' "$run_status"
if [[ "$run_status" -eq 0 ]]; then
    printf '%s\n' 'Diagnostic script exited normally. Review the .out and .err files.'
else
    printf '%s\n' 'Diagnostic run failed. The detailed error is in the .out/.err files.'
fi
printf '%s\n' '========================================'

exit "$run_status"
