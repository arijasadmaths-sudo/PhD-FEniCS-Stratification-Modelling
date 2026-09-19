#!/bin/bash
#SBATCH --job-name=rect3d_trial
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=rectangular_plume_3d_trial_%j.out
#SBATCH --error=rectangular_plume_3d_trial_%j.err
#SBATCH --export=ALL

# Put BOTH files in the same directory, cd there, then submit:
#   sbatch submit_rectangular_plume_3d_trial.sh
# Re-submit the same command AFTER the previous job ends to resume this trial.
# No automatic job submission or changes to the existing 3D run are performed.
# The cluster's default partition/account/memory allocation is used. Copy any
# required #SBATCH partition/account/memory directives from your working job
# into the header above, before the first executable line.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "Submit this file with sbatch from the directory containing both files." >&2
    exit 2
fi

# Slurm executes a spooled copy of this script. BASH_SOURCE is NOT the location
# of the original Python file. SLURM_SUBMIT_DIR is the submission directory:
# https://slurm.schedmd.com/sbatch.html#OPT_SLURM_SUBMIT_DIR
cd -- "${SLURM_SUBMIT_DIR}"
solver_path="${SLURM_SUBMIT_DIR}/rectangular_plume_3d_trial.py"
if [[ ! -f "${solver_path}" ]]; then
    echo "Cannot find ${solver_path}. Put both files together and cd there before sbatch." >&2
    exit 2
fi

# ================= YOUR EXISTING FEniCS ENVIRONMENT =================
# If FEniCS is not already available in the submitted environment, paste the
# SAME module/virtualenv activation commands as your successful 3D job here.
# No module name, partition or installation path has been guessed.
# If your working job launches inside a container, copy its container command
# into BOTH srun calls below; the preflight and solver must use one environment.
# ==================================================================

python_bin="${FENICS_PYTHON:-python3}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL

export FLOW_RATE_CC_MIN="${FLOW_RATE_CC_MIN:-20}"
export T_GLOBAL="${T_GLOBAL:-20.0}"
export SEGMENT_DURATION="${SEGMENT_DURATION:-0.5}"
export INLET_PERTURBATION="${INLET_PERTURBATION:-0.0}"
export CHECKPOINT_WALL_SECONDS="${CHECKPOINT_WALL_SECONDS:-600}"
# Best-effort stop after an accepted step, leaving 30 min of the 2 h allocation
# for output/checkpoint work. A single long solve can still exceed the margin.
# Adjust this if you change #SBATCH --time. Zero disables the soft stop.
export WALL_TIME_BUDGET_SECONDS="${WALL_TIME_BUDGET_SECONDS:-5400}"

# Uses results_rectangular_plume_3d_trial_Q20ccmin_eps0_dt0p001 by default.
# Other flow/perturbation/dt settings select separate trial directories.
# For another independent trial with identical parameters, explicitly set
# RECT3D_OUTPUT_ROOT to a NEW directory. Never point it at your production run.

echo "Rectangular plume 3D trial | job ${SLURM_JOB_ID} | ranks ${SLURM_NTASKS}"
echo "Box: 0.800 x 0.400 x 0.450 m; distributed floor return (approximation)"
echo "Q=${FLOW_RATE_CC_MIN} cc/min; target=${T_GLOBAL} s; segment<=${SEGMENT_DURATION} s"
echo "DT=${DT:-automatic}; soft wall budget=${WALL_TIME_BUDGET_SECONDS} s"

# Fail early if the job cannot import the legacy FEniCS stack.
if ! srun --nodes=1 --ntasks=1 --cpu-bind=cores "${python_bin}" -c \
    'import dolfin, mpi4py, petsc4py, numpy; print("DOLFIN:", dolfin.__version__); assert hasattr(dolfin.BoxMesh, "create"), "Legacy DOLFIN BoxMesh.create is required"'; then
    echo "FEniCS preflight failed. Copy the environment setup from your working 3D .sh into the marked section." >&2
    exit 2
fi

srun --nodes=1 --ntasks="${SLURM_NTASKS}" --cpu-bind=cores \
    "${python_bin}" -u "${solver_path}"

echo "Job segment finished. Check the log, then re-submit this .sh to continue."
