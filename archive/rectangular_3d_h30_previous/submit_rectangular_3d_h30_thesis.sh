#!/bin/bash -l
#SBATCH --job-name=rect3d_h30_thesis
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@1800
#SBATCH --export=ALL
#SBATCH --output=rect3d_h30_thesis_%j.out
#SBATCH --error=rect3d_h30_thesis_%j.err

# Put this file beside rectangular_3d_h30_thesis.py, then submit from that directory:
#   sbatch submit_rectangular_3d_h30_thesis.sh
# To queue three consecutive allocations with one command (explicit opt-in):
#   bash submit_rectangular_3d_h30_thesis.sh --chain 3
# Append your usual options, e.g. --account=CODE --partition=YOUR_PARTITION.
# Add the SAME account/project, partition and memory options used by your working
# BluePebble job. For example, pass your existing project with --account=CODE.
# The 48-hour request is configurable; your partition's limit still applies.
# Re-submit the same script to continue the saved simulation towards T_GLOBAL.
# Successors are queued only in --chain mode; earlier 3D results are not reused.

set -eo pipefail

if [[ "${1:-}" == "--chain" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Use --chain from the login node, not from inside an existing job." >&2
        exit 2
    fi
    if [[ $# -lt 2 || ! "${2}" =~ ^([1-9]|10)$ ]]; then
        echo "Usage: bash $0 --chain N [--account=CODE] [--partition=NAME] [--time=HH:MM:SS] [--mem=SIZE]; N must be 1 to 10." >&2
        exit 2
    fi
    chain_length="${2}"
    shift 2
    chain_options=()
    for chain_option in "$@"; do
        case "${chain_option}" in
            --account=*|--partition=*|--time=*|--mem=*|--mem-per-cpu=*|--qos=*)
                chain_options+=("${chain_option}") ;;
            *)
                echo "Unsupported --chain option: ${chain_option}. Use the --name=value form for account, partition, time, memory or qos." >&2
                exit 2 ;;
        esac
    done
    # Here BASH_SOURCE is the original file: this branch runs before submission.
    launcher_path="$(readlink -f -- "${BASH_SOURCE[0]}")"
    chain_dir="$(dirname -- "${launcher_path}")"
    if [[ ! -f "${chain_dir}/rectangular_3d_h30_thesis.py" ]]; then
        echo "Put the Python file beside this script before submitting the chain." >&2
        exit 2
    fi
    export SIMULATION_DIR="${chain_dir}"
    previous_job=""
    for ((chain_index=1; chain_index<=chain_length; chain_index++)); do
        dependency_options=()
        if [[ -n "${previous_job}" ]]; then
            dependency_options=("--dependency=afterok:${previous_job}" "--kill-on-invalid-dep=yes")
        fi
        if ! submission="$(sbatch --parsable --export=ALL "${chain_options[@]}" --chdir="${chain_dir}" "${dependency_options[@]}" "${launcher_path}")"; then
            echo "Submission stopped at job ${chain_index}. Previously printed job IDs remain submitted." >&2
            exit 1
        fi
        previous_job="${submission%%;*}"
        if [[ ! "${previous_job}" =~ ^[0-9]+$ ]]; then
            echo "Unexpected sbatch response: ${submission}. Check squeue before submitting again." >&2
            exit 1
        fi
        echo "Queued allocation ${chain_index}/${chain_length}: job ${previous_job}"
    done
    echo "Each allocation resumes the saved run. Later jobs start only after successful completion of the preceding job."
    echo "Once the 120 s target (or your T_GLOBAL override) is reached, remaining jobs exit without rebuilding the mesh."
    exit 0
fi

if [[ $# -ne 0 ]]; then
    echo "Unknown arguments. Use sbatch $0, or bash $0 --chain N." >&2
    exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch; do not run the solver on the login node." >&2
    exit 2
fi

# Slurm copies the batch script to its spool directory. SLURM_SUBMIT_DIR identifies
# the directory you submitted from; BASH_SOURCE inside a job does not locate .py.
run_dir="${SIMULATION_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "Cannot find the submission directory. Set SIMULATION_DIR to the code directory." >&2
    exit 2
fi
cd -- "${run_dir}"
run_dir="${PWD}"
solver_path="${run_dir}/rectangular_3d_h30_thesis.py"
if [[ ! -f "${solver_path}" ]]; then
    echo "Missing ${solver_path}. Submit from the directory containing both files." >&2
    exit 2
fi

# ENVIRONMENT SETUP: retain your proven FEniCS 2019.1 activation commands here.
# An already working inherited environment is used directly. Alternatively set
# FENICS_ENV_SCRIPT to a file containing ONLY environment activation commands.
# Do not point it at an old batch script, which would launch the old simulation.
if [[ -n "${FENICS_ENV_SCRIPT:-}" ]]; then
    if [[ ! -f "${FENICS_ENV_SCRIPT}" ]]; then
        echo "FENICS_ENV_SCRIPT does not exist: ${FENICS_ENV_SCRIPT}" >&2
        exit 2
    fi
    source "${FENICS_ENV_SCRIPT}"
fi

FENICS_PYTHON="${FENICS_PYTHON:-python3}"
fenics_import_check='import dolfin, mpi4py, petsc4py; assert dolfin.__version__.startswith("2019.1"), dolfin.__version__'
if ! "${FENICS_PYTHON}" -c "${fenics_import_check}" >/dev/null 2>&1; then
    if type module >/dev/null 2>&1; then
        # This is the module name recorded for your previous FEniCS workflow.
        # Override FENICS_MODULE if your working job uses a fully qualified name.
        if ! module load "${FENICS_MODULE:-fenics-2019.1.0}"; then
            echo "FEniCS module activation failed. Copy the activation lines from your working job into the marked section above." >&2
            exit 2
        fi
    else
        echo "FEniCS 2019.1 is unavailable. Copy your working job's activation lines into the marked section above." >&2
        exit 2
    fi
fi

# Fail before mesh generation if the environment is not the legacy solver stack.
"${FENICS_PYTHON}" -c "${fenics_import_check}; print('DOLFIN version:', dolfin.__version__)"
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL
export FLOW_RATE_CC_MIN="${FLOW_RATE_CC_MIN:-20}"
export DT="${DT:-0.004}"
export T_GLOBAL="${T_GLOBAL:-120.0}"
export SEGMENT_DURATION="${SEGMENT_DURATION:-${T_GLOBAL}}"
export CHECKPOINT_WALL_SECONDS="${CHECKPOINT_WALL_SECONDS:-1800}"
export SAVE_MARGIN_SECONDS="${SAVE_MARGIN_SECONDS:-1800}"
# Fallback limit includes startup and mesh generation, not just time stepping.
export WALL_TIME_BUDGET_SECONDS="${WALL_TIME_BUDGET_SECONDS:-165600}"

# Recent Slurm versions supply the actual allocation end time. On older versions,
# obtain it read-only so --time overrides also receive the correct save deadline.
if [[ ! "${SLURM_JOB_END_TIME:-}" =~ ^0*[1-9][0-9]*$ ]]; then
    unset SLURM_JOB_END_TIME
    job_description="$(scontrol show job --oneliner "${SLURM_JOB_ID}" 2>/dev/null || true)"
    if [[ "${job_description}" =~ (^|[[:space:]])EndTime=([^[:space:]]+) ]]; then
        allocation_end_text="${BASH_REMATCH[2]}"
        allocation_end_epoch="$(date --date="${allocation_end_text}" +%s 2>/dev/null || true)"
        if [[ "${allocation_end_epoch}" =~ ^0*[1-9][0-9]*$ ]]; then
            export SLURM_JOB_END_TIME="${allocation_end_epoch}"
        fi
    fi
fi

echo "Rectangular 3D, 600 x 300 x 300 mm, ceiling-refined mesh"
echo "Job ${SLURM_JOB_ID}; tasks ${SLURM_NTASKS:-24}; started $(date --iso-8601=seconds)"
echo "Q=${FLOW_RATE_CC_MIN} cc/min; dt=${DT} s; target=${T_GLOBAL} simulated seconds"
echo "Checkpoint interval ${CHECKPOINT_WALL_SECONDS} wall seconds; save margin ${SAVE_MARGIN_SECONDS} seconds"
echo "Allocation end epoch: ${SLURM_JOB_END_TIME:-unavailable; elapsed-time fallback active}"
echo "Solver: ${solver_path}"

# --signal above sends USR1 to the job-step processes (not only the batch shell).
# Python records a stop request and saves collectively at a completed time step.
# Periodic checkpoints also protect progress if a long solve delays that request.
srun --cpu-bind=cores "${FENICS_PYTHON}" -u "${solver_path}"

echo "Solver finished normally at $(date --iso-8601=seconds)."
echo "If the target is unfinished, resubmit this script to resume the saved state."
