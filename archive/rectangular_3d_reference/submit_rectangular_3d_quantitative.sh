#!/bin/bash -l
#SBATCH --job-name=rect3d_quant_fine
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@1800
#SBATCH --export=ALL
#SBATCH --output=rect3d_quant_%x_%j.out
#SBATCH --error=rect3d_quant_%x_%j.err

# Keep these four files together:
#   submit_rectangular_3d_quantitative.sh
#   rectangular_3d_h30_quantitative.py
#   mixed_scalar_transport.py
#   verify_mixed_scalar_transport.py
#
# Queue two spatial cases and a shorter time-step comparison:
#   bash submit_rectangular_3d_quantitative.sh --study 8
# This queues TWO independent continuation chains, fine and reference, both
# starting from salt water, with up to eight allocations each. Two further
# allocations branch from the fine case's saved 60-second state and advance
# to 80 seconds at dt=0.0005. This tests developed-flow time-step sensitivity,
# not convergence of the earlier transient. No case repeats its completed run.
# Eight 48-hour allocations provide capacity, not a runtime guarantee.
#
# Queue one case only (fine by default):
#   bash submit_rectangular_3d_quantitative.sh --chain 8
# Or submit one allocation from the directory containing the files:
#   sbatch submit_rectangular_3d_quantitative.sh
#
# Add the SAME account/project, partition and memory options as your working
# BluePebble job; --study/--chain accept --account=CODE --partition=NAME
# --time=HH:MM:SS --mem=SIZE --mem-per-cpu=SIZE and --qos=NAME.
# Your partition's maximum allocation time still applies to the 48-hour request.
# The small numerical verification runs under MPI before each production job.
# To check MPI startup and numerical verification in one short allocation:
#   RECT3D_VERIFY_ONLY=1 sbatch --time=00:30:00 --signal=USR1@60 submit_rectangular_3d_quantitative.sh
# The verification-only option exits before building the production mesh.

set -eo pipefail

required_files=(rectangular_3d_h30_quantitative.py mixed_scalar_transport.py verify_mixed_scalar_transport.py)
RECT3D_VERIFY_ONLY="${RECT3D_VERIFY_ONLY:-0}"
if [[ "${RECT3D_VERIFY_ONLY}" != 0 && "${RECT3D_VERIFY_ONLY}" != 1 ]]; then
    echo "RECT3D_VERIFY_ONLY must be 0 or 1." >&2
    exit 2
fi

if [[ "${1:-}" == "--study" || "${1:-}" == "--chain" ]]; then
    submission_mode="${1}"
    if [[ "${RECT3D_VERIFY_ONLY}" == 1 ]]; then
        echo "Use RECT3D_VERIFY_ONLY=1 with one sbatch allocation, not --study or --chain." >&2
        exit 2
    fi
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Use ${submission_mode} from the login node, not inside an existing job." >&2
        exit 2
    fi
    if [[ $# -lt 2 || ! "${2}" =~ ^([1-9]|1[0-9]|20)$ ]]; then
        echo "Usage: bash $0 ${submission_mode} N [--account=CODE] [--partition=NAME] [--time=HH:MM:SS]; N must be 1 to 20." >&2
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
                echo "Unsupported option: ${chain_option}. Use --name=value for account, partition, time, memory or qos." >&2
                exit 2 ;;
        esac
    done
    if [[ "${submission_mode}" == --study ]]; then
        if [[ -n "${RECT3D_OUTPUT_ROOT:-}" ]]; then
            echo "Unset RECT3D_OUTPUT_ROOT for --study so the two cases use distinct automatic result directories." >&2
            exit 2
        fi
        if [[ -n "${SEED_CHECKPOINT:-}" ]]; then
            echo "Unset SEED_CHECKPOINT for --study; only the time-step comparison uses a seeded state." >&2
            exit 2
        fi
        mesh_cases=(fine reference)
    else
        mesh_cases=("${MESH_LEVEL:-fine}")
    fi
    for mesh_case in "${mesh_cases[@]}"; do
        if [[ "${mesh_case}" != fine && "${mesh_case}" != reference ]]; then
            echo "MESH_LEVEL must be fine or reference." >&2
            exit 2
        fi
    done
    # This branch runs before submission, so BASH_SOURCE is the original file.
    launcher_path="$(readlink -f -- "${BASH_SOURCE[0]}")"
    chain_dir="$(dirname -- "${launcher_path}")"
    for required_file in "${required_files[@]}"; do
        if [[ ! -f "${chain_dir}/${required_file}" ]]; then
            echo "Missing ${chain_dir}/${required_file}. Keep all four supplied files together." >&2
            exit 2
        fi
    done
    export SIMULATION_DIR="${chain_dir}"
    if [[ "${submission_mode}" == --study ]]; then
        export DT="${DT:-0.001}"
        export T_GLOBAL="${T_GLOBAL:-120.0}"
        export FLOW_RATE_CC_MIN="${FLOW_RATE_CC_MIN:-20}"
        export INLET_PERTURBATION="${INLET_PERTURBATION:-0.0}"
        export SCALAR_DIFFUSIVITY="${SCALAR_DIFFUSIVITY:-1.5e-9}"
        export DYE_DIFFUSIVITY="${DYE_DIFFUSIVITY:-4.14e-10}"
        # Standard-library Python only; no solver or large computation runs here.
        # Use exactly the same floating-point tags as the production solver.
        study_root_names="$(python3 -c '
import math, os
dt = float(os.environ["DT"])
target = float(os.environ["T_GLOBAL"])
q = float(os.environ["FLOW_RATE_CC_MIN"])
eps = float(os.environ["INLET_PERTURBATION"])
d = float(os.environ["SCALAR_DIFFUSIVITY"])
dye = float(os.environ["DYE_DIFFUSIVITY"])
if not all(math.isfinite(x) for x in (dt, target, q, eps, d, dye)):
    raise SystemExit("Study parameters must be finite.")
if not math.isclose(dt, 0.001, rel_tol=0.0, abs_tol=1e-15):
    raise SystemExit("--study requires primary DT=0.001; use --chain for other timesteps.")
if target < 80.0 or abs(target / dt - round(target / dt)) > 1e-7:
    raise SystemExit("--study requires T_GLOBAL >= 80 and an integer number of primary timesteps.")
if q <= 0.0 or d <= 0.0 or dye <= 0.0 or not 0.0 <= eps <= 0.05:
    raise SystemExit("Invalid flow rate, diffusivity or inlet perturbation.")
def tag(x):
    return ("{:g}".format(x)).replace(".", "p").replace("-", "m")
for mesh, timestep, suffix in (("fine", 0.001, ""), ("reference", 0.001, ""), ("fine", 0.0005, "_from60")):
    print("results_rect3d_quant_{}_Q{}_eps{}_dt{}_D{}_Dye{}{}".format(mesh, tag(q), tag(eps), tag(timestep), tag(d), tag(dye), suffix))
')"
        readarray -t study_roots <<< "${study_root_names}"
        fine_root="${chain_dir}/${study_roots[0]}"
        reference_root="${chain_dir}/${study_roots[1]}"
        timecheck_root="${chain_dir}/${study_roots[2]}"
        export DT=0.001
        export SEGMENT_DURATION="${T_GLOBAL}"
        last_fine_job=""
    fi
    for mesh_case in "${mesh_cases[@]}"; do
        # Keep the real environment aligned with --export=ALL,NAME=value: ALL
        # can otherwise preserve an inherited value instead of the requested one.
        export MESH_LEVEL="${mesh_case}"
        if [[ "${submission_mode}" == --study ]]; then
            if [[ "${mesh_case}" == fine ]]; then
                export RECT3D_OUTPUT_ROOT="${fine_root}" PIN_CHECKPOINT_TIME=60
            else
                export RECT3D_OUTPUT_ROOT="${reference_root}" PIN_CHECKPOINT_TIME=0
            fi
        fi
        previous_job=""
        for ((chain_index=1; chain_index<=chain_length; chain_index++)); do
            dependency_options=()
            if [[ -n "${previous_job}" ]]; then
                dependency_options=("--dependency=afterok:${previous_job}" "--kill-on-invalid-dep=yes")
            fi
            if ! submission="$(sbatch --parsable "--export=ALL,MESH_LEVEL=${mesh_case}" --job-name="rect3d_quant_${mesh_case}" "${chain_options[@]}" --chdir="${chain_dir}" "${dependency_options[@]}" "${launcher_path}")"; then
                echo "Submission stopped at ${mesh_case} allocation ${chain_index}. Previously printed job IDs remain submitted." >&2
                exit 1
            fi
            previous_job="${submission%%;*}"
            if [[ ! "${previous_job}" =~ ^[0-9]+$ ]]; then
                echo "Unexpected sbatch response: ${submission}. Check squeue before submitting again." >&2
                exit 1
            fi
            echo "Queued ${mesh_case} allocation ${chain_index}/${chain_length}: job ${previous_job}"
        done
        if [[ "${submission_mode}" == --study && "${mesh_case}" == fine ]]; then
            last_fine_job="${previous_job}"
        fi
    done
    if [[ "${submission_mode}" == --study ]]; then
        export MESH_LEVEL=fine DT=0.0005 T_GLOBAL=80 SEGMENT_DURATION=80
        export RECT3D_OUTPUT_ROOT="${timecheck_root}" PIN_CHECKPOINT_TIME=0
        export SEED_CHECKPOINT="${fine_root}/seeds/state_time_60s.npz"
        previous_job="${last_fine_job}"
        for ((chain_index=1; chain_index<=2; chain_index++)); do
            if ! submission="$(sbatch --parsable --export=ALL,MESH_LEVEL=fine,DT=0.0005,T_GLOBAL=80,SEGMENT_DURATION=80 --job-name=rect3dq_timecheck "${chain_options[@]}" --chdir="${chain_dir}" "--dependency=afterok:${previous_job}" --kill-on-invalid-dep=yes "${launcher_path}")"; then
                echo "Time-step branch submission stopped at allocation ${chain_index}. Previously printed job IDs remain submitted." >&2
                exit 1
            fi
            previous_job="${submission%%;*}"
            if [[ ! "${previous_job}" =~ ^[0-9]+$ ]]; then
                echo "Unexpected sbatch response: ${submission}. Check squeue before submitting again." >&2
                exit 1
            fi
            echo "Queued time-step comparison allocation ${chain_index}/2: job ${previous_job}"
        done
        echo "Time-step comparison: fine mesh, 60 to 80 seconds, dt=0.0005; shared state before 60 seconds."
        echo "Primary and reference roots: ${fine_root} ; ${reference_root}"
        echo "Time-step comparison root: ${timecheck_root}"
    fi
    echo "Each allocation continues its case from the latest compatible checkpoint."
    echo "Once a case reaches its target, later allocations exit before rebuilding the production mesh."
    echo "A failed verification or production job blocks its dependent continuation jobs."
    exit 0
fi

if [[ $# -ne 0 ]]; then
    echo "Unknown arguments. Use sbatch $0, or bash $0 --study N / --chain N." >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit with sbatch, --study or --chain; do not run the solver on the login node." >&2
    exit 2
fi

# Slurm copies the batch script into a spool directory. Locate the Python files
# using the submitted directory, not BASH_SOURCE inside an allocated job.
run_dir="${SIMULATION_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "Cannot find the submission directory. Set SIMULATION_DIR to the code directory." >&2
    exit 2
fi
cd -- "${run_dir}"
run_dir="${PWD}"
for required_file in "${required_files[@]}"; do
    if [[ ! -f "${run_dir}/${required_file}" ]]; then
        echo "Missing ${run_dir}/${required_file}. Keep all four supplied files together." >&2
        exit 2
    fi
done

# ENVIRONMENT SETUP: retain your proven FEniCS 2019.1 activation commands here.
# An inherited working environment is used directly. Alternatively set
# FENICS_ENV_SCRIPT to a file containing ONLY environment activation commands.
# Do not point it at an old batch script, which could launch the old solver.
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
        # Fully qualified name from the working Blue Pebble axisymmetric job.
        FENICS_MODULE="${FENICS_MODULE:-languages/python/fenics-2019.1.0}"
        echo "Loading FEniCS module: ${FENICS_MODULE}"
        if ! module load "${FENICS_MODULE}"; then
            echo "FEniCS activation failed for module ${FENICS_MODULE}. Set FENICS_MODULE or FENICS_ENV_SCRIPT to your working environment." >&2
            exit 2
        fi
    else
        echo "FEniCS 2019.1 is unavailable. Copy your working job's activation lines into the marked section." >&2
        exit 2
    fi
fi
"${FENICS_PYTHON}" -c "${fenics_import_check}; print('DOLFIN version:', dolfin.__version__)"
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL
# Match the explicit PMI selection in the working Blue Pebble FEniCS launchers.
# An alternate site configuration can override FENICS_MPI_TYPE explicitly.
FENICS_MPI_TYPE="${FENICS_MPI_TYPE:-pmi2}"
launch_ranks="${SLURM_NTASKS:-24}"
if [[ ! "${FENICS_MPI_TYPE}" =~ ^[[:alnum:]_.-]+$ || ! "${launch_ranks}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid FENICS_MPI_TYPE or SLURM_NTASKS." >&2
    exit 2
fi
srun_options=("--mpi=${FENICS_MPI_TYPE}" --cpu-bind=cores "--ntasks=${launch_ranks}" --kill-on-bad-exit=1)
export MESH_LEVEL="${MESH_LEVEL:-fine}"
if [[ "${MESH_LEVEL}" != fine && "${MESH_LEVEL}" != reference ]]; then
    echo "MESH_LEVEL must be fine or reference." >&2
    exit 2
fi
export FLOW_RATE_CC_MIN="${FLOW_RATE_CC_MIN:-20}"
export DT="${DT:-0.001}"
export T_GLOBAL="${T_GLOBAL:-120.0}"
export SEGMENT_DURATION="${SEGMENT_DURATION:-${T_GLOBAL}}"
# Approximate 25 C diffusivities cited in the experimental apparatus source;
# the index-matched mixture's actual values have not been measured here.
export SCALAR_DIFFUSIVITY="${SCALAR_DIFFUSIVITY:-1.5e-9}"
export DYE_DIFFUSIVITY="${DYE_DIFFUSIVITY:-4.14e-10}"
export CHECKPOINT_WALL_SECONDS="${CHECKPOINT_WALL_SECONDS:-1800}"
export SAVE_MARGIN_SECONDS="${SAVE_MARGIN_SECONDS:-1800}"
export WALL_TIME_BUDGET_SECONDS="${WALL_TIME_BUDGET_SECONDS:-165600}"

# Use the actual allocation deadline if available, including --time overrides.
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

echo "Rectangular 3D quantitative study: 600 x 300 x 300 mm; mesh=${MESH_LEVEL}"
echo "Job ${SLURM_JOB_ID}; tasks ${SLURM_NTASKS:-24}; started $(date --iso-8601=seconds)"
echo "Q=${FLOW_RATE_CC_MIN} cc/min; dt=${DT} s; target=${T_GLOBAL} simulated seconds"
echo "Density-scalar D=${SCALAR_DIFFUSIVITY} m^2/s; passive dye D=${DYE_DIFFUSIVITY} m^2/s"
echo "Checkpoint interval ${CHECKPOINT_WALL_SECONDS} wall seconds; save margin ${SAVE_MARGIN_SECONDS} seconds"
echo "Allocation end epoch: ${SLURM_JOB_END_TIME:-unavailable; elapsed-time fallback active}"
echo "srun executable: $(command -v srun); Python executable: $(command -v "${FENICS_PYTHON}")"
srun --version
srun --mpi=list
echo "Checking MPI startup: plugin=${FENICS_MPI_TYPE}; expected ranks=${launch_ranks}"
srun "${srun_options[@]}" "${FENICS_PYTHON}" -u -c '
import sys
from mpi4py import MPI
import dolfin
from petsc4py import PETSc
comm = MPI.COMM_WORLD
expected = int(sys.argv[1])
sizes = (comm.Get_size(), dolfin.MPI.size(dolfin.MPI.comm_world), PETSc.COMM_WORLD.getSize())
ranks = (comm.Get_rank(), dolfin.MPI.rank(dolfin.MPI.comm_world), PETSc.COMM_WORLD.getRank())
if any(size != expected for size in sizes) or len(set(ranks)) != 1:
    raise RuntimeError("MPI communicator mismatch: expected {} ranks; sizes={}, ranks={}".format(expected, sizes, ranks))
checksum = comm.allreduce(comm.Get_rank() + 1, op=MPI.SUM)
if checksum != expected * (expected + 1) // 2:
    raise RuntimeError("MPI allreduce check failed.")
hosts = comm.gather(MPI.Get_processor_name(), root=0)
if comm.Get_rank() == 0:
    print("MPI library:", MPI.Get_library_version().strip(), flush=True)
    print("MPI LAUNCH CHECK PASSED: {} ranks; hosts={}".format(expected, sorted(set(hosts))), flush=True)
' "${launch_ranks}"
echo "Running the small numerical verification on the allocated MPI ranks."
srun "${srun_options[@]}" "${FENICS_PYTHON}" -u "${run_dir}/verify_mixed_scalar_transport.py" --report "${run_dir}/verification_${SLURM_JOB_ID}.json"

if [[ "${RECT3D_VERIFY_ONLY}" == 1 ]]; then
    echo "RECT3D STARTUP VERIFICATION PASSED: MPI and mixed scalar checks completed; verification-only job finished."
    exit 0
fi

# Verification failure stops this script before the production solver starts.
# USR1 is sent to job steps, and the production solver saves collectively at a
# completed time step. Periodic checkpoints also protect earlier progress.
echo "Verification passed. Starting or resuming the ${MESH_LEVEL} production case."
srun "${srun_options[@]}" "${FENICS_PYTHON}" -u "${run_dir}/rectangular_3d_h30_quantitative.py"
echo "Solver finished normally at $(date --iso-8601=seconds)."
