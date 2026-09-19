#!/bin/bash -l
#SBATCH --job-name=rect3d_bounds_followup
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@180
#SBATCH --export=ALL
#SBATCH --output=rect3d_bounds_%j.out
#SBATCH --error=rect3d_bounds_%j.err
set -eo pipefail

if [[ $# -ne 0 ]]; then
    echo "Use bash submit_rect3d_bounds_continuation.sh with no arguments." >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    bounds_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    for bounds_required in continue_rect3d_scalar_bounds.py rect3d_bounds_followup_report.py rect3d_bounds_report.py mixed_scalar_transport.py verify_mixed_scalar_transport.py; do
        [[ -f "${bounds_code_dir}/${bounds_required}" ]] || { echo "Missing ${bounds_code_dir}/${bounds_required}" >&2; exit 2; }
    done
    bounds_source="${bounds_code_dir}/diagnostic_rect3d_bounds_18989285/checkpoint/state_step_000000050.npz"
    [[ -f "${bounds_source}" ]] || { echo "Missing saved diagnostic checkpoint: ${bounds_source}" >&2; exit 2; }
    export SIMULATION_DIR="${bounds_code_dir}"
    exec sbatch --account=YOUR_ACCOUNT --partition=compute --chdir="${bounds_code_dir}" "${bounds_code_dir}/submit_rect3d_bounds_continuation.sh"
fi
run_dir="${SIMULATION_DIR:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "${run_dir}" && -d "${run_dir}" ]] || { echo "Missing code directory" >&2; exit 2; }
cd -- "${run_dir}"
run_dir="${PWD}"
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

# Settings are also fixed in the diagnostic Python driver.
export RECT3D_VERIFY_ONLY=0 MESH_LEVEL=fine DT=0.001 T_GLOBAL=0.1 SEGMENT_DURATION=0.05
export FLOW_RATE_CC_MIN=20 SCALAR_DIFFUSIVITY=1.5e-9 DYE_DIFFUSIVITY=4.14e-10 INLET_PERTURBATION=0
export PIN_CHECKPOINT_TIME=0 SAVE_MARGIN_SECONDS=180 WALL_TIME_BUDGET_SECONDS=1440 CHECKPOINT_WALL_SECONDS=600
unset RECT3D_OUTPUT_ROOT SEED_CHECKPOINT
echo "Fine-mesh bounds continuation: job ${SLURM_JOB_ID}; 0.050 to 0.100 s; 50 further steps; maximum 30 minutes."
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

"${FENICS_PYTHON}" -c 'from mixed_scalar_transport import SCALAR_SOLVER_REVISION; assert SCALAR_SOLVER_REVISION == "equilibrated-2026-09-17", "Replace mixed_scalar_transport.py with the equilibrated revision first."'
echo "Running the established verification before the diagnostic."
srun "${srun_options[@]}" "${FENICS_PYTHON}" -u "${run_dir}/verify_mixed_scalar_transport.py" --report "${run_dir}/verification_bounds_${SLURM_JOB_ID}.json"
diagnostic_status=0
srun "${srun_options[@]}" "${FENICS_PYTHON}" -u "${run_dir}/continue_rect3d_scalar_bounds.py" || diagnostic_status=$?

# Package the return files after all MPI ranks have finished, including when
# the diagnostic hit its emergency ceiling. Logs alone are still useful if
# failure occurred before a report could be written.
"${FENICS_PYTHON}" - "${run_dir}" "${SLURM_JOB_ID}" "${diagnostic_status}" <<'PY'
import json
from pathlib import Path
import shutil
import sys
import zipfile
root, job, status = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
ret = root/('RETURN_rect3d_bounds_'+job)
ret.mkdir(exist_ok=True)
for name in ('rect3d_bounds_'+job+'.out', 'rect3d_bounds_'+job+'.err', 'verification_bounds_'+job+'.json'):
    path = root/name
    if path.is_file():
        shutil.copyfile(str(path), str(ret/name))
(ret/'launcher_status.json').write_text(json.dumps(dict(job_id=job, diagnostic_exit_code=status, production_eligible=False), indent=2))
archive = root/(ret.name+'.zip')
with zipfile.ZipFile(str(archive), 'w', zipfile.ZIP_DEFLATED) as z:
    for path in sorted(ret.iterdir()):
        if path.is_file():
            z.write(str(path), ret.name+'/'+path.name)
print('RETURN THIS FILE:', archive, flush=True)
PY
exit "${diagnostic_status}"
