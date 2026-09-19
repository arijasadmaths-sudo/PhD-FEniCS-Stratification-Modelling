#!/bin/bash
#SBATCH --job-name=cart_fv_5s
#SBATCH --account=SEMT035584
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=cartesian_conservative_5s_%j.out
#SBATCH --error=cartesian_conservative_5s_%j.err

# Put this file and cartesian_ipcs_conservative_5s.py in the same directory.
# From that directory, submit:
#   sbatch run_cartesian_ipcs_conservative_5s.sh
# Exactly ONE MPI rank is required by the serial FV / DOF coupling.
# 24 hours is the job's scheduler limit, not a runtime prediction.
# The simulation is 5000 steps x 0.001 s = 5 physical seconds, from rest.
set -euo pipefail

module purge
module add languages/python/fenics-2019.1.0

cd "${SLURM_SUBMIT_DIR:?Submit this script using sbatch}"
fv_script="${SLURM_SUBMIT_DIR}/cartesian_ipcs_conservative_5s.py"
if [[ ! -f "$fv_script" ]]; then
    echo "Missing Python script: $fv_script" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CARTESIAN_FV_OUTPUT="/user/work/kn22417/cartesian_conservative_5s_${SLURM_JOB_ID:?Missing Slurm job ID}"

echo "Five-second Cartesian pilot: conservative FV scalar; NO CLIPPING"
echo "Output directory: $CARTESIAN_FV_OUTPUT"

# Fail early on missing dependencies, then verify scalar/flux kernels before
# starting the coupled calculation. The Python code also checks its FEniCS
# face-integration and DG0 mappings at startup.
srun --mpi=pmi2 python3 -c 'import dolfin, numpy, scipy, mpi4py; print("DOLFIN", dolfin.__version__, "NumPy", numpy.__version__, "SciPy", scipy.__version__, flush=True)'
srun --mpi=pmi2 python3 -u "$fv_script" --self-test
srun --mpi=pmi2 python3 -u "$fv_script" --output "$CARTESIAN_FV_OUTPUT" --steps 5000
