#!/bin/bash
#SBATCH --job-name=cart_mesh_5s
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=cartesian_mesh_check_5s_%j.out
#SBATCH --error=cartesian_mesh_check_5s_%j.err

# Put this file and cartesian_ipcs_verification_5s.py in the same directory.
# Submit from that directory: sbatch run_cartesian_mesh_check_5s.sh
# 240 x 120 cells; dt=0.001 s; 5000 steps. Five physical seconds from rest.
# Exactly ONE MPI rank. The 24-hour scheduler limit is not a runtime estimate.
set -euo pipefail

module purge
module add languages/python/fenics-2019.1.0

cd "${SLURM_SUBMIT_DIR:?Submit this script using sbatch}"
fv_script="${SLURM_SUBMIT_DIR}/cartesian_ipcs_verification_5s.py"
if [[ ! -f "$fv_script" ]]; then
    echo "Missing Python script: $fv_script" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CARTESIAN_FV_OUTPUT="/path/to/work/cartesian_mesh_check_5s_${SLURM_JOB_ID:?Missing Slurm job ID}"

echo "Cartesian mesh verification: 240 x 120 cells; dt=0.001 s; 5000 steps; NO CLIPPING"
echo "Output directory: $CARTESIAN_FV_OUTPUT"

# Dependencies and scalar/flux tests before the coupled FEniCS calculation.
srun --mpi=pmi2 python3 -c 'import dolfin, numpy, scipy, mpi4py; print("DOLFIN", dolfin.__version__, "NumPy", numpy.__version__, "SciPy", scipy.__version__, flush=True)'
srun --mpi=pmi2 python3 -u "$fv_script" --self-test
srun --mpi=pmi2 python3 -u "$fv_script" --case mesh --output "$CARTESIAN_FV_OUTPUT"
