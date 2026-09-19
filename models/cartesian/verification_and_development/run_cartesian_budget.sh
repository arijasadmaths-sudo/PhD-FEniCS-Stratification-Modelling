#!/bin/bash
#SBATCH --job-name=cartesian_budget
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=cartesian_budget_%j.out
#SBATCH --error=cartesian_budget_%j.err

set -euo pipefail

module purge
module add languages/python/fenics-2019.1.0

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Each submitted job gets a separate results directory.
export CARTESIAN_BUDGET_OUTPUT="/path/to/work/cartesian_budget_${SLURM_JOB_ID}"
echo "Output directory: $CARTESIAN_BUDGET_OUTPUT"

srun --mpi=pmi2 python3 -u IPSC_0.001_budget_diagnostic.py
