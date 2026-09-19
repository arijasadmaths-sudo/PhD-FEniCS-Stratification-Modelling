#!/bin/bash
#SBATCH --job-name=cart_no_clip_5s
#SBATCH --account=SEMT035584
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=cartesian_no_clipping_5s_%j.out
#SBATCH --error=cartesian_no_clipping_5s_%j.err

# The 24-hour value is a scheduler wall-time limit, not simulated duration.
# The Python program runs 5000 steps of 0.001 s = 5 simulated seconds.
# Submit from the directory containing BOTH this file and its Python script:
#   sbatch run_cartesian_ipcs_no_clipping_5s.sh

set -euo pipefail

module purge
module add languages/python/fenics-2019.1.0

cd "${SLURM_SUBMIT_DIR:?Submit this script using sbatch}"
noclip_script="${SLURM_SUBMIT_DIR}/cartesian_ipcs_no_clipping_5s.py"
if [[ ! -f "$noclip_script" ]]; then
    echo "Missing Python script: $noclip_script" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Each job gets its own folder. The Python program refuses a nonempty folder.
export CARTESIAN_NOCLIP_OUTPUT="/user/work/kn22417/cartesian_no_clipping_5s_${SLURM_JOB_ID:?Missing Slurm job ID}"
echo "Five-second Cartesian control: scalar clipping DISABLED"
echo "Output directory: $CARTESIAN_NOCLIP_OUTPUT"

srun --mpi=pmi2 python3 -u "$noclip_script"
