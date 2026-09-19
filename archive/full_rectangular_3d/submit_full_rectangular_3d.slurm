#!/bin/bash
#SBATCH --job-name=full_rect_3d
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=full_rect_3d_%A.out
#SBATCH --error=full_rect_3d_%A.err

set -euo pipefail

# Keep the same module/container activation commands used by the working
# DOLFIN 2019.1 job before this point if your login shell does not load them.
export OMP_NUM_THREADS=1

# Override any of these at submission, for example:
#   sbatch --export=ALL,FLOW_RATE_CC_MIN=40 submit_full_rectangular_3d.slurm
export FLOW_RATE_CC_MIN="${FLOW_RATE_CC_MIN:-20}"
export T_GLOBAL="${T_GLOBAL:-20.0}"
export SEGMENT_DURATION="${SEGMENT_DURATION:-0.5}"
export INLET_PERTURBATION="${INLET_PERTURBATION:-0.0}"
export CHECKPOINT_WALL_SECONDS="${CHECKPOINT_WALL_SECONDS:-1200}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

echo "Full rectangular 3D experiment"
echo "Flow rate: ${FLOW_RATE_CC_MIN} cc/min"
echo "Target time: ${T_GLOBAL} s"
echo "MPI ranks: ${SLURM_NTASKS}"
if [[ -n "${DT:-}" ]]; then
    echo "Time step override: ${DT} s"
else
    echo "Time step: automatic inlet-CFL default"
fi

srun --cpu-bind=cores python3 "${script_dir}/full_rectangular_3d_physical.py"
