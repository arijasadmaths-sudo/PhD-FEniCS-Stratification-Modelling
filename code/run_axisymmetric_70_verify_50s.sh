#!/bin/bash
#SBATCH --job-name=axisym70_verify50
#SBATCH --account=SEMT035584
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-1
#SBATCH --output=axisym70_verify50_%A_%a.out
#SBATCH --error=axisym70_verify50_%A_%a.err

# Default: two independent fresh starts, timestep and mesh, each stopping at50 s.
# Optional uncorrected diagnostic ONLY: sbatch --array=2 this_script.sh
# Optional original-case reproduction ONLY: sbatch --array=3 this_script.sh
# 72 h is a scheduler ceiling, not a guaranteed runtime or a request to run1000 s.
set -euo pipefail

module purge
module add languages/python/fenics-2019.1.0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the directory containing both files.}"
axisym_script="${SLURM_SUBMIT_DIR}/axisymmetric_70_verify_50s.py"
if [[ ! -f "$axisym_script" ]]; then
    printf 'Missing Python script: %s\n' "$axisym_script" >&2
    exit 1
fi
case "${SLURM_ARRAY_TASK_ID:?This launcher must run as an array job.}" in
    0) axisym_case=timestep ;;
    1) axisym_case=mesh ;;
    2) axisym_case=uncorrected ;;
    3) axisym_case=baseline ;;
    *) printf 'Unknown array index: %s\n' "$SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
axisym_output="/user/work/kn22417/axisymmetric_70_verify_50s_${SLURM_ARRAY_JOB_ID:?Missing array job ID}/${axisym_case}"

python3 - <<'PY'
import dolfin
import numpy
print('FEniCS/DOLFIN:', dolfin.__version__)
print('NumPy:', numpy.__version__)
if not dolfin.__version__.startswith('2019.1'):
    raise RuntimeError('This verification preserves the legacy FEniCS2019.1 solver.')
PY
python3 "$axisym_script" --self-test
printf 'Case: %s\nOutput: %s\n' "$axisym_case" "$axisym_output"
srun --mpi=pmi2 python3 "$axisym_script" --case "$axisym_case" --output "$axisym_output"
