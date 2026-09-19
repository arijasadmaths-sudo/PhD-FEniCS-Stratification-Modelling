#!/bin/bash
#SBATCH --job-name=axisym70_flux2s
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=axisym70_flux2s_%j.out
#SBATCH --error=axisym70_flux2s_%j.err

# One short fresh-start diagnostic. No job array and no restart.
set -euo pipefail
module purge
module add languages/python/fenics-2019.1.0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:?Submit from the directory containing the Python and shell files.}"
axisym_script="${SLURM_SUBMIT_DIR}/axisymmetric_70_flux_2s.py"
if [[ ! -f "$axisym_script" ]]; then
    printf 'Missing Python script: %s\n' "$axisym_script" >&2
    exit 1
fi
axisym_output="/path/to/work/axisymmetric_70_flux_2s_${SLURM_JOB_ID:?Missing Slurm job ID}"

python3 - <<'PY'
import dolfin
import numpy
import scipy
print('DOLFIN:', dolfin.__version__)
print('NumPy:', numpy.__version__)
print('SciPy:', scipy.__version__)
if not dolfin.__version__.startswith('2019.1'):
    raise RuntimeError('Load the legacy FEniCS 2019.1 module.')
PY
python3 "$axisym_script" --self-test
printf 'Output: %s\n' "$axisym_output"
srun --mpi=pmi2 python3 "$axisym_script" --output "$axisym_output"
