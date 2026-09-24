# FEniCS stratification modelling

Source, launchers, verification utilities and post-processing for the numerical
models in this repository.

## Layout

- `models/cartesian/`: Cartesian solvers and verification cases
- `models/axisymmetric/`: current, verification and earlier axisymmetric sources
- `models/rectangular_3d_earlier/`: reduced-height rectangular 3D sources
- `models/rectangular_3d_quantitative/`: current full-height rectangular 3D source
- `postprocessing/`: MATLAB and Python analysis utilities
- `archive/`: superseded sources retained for provenance and old checkpoints
- `tests/`: local repository and launcher checks

## Requirements

The simulations use legacy FEniCS 2019.1.0 with PETSc and MPI. Later 2D checks
also use NumPy and SciPy. Python post-processing uses NumPy, SciPy, h5py and
Matplotlib; axisymmetric overlap calculations require Shapely 2 or later.

Use the instructions in the relevant model folder. Set the Slurm account, module
and work paths for the target cluster before submission. The 2D verification cases
use one MPI rank; the 3D launchers request 24.

## Checks

Run from the repository root with Python 3.9 or later:

```sh
python tests/check_repository.py
```

This checks syntax, file dependencies, numerical kernels and mocked launcher
control paths. It does not run coupled FEniCS simulations or MATLAB.

## Outputs and restarts

Simulation outputs, checkpoints and generated figures are not included. The
post-processing scripts require the corresponding return folders or XDMF/HDF5
pairs.

Only saved output times should be treated as completed. Checkpoints are
version-specific and require their matching source and helper files.

## Citation

Citation details are in `CITATION.cff`. Include the commit used.
