# PhD FEniCS Stratification Modelling

Source code for the numerical models used in my PhD thesis.

The repository contains the main Cartesian, axisymmetric and rectangular 3D model families implemented in FEniCS, together with the MATLAB and Python post-processing used for the numerical results.

## Structure

- `models/cartesian/` - Cartesian model and numerical checks
- `models/axisymmetric/` - axisymmetric model, verification cases and earlier versions
- `models/rectangular_3d_quantitative/` - current 600 x 300 x 300 mm rectangular 3D quantitative model
- `models/rectangular_3d_earlier/` - earlier rectangular 3D model family
- `postprocessing/` - MATLAB and Python post-processing scripts
- `archive/` - older numerical model versions retained for reference

## Software

The simulations were developed using FEniCS 2019.1.0 and run using Slurm on a high-performance computing cluster. MATLAB was used for the main thesis post-processing.

## Notes

Simulation output, checkpoints, returned run bundles and large analysis files are not included. Some archived scripts reflect earlier stages of the numerical development and may require paths or cluster settings to be changed before use. Machine-specific usernames, project codes and absolute cluster paths have been replaced with placeholders for the public repository.
