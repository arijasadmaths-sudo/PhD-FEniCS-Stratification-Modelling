# PhD FEniCS Stratification Modelling

Source code for the numerical models in Chapter 6 and Appendices B-D of my PhD thesis, with the associated post-processing and numerical checks.

## Contents

| Thesis calculation | Code |
| --- | --- |
| Original Cartesian models, Section 6.3 | `archive/early_fenics_ipcs/` |
| Original axisymmetric models, Section 6.4 | `models/axisymmetric/earlier/`; the 70 cc/min source is `archive/early_fenics_ipcs/HPC.py` |
| Cartesian checks, Section 6.5.3 | `models/cartesian/` |
| Axisymmetric checks, Section 6.5.4 | `models/axisymmetric/current/` and `verification/` |
| Reduced-height 3D model, Section 6.6 | `models/rectangular_3d_earlier/` |
| Full-height 3D reference, Section 6.6.5 | `archive/rectangular_3d_reference/` |

`models/rectangular_3d_quantitative/` contains the later full-height solver updates. These are kept separate from the earlier reference version, not treated as another model.

`postprocessing/` contains the MATLAB scripts and Python comparisons. Other earlier versions are retained in `archive/`.

## Running the code

The simulations use legacy FEniCS 2019.1.0 with PETSc/MPI. The later 2D checks also use NumPy and SciPy. Python post-processing uses NumPy, SciPy, h5py and Matplotlib; the axisymmetric overlap calculations need Shapely 2 or later.

Use the instructions in the relevant model folder. Set the Slurm account and work paths for your machine before submitting. The 2D verification cases use one MPI rank; the 3D launchers request 24.

Run the local checks with Python 3.9 or later from the repository root:

```sh
python tests/check_repository.py
```

These check syntax, file dependencies and the available numerical kernels. They do not rerun the coupled FEniCS simulations or MATLAB.

## Notes

Simulation outputs and checkpoints are not included. A configured end time is not a completed result. Existing HPC checkpoints require their original matching source and helper files; do not bypass the restart checks.
