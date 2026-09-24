# Post-processing

`matlab/` contains MATLAB plotting scripts.

`python/` contains scripts for the Cartesian half-step comparison, axisymmetric mesh and time-step checks, and reduced-height 3D analysis. Tools for the full-height 3D case are kept with the solver in `models/rectangular_3d_quantitative/`.

Run each script with `--help` for its input paths. These tools read saved outputs; they do not run FEniCS. Supply the corresponding return folders or XDMF/HDF5 pairs. Results and figures are not bundled here.

The Cartesian scripts compare raw freshwater fractions using conservative rectangular overlaps. The axisymmetric scripts report both projected and exact-overlap differences. These measures are not interchangeable.
