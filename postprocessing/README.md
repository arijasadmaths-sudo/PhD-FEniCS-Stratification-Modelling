# Post-processing

`matlab/` contains the original thesis plotting scripts.

The Python folders contain the final Cartesian half-step comparison, axisymmetric mesh/time-step comparisons, earlier axisymmetric checks and reduced-height 3D analysis. The full-height geometry and profile tools remain beside their solver in `models/rectangular_3d_quantitative/`.

Run each script with `--help` for its input paths. These tools read saved outputs; they do not run FEniCS. Supply the original return folders or XDMF/HDF5 pairs. Results and figures are not bundled here.

The Cartesian comparisons use raw freshwater fractions and conservative rectangular overlaps. The axisymmetric comparisons retain both projected and exact-overlap differences. These are different measures and should not be interchanged.
