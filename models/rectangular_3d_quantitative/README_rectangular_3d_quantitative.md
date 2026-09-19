# Solver settings and outputs

The full-height model uses dt = 0.001 s, Q = 20 cc/min, density diffusivity 1.5e-9 m2/s and dye diffusivity 4.14e-10 m2/s. Buoyancy depends on the density scalar only. The flow is Taylor-Hood with an RT transport reconstruction; scalar diffusion uses the mixed RT/DG0 solve.

The equilibrated scalar solve retains an independent residual check in the original equations. Verification runs before each production allocation. Passing these checks does not establish mesh independence.

For the tested fine case only, the density upper margin is 0.003 during 0 < t < 0.100 s. All other scalar margins remain 0.001. Cells above 1.001 must lie within 1.5 mm of the source axis and 0.25 mm of the floor, with at most 64 cells and 1e-10 m3 affected. The total excess above one must not exceed 2e-13 m3. At 0.100 s the standard bounds apply again. No field values are changed by these checks.

Results include configuration, diagnostics, accepted checkpoints and XDMF/HDF5 fields. Keep each XDMF file with its HDF5 companion. Diagnostic bounds checkpoints cannot restart production.

The supplied analyser and plotter read the saved results. The analyser's normalised-profile crossing is not the experimental layer detector described in the thesis. The thesis full-height comparison uses raw profiles and fixed-region inventories, not that crossing.

The optional half-step branch requires the fine case's actual 60 s seed. It is not a completed comparison in the supplied thesis, and the 20 s command in README.md does not create that seed.
