# Solver settings and outputs

The full-height model uses dt = 0.001 s, Q = 20 cc/min, density diffusivity 1.5e-9 m2/s and dye diffusivity 4.14e-10 m2/s. Buoyancy depends on the density scalar only. The flow is Taylor-Hood with an RT transport reconstruction; scalar diffusion uses the mixed RT/DG0 solve.

The equilibrated scalar solve retains an independent residual check in the original equations. Verification runs before each production allocation. Passing these checks does not establish mesh independence.

The monolithic Taylor-Hood flow solve uses a Schur field split. Its production default is PETSc block Jacobi with a local ILU solve on each MPI subdomain. HYPRE was replaced for these two blocks after intermittent PETSc error 76 failures occurred on several Blue Pebble nodes and from several valid checkpoints. The H(div) reconstruction still uses HYPRE. The flow-block choice can be set explicitly with `RECT3D_MONO_BLOCK_PC=bjacobi`, `hypre` or `gamg`; the validated production calculations used `bjacobi`.

For the tested fine case only, the density upper margin is 0.003 during 0 < t < 0.100 s. All other scalar margins remain 0.001. Cells above 1.001 must lie within 1.5 mm of the source axis and 0.25 mm of the floor, with at most 64 cells and 1e-10 m3 affected. The total excess above one must not exceed 2e-13 m3. At 0.100 s the standard bounds apply again. No field values are changed by these checks.

Results include configuration, diagnostics, accepted checkpoints and XDMF/HDF5 fields. Keep each XDMF file with its HDF5 companion. Diagnostic bounds checkpoints cannot restart production.

The supplied analyser and plotter read the saved results. The analyser's normalised-profile crossing is not the experimental layer detector described in the thesis. The thesis full-height comparison uses raw profiles and fixed-region inventories, not that crossing.

The bounded time-step comparison uses the fine-mesh state at 6.979 s as its common seed and advances with `dt = 0.0005 s` toward the same 20 s target as the primary fine case. Keep it in a distinct output root:

```sh
RECT3D_MONO_BLOCK_PC=bjacobi MESH_LEVEL=fine DT=0.0005 T_GLOBAL=20 SEGMENT_DURATION=20 PIN_CHECKPOINT_TIME=0 RECT3D_OUTPUT_ROOT=/path/to/results_rect3d_quant_fine_Q20_eps0_dt0p0005_D1p5em09_Dye4p14em10_from6p979 SEED_CHECKPOINT=/path/to/seeds/fine_time_6p979s.npz bash submit_rectangular_3d_quantitative.sh --chain 1 --account=YOUR_ACCOUNT --partition=compute --mem=64G
```

The seed is an external simulation output and is not included in this repository. A different compatible seed may be used, but comparisons must begin at its actual saved time and use separate result directories.
