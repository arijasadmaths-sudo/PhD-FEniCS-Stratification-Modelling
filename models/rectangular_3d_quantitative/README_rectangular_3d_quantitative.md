# Configuration and outputs

## Default case

- `MESH_LEVEL=fine`
- `DT=0.001`
- `FLOW_RATE_CC_MIN=20`
- `SCALAR_DIFFUSIVITY=1.5e-9`
- `DYE_DIFFUSIVITY=4.14e-10`

The density scalar drives buoyancy; the dye scalar is passive. The flow space is
Taylor-Hood, velocity is reconstructed in RT, and scalar diffusion uses the mixed
RT/DG0 solve.

## Solver options

The monolithic flow solve uses a Schur field split. The default flow-block
preconditioner is PETSc block Jacobi with local ILU. Set
`RECT3D_MONO_BLOCK_PC=bjacobi`, `hypre` or `gamg` to select another supported
option. The H(div) reconstruction uses HYPRE independently of this setting.

Each allocation runs the scalar verifier before production.

## Startup-bound policy

For the fine case at the default parameters, the density upper margin is 0.003 for
`0 < t < 0.100 s`; all other scalar margins remain 0.001. During this interval,
cells above 1.001 must satisfy all of these limits:

- within 1.5 mm of the source axis
- within 0.25 mm of the floor
- at most 64 cells
- affected volume no greater than 1e-10 m3
- integrated excess above one no greater than 2e-13 m3

Standard limits resume at 0.100 s. These checks do not modify scalar values.

## Outputs and restarts

The output root contains `configuration.json`, `run_status.json`,
`diagnostics.csv`, accepted checkpoints and XDMF/HDF5 fields. Keep each XDMF file
with its HDF5 companion. Checkpoints written by the separate bounds diagnostic
cannot restart production.

## Analysis utilities

`analyse_rectangular_3d_quantitative.py` reads saved fields and writes profiles,
inventories and comparison data. `plot_rectangular_3d_quantitative.py` plots those
outputs. The normalised-profile crossing reported by the analyser is a numerical
diagnostic.

## Seeded run

Use a new output root when starting from a compatible seed:

```sh
RECT3D_MONO_BLOCK_PC=bjacobi MESH_LEVEL=fine DT=0.0005 T_GLOBAL=20 SEGMENT_DURATION=20 PIN_CHECKPOINT_TIME=0 RECT3D_OUTPUT_ROOT=/path/to/new_output SEED_CHECKPOINT=/path/to/compatible_seed.npz bash submit_rectangular_3d_quantitative.sh --chain 1 --account=YOUR_ACCOUNT --partition=compute --mem=64G
```

The seed is external simulation output and is not included in this repository.
Seed metadata and solver settings are checked before loading.
