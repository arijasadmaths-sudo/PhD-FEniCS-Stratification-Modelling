# Full-height rectangular 3D model

Keep the driver, scalar module, verifier and launcher together. The flow solve
defaults to PETSc block Jacobi with local ILU on each MPI subdomain; set
`RECT3D_MONO_BLOCK_PC` to override it.

Submit one fine-mesh allocation after setting the account and partition:

```sh
RECT3D_MONO_BLOCK_PC=bjacobi MESH_LEVEL=fine DT=0.001 T_GLOBAL=20 PIN_CHECKPOINT_TIME=0 bash submit_rectangular_3d_quantitative.sh --chain 1 --account=YOUR_ACCOUNT --partition=compute --mem=64G
```

Confirm completion from `run_status.json`. Do not modify files in an active run or
write a second run to the same output directory.

The superseded reference source is in `archive/rectangular_3d_reference/`. See
`README_rectangular_3d_quantitative.md` for configuration, outputs and seeded-run
instructions.
