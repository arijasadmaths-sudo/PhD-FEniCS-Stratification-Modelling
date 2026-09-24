# Full-height rectangular 3D model

This folder contains the later 600 x 300 x 300 mm model, with separate density and dye transport at 20 cc/min. Keep the solver, scalar module, verifier and launcher together.

The scalar solver includes the September 17 equilibration and residual correction. The September 18 driver adds a monitored fine-mesh startup allowance. Neither change clips or redistributes the concentration. The monolithic flow solve now defaults to PETSc block Jacobi with local ILU on each MPI subdomain. This was the stable production configuration after HYPRE intermittently returned PETSc error 76 on Blue Pebble.

For one new fine allocation, with the account and work directory set for your machine:

```sh
RECT3D_MONO_BLOCK_PC=bjacobi MESH_LEVEL=fine DT=0.001 T_GLOBAL=20 PIN_CHECKPOINT_TIME=0 bash submit_rectangular_3d_quantitative.sh --chain 1 --account=YOUR_ACCOUNT --partition=compute --mem=64G
```

This is a 20 s target, not a claim of completion. Do not replace files in an active HPC run or launch another copy into its output directory.

The thesis reference fields use the earlier source in `archive/rectangular_3d_reference/`, not this later revision. See `README_rectangular_3d_quantitative.md` for the solver choice, startup limits, time-step comparison and output interpretation.
