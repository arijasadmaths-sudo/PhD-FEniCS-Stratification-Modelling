# Startup bounds diagnostic

`diagnose_rect3d_scalar_bounds.py` investigates the fine-case startup excursion using the equilibrated scalar solver. Submit with `submit_rect3d_bounds_diagnostic.sh` from this folder after setting the cluster options.

The diagnostic saves raw extrema, affected cells and independent residual checks. It does not clip the scalar or replace the production result. Keep its checkpoint separate from production checkpoints.
