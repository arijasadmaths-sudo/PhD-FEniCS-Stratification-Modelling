# Startup bounds diagnostic

`diagnose_rect3d_scalar_bounds.py` investigates the fine-case startup excursion using the equilibrated scalar solver. Submit with `submit_rect3d_bounds_diagnostic.sh` from this folder after setting the cluster options.

Outputs include raw extrema, affected-cell data and independent residual checks. Treat them as diagnostic output only: scalar values are unchanged, and the checkpoint directory must remain separate from production.
