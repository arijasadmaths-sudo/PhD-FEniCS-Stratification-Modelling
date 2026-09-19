# Bounds continuation

`continue_rect3d_scalar_bounds.py` continues the separate startup diagnostic from 0.050 to 0.100 s. It requires the real diagnostic checkpoint and the matching equilibrated scalar module.

Use `submit_rect3d_bounds_continuation.sh` with the checkpoint path set for your machine. A diagnostic checkpoint is not a production restart. The reporting scripts retain the raw fields and residual checks.
