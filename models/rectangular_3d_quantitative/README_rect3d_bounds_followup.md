# Bounds continuation

`continue_rect3d_scalar_bounds.py` continues the separate startup diagnostic from 0.050 to 0.100 s. It requires the matching diagnostic checkpoint and equilibrated scalar module.

Set the diagnostic checkpoint path in `submit_rect3d_bounds_continuation.sh`. Keep diagnostic and production restarts separate; reporting retains the raw fields and residual checks.
