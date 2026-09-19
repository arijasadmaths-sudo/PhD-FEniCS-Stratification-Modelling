# Axisymmetric models

`current/` contains the locally balanced transport used for Section 6.5.4, including the underflow repair and extra-fine quarter-step check. `verification/` contains the earlier controls and flux pilots.

The two named sources in `earlier/` retain the original 20 cc/min calculation and its continuation. The original 70 cc/min source is `archive/early_fenics_ipcs/HPC.py` from the repository root. Its executable time step is 0.001 s despite the older wording in its header.

The final quarter-step comparisons are in `postprocessing/python/axisymmetric_verification/`.
