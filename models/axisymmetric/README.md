# Axisymmetric models

- `current/`: current solver, launchers and restart utilities
- `verification/`: earlier controls and flux checks
- `earlier/`: archived 20 cc/min source and continuation

The archived 70 cc/min source is `archive/early_fenics_ipcs/HPC.py` from the
repository root. Its executable time step is 0.001 s despite the older header.

Comparison scripts are in `postprocessing/python/axisymmetric_verification/`.
