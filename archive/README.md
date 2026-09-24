# Archived sources

Files in this directory are retained for provenance and checkpoint compatibility.
Sources, helpers and checkpoints are version-specific.

`early_fenics_ipcs/IPSC_0.001.py` uses `U_in = 0.001 m/s` and
`early_fenics_ipcs/IPSC_Final_Parameters_60000.py` uses `U_in = 0.020 m/s`.
Both execute with `dt = 0.001 s` and `beta = -6.27e-3`.

`early_fenics_ipcs/HPC.py` is the archived 70 cc/min axisymmetric source. Its
executable time step is 0.001 s despite the older header.

`rectangular_3d_reference/` contains the superseded full-height reference source.
The current full-height source is in `models/rectangular_3d_quantitative/`.
