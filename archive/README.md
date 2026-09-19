# Earlier sources

These files retain the earlier model versions, including the original qualitative calculations in Chapter 6. They are not all interchangeable.

For the original Cartesian cases, `IPSC_0.001.py` has U_in = 0.001 m/s and `IPSC_Final_Parameters_60000.py` has U_in = 0.020 m/s. Both use dt = 0.001 s and beta = -6.27e-3; the thesis parameter table gives beta = -6e-3. The lower-flow file is configured for 600 s although the thesis figures stop at 60 s.

`early_fenics_ipcs/HPC.py` is the original 70 cc/min axisymmetric source, despite its location and older header. Its executable dt is 0.001 s.

`rectangular_3d_reference/` retains the earlier full-height reference formulation. The later solver updates are under `models/rectangular_3d_quantitative/`.
