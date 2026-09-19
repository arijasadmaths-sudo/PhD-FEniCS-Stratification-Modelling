# Full-height reference source

Earlier 600 x 300 x 300 mm source for the full-height formulation described in Section 6.6.5 and Appendix D.7. It uses the right-Jacobi mixed scalar solve, before the later equilibration and fine-startup changes.

Keep these four files together. Set MESH_LEVEL=reference and DT=0.001 for the reference case, and use a new output directory. The supplied thesis reports fields through 36 s and a checkpoint at 36.197 s; the configured 120 s target is not an attained result.

The analysis and plotting helpers are in `models/rectangular_3d_quantitative/` from the repository root. Existing checkpoints require the matching original sources and settings.
