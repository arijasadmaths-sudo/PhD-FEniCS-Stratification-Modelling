Axisymmetric solver and verification launcher

The solver supports these --case values:

  coarse, fine, fine_half, fine_quarter,
  extra_fine, extra_fine_half, extra_fine_quarter

Set --end-time explicitly for a new direct run.

Extra-fine quarter-step launcher
--------------------------------

Set the Slurm account and submit from this directory:

  AXISYM_WORK_PARENT=/absolute/existing/work sbatch run_axisymmetric_extra_fine_quarter_50s.sh

The launcher filename is retained for compatibility, but its default review stop is
5 s. A paused status with exit code 75 is normal. Each run is written to a new
return folder and ZIP.

To restart, set AXISYM_EXTRA_FINE_QUARTER_RESUME to the absolute checkpoint path
and keep the matching budget and field files beside it. Old checkpoints require
their matching source because the source hash is part of the restart identity.

Local checks
------------

  python axisymmetric_70_study_50s.py --self-test
  python check_underflow.py --output /absolute/new/underflow_check.json

When FEniCS is available, the launcher also runs the coupled restart check. The
earlier repaired source and checkpoint helper are in provenance/.
