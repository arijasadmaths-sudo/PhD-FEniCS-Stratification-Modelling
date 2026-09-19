Axisymmetric verification

The current solver uses locally balanced DG0 transport without redistribution.
The underflow-aware residual check and restart guards are retained.

For the extra-fine quarter-step check, set the Slurm account and submit here:

  AXISYM_WORK_PARENT=/absolute/existing/work sbatch run_axisymmetric_extra_fine_quarter_50s.sh

This starts from rest with 200,736 triangles and dt = 0.00025 s. It stops at
5 s, not 50 s. Status paused and exit code 75 are expected at that review point.
The launcher writes a separate return folder and ZIP and will not overwrite one.

For a compatible restart, set AXISYM_EXTRA_FINE_QUARTER_RESUME to the absolute
checkpoint path, keeping its budget and earlier fields beside it. The default
stop remains absolute 5 s. Use the original matching sources for an old HPC
checkpoint; source hashes are part of the restart identity.

Other cases are selected with --case. Coarse, fine and fine_half were compared
to 50 s; fine_quarter, extra_fine, extra_fine_half and extra_fine_quarter were
compared to 5 s in the thesis. Set --end-time explicitly for a new run.

Local checks:
  python axisymmetric_70_study_50s.py --self-test
  python check_underflow.py --output /absolute/new/underflow_check.json

The Slurm launcher also runs the coupled restart check when FEniCS is available.
provenance/ retains the earlier repaired source and its checkpoint helper.
