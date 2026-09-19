AXISYMMETRIC EXTRA-FINE QUARTER-TIMESTEP CHECK

Purpose
One new run: 200,736 triangular cells, refinement factor 4,
dt = 0.00025 s, starting from salt water at rest with zero freshwater.
Compare this with the returned extra-fine dt=.0005 and .001 s runs and
the fine dt=.00025 s run at matched physical times of 2 and 5 seconds.
All physical parameters, the 2 s inlet ramp, momentum equations, scalar
transport and the reviewed underflow residual repair remain unchanged.

Submit from a new extracted folder on Blue Pebble:

  cd axisymmetric_extra_fine_quarter_check
  AXISYM_EXTRA_FINE_QUARTER_RESUME= AXISYM_EXTRA_FINE_QUARTER_CONTINUE_50=0 sbatch run_axisymmetric_extra_fine_quarter_50s.sh

Only this case is submitted. Do not add --array.
The job requests 1 rank, 1 CPU, 64 GB and 72 hours under YOUR_ACCOUNT,
using languages/python/fenics-2019.1.0. Estimated full runtime is 60-70 h;
runtime can vary, so a second allocation may be needed.

The normal review stop is exactly t=5 s (step 20,000). The target stored
in the checkpoint identity is 50 s solely to retain the option of a later
reviewed continuation. The default launcher does NOT run to 50 s.
Status paused and exit code 75 are expected at the 5 s review stop.

Return
Send the whole ZIP created beside the scripts:

  axisymmetric50_extra_fine_quarter_RETURN_<job-ID>.zip

It contains the cumulative CSV, comparison fields at 0/2/5 s when reached,
latest accepted checkpoint, status, exact source files and hashes,
restart/underflow preflight results, analysis and supplied reference fields.
Do not send only the checkpoint. Keep the complete return folder too.

Wall-time pause and restart
Slurm sends the batch script USR1 ten minutes before the wall-time limit.
The script forwards it to the serial Python solver. The solver finishes
its current accepted step, saves an atomic checkpoint and exits 75;
the launcher creates the return ZIP. Checkpoints are also saved every 0.5 s.
There is no automatic resubmission.

For an early pause, use an ABSOLUTE path to the latest quarter-case
checkpoint in its RETURN folder, with verification_budget.csv and all
earlier field_step files still beside it:

  AXISYM_EXTRA_FINE_QUARTER_RESUME=/absolute/path/axisymmetric50_extra_fine_quarter_RETURN_<job-ID>/checkpoint_latest.npz AXISYM_EXTRA_FINE_QUARTER_CONTINUE_50=0 sbatch run_axisymmetric_extra_fine_quarter_50s.sh

Replace <job-ID> and /absolute/path with the actual values. Put these
environment variables BEFORE sbatch as shown. Do not append them after
the script name. A resumed allocation still pauses at absolute 5 s:
the launcher subtracts the accepted checkpoint step from 20,000.
Do not alter solver/helper files between allocations. All older half-step,
full-step and other-case checkpoints are rejected. The solver also checks
the full physical, mesh/DOF and software identity before accepting a resume.
Each new job gets a new work/return directory; existing paths are protected.

Hard kill/manual packaging
If no ZIP was created after a hard kill, recover the accepted checkpoint
and accumulated evidence from the work directory printed in the log:

  python3 pack_return.py --run-root /path/to/work/axisymmetric50_extra_fine_quarter_<job-ID> --destination . --label recovered

This creates a new ZIP without replacing an earlier return. The CSV may
have rows after the checkpoint; resume carries only the accepted history
through that checkpoint. Failure snapshots are diagnostic, not restartable.

After the 5 s review
Send the ZIP for comparison. No 50 s continuation is requested now.
An explicit continuation remains technically available only with a
quarter-case checkpoint at or beyond 5 s and the opt-in environment
variable AXISYM_EXTRA_FINE_QUARTER_CONTINUE_50=1. The default remains 0.

Validation and provenance
Local checks cover unchanged equations, kernel transport/conservation,
underflow guard regression, case/time mappings, restart identity, the
absolute review stop and launcher/ZIP paths using synthetic simulator
outputs. No full DOLFIN coupled trajectory has been run locally.
On Blue Pebble the launcher repeats the numerical self-tests and underflow
regression, then check_restart.py compares four uninterrupted coarse steps
with two steps + checkpoint + two steps. Production starts only after that
coupled restart gate passes. This gate does not establish convergence.
provenance/ contains the exact repaired baseline source and repair evidence;
SOURCE_PROVENANCE.json records hashes. Fresh/continued production files
carry their own source hashes and allocation history.
