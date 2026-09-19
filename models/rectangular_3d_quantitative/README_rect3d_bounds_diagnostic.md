# Rectangular fine-mesh scalar-bounds diagnostic

## Launcher correction after job 18989260

That job stopped before MPI startup completed, so it produced no verification
or scalar-bounds results. The earlier diagnostic launcher accidentally omitted
the environment setup block. Its Python executable, MPI plugin and expected
rank count were empty. This corrected version restores the working production
launcher's FEniCS activation, default `python3`, `--mpi=pmi2`, 24-rank launch
settings and single-thread configuration. It also uses the same login-shell
interpreter as that launcher. The diagnostic Python files are unchanged.

If the three diagnostic files are already on the HPC, replace only
`submit_rect3d_bounds_diagnostic.sh`, keeping its exact filename, then submit
the single line below. The job remains capped at 30 minutes with 64 GB total.

## Numerical issue being investigated

Job 18988659 passed all six verification checks in 30.3 seconds. It used
`equilibrated-2026-09-17` and got past the earlier GMRES failure. Production
stopped at step 10, t=0.010 s, because the density complement reached
1.0010806175, exceeding the configured upper limit 1.001. The dye maximum was
1.0003000606. This log reports neither an out-of-memory event nor a linear
solver failure. Its last full diagnostic, at 0.005 s, had density/dye budget
errors of 4.03e-9 and 3.06e-9 under the code's normalization, which uses a
minimum denominator of Q*dt during startup.

Conservation and small algebraic residuals do not guarantee bounded scalar
values. Mixed diffusion methods can violate a discrete maximum principle:
[Nakshatrala and Valocchi, 2009](https://arxiv.org/abs/0810.0322).
An independent RT0/DG0 calculation on a graded 1 mm box, with the physical
density diffusivity and a slow velocity ramp, produced a peak 1.0014733 even
with a balanced sparse direct solve and residual below 1.6e-20. This reproduces
the kind of startup overshoot, not the full 608,706-cell production calculation.
The actual fine-mesh magnitude, duration, location and affected volume still
need measurement.

## What this package does

It runs the same full fine-mesh equations for at most 50 timesteps:
dt=0.001 s, ending by 0.050 s. It starts at t=0 in a separate directory and
records density and dye extrema, affected cell counts and volumes, integrated
out-of-bounds amounts, and the existing scalar budgets at every timestep.
At the first crossing of the original production limit, it repeats both
scalar solves at tighter tolerances, then restores the baseline states and
solver parameters before advancing the diagnostic.

This diagnostic copy uses a 2% emergency scalar margin. That is an observation
ceiling, not an approved accuracy tolerance. The production script still has
its 0.1% margin. The copied diagnostic applies no clipping, redistribution,
physical parameter adjustment or change to the finite-element forms. Its
checkpoint schema is distinct, so its states cannot be resumed as production
checkpoints. Its output is marked `production_eligible=false`.

The allocation is limited to 30 minutes on 24 ranks with 64 GB. At the uploaded
job's early speed, 50 timesteps alone would take about 15 minutes, plus startup
and reporting. Queue time is additional, and completion within one allocation
is not guaranteed. A 24-minute soft wall budget can stop the diagnostic early.
There is no continuation chain.

## Submit

Copy these three files into the existing
`/path/to/home/rectangular_3d_quantitative_bundle` folder:

- `diagnose_rect3d_scalar_bounds.py`
- `rect3d_bounds_report.py`
- `submit_rect3d_bounds_diagnostic.sh`

They use the corrected `mixed_scalar_transport.py` and
`verify_mixed_scalar_transport.py` already used by job 18988659. The diagnostic
requires `SCALAR_SOLVER_REVISION=equilibrated-2026-09-17` and runs the established
verification before constructing the production mesh. Replace earlier copies
of the diagnostic launcher with the corrected version. Keep the reference
chain and the separate older 3D job running.

Run this single line from the existing bundle folder:

```bash
bash submit_rect3d_bounds_diagnostic.sh
```

The launcher submits exactly one allocation using account `YOUR_ACCOUNT` and
partition `compute`. It prints `Submitted batch job` and a job number.

## Return

When the job ends, upload `RETURN_rect3d_bounds_JOBID.zip` from the same folder.
The launcher includes the `.out`, `.err`, verification report and:

- `report.json`: peak, first threshold crossing, timestep history, affected
  volume, original-solver residuals and the tighter-solve comparison.
- `bounds_history.csv`: per-step extrema and integrated bounds errors.
- `largest_excursions.csv`: coordinates of up to 500 worst cells per scalar
  and snapshot. This table is a selection, not the full affected population.
- `scalar_bounds_snapshots.npz`: full coordinate-keyed fields at the first
  crossing, largest combined excursion and final observed step, with owned
  cell volumes and tighter-solve fields when available.

The volume-weighted metrics include the integrated excess above 1 and deficit
below 0. Their ratio to net source volume can be large at startup because the
source-fluid amount is initially very small; it is not a direct layer-height
error estimate. Counts outside [0,1] use a stated 1e-10 threshold to avoid
counting floating-point noise. Production-limit counts use the original
0.001 margin exactly.

If the launcher stops before producing the ZIP, return its
`rect3d_bounds_JOBID.out` and `.err`. No diagnostic output should be used as
thesis production data. The next decision depends on the measured overshoot;
this package does not declare the concentration issue resolved.

## Checks performed locally

Python compilation and shell syntax passed. Focused array/report checks cover
volume weighting, threshold classification, snapshot coordinate ordering, and
restoration of the physical mixed state and solver counters after a successful
or failed tighter solve. Launcher checks use fake commands and do not submit
real jobs. The launcher test now starts with all FEniCS settings unset and a
simulated inactive environment. It reproduced the empty-command failure with
the previous launcher. With the corrected launcher, it checks module loading,
the Python executable and all MPI arguments, thread settings, failure gates
and return-archive creation. Shell syntax and these checks pass.

The original production source is unchanged. DOLFIN/PETSc/MPI are not
available locally, so execution of this diagnostic still requires HPC.
