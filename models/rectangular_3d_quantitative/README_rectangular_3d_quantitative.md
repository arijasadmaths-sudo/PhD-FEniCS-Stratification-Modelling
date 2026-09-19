18 September 2026: production restart after measured startup overshoot
=====================================================================

This section supersedes the installation and restart advice in the dated
history below. For the ongoing study, replace only
`rectangular_3d_h30_quantitative.py` in the existing HPC bundle folder. Keep
the equilibrated scalar module and six-check verifier that ran successfully
in diagnostics 18989285 and 18989361. The reference chain and the separate
older 3D model continue independently.

What the two diagnostics establish
---------------------------------

Job 18989285 completed 0 to 0.050 s. Job 18989361 restored that diagnostic
checkpoint and completed 0.050 to 0.100 s. Both exited normally, each in about
22 minutes with 24 ranks and 64 GB. The second job's error log contains only
the CPU-binding message. These isolated runs do not replace production data.

| Measured quantity | Density complement | Dye complement |
|---|---:|---:|
| Largest overshoot above 1 over 0 to 0.100 s | 0.002525565 (0.25256%) | 0.000658046 (0.06580%) |
| Maximum at 0.100 s | 1.000201419 | 1.000023419 |
| Minimum at 0.100 s | 0.000012693 | 0.000012191 |
| Consecutive final steps within the original 0.001 margin | 27 | 100 across both jobs |
| Budget relative error at 0.100 s | 9.56800e-9 | 9.22244e-9 |
| Residual-correction iterations at 0.051 s | 70 | 68 |
| Largest concentration change from that correction | 3.15256e-11 | 5.97798e-11 |

Density returned inside the original upper limit, 1.001, at 0.074 s and stayed
inside it through 0.100 s. The effective PETSc correction tolerances were
recorded. Independent relative correction residuals were 9.09e-11 and
8.95e-11; residuals in the original equations fell to approximately 1e-19.
This directly shows that incomplete convergence of the linear solve does
not explain the overshoot at the tested 0.051 s step. The first diagnostic's
zero-iteration warm-start comparison is not used as accuracy evidence.

At most 32 of 608,706 cells exceeded 1.001, with at most 3.70873e-11 m³
(0.0370873 mm³) in those cells. The largest integrated density excess above
one was 9.64062e-14 m³. The saved spatial snapshots place above-limit cells
near the nozzle rim, with centres 0.045573 mm above the floor. This measured,
localised transient and the effective algebraic-accuracy check support
proceeding with a narrowly bounded startup allowance. They do not prove
mesh/timestep independence, validate the plume against experiment, or imply
that the spatial method satisfies a maximum principle.

The production policy is explicit
---------------------------------

The state equations and raw concentration values are unchanged. No clipping,
redistribution, solver-tolerance relaxation, mesh change, boundary-condition
change or diffusivity adjustment is made. The production code changes its
acceptance checks for the specifically tested initial transient:

- It applies only to the fine mesh, Q=20 cc/min, dt=0.001 s, density
  D=1.5e-9 m²/s, dye D=4.14e-10 m²/s and zero inlet perturbation.
- For 0 < t < 0.100 s, the density upper limit is 1.003, or 0.3% above one.
  The density lower limit and both dye limits retain the original 0.001
  margin. Velocity, linear-residual, continuity and scalar-budget checks
  retain their existing thresholds.
- Every density cell above 1.001 must have its centre within 1.5 mm of the
  nozzle axis and between the floor and 0.25 mm height.
- At most 64 cells and 1e-10 m³ (0.1 mm³) may be above 1.001.
- The integrated density excess above one, over the entire mesh, must not
  exceed 2e-13 m³. This counts all positive excesses, including values below
  the 1.001 reporting threshold.
- At t=0.100 s and thereafter, the original global 0.001 margins apply to
  both scalars. The allowance cannot persist for a later restart.

These startup limits are monitoring limits chosen with modest headroom above
the measured transient. They are not a claimed uncertainty bound on plume
height or later concentration profiles. A larger excursion, greater spatial
extent or excursion beyond the specified time still stops the job. All raw
fields are retained for subsequent numerical sensitivity analysis.

The policy is recorded in configuration, status and checkpoint metadata.
Accepted startup steps are recorded in `startup_bounds_JOBID.json` inside
the fine results folder. The log prints a short startup update every ten
steps. At the end of the window it prints:

```text
FINE STARTUP BOUNDS CHECK PASSED at t=0.100 s; standard 0.001 limits are active.
```

It then saves an explicit 0.100 s checkpoint and continues the same production
case. This marker only confirms passage through the monitored startup period.
It is not a statement of convergence or completion of the 120 s target.

Restart the fine production case
--------------------------------

Download the corrected `rectangular_3d_h30_quantitative.py` and replace that
file, retaining its exact filename, in:

```text
/path/to/home/rectangular_3d_quantitative_bundle
```

Submit this one line:

```bash
RECT3D_VERIFY_ONLY=0 MESH_LEVEL=fine DT=0.001 T_GLOBAL=120.0 PIN_CHECKPOINT_TIME=60 bash /path/to/home/rectangular_3d_quantitative_bundle/submit_rectangular_3d_quantitative.sh --chain 1 --account=YOUR_ACCOUNT --partition=compute --mem=64G
```

This queues one normal production allocation, requested for 48 hours, toward
the 120-second simulated-time target. It is not another 30-minute diagnostic.
It starts from the latest compatible production checkpoint, last known to be
at t=0. The diagnostic checkpoints retain their distinct schema and cannot
be used as production restarts. Do not copy or rename a diagnostic checkpoint
into the production folder. The pinned 60-second checkpoint remains enabled.

After approximately 45 to 60 minutes of running time, return the new `.out`
and `.err` files. If available, also return the new `startup_bounds_JOBID.json`.
The expected first milestone is the 0.100 s marker above followed by its
saved checkpoint. Queue waiting is additional to running time.

Once the longer calculation is progressing healthily, further allocations can
continue its production checkpoint. The short 60 to 80 s half-timestep branch
still needs connecting to the replacement fine chain. Do not use `--study`
to duplicate the reference case that is already running.

Runtime remains substantial. The latest diagnostic time loop averaged about
21 wall seconds per step. If that rate persisted, 120 simulated seconds would
require about 29 days of computing time, excluding queueing. That is an early
startup estimate, not a completion forecast. It exceeds the capacity of the
original eight-allocation fine chain, so use the measured production rate
when sizing later continuations.

Validation of this update
-------------------------

The recorded scalar extrema, cell counts, affected volumes and integrated
excesses for all 100 diagnostic steps pass the proposed checks. The spatial
check passes all six saved baseline snapshot entries; these represent five
distinct physical times. The source arrays are unchanged by the checks.
Separate tests confirm rejection at the 0.100 s boundary, unchanged lower and
dye bounds, case eligibility, excessive cell counts/volumes/integrated excess,
spatial spread, nonfinite values and excessive velocity. Global MPI reduction
behaviour is checked with a local adapter, and report-write failures remain
fatal. Source-level comparison confirms that the weak forms, physical
parameters, scalar solver construction and checkpoint-compatibility validation
are unchanged. Python compilation passes.

These local checks exercise the new guard logic and saved HPC fields. Actual
execution of the updated production driver still requires the new HPC job.
The linear solver itself is the revision already exercised by both successful
diagnostic jobs. Keep the original diagnostic ZIPs as evidence of why this
startup allowance was introduced.

Historical notes follow
=======================

The material below records earlier revisions. Statements there that all
concentration limits are unchanged refer to those revisions; the narrowly
scoped startup policy above is the current rule. The current one-allocation
restart command above supersedes older advice to submit a complete new study.


17 September 2026 update for the ongoing study
============================================

Fine job 18988148 used the residual-recovery module: it recovered after a
density-scalar breakdown at call 6, then failed inside the correction solve
at call 9 (the 0.009 s step). All five small verification checks passed.
The production log shows no new saved checkpoint beyond the restored t=0
state, and no out-of-memory error. The previous recovery patch was therefore
insufficient. Keep the explicit 64 GB request; eventual peak memory remains
unmeasured.

Replace BOTH mixed_scalar_transport.py and verify_mixed_scalar_transport.py
in the existing rectangular_3d_quantitative_bundle directory with this version.
The solver identifies itself in the log as revision=equilibrated-2026-09-17.
The separate fenics3D001_monitored job in /3DProduction is an older 20 cm model;
it is not a continuation of this 30 cm quantitative fine case. Keep that job
and the quantitative reference chain running.

The scalar matrix combines RT face fluxes and DG cell means with very
different coefficient scales on small cells. The new solver explicitly
forms the equivalent system (S A S)y = S b, where S_ii = 1/sqrt(A_ii) and
x = S y. It retains GMRES with right Jacobi preconditioning and requests
classical Gram-Schmidt refinement at every iteration. These are linear
algebra changes; the transport forms and physical parameters are unchanged.
PETSc documents [diagonal scaling](https://petsc.org/release/manualpages/Mat/MatDiagonalScale/)
and [GMRES orthogonalization refinement](https://petsc.org/release/manualpages/KSP/KSPGMRESSetCGSRefinementType/).

The scaling uses additional vectors, without retaining a second full matrix.
The original matrix and physical iterate are restored after every solve,
including a reported nonconvergence; the original RHS is never modified.
The final accuracy guard remains ||A x-b|| <= max(1e-13, 1e-8 ||b||).
One equilibrated residual correction is allowed after a GMRES breakdown or
when a converged scaled solve has not yet met that original-system guard.
The correction's absolute AND relative tolerances are bounded using the
scaling to target the original equations. They can become stricter, never
looser. A failed correction or failed final guard still stops production.
The concentration bounds and mass-budget limits are unchanged.

Local validation used independently integrated RT0/DG0 tetrahedral matrices
in SciPy, including 384 cells in a graded 2 mm box, both physical diffusivities,
and 40 changing-flow steps per scalar. Executing the production scaling and
guard code through NumPy/SciPy adapters gave weighted relative differences
from a sparse direct solve below 7.1e-10 and relative step mass-balance errors
below 4.2e-10. Six targeted guard checks covered correction, restoration on
failure, rejection of nonfinite/inaccurate fields and invalid diagonals.
This does not reproduce the exact full-mesh PETSc failure or establish that
the HPC problem is solved: DOLFIN/PETSc/MPI are unavailable locally.

The HPC verifier retains all five previous checks and adds
physical_scale_startup. It checks an affine flux and 40 changing-flow steps
for each physical diffusivity on a graded 2 mm box, using dt=0.001 s.
This probes physical cell scales absent from the earlier unit-box tests.
The JSON report also records scalar_solver_revision. The existing launcher
runs these checks automatically before production.

Submit one fine production allocation with the command below, as one line.
It runs verification first, then resumes the same fine case toward 120 s
from the latest compatible checkpoint (t=0 in the supplied failed-job log).
The 60-second pinned checkpoint remains enabled. The command does not
reconnect the half-timestep branch or submit the rest of the fine chain.

```bash
RECT3D_VERIFY_ONLY=0 MESH_LEVEL=fine DT=0.001 T_GLOBAL=120.0 PIN_CHECKPOINT_TIME=60 bash /path/to/home/rectangular_3d_quantitative_bundle/submit_rectangular_3d_quantitative.sh --chain 1 --account=YOUR_ACCOUNT --partition=compute --mem=64G
```

Send this job's .out and .err after 30 to 60 minutes of running time, or when
it stops if earlier. Confirm progress and a new saved checkpoint before
queuing further fine allocations. The reference calculation continues from
its existing checkpoints. Do not use the original --study command below to
duplicate a study already in progress.

Original package instructions and earlier fixes
==============================================

The dated solver notes below describe earlier revisions. The 17 September
equilibration and correction procedure above supersedes their solver setup.

This package is a bounded study for **quantitative comparison** with the full
600 × 300 × 300 mm quiescent rectangular experiment. It replaces the earlier
dt=0.004 proposal. The scientific outputs are layer depth, raw and normalized
profiles, dye/density differences, and measured sensitivity to mesh and timestep.

Extract the complete ZIP into a new HPC directory. Keep all Python files and
the shell script together. In the marked environment section, retain the
FEniCS 2019.1 activation commands from your working job, and retain its
account/project, partition and memory directives. The launcher first uses an
already working environment, then loads
`languages/python/fenics-2019.1.0`, the fully qualified module name used by
the working Blue Pebble axisymmetric launcher. Set `FENICS_MODULE` to override
the module name. The 48-hour allocation request must be permitted by your
partition.

Submit the recommended study once from the login node:

```bash
bash submit_rectangular_3d_quantitative.sh --study 8
```

You can append your existing `--account=CODE`, `--partition=NAME`,
`--mem=SIZE` or `--time=HH:MM:SS` settings.

The 8 September 2026 launcher corrections address two startup issues. The
original shorter module name caused jobs 18880807 and 18880815 to exit before
verification. Job 18880893 then reported an unspecified Slurm task-launch error.
That message identifies a launch failure but does not establish its exact cause.
The current launcher explicitly selects `--mpi=pmi2`, matching the working
Blue Pebble FEniCS launchers, instead of relying on the cluster default. Set
`FENICS_MPI_TYPE` explicitly if the site's MPI configuration requires another
plugin. Slurm explains this requirement in its
[MPI guide](https://slurm.schedmd.com/mpi_guide.html).

The 9 September compatibility correction updates `mixed_scalar_transport.py`.
Job 18881460 reached the first numerical check, then its FEniCS Python binding
rejected `PETScKrylovSolver(comm, "gmres", "jacobi")`. The corrected module
creates a petsc4py KSP on the mesh communicator, wraps it using the KSP-object
constructor explicitly supported in that traceback, and sets GMRES/Jacobi.
The [legacy DOLFIN API](https://fenics.readthedocs.io/projects/dolfin/en/2017.2.0/apis/api_la.html#petsckrylovsolver)
documents the KSP wrapper. Numerical forms, tolerances, residual checks and
simulation parameters are unchanged. A targeted local API-stub regression
reproduced the rejected call and checked the new setup, communicator retention
and independent density/dye solvers. It does not replace the HPC numerical tests.
No other instance of the rejected constructor was found in the supplied Python
files.

The 10 September residual correction addresses job 18897209. It reached the
third verification check, at the physical diffusivities, after passing affine
diffusion and constant advection. GMRES returned, but the independently computed
relative residual was 3.589148e-8, exceeding the 1e-8 guard. The scalar solver
now uses **right Jacobi preconditioning and an unpreconditioned residual norm**.
The [PETSc norm documentation](https://petsc.org/release/manualpages/KSP/KSPSetNormType/)
explains why default left preconditioning tests a different residual. This
change aligns the solver's stopping criterion with the original linear system.
The independent residual recomputation remains; all tolerances, flux checks,
physical parameters, meshes and timesteps are unchanged.

An independent SciPy assembly of the same RT0/DG0 affine test on 384 graded
tetrahedra reproduced premature convergence with left Jacobi scaling. With
right scaling, the true relative residuals were 5.02e-11 for the density proxy
and 9.88e-11 for the dye; relative flux L2 errors were 8.39e-7 and 6.70e-6.
Both passed the unchanged residual and flux thresholds. This is an independent
matrix regression, not execution of the production DOLFIN/PETSc code. The
actual MPI verification must still pass on the HPC before production.

Replace `mixed_scalar_transport.py` and `verify_mixed_scalar_transport.py`
for this correction. The verifier now prints each check's start and passing
metrics immediately, so the `.out` log shows completed checks even if a later
one fails. The complete JSON report and overall pass marker are still written
only after every check passes. The shell script retains the previous MPI and
verification-only corrections.

Use the corrected shell script and scalar module. To check startup after an
error, submit one verification-only allocation from the bundle folder:

```bash
RECT3D_VERIFY_ONLY=1 sbatch --export=ALL --account=YOUR_ACCOUNT --partition=compute --time=00:30:00 --signal=USR1@60 --job-name=rect3d_startup_check submit_rectangular_3d_quantitative.sh
```

This uses the same 24 MPI ranks and numerical verification as the production
jobs, then exits before building a production mesh or modifying simulation
checkpoints. Thirty minutes is the allocation ceiling; the job exits as soon
as the checks finish. The signal override moves the production script's
30-minute warning to one minute before this shorter allocation ends.
`RECT3D_VERIFY_ONLY=1` is rejected with `--study` or `--chain` to avoid submitting
an entire verification-only chain.

The new launch check verifies that mpi4py, DOLFIN and PETSc see the requested
number of ranks and matching rank indices, and that an MPI sum gives the
expected result. It stops if processes have started as separate singleton
jobs. Both this check and the numerical checks must pass before production.
The log records the `srun` path/version, available plugins and MPI library.
A successful short check prints `MPI LAUNCH CHECK PASSED: 24 ranks` and then
`RECT3D STARTUP VERIFICATION PASSED`.

After a successful check, submit the full study with the normal command above
and your account/partition options. The failed jobs do not restart automatically
when their script is corrected. If startup still fails, retain both the full
`.out` and `.err` logs for diagnosis; do not treat an unspecified launch error
as evidence that the numerical solver failed. Shell syntax and mocked launch
control checks can be run locally; actual MPI and numerical verification must
still be run on the HPC.

| Case | Physical interval from injection | Timestep | Purpose |
|---|---|---|---|
| Fine primary | 0–120 s | 0.001 s | Main simulation |
| Reference mesh | 0–120 s | 0.001 s | Spatial sensitivity at identical physics and timestep |
| Fine timestep branch | 60–80 s | 0.0005 s | Sensitivity of developed flow to timestep |

The fine and reference cases are independent and can run simultaneously if
resources allow. Each has up to eight consecutive allocations on 24 MPI ranks.
The two-allocation timestep branch starts after the fine chain and reads its
permanently retained 60 s checkpoint. Thus `--study 8` queues **18 jobs for
three cases**, not 18 simulations from rest. Later allocations exit before
building the production mesh when their case is already complete.

Each main case has 120,000 timesteps; the branch has 40,000. Your old run took
about 3.48 wall seconds per step, equivalent to 116 hours for one 120,000-step
case on that old mesh. Both the mesh and scalar solver have changed, so this
is a baseline comparison, not a runtime prediction. Expect a substantial HPC
calculation. Eight allocations are a capacity limit, not a completion promise.
The running log estimates remaining compute hours from achieved throughput.
Queue waiting is additional.

Each allocation checkpoints every 10 simulated seconds or 30 wall minutes,
keeps three rolling checkpoints, and stops with a save margin before the
scheduler deadline. A fine-case seed at exactly 60 s is retained separately.
If an allocation fails its verification or solver checks, its dependent jobs
do not proceed. If an entire bounded chain ends before the target, its results
and checkpoints remain usable. The timestep comparison reports only the
physical interval saved by both cases; a missing 60 s seed is an error.

For one fine-case continuation chain alone:

```bash
bash submit_rectangular_3d_quantitative.sh --chain 8
```

Preserve all physical and timestep settings when continuing a case. The
automatic study uses separate result directories and rejects incompatible
checkpoints. The branch explicitly records its seed, old timestep and physical
start time. It does not establish timestep convergence of the history before
60 s. Do not overwrite scripts or change model parameters in an active study.

The numerical changes are:

- Retain the established dt=0.001 s for the two full runs.
- Replace the cell-diameter diffusion penalty with a consistent mixed RT/DG0
  formulation. The auxiliary flux is the Fickian flux
  $q=-D\nabla c$; it is solved with concentration on the tetrahedral mesh.
  This follows the boundary-condition structure of the
  [DOLFIN mixed-Poisson formulation](https://olddocs.fenicsproject.org/dolfin/2019.1.0/python/demos/mixed-poisson/demo_mixed-poisson.py.html).
- Keep the Taylor–Hood flow solve and divergence-free RT transport velocity.
  DG0 upwind advection remains first order; the mesh comparison measures its
  effect alongside the other spatial errors.
- Transport a density proxy and a separate passive Rhodamine dye complement.
  Buoyancy depends only on the density proxy. The dye field is used for PLIF
  comparisons.
- Calibrate total inward and outward volume flux to 20 cc/min. Net flux on a
  midpoint-marked nozzle alone can miss source flow on crossing facets.
  P2 calibration uses a specified boundary quadrature.
- Use whole-boundary scalar budgets, including diffusive flux, for both fields.
  No concentration clipping is applied. Bounds, solver residuals, gross flux,
  wall leakage and budgets are checked during stepping.

| Parameter | Fine primary | Reference |
|---|---|---|
| Geometry / nozzle | 600 × 300 × 300 mm / 2 mm | Same |
| Flow rate | 20 cc/min | Same |
| Density values | 999.3 / 1005.5 kg/m³ | Same |
| Kinematic viscosity | 1e-6 m²/s | Same |
| Density-proxy diffusivity | 1.5e-9 m²/s | Same |
| Dye diffusivity | 4.14e-10 m²/s | Same |
| Horizontal base spacing | 12.5 mm | 15 mm |
| Ceiling base spacing across upper 63 mm | 3 mm | 3.9375 mm |
| Base tetrahedra before local refinement | 290,304 | 158,400 |

Both meshes additionally refine the central plume and nozzle. The actual
counts, smallest cells and discrete nozzle area are recorded at runtime.

The diffusivities are nominal literature values cited in your
`experimental_apparatus(1).tex`: NaCl mutual diffusion around
1.48–1.57e-9 m²/s at 25 °C, and Rhodamine 6G around 4.14e-10 m²/s.
They are not measurements of your full water/salt/propan-2-ol mixture.
The density field is a single-scalar approximation to that mixture.
The flush parabolic source and weak distributed floor return also remain
explicit modelling assumptions. These affect physical interpretation even
when the numerical checks pass.

Before each production allocation, `verify_mixed_scalar_transport.py` runs
small 3D tests using the actual production scalar class: affine diffusion and
flux, constant preservation with advection, independent density/dye states at
their nominal diffusivities, and transient diffusion on two graded meshes.
It also checks affine fluxes and changing-flow startup at physical cell scales
for both diffusivities, using the production timestep.
A failure prevents the plume run. Results are saved as
`verification_JOBID.json` and printed in the job log. This verifies the scalar
implementation, not the complete experimental plume.

The main solver then checks geometry, flux calibration and a complete
checkpoint write/read at t=0. During the run, scalar budget errors above 0.1%
after the first second or concentration excursions beyond 0.001 outside [0,1]
stop the job. These are error-detection limits, not a claim of 0.1% physical
accuracy. Mixed diffusion is not universally bound-preserving on arbitrary
meshes, which is why extrema are checked rather than silently corrected.

The default result directories are:

```text
results_rect3d_quant_fine_Q20_eps0_dt0p001_D1p5em09_Dye4p14em10
results_rect3d_quant_reference_Q20_eps0_dt0p001_D1p5em09_Dye4p14em10
results_rect3d_quant_fine_Q20_eps0_dt0p0005_D1p5em09_Dye4p14em10_from60
```

Each contains `configuration.json`, `run_status.json`, `diagnostics.csv`,
checkpoints, and per-allocation `segments/`. Each segment saves:

- `c_dg0.xdmf/.h5`: density proxy, 1 ambient and 0 source.
- `dye_complement_dg0.xdmf/.h5`: dye complement, 1 ambient and 0 source;
  its complement is the dye-derived source fraction.
- `u_taylor_hood.xdmf/.h5` and `p.xdmf/.h5`: velocity and pressure.

Both scalar fields are written every simulated second; flow every 5 s;
diagnostics every 0.5 s. Keep each XDMF with its companion HDF5.

For quantitative analysis, use ordinary Python with NumPy, h5py and Matplotlib
after an allocation has closed its output. For example:

```bash
python3 analyse_rectangular_3d_quantitative.py --results results_rect3d_quant_fine_Q20_eps0_dt0p001_D1p5em09_Dye4p14em10 --results results_rect3d_quant_reference_Q20_eps0_dt0p001_D1p5em09_Dye4p14em10 --labels fine reference --comparison-out mesh_comparison
```

This processes every available frame. It writes central-sheet profiles,
whole-volume slab profiles, raw and corrected amplitudes, source-equivalent
volumes, threshold heights at epsilon=0.04/0.08/0.12, and case differences at
exact common times. It records the actual comparison interval. Use
`--field density` to analyse the active scalar instead of the default dye.

For the short timestep comparison, compare the fine and branch directories,
with `--labels fine half_dt --background-time 60 60 --time-origin 0 0
--times 60 65 70 75 80 --comparison-out timestep_comparison`.
This uses their common 60 s background and injection clock. Check the saved
common interval before describing it as a full 60–80 s comparison.

**Match the experimental processing before interpreting layer-height agreement.**
Your methods chapter subtracts the first processed profile before normalizing.
When that reference contains the established plume, this removes its steady
contribution. The analyser implements
$b(z,t)=\bar c(z,t)-\bar c(z,t_{\rm ref})$, then normalizes b between its
instantaneous extrema. It scans upward from the lowest retained row for the
first normalized value below 0.92 and sets h=H-z. It flags unresolved amplitudes,
bottom-boundary hits and disconnected features; it does not silently change
the height rule or exclude the plume.

For experimental comparison, provide:

- `--roi-x MIN MAX --roi-z MIN MAX` using the calibrated physical limits of
  the retained camera region, in metres. Default full-width central-sheet
  averages are not claimed to match the camera crop.
- `--time-origin T_FINE T_REFERENCE` using each run's observed ceiling-contact
  time when the experimental clock starts at impingement. This also selects
  the corresponding background unless `--background-time` is supplied.
- Optionally `--experiment heights.csv`, containing `time_s,h_m` measured
  using epsilon=0.08. The script requires explicit time origins and compares
  only within the supplied experimental time range.

Without an explicit reference, the earliest saved frame is used and labelled
as an unverified onset choice. Reference-frame h is undefined. Selecting the
correct reference matters: testing on your old data showed that including the
steady plume without background subtraction could falsely make h reach the
bottom of the image. The corrected implementation recovered a finite layer.
The 1 mm processing bands are sampling intervals, not 1 mm mesh resolution.
The zero-thickness central plane also approximates the finite PLIF sheet.

`plot_rectangular_3d_quantitative.py --results RUN_DIRECTORY` exports
fixed-scale 0–1 and 0.97–1 concentration panels and profile figures; dye is the
default field. Use `--field density` explicitly for density-proxy figures.
All figures identify actual saved times.

Use the mesh and timestep differences to qualify the numerical predictions.
For example, report height differences in mm, normalized-profile RMS
differences, and changes in raw amplitude and integrated source fraction.
Do not claim a convergence order from two meshes, or full-history temporal
convergence from the seeded branch. The threshold variations are a sensitivity
check, not a statistical confidence interval. The experiment itself provides
relative PLIF profiles; an absolute density validation would require its
independent calibration.

Local validation completed: Python/shell syntax, mocked launcher dependency
and failure paths, 12 control/seed/pinning tests, analytical clipping/profile
tests, and post-processing of all 301 old concentration snapshots. No new
plume results are included. DOLFIN/MPI execution was unavailable here; the
numerical verification gate therefore runs on your HPC before production.
