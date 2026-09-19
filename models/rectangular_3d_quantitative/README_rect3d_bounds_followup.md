# Fine-mesh bounds diagnostic: findings and one continuation

## What job 18989285 established

The diagnostic completed all 50 steps, from 0 to 0.050 s, with exit code 0.
All six numerical verification checks passed. The error log contains only
the CPU-binding message. Total wall time was 1329.95 s, about 22.2 minutes;
the time loop took 1057.73 s. This was an isolated diagnostic with a 2%
emergency ceiling, not a production run passing the original 0.1% limit.

| Quantity | Density complement | Dye complement |
|---|---:|---:|
| Largest value during the diagnostic | 1.002525565 at 0.037 s | 1.000658046 at 0.033 s |
| Largest excursion above 1 | 0.25256% of the normalized range | 0.06580% of the normalized range |
| Maximum value at 0.050 s | 1.001867866 | 1.000319352 |
| Most cells exceeding the production margin at once | 32 of 608,706 | 0 |
| Largest volume exceeding that margin | 3.70873e-11 m³ = 0.0370873 mm³ | 0 |
| Reported budget error at 0.050 s | 1.97186e-8 | 1.81884e-8 |

The production upper limit is 1.001. Density first crossed it at 0.010 s and
was still above it at the endpoint. The peak began falling after 0.037 s.
No concentration fell below zero during these 50 steps.

An independent calculation from the full NPZ snapshots confirms the recorded
extrema, threshold counts and volume-weighted excesses. In the saved first
crossing, peak and final snapshots, cells with density complement above
1.001 are in the first cell layer near the 1 mm-radius nozzle rim. Their cell
centres are 0.045573 mm above the floor. The three snapshots establish
localisation at those times; they do not provide a spatial snapshot at every
timestep.

At the density peak, the integrated excess above 1 is 8.29123e-14 m³,
equivalent to 0.11755% of the net source-fluid volume then present. At 0.050 s
it is 6.54273e-14 m³, or 0.03927% of the net source volume. These are integrated
scalar-error measures, not estimates of plume-height error. Ratios to the
source amount are larger at the very beginning, when almost no source fluid
has entered. The budget-error denominator is the larger of expected source
volume and Q*dt; the listed budget errors use that startup floor.

## Limitation of the original tighter-tolerance check

Both requested tighter solves reported zero iterations. Their maximum
changes, 2.22e-16, therefore do not demonstrate that a fresh solve confirmed
the overshoot. The check reused the converged state as its initial guess.
PETSc's default convergence test can accept an initial guess immediately
when its residual already meets the tolerance relative to the full RHS.
See [PETSc's convergence-test documentation](https://petsc.org/release/manualpages/KSP/KSPConvergedDefault/).
The original report does not record enough effective solver settings to
attribute the zero iterations more specifically.

The localisation, decline and previous independent small-mesh calculations
are consistent with a spatial-discretisation overshoot. They do not establish
that interpretation conclusively on this fine mesh or justify an unrestricted
production run. The existing production guard remains 0.001.

## What the supplied continuation does

One allocation advances the saved diagnostic from 0.050 to 0.100 s, at
dt=0.001 s: 50 further steps. It copies this existing HPC checkpoint into a
new diagnostic directory:

```text
diagnostic_rect3d_bounds_18989285/checkpoint/state_step_000000050.npz
```

The checkpoint's schema, physical parameters, mesh identity and step/time are
checked. Missing or incompatible data stop the job; it cannot silently start
from zero. The original checkpoint is preserved. The new results directory
is `diagnostic_rect3d_bounds_followup_NEWJOBID`. There is no continuation chain.
The job uses 24 ranks, 64 GB total and a 30-minute allocation, with the same
24-minute soft wall budget and 3-minute save margin as the successful test.
Completion within that allocation is not guaranteed.

The numerical forms, mesh, physical parameters and accepted baseline fields
are unchanged. The diagnostic still has its separate schema and 2% emergency
ceiling. None of its checkpoints is eligible as a production restart.

The revised accuracy probe forms the original residual r=b-A*x and solves
A*delta=r from a zero initial correction. Tolerances then apply to this small
correction equation. It records actual iterations, effective PETSc tolerances,
an independently recomputed correction residual, and the concentration change
in x+delta. A zero-iteration or inaccurate correction is marked inconclusive.
The candidate fields are saved for comparison, then the original state and
solver parameters are restored. The diagnostic does not advance with corrected
or clipped concentration values. This probes one timestep's algebraic error;
it is not a mesh or timestep convergence study.

## Install and submit

Extract `rect3d_bounds_followup.zip` directly into the existing folder:

```text
/path/to/home/rectangular_3d_quantitative_bundle
```

It contains:

- `continue_rect3d_scalar_bounds.py`
- `rect3d_bounds_followup_report.py`
- `submit_rect3d_bounds_continuation.sh`
- `rect3d_bounds_report.py`, the unchanged common reporting helper
- this README

It uses the existing `mixed_scalar_transport.py` with revision
`equilibrated-2026-09-17` and the existing six-check verifier. The launcher
retains the FEniCS activation and `--mpi=pmi2` setup that succeeded in 18989285.

Submit this single line:

```bash
bash /path/to/home/rectangular_3d_quantitative_bundle/submit_rect3d_bounds_continuation.sh
```

## Return

When it finishes, upload `RETURN_rect3d_bounds_NEWJOBID.zip` from the bundle
folder. This includes the extrema/volume history, full spatial snapshots,
residual-correction results, completion status and logs. If no ZIP appears,
return `rect3d_bounds_NEWJOBID.out` and `.err`.

The next decision will use whether density returns inside the original margin,
how many final steps remain inside it, and how much the residual correction
changes the concentration. This diagnostic does not automatically authorise
or submit a long fine-mesh run. The reference chain and separate older 3D run
do not need restarting for this check.

## Local validation and its limits

The revised probe was executed with SciPy adapters on independently assembled
small RT0/DG0 systems for both physical diffusivities. It performed 25 and 24
iterations, respectively, and removed an injected concentration error of
approximately 1e-5, agreeing with a balanced direct solve within 1e-10.
Independent relative correction residuals were below 1e-10. Checks also cover
state/parameter restoration after success and failure, rejection of a
zero-iteration comparison, checkpoint identity and source preservation,
and refusal to overwrite an existing destination.

Mock launcher checks cover default module activation, exact MPI arguments,
missing checkpoints, one-job submission, failure gates and returned ZIP/exit
status. Python compilation and shell syntax checks pass. The diagnostic and
production source equations remain unchanged. Actual DOLFIN/PETSc/MPI
execution of this continuation still requires the HPC job.

Evidence: `RETURN_rect3d_bounds_18989285.zip`, supplied by the user, including
`report.json`, `scalar_bounds_snapshots.npz`, the verifier report and logs.
