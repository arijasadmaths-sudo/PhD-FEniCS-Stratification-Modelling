This package is a bounded study for **quantitative comparison** with the full
600 × 300 × 300 mm quiescent rectangular experiment. It replaces the earlier
dt=0.004 proposal. The scientific outputs are layer depth, raw and normalized
profiles, dye/density differences, and measured sensitivity to mesh and timestep.

Extract the complete ZIP into a new HPC directory. Keep all Python files and
the shell script together. In the marked environment section, retain the
FEniCS 2019.1 activation commands from your working job, and retain its
account/project, partition and memory directives. The launcher first uses an
already working environment, then tries `fenics-2019.1.0`. The 48-hour
allocation request must be permitted by your partition.

Submit the recommended study once from the login node:

```bash
bash submit_rectangular_3d_quantitative.sh --study 8
```

You can append your existing `--account=CODE`, `--partition=NAME`,
`--mem=SIZE` or `--time=HH:MM:SS` settings.

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
