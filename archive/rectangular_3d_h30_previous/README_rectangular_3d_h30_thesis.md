This package runs the full rectangular tank to **120 simulated seconds**, with a
30 cm height and targeted vertical refinement below the ceiling. The aim is to
capture plume rise, ceiling impingement and subsequent spreading in one continuing
simulation. Reaching a particular flow regime by 120 s is not guaranteed.

Keep all three scripts together on the HPC filesystem. In the marked section of
`submit_rectangular_3d_h30_thesis.sh`, retain the FEniCS activation and
account/project, partition and memory settings from your working job. The launcher
first tries the inherited DOLFIN 2019.1 environment, then the module
`fenics-2019.1.0`; use your working module name if it differs. The requested
48-hour limit must be allowed by your partition.

From that directory on the login node, queue three consecutive allocations:

```bash
bash submit_rectangular_3d_h30_thesis.sh --chain 3
```

This queues up to three 48-hour allocations on 24 MPI ranks. Each resumes the same
simulation; an unused successor exits before mesh generation once the target is
already reached. The chain uses successful-job dependencies, so a solver error
stops progression. Normal checkpointed stops before the wall limit count as
successful exits. This behaviour uses Slurm's
[afterok dependencies and pre-timeout signals](https://slurm.schedmd.com/sbatch.html).

You can append existing site settings as `--account=YOUR_CODE`,
`--partition=YOUR_PARTITION`, `--mem=YOUR_MEMORY` or `--time=HH:MM:SS`.
For a single allocation, use:

```bash
sbatch submit_rectangular_3d_h30_thesis.sh
```

| Setting | Default |
|---|---|
| Tank length × width × height | 600 × 300 × 300 mm; full domain |
| Freshwater inflow / nozzle diameter | 20 cc/min / 2 mm |
| Physical target / time step | 120 s / 0.004 s; 30,000 steps |
| Mesh | 158,400 base tetrahedra before local refinement |
| Vertical ceiling spacing | 3.9375 mm across upper 63 mm, before local refinement |
| Bulk horizontal spacing | 15 mm, with local plume/nozzle refinement |
| Concentration output | Every 1 simulated second |
| Velocity and pressure output | Every 5 simulated seconds |
| Diagnostics | Every 0.5 simulated second, plus initial steps |
| Checkpoints | Every 10 simulated seconds or 30 wall minutes; three retained |
| Planned soft stop | 46 wall hours, or 30 minutes before allocation end |

The final mesh count is recorded in `configuration.json`. The earlier 30,000-step
segment took about 29 wall hours on 24 ranks. This mesh is larger, so this package
can take multiple days; three allocations are an allowance, not a runtime
guarantee. The log reports an estimated remaining compute time after stepping
begins. Queue waiting is additional.

The default results directory is:

```text
results_rectangular_3d_h30_thesis_Q20ccmin_eps0_dt0p004
```

`run_status.json` records current physical time and completion.
`diagnostics.csv` contains fluxes, concentration bounds, freshwater budget and
centroids. `segments/` contains the XDMF/HDF5 visualization files. Old runs are
kept separate. Continue with the same scripts and parameter settings; the solver
checks restart compatibility and reconstructs fields on a regenerated mesh.
Periodic checkpoints limit lost work if a job is killed before a graceful stop.

After an allocation finishes, run the figure script on a machine with NumPy,
h5py and Matplotlib, using one Python process:

```bash
python3 plot_rectangular_3d_h30_thesis.py --results results_rectangular_3d_h30_thesis_Q20ccmin_eps0_dt0p004
```

It exports PNG/PDF central sections at 20, 40, 60, 80, 100 and 120 s, fixed
concentration scales of 0–1 and 0.97–1, horizontally averaged freshwater profiles,
CSV values and source/time metadata into `figures/`. Unavailable times are
identified. For early output, select saved times, for example `--times 5 10 15`.
Keep each XDMF with its companion HDF5. Concentration is normalized salinity:
`c=1` is ambient saltwater and `c=0` is freshwater.

For the thesis, describe this as a qualitative numerical simulation. It retains
the Taylor–Hood flow solve, divergence-free lowest-order RT transport flux and
implicit upwind DG0 scalar scheme. The 0.004 s step is a deliberate cost versus
temporal-accuracy choice; mesh and time convergence are not established. The
flush nozzle and distributed floor return approximate the experiment's inlet
and outlet. The scalar diffusivity of 1e-7 m²/s is a modelling value; the inherited
DG0 jump-diffusion treatment is approximate on the graded tetrahedra.

Python syntax, shell syntax and mocked launcher control flow were checked. Eight
tests of the extracted Python controls passed, including invalid parameters,
graded-layer geometry, damaged-checkpoint fallback and completed-run early exit. The
figure script was executed and visually checked on your existing concentration
data. The solver includes geometry, flux, finite-value, scalar-budget and
checkpoint round-trip checks at runtime. The complete new DOLFIN/MPI simulation
could not be executed in this environment; no new-run results are included.
