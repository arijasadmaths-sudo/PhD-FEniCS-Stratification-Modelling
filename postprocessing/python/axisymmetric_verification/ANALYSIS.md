# Axisymmetric extra-fine half-step review

The repaired run `axisymmetric50_extra_fine_half_RETURN_18954282.zip` reached the intended **5 s review pause**, with 200,736 triangular cells and Δt = 0.0005 s. Its status is `paused` and exit code 75 is expected. It has not completed the 50 s target. The checkpoint and full accepted budget history are intact.

**Keep the 50 s continuation on hold.** The new result completes the fine/extra-fine comparison at Δt = 0.001 and 0.0005 s, but appreciable mesh and time-step sensitivity remain. This bundle prepares one extra-fine run at Δt = 0.00025 s from the original initial condition through 5 s, so that the next temporal halving and the mesh comparison at the quarter step can both be evaluated.

## Integrity, repair and conservation

The return passed the following independent checks:

- All 47 return-manifest entries, eight prepared-source hashes and three seed hashes agree.
- The accepted budget contains 10,000 contiguous steps to 5 s. Its initial 2,000 rows are identical to the preserved 1 s seed history. Cumulative input and output recompute exactly.
- Checkpoint velocity, pressure and reordered DG0 scalar agree with the returned paused state and 5 s field. Mesh/DOF identity and geometric cell volumes agree.
- The maximum relative cumulative freshwater budget residual is **1.213 × 10⁻¹³**. Final inventory is **4.66695833333327 × 10⁻⁶ m³**. Concentration stayed between zero and one to roundoff; the largest excess over one was **6.04 × 10⁻¹⁴**.
- Independent replay of the saved underflow failure gives the same concentration array before and after the repair. The repair changes the residual measurement for subnormal values, without changing the scalar solve or its candidate. Corrupted populated and tiny-tail solutions still fail the check.
- The recorded coupled restart smoke test on Blue Pebble passed with zero reported state differences. The resumed 1–5 s segment took **25.48 hours**.

The transport uses balanced integrated face fluxes. The final correction has relative L₂ magnitude 5.43%, with a maximum of 18.61% during this history. The unbalanced-flux shadow diagnostic is a one-step calculation and is never evolved or fed back into the solver; its overshoots are not overshoots of the accepted solution and do not estimate global discretization error. Conservation and boundedness establish useful integrity properties, not convergence.

## Time-step comparison

These are direct, volume-weighted freshwater-field L₂ differences on identical meshes, divided by the norm of the first (larger-time-step) field. Comparisons use the same physical times and the original initial condition.

| Mesh | Time-step change (s) | At 2 s | At 5 s |
|---|---|---:|---:|
| Fine, 50,184 cells | 0.001 → 0.0005 | 16.8335% | 2.5045% |
| Fine, 50,184 cells | 0.0005 → 0.00025 | 9.1172% | 1.0445% |
| Extra-fine, 200,736 cells | 0.001 → 0.0005 | **18.3710%** | **7.6306%** |

The temporal changes shrink on the fine mesh. The extra-fine mesh remains appreciably sensitive at 5 s, so the fine-mesh late-time result cannot establish time-step independence on the extra-fine mesh. Only one time-step halving has been measured on the extra-fine mesh so far.

The extra-fine freshwater vertical centroid rises by 1.006 mm at 2 s and 0.447 mm at 5 s when the time step is halved. Differences in total injection between time steps are small consequences of the existing discrete inlet ramp: the full/half-step inventory difference is 0.02499% at 2 s and 0.006249% at 5 s. They are not budget leaks. Raw concentrations are compared without normalizing their inventories.

## Mesh comparison

The triangular meshes are not nested in the computational coordinates. Two field measures are retained:

1. **Exact cell-overlap L₂:** integrate the squared difference of the two piecewise-constant fields over every intersecting cell pair. This compares the full returned fields without averaging either one first.
2. **Projected L₂:** conservatively average the higher-resolution field onto the lower-resolution cells, then compare. This retains the metric used in previous reviews, but the averaging smooths subcell differences.

| Time | Fine → extra-fine time step | Exact cell-overlap L₂ | L₂ after projection onto fine cells |
|---|---:|---:|---:|
| 2 s | 0.001 s | 16.6799% | 12.4922% |
| 2 s | 0.0005 s | 18.3783% | 14.3033% |
| 5 s | 0.001 s | **33.7768%** | **27.8641%** |
| 5 s | 0.0005 s | **33.2593%** | **27.2940%** |

Halving the time step does not remove the large spatial difference at 5 s. The earlier coarse → fine comparison at Δt = 0.001 s has exact field difference 32.7424% at 5 s (23.7515% after projection). These data do not establish a decreasing asymptotic spatial error or support a reliable Richardson extrapolation.

At Δt = 0.0005 s, the extra-fine freshwater centroid is 2.864 mm above the fine-mesh centroid at 5 s. Good agreement of inventory alone therefore does not imply agreement of its distribution.

## Mean-profile averaging sensitivity

Profiles are exact volume averages over full-radius horizontal slabs. Both meshes are integrated over identical slab boundaries. The previous review used 51 vertical slabs; this review also retains 102 slabs to expose sensitivity to vertical averaging. Profile L₂ uses slab-height weights and the first profile as denominator.

| Fine → extra-fine at Δt = 0.0005 s | 51 slabs | 102 slabs |
|---|---:|---:|
| At 2 s | 6.7222% | 7.3370% |
| At 5 s | **5.3274%** | **8.5127%** |

The more heavily averaged 5.33% profile difference should not be used alone to claim mesh independence. The included profile plot uses 102 slabs and shows the averages as steps, preserving their piecewise-constant meaning.

## Method and reproducibility

For coordinates s = r²/2 and height z, the physical volume measure is dV = 2π ds dz. No extra radial weight is applied. Exact polygon intersections use Shapely; their row and column sums close to the supplied physical cell volumes within 8.44 × 10⁻¹⁴ relative error. Temporal comparisons assert identical mesh geometry, topology and cell volumes. All compared physical coefficients and inlet settings match.

Source differences were independently inspected. They add mesh/time-step choices, metadata, restart compatibility and the audited residual-check repair. They do not change physical coefficients, weak forms, the transport matrix/right-hand side, flux-projection solve or scalar candidate calculation.

The `reference_fields` directory preserves the 0, 2 and 5 s snapshots, configurations and source/helper files for all six compared cases, with SHA-256 hashes. Original return archives and checkpoints remain the authoritative full histories. The metrics JSON records the original local source paths and hashes; the preserved snapshots have identical hashes in their corresponding `reference_fields` subdirectories.

To recompute the comparison from this bundle, using Python with NumPy, SciPy and Shapely 2 or later, run from the bundle directory:

```bash
python analysis/analyze_half_matrix.py \
  --data reference_fields \
  --current reference_fields/axisymmetric50_extra_fine_half_RETURN_18954282 \
  --out analysis
python analysis/plot_half_matrix.py
```

Plot generation also needs Matplotlib. These analysis dependencies are separate from the FEniCS production environment; analysis does not run in the production launcher. `current_return_integrity.json`, `underflow_replay_independent.json` and `comparison_independent.json` record the separate audits.

## Next test and decision

Run **extra-fine, 200,736 cells, Δt = 0.00025 s, from rest through 5 s**. It needs 20,000 accepted steps. Retain the 50 s checkpoint identity for a possible later continuation, but the launcher defaults to an absolute 5 s review pause, including on restarts. No existing half-step checkpoint is used to initialize this new time-step test.

Allow approximately **60–70 hours** based on the measured half-step segment; runtime can vary. The Slurm request is 72 hours, and checkpoint restart is available if another allocation is needed. The full coupled calculation has not run locally. The launcher performs the coupled FEniCS restart smoke test on Blue Pebble before production.

After the return, compare extra-fine 0.0005 → 0.00025 s at both 2 and 5 s, and fine → extra-fine at 0.00025 s. Assess field differences, profiles at a fixed stated averaging scale and integral measures together. If the temporal difference shrinks, it gives stronger evidence for selecting a time step; remaining spatial sensitivity must still be reported and may require another mesh or discretization investigation. This test does not guarantee that the 50 s continuation will be justified.
