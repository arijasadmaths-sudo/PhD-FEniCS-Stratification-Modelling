# Axisymmetric verification post-processing

These scripts check external axisymmetric return bundles and generate mesh and
time-step comparison data. They do not run FEniCS. Simulation returns and generated
results are not stored in this repository.

## Requirements

- Python 3
- NumPy
- SciPy
- Shapely 2 or later
- Matplotlib for plotting

Run the commands from the repository root.

## Half-step comparison

`analyze_half_matrix.py` expects:

- `--data`: a directory containing the reference return folders named in the
  script's `names` mapping
- `--current`: the extra-fine half-step return folder
- `--out`: an output directory

```bash
python postprocessing/python/axisymmetric_verification/analyze_half_matrix.py \\
  --data /path/to/reference_fields \\
  --current /path/to/extra_fine_half_return \\
  --out /path/to/output

python postprocessing/python/axisymmetric_verification/plot_half_matrix.py \\
  /path/to/output
```

Generated files:

- `axisymmetric_half_matrix_metrics.json`: comparison metadata and field metrics
- `axisymmetric_half_matrix_profiles.npz`: matched-time horizontal profiles
- `field_sensitivity.png` and `field_sensitivity.pdf`: field-comparison plots
- `mean_profiles.png` and `mean_profiles.pdf`: profile plots

## Quarter-step comparison

`analyze_quarter_check.py` expects an extracted quarter-step return. Its
`reference_fields` directory must contain the earlier cases named in the script.

```bash
python postprocessing/python/axisymmetric_verification/analyze_quarter_check.py \\
  --return-dir /path/to/extracted_return \\
  --out /path/to/output

python postprocessing/python/axisymmetric_verification/plot_quarter_check.py \\
  /path/to/output
```

Generated files:

- `integrity.json`: input and checkpoint audit summary
- `metrics.json`: comparison metadata and metrics
- `comparisons.csv`: flattened temporal and spatial comparisons
- `integrals.csv`: per-case field integrals and summary quantities
- `profiles.npz`: matched-time horizontal profiles
- `field_sensitivity.png` and `field_sensitivity.svg`: field-comparison plots
- `mean_profiles.png` and `mean_profiles.svg`: profile plots

## Comparison operations

Same-mesh temporal comparisons use direct volume-weighted L2 differences.
Cross-mesh comparisons use exact cell intersections and conservative projection
onto the coarser mesh. Horizontal profiles use full-radius slab averages.
`comparison_geometry.py` contains the shared geometry and integration routines.

The analysis scripts validate their expected manifests, hashes, mesh relationships
and configuration values. They stop with an assertion if an input is missing or
incompatible.
