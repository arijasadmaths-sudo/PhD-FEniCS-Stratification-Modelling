# Axisymmetric verification post-processing

These scripts read saved axisymmetric return bundles and generate mesh and
time-step comparisons. Run the commands from the repository root. The return
bundles are not included.

## Requirements

- Python 3
- NumPy
- SciPy
- Shapely 2 or later
- Matplotlib for plotting

## Half-step comparison

`analyze_half_matrix.py` expects:

- `--data`: a directory containing the reference return folders named in the
  script's `names` mapping
- `--current`: the extra-fine half-step return folder
- `--out`: an output directory

```bash
python postprocessing/python/axisymmetric_verification/analyze_half_matrix.py \
  --data /path/to/reference_fields \
  --current /path/to/extra_fine_half_return \
  --out /path/to/output

python postprocessing/python/axisymmetric_verification/plot_half_matrix.py \
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
python postprocessing/python/axisymmetric_verification/analyze_quarter_check.py \
  --return-dir /path/to/extracted_return \
  --out /path/to/output

python postprocessing/python/axisymmetric_verification/plot_quarter_check.py \
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

The shared geometry and integration routines are in `comparison_geometry.py`.
The scripts check input manifests, hashes, mesh relationships and configurations
before comparing fields.
