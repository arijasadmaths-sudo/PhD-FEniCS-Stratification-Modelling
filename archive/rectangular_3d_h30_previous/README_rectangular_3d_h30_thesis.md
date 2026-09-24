# Archived full-height 3D package

This directory contains a superseded full-height solver, launcher and plotting
script. Keep the three files together; checkpoints from this version require the
same source and settings.

## Cluster setup

Set the FEniCS module, Slurm account, partition, memory and permitted wall time in
`submit_rectangular_3d_h30_thesis.sh`. The launcher uses an active DOLFIN 2019.1
environment or loads `fenics-2019.1.0`.

## Submission

Submit three dependent allocations from the package directory:

```bash
bash submit_rectangular_3d_h30_thesis.sh --chain 3
```

The default chain uses 24 MPI ranks and successful-job dependencies. Each job
resumes from the latest compatible checkpoint. A successor exits before mesh
generation if the configured target has already been reached.

Site settings can be supplied as `--account=YOUR_CODE`,
`--partition=YOUR_PARTITION`, `--mem=YOUR_MEMORY` and `--time=HH:MM:SS`.
For one allocation, use:

```bash
sbatch submit_rectangular_3d_h30_thesis.sh
```

## Outputs and restarts

The default output directory is:

```text
results_rectangular_3d_h30_thesis_Q20ccmin_eps0_dt0p004
```

`run_status.json` records physical time and completion, `diagnostics.csv` records
runtime diagnostics, and `segments/` contains XDMF/HDF5 fields. Periodic
checkpoints are retained in the result directory. Restart compatibility is checked
before fields are reconstructed on a regenerated mesh.

## Plotting archived output

The plotting script requires NumPy, h5py and Matplotlib and should be run with one
Python process:

```bash
python3 plot_rectangular_3d_h30_thesis.py \
  --results results_rectangular_3d_h30_thesis_Q20ccmin_eps0_dt0p004
```

Use `--times` to select saved times. Keep each XDMF file with its companion HDF5
file.
