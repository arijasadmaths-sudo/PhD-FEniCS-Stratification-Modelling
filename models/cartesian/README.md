# Cartesian compact half-step source bundle

This directory preserves the supplied half-step solver and its launcher/helpers.
Both compact cases use dt=0.0005 s to t=5 s and one MPI rank.
The original full-step source was present at `../cartesian_compact_5s.py`;
its exact original is also provided at `provenance/cartesian_compact_dt001.py`
to restore the existing validator's expected layout. The original full-step
SHA-256 is 3051033140e9e158da891416749a167d9f9f7654a6c1e278cc0c3f490611689d.

On the original Bristol environment, submit from this directory:

```sh
export CARTESIAN_WORK_PARENT=/absolute/existing/work/directory
sbatch run_cartesian_half_step_5s.sh
```

Check the scheduler account and module on your system before submitting.
The array runs compact312p5 and compact156p25 from rest. Each case packages
its accepted checkpoint, scalar budget and comparison fields in a separate
RETURN directory and ZIP. Existing output paths are protected.

`python3 cartesian_half_step_5s.py --self-test` needs NumPy/SciPy only.
`python3 validate_bundle.py` also checks the restored full-step source,
restart guards and mocked launcher packaging. Mock checks are not coupled
finite-element simulations. The validator requires Python 3.9 or newer.

Essential comments are retained; decorative separators are removed. Executable code, docstrings,
source-hash guards and scheduler directives are preserved. Original checkpoints
with hashes of the commented source are not interchangeable with this cleaned
source. Use the original-source Git reference to restore the complete matching sources for those runs; never disable
these checks. See `../../docs/AUDIT.md` and `../../docs/source_manifest.json`.
