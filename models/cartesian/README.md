# Cartesian models

`provenance/cartesian_compact_dt001.py` is the protected full-step source and `cartesian_half_step_5s.py` is the half-step source. Both compact cases start from rest and stop at 5 s.

Run a full-step case with:

```sh
python provenance/cartesian_compact_dt001.py --case compact312p5 --output /absolute/new/run
```

Use `compact156p25` for the finer mesh. For both half-step cases, set the account in the launcher and submit from this folder:

```sh
CARTESIAN_WORK_PARENT=/absolute/existing/work sbatch run_cartesian_half_step_5s.sh
```

The array writes a separate return folder and ZIP for each case. Existing outputs are protected. Earlier uniform and focused checks are in `verification_and_development/`.

`verify_finish_input.py` is only for the historical compact156p25 full-step restart. Supply its checkpoint explicitly with `--checkpoint`; the original source-hash check is retained.

Comparison scripts are in `postprocessing/python/cartesian_verification/` from the repository root. `python validate_bundle.py` checks the compact solver, restart guards and mocked launcher packaging without FEniCS.
