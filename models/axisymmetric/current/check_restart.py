#!/usr/bin/env python3
"""HPC gate: compare four uninterrupted steps against two + checkpoint + two."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    root.mkdir()
    solver = Path(__file__).with_name("axisymmetric_70_study_50s.py")
    base = [sys.executable, "-u", str(solver), "--case", "coarse", "--end-time", "0.004"]
    report = dict(status="running", purpose="Four coarse steps, uninterrupted versus accepted checkpoint restart")
    report_path = root.parent/"restart_check.json"
    try:
        for name, extra, expected in (
                ("full", [], 0),
                ("first", ["--stop-after-steps", "2"], 75),
                ("resumed", ["--resume", str(root/"first"/"checkpoint_latest.npz")], 0)):
            result = subprocess.run(base+["--output", str(root/name)]+extra)
            if result.returncode != expected:
                raise RuntimeError("Restart check {} exited {}, expected {}".format(name, result.returncode, expected))
        with np.load(str(root/"full"/"checkpoint_latest.npz"), allow_pickle=False) as left, \
                np.load(str(root/"resumed"/"checkpoint_latest.npz"), allow_pickle=False) as right:
            metrics = {}
            for key in ("velocity_dofs", "pressure_dofs", "fresh_dofs"):
                a, b = left[key], right[key]
                difference = float(np.max(np.abs(a-b)))
                scale = max(float(np.max(np.abs(a))), 1e-12)
                if difference > 1e-10*scale+1e-14:
                    raise RuntimeError("Restart differs from uninterrupted run for "+key)
                metrics[key+"_max_abs_difference"] = difference
            lm, rm = (json.loads(str(data["metadata_json"].item())) for data in (left, right))
            if lm["step"] != 4 or rm["step"] != 4 or lm["time_s"] != rm["time_s"]:
                raise RuntimeError("Restart lost the physical step/time")
            for key in ("cumulative_in", "cumulative_out"):
                np.testing.assert_allclose(lm[key], rm[key], rtol=1e-10, atol=1e-25)
        for name in ("full", "resumed"):
            status = json.loads((root/name/"verification_status.json").read_text())
            if status["status"] != "completed":
                raise RuntimeError("Restart check did not complete: "+name)
            # Full CSV history must survive the restart without duplicated rows.
            import csv
            with (root/name/"verification_budget.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if [int(row["step"]) for row in rows] != [1, 2, 3, 4]:
                raise RuntimeError("Restart budget has missing or duplicated rows")
        report.update(status="passed", metrics=metrics)
        print("CHECKPOINT RESTART CHECK PASSED", flush=True)
    except BaseException as error:
        report.update(status="failed", error=str(error), exception_type=type(error).__name__)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
