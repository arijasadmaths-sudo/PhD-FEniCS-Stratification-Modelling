#!/usr/bin/env python3
"""Check launcher identity and print the remaining steps to the 5 s review."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CASE = "extra_fine_quarter"
DT = 0.00025
REVIEW_STEP = 20000
TARGET_STEP = 200000


def remaining_steps(source, resume=None, continue_50=False):
    source = Path(source).resolve()
    step = 0
    if resume:
        checkpoint = Path(resume).expanduser()
        if not checkpoint.is_absolute():
            raise ValueError("Use an absolute path to checkpoint_latest.npz")
        with np.load(str(checkpoint), allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"].item()))
        identity = metadata.get("identity", {})
        expected = dict(case=CASE, dt_s=DT, target_time_s=50.0,
                        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        checkpoint_helper_sha256=hashlib.sha256(
                            source.with_name("study_checkpoint.py").read_bytes()).hexdigest())
        if any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError("Resume requires this exact extra_fine_quarter source, dt=.00025 and target=50 s; other cases cannot be resumed")
        step = metadata.get("step")
        if (metadata.get("schema") != "axisym-balanced-50s-v1"
                or isinstance(step, bool) or not isinstance(step, int)
                or not 0 <= step < TARGET_STEP or metadata.get("dt_s") != DT
                or abs(float(metadata.get("time_s", -1))-step*DT) > 1e-12):
            raise ValueError("Invalid accepted checkpoint step/time")
        if step and not (checkpoint.parent/"verification_budget.csv").is_file():
            raise ValueError("Keep the complete sibling verification_budget.csv with the checkpoint")
        for field_step in (0, 8000, 20000, 40000, 80000, 120000, 160000):
            if field_step <= step and not (checkpoint.parent/("field_step_%09d.npz" % field_step)).is_file():
                raise ValueError("Missing historical matched field at step %d" % field_step)
    if continue_50:
        if not resume or step < REVIEW_STEP:
            raise ValueError("Complete and review 5 s before an explicit 50 s continuation")
        return 0
    if step >= REVIEW_STEP:
        raise ValueError("This diagnostic has reached 5 s; send its RETURN ZIP for review")
    return REVIEW_STEP-step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--continue-50", action="store_true")
    args = parser.parse_args()
    print(remaining_steps(args.source, args.resume, args.continue_50))


if __name__ == "__main__":
    main()
