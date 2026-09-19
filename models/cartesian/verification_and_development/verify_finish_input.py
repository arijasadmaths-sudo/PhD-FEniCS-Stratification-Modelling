#!/usr/bin/env python3
"""Validate the saved restart state without requiring DOLFIN.

The unchanged production solver additionally regenerates and checks the exact
DOLFIN mesh and DOF ordering before accepting this checkpoint on Blue Pebble.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


EXPECTED_SOURCE = "3051033140e9e158da891416749a167d9f9f7654a6c1e278cc0c3f490611689d"
CASE = "compact156p25"


def check(checkpoint, source):
    source = Path(source).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SOURCE:
        raise ValueError("Solver source differs from the original returned solver")
    fingerprint = hashlib.sha256((digest + "|" + CASE + "|dt=.001|T=5").encode()).hexdigest()
    spec = importlib.util.spec_from_file_location("cartesian_original", str(source))
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    with np.load(checkpoint, allow_pickle=False) as archive:
        state = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(state["metadata_json"]))
    if metadata.get("source_sha256") != digest or metadata.get("dt_s") != 0.001:
        raise ValueError("Checkpoint source or time step differs")
    edges = solver.make_graded_grid(solver.CASE_H[CASE])
    grid = solver.BalancedFluxGrid(*edges)
    for name in ("velocity_dof_xy_m", "pressure_dof_xy_m", "mesh_coordinates_m"):
        coords = state[name]
        if coords.ndim != 2 or coords.shape[1] != 2 or not np.all(np.isfinite(coords)):
            raise ValueError("Invalid coordinate array: " + name)
        if np.min(coords) < -1e-12 or np.max(coords[:, 0]) > 0.6+1e-12 or np.max(coords[:, 1]) > 0.3+1e-12:
            raise ValueError("Coordinate outside the domain: " + name)
    if state["mesh_cells"].shape != (265524, 3):
        raise ValueError("Wrong triangular mesh size")
    if state["mesh_cells"].dtype.kind not in "iu" or np.min(state["mesh_cells"]) < 0 or np.max(state["mesh_cells"]) >= len(state["mesh_coordinates_m"]):
        raise ValueError("Invalid triangle indices")
    if state["u_dofs"].size != 1065030 or state["p_dofs"].size != 133496:
        raise ValueError("Wrong number of finite-element DOFs")
    step, cumulative = solver.validate_restart(
        state, CASE, fingerprint, grid,
        state["velocity_dof_xy_m"], state["pressure_dof_xy_m"],
        state["mesh_coordinates_m"], state["mesh_cells"])
    if step < 4750:
        raise ValueError("This finish bundle must resume the 4.75 s checkpoint or a later accepted checkpoint")
    wall = float(state["cumulative_wall_s"])
    if not np.isfinite(wall) or wall < 0:
        raise ValueError("Invalid saved cumulative wall time")
    inventory = float(np.sum(grid.volume * (1-state["c"])))
    residual = inventory - (cumulative[0]-cumulative[1]+cumulative[2])
    return {
        "check": "passed", "case": CASE, "source_sha256": digest,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_fingerprint": fingerprint, "checkpoint_step": step,
        "checkpoint_time_s": float(state["time_s"]), "remaining_steps": 5000-step,
        "target_time_s": 5.0, "dt_s": 0.001,
        "fv_grid_cells": [grid.nx, grid.ny], "triangular_cells": 265524,
        "c_min": float(state["c"].min()), "c_max": float(state["c"].max()),
        "fresh_inventory_m2": inventory, "cumulative_budget_residual_m2": float(residual),
        "relative_budget_residual": abs(float(residual))/inventory,
        "cumulative_wall_s": wall,
        "limitation": "Saved arrays and regenerated FV grid checked locally; exact regenerated DOLFIN mesh/DOF ordering is checked by production solver on Blue Pebble."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--checkpoint", required=True, help="Accepted compact156p25 full-step checkpoint")
    parser.add_argument("--source", default=str(root.parent/"provenance/cartesian_compact_dt001.py"))
    parser.add_argument("--report")
    args = parser.parse_args()
    report = check(args.checkpoint, args.source)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(encoded, end="")
    if args.report:
        with open(args.report, "x", encoding="utf-8") as handle:
            handle.write(encoded)


if __name__ == "__main__":
    main()
