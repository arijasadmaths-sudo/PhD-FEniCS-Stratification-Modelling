"""Accepted-state checkpoint support; NumPy plus the Python standard library."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SCHEMA = "axisym-balanced-50s-v1"


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def fingerprint(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_checkpoint(path, identity, step, time_s, dt_s, velocity, pressure,
                     fresh_dofs, cumulative_in, cumulative_out,
                     max_budget_relative, max_flux_change_relative, last_row,
                     segments):
    metadata = dict(schema=SCHEMA, identity=identity, step=int(step),
                    time_s=float(time_s), dt_s=float(dt_s),
                    cumulative_in=float(cumulative_in),
                    cumulative_out=float(cumulative_out),
                    max_budget_relative=float(max_budget_relative),
                    max_flux_change_relative=float(max_flux_change_relative),
                    last_row=last_row, segments=segments)
    atomic_npz(path, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True,
                                                        allow_nan=False)),
               velocity_dofs=velocity, pressure_dofs=pressure,
               fresh_dofs=fresh_dofs)


def load_checkpoint(path, identity, dt_s, num_steps, dimensions):
    with np.load(str(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {key: archive[key].copy() for key in
                  ("velocity_dofs", "pressure_dofs", "fresh_dofs")}
    if metadata.get("schema") != SCHEMA or metadata.get("identity") != identity:
        raise ValueError("Checkpoint source, case, physical parameters, mesh or DOF ordering differs")
    step = metadata["step"]
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < num_steps:
        raise ValueError("Checkpoint must be an incomplete accepted state of this case")
    if metadata["dt_s"] != dt_s or abs(metadata["time_s"] - step*dt_s) > 1e-12:
        raise ValueError("Checkpoint time or timestep differs; a half-step run starts from rest")
    for key, length in zip(("velocity_dofs", "pressure_dofs", "fresh_dofs"), dimensions):
        if arrays[key].shape != (length,) or not np.all(np.isfinite(arrays[key])):
            raise ValueError("Invalid checkpoint array: " + key)
    fresh = arrays["fresh_dofs"]
    if np.min(fresh) < -1e-10 or np.max(fresh) > 1+1e-10:
        raise ValueError("Checkpoint scalar outside accepted bounds")
    for key in ("cumulative_in", "cumulative_out", "max_budget_relative", "max_flux_change_relative"):
        value = metadata[key]
        if not np.isfinite(value) or value < 0:
            raise ValueError("Invalid checkpoint metadata: " + key)
    if step and (not isinstance(metadata["last_row"], dict) or
                 metadata["last_row"].get("step") != step):
        raise ValueError("Missing accepted budget row in checkpoint")
    return metadata, arrays


def carry_budget(source, destination, metadata, dt_s):
    """Carry only rows through the checkpoint; discard later/unaccepted rows."""
    step = metadata["step"]
    if not step:
        return None
    source, destination = Path(source), Path(destination)
    with source.open(newline="") as incoming, destination.open("w", newline="") as outgoing:
        reader = csv.DictReader(incoming)
        if not reader.fieldnames:
            raise ValueError("Resume requires the checkpoint's complete sibling budget CSV")
        writer = csv.DictWriter(outgoing, fieldnames=reader.fieldnames)
        writer.writeheader()
        count = 0
        last = None
        for row in reader:
            if count == step:
                break
            current = int(row["step"])
            if current != count+1 or float(row["dt_s"]) != dt_s or abs(float(row["time_s"])-current*dt_s) > 1e-12:
                raise ValueError("Resume budget has missing rows or different times")
            writer.writerow(row)
            count += 1
            last = row
        if count != step:
            raise ValueError("Resume budget ends before the accepted checkpoint")
        for key in ("fresh_in_cumulative_m3", "fresh_out_cumulative_m3", "fresh_stored_m3"):
            if float(last[key]) != float(metadata["last_row"][key]):
                raise ValueError("Budget does not match the checkpoint: " + key)
    return reader.fieldnames
