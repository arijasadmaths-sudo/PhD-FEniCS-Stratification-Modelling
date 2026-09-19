#!/bin/bash
#SBATCH --job-name=cart_focus5s
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#SBATCH --array=0-1%2
#SBATCH --signal=TERM@180
#SBATCH --output=cart_focus5s_%A_%a.out
#SBATCH --error=cart_focus5s_%A_%a.err

# Submit both independent cases from the directory containing both source files:
#   sbatch run_cartesian_focused_5s.sh
#   task 0: focus1250; task 1: focus625.
# Resume ONE case in a NEW Slurm job, keeping the same task number:
#   sbatch --array=0 --export=ALL,CARTESIAN_RESUME=/absolute/path/checkpoint_latest.npz run_cartesian_focused_5s.sh
# A normal completion, solver error, or graceful pause produces a return folder
# and ZIP in the submission directory. A scheduler hard kill can prevent this.
set -uo pipefail
umask 022

cart_submit_dir="${SLURM_SUBMIT_DIR:?Submit with sbatch from the directory containing both source files.}"
cart_job_id="${SLURM_JOB_ID:?This launcher requires a Slurm job ID.}"
cart_array_id="${SLURM_ARRAY_JOB_ID:?Submit this script as a Slurm array, including for a one-case restart.}"
cart_task_id="${SLURM_ARRAY_TASK_ID:?A Slurm array task ID is required.}"
if [[ ! "$cart_job_id" =~ ^[0-9]+$ || ! "$cart_array_id" =~ ^[0-9]+$ ]]; then
    printf 'Unexpected Slurm job identifiers.\n' >&2
    exit 2
fi
case "$cart_task_id" in
    0) cart_case="focus1250" ;;
    1) cart_case="focus625" ;;
    *) printf 'Unsupported array task: %s (use 0 or 1).\n' "$cart_task_id" >&2; exit 2 ;;
esac
cart_resume="${CARTESIAN_RESUME:-}"
if [[ -n "$cart_resume" ]]; then
    if [[ "${SLURM_ARRAY_TASK_COUNT:-}" != "1" ]]; then
        printf 'CARTESIAN_RESUME requires exactly one array task: submit with --array=0 or --array=1.\n' >&2
        exit 2
    fi
    if [[ "$cart_resume" != /* || ! -f "$cart_resume" || ! -r "$cart_resume" ]]; then
        printf 'CARTESIAN_RESUME must name a readable checkpoint using its absolute path: %s\n' "$cart_resume" >&2
        exit 2
    fi
fi
cd -- "$cart_submit_dir" || exit $?
cart_submit_dir="$PWD"
cart_python_name="cartesian_focused_5s.py"
cart_shell_name="run_cartesian_focused_5s.sh"
for cart_name in "$cart_python_name" "$cart_shell_name"; do
    if [[ ! -f "$cart_submit_dir/$cart_name" ]]; then
        printf 'Missing source file: %s\n' "$cart_submit_dir/$cart_name" >&2
        exit 2
    fi
done

cart_identity="${cart_array_id}_${cart_task_id}"
cart_run_root="/path/to/work/cartesian_focused_5s_${cart_identity}"
cart_simulation_dir="$cart_run_root/simulation"
cart_return_name="cartesian_${cart_case}_RETURN_${cart_identity}"
cart_return_dir="$cart_submit_dir/$cart_return_name"
cart_return_zip="$cart_return_dir.zip"
for cart_path in "$cart_run_root" "$cart_return_dir" "$cart_return_zip"; do
    if [[ -e "$cart_path" || -L "$cart_path" ]]; then
        printf 'Refusing to overwrite an existing path: %s\n' "$cart_path" >&2
        exit 2
    fi
done
mkdir -- "$cart_run_root" || exit $?
cart_run_log="$cart_run_root/run.log"
cart_stage="source_snapshot"
cart_started_epoch="$(date +%s)"

cart_run_workload() {
    printf 'Job: %s; array: %s; task: %s; case: %s\n' \
        "$cart_job_id" "$cart_array_id" "$cart_task_id" "$cart_case"
    printf 'Submission directory: %s\nWork directory: %s\n' "$cart_submit_dir" "$cart_run_root"
    printf 'Started (UTC): %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'Requested final physical time: 5 seconds.\n'
    printf 'Resume checkpoint: %s\n' "${cart_resume:-NONE; fresh start}"

    # Execute the captured source; later edits in the submit directory cannot
    # change either the running program or its archived source. Under Slurm,
    # BASH_SOURCE[0] identifies the actual submitted script in the Slurm spool.
    cp -- "$cart_submit_dir/$cart_python_name" "$cart_run_root/$cart_python_name" || return $?
    cp -- "${BASH_SOURCE[0]}" "$cart_run_root/$cart_shell_name" || return $?

    cart_stage="module_setup"
    module purge || return $?
    module add languages/python/fenics-2019.1.0 || return $?
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export PYTHONUNBUFFERED=1

    cart_stage="preflight"
    python3 - <<'PY' || return $?
import platform
import sys
import dolfin
import numpy
import scipy

print("Host:", platform.node())
print("Python:", sys.version.replace("\n", " "))
print("DOLFIN:", dolfin.__version__)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
if not dolfin.__version__.startswith("2019.1"):
    raise RuntimeError("Load the legacy FEniCS 2019.1 module.")
if dolfin.MPI.size(dolfin.MPI.comm_world) != 1:
    raise RuntimeError("This solver requires exactly one MPI rank.")
PY

    cart_stage="self_test"
    python3 "$cart_run_root/$cart_python_name" --self-test || return $?

    cart_stage="simulation"
    cart_arguments=(--case "$cart_case" --output "$cart_simulation_dir")
    if [[ -n "$cart_resume" ]]; then
        cart_arguments+=(--resume "$cart_resume")
    fi
    srun --mpi=pmi2 python3 "$cart_run_root/$cart_python_name" \
        "${cart_arguments[@]}" || return $?
    cart_stage="completed"
}

# Keep module changes in this shell and wait for all log output before packing.
exec 3>&1 4>&2
exec > >(tee --output-error=warn "$cart_run_log") 2>&1
cart_tee_pid=$!
if cart_run_workload; then
    cart_workload_rc=0
else
    cart_workload_rc=$?
fi
cart_finished_epoch="$(date +%s)"
printf 'Workload exit code: %s; last stage: %s\n' "$cart_workload_rc" "$cart_stage"
printf 'Workload wall seconds: %s\n' "$((cart_finished_epoch-cart_started_epoch))"
printf 'Finished (UTC): %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec 1>&3 2>&4 3>&- 4>&-
if wait "$cart_tee_pid"; then
    cart_log_rc=0
else
    cart_log_rc=$?
    printf 'Log capture failed with exit code %s.\n' "$cart_log_rc" >&2
fi

# Packaging uses only the Python standard library, even after an import error.
# The final ZIP is published atomically without overwriting an existing file.
if python3 - "$cart_run_root" "$cart_submit_dir" "$cart_job_id" \
    "$cart_array_id" "$cart_task_id" "$cart_case" "$cart_workload_rc" \
    "$cart_log_rc" "$cart_stage" "$cart_resume" "$cart_started_epoch" \
    "$cart_finished_epoch" <<'PY'
import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import zipfile

run_root = Path(sys.argv[1])
submit_dir = Path(sys.argv[2])
job_id, array_id, task_id, case = sys.argv[3:7]
workload_rc, log_rc = map(int, sys.argv[7:9])
stage, resume = sys.argv[9:11]
started_epoch, finished_epoch = map(int, sys.argv[11:13])
simulation_dir = run_root / "simulation"
return_name = "cartesian_{}_RETURN_{}_{}".format(case, array_id, task_id)
return_dir = submit_dir / return_name
return_zip = submit_dir / (return_name + ".zip")
if os.path.lexists(str(return_dir)) or os.path.lexists(str(return_zip)):
    raise RuntimeError("Refusing to overwrite an existing return folder or ZIP.")
return_dir.mkdir()

copied, missing = [], []
for source in [
    simulation_dir / "budget_configuration.json",
    simulation_dir / "run_status.json",
    simulation_dir / "scalar_budget.csv",
    run_root / "run.log",
    run_root / "cartesian_focused_5s.py",
    run_root / "run_cartesian_focused_5s.sh",
]:
    if source.is_file():
        shutil.copy2(str(source), str(return_dir / source.name))
        copied.append(source.name)
    else:
        missing.append(source.name)

status, status_error = None, None
status_path = return_dir / "run_status.json"
if status_path.is_file():
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        status_error = str(error)

def first_value(mapping, keys):
    if isinstance(mapping, dict):
        for key in keys:
            if key in mapping:
                return mapping[key]
    return None

# Reading numeric scalar NPY headers here does not import NumPy or unpickle
# objects. Fields are inspected only to match an accepted snapshot to status.
def npz_scalar(path, names):
    with zipfile.ZipFile(str(path), "r") as archive:
        for name in names:
            member = name + ".npy"
            if member not in archive.namelist():
                continue
            with archive.open(member) as handle:
                if handle.read(6) != b"\x93NUMPY":
                    raise ValueError("Invalid NPY header: " + member)
                version = tuple(handle.read(2))
                if version == (1, 0):
                    size = struct.unpack("<H", handle.read(2))[0]
                elif version in [(2, 0), (3, 0)]:
                    size = struct.unpack("<I", handle.read(4))[0]
                else:
                    raise ValueError("Unsupported NPY format version")
                header = ast.literal_eval(handle.read(size).decode("utf-8" if version == (3, 0) else "latin1"))
                if header.get("shape") not in [(), (1,)]:
                    raise ValueError("Expected scalar metadata: " + member)
                dtype = header.get("descr", "")
                match = re.fullmatch(r"([<>|=])([fiu])(\d+)", dtype)
                if not match:
                    raise ValueError("Unsupported scalar dtype: " + str(dtype))
                order, kind, width = match.groups()
                width = int(width)
                codes = {("f", 4): "f", ("f", 8): "d", ("i", 1): "b", ("i", 2): "h",
                         ("i", 4): "i", ("i", 8): "q", ("u", 1): "B", ("u", 2): "H",
                         ("u", 4): "I", ("u", 8): "Q"}
                code = codes.get((kind, width))
                if code is None:
                    raise ValueError("Unsupported scalar dtype: " + dtype)
                value = struct.unpack(("=" if order == "|" else order) + code, handle.read(width))[0]
                if not math.isfinite(value):
                    raise ValueError("Non-finite snapshot metadata")
                return value
    return None

accepted_step = first_value(status, ["accepted_step", "step", "completed_steps", "steps_completed", "last_completed_step"])
accepted_time = first_value(status, ["accepted_time_s", "time_s", "t_s", "time", "t", "last_completed_time_s"])
snapshot, snapshot_kind = None, None
snapshot_candidates, snapshot_rejections = [], []
final = simulation_dir / "fv_final.npz"
if final.is_file():
    snapshot_candidates.append(("final", final))
if simulation_dir.is_dir():
    steps = []
    for candidate in simulation_dir.iterdir():
        match = re.fullmatch(r"fv_step_(\d+)\.npz", candidate.name)
        if match and candidate.is_file():
            steps.append((int(match.group(1)), candidate))
    snapshot_candidates.extend(("accepted_step", path) for step, path in sorted(steps, reverse=True))

for kind, candidate in snapshot_candidates:
    try:
        saved_step = npz_scalar(candidate, ["step", "accepted_step"])
        saved_time = npz_scalar(candidate, ["time_s", "t", "time", "t_s", "accepted_time_s"])
        if accepted_step is None or accepted_time is None:
            raise ValueError("Status does not specify both accepted step and time")
        if saved_step is None or saved_time is None:
            raise ValueError("Snapshot does not specify both step and time")
        if int(saved_step) != int(accepted_step) or not math.isclose(
                float(saved_time), float(accepted_time), rel_tol=1e-11, abs_tol=1e-12):
            raise ValueError("Snapshot step/time differs from accepted status")
        snapshot, snapshot_kind = candidate, kind
        shutil.copy2(str(snapshot), str(return_dir / snapshot.name))
        copied.append(snapshot.name)
        break
    except (OSError, ValueError, KeyError, TypeError, struct.error, zipfile.BadZipFile) as error:
        snapshot_rejections.append({"file": candidate.name, "reason": str(error)})
if snapshot is None:
    missing.append("FV snapshot matching the accepted status step and time")

checkpoint = simulation_dir / "checkpoint_latest.npz"
if checkpoint.is_file():
    shutil.copy2(str(checkpoint), str(return_dir / checkpoint.name))
    copied.append(checkpoint.name)
else:
    missing.append("checkpoint_latest.npz")

source_hashes = {}
for name in ["cartesian_focused_5s.py", "run_cartesian_focused_5s.sh"]:
    source = return_dir / name
    if source.is_file():
        source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
metadata = {
    "case": case, "target_time_s": 5.0,
    "slurm_job_id": job_id, "slurm_array_job_id": array_id, "slurm_array_task_id": task_id,
    "work_directory": str(run_root), "submission_directory": str(submit_dir),
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "workload_exit_code": workload_rc, "log_capture_exit_code": log_rc,
    "last_workload_stage": stage, "workload_wall_seconds": finished_epoch-started_epoch,
    "resume_input": resume or None, "simulation_status": status,
    "simulation_status_read_error": status_error,
    "selected_snapshot": snapshot.name if snapshot else None,
    "selected_snapshot_kind": snapshot_kind,
    "snapshot_rejections": snapshot_rejections,
    "copied_files": copied, "missing_files": missing, "source_sha256": source_hashes,
}
(return_dir / "launcher_status.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
readme = """CARTESIAN FOCUSED-GRID 5-SECOND RETURN PACKAGE

Case: {case}; Slurm array/task: {array_id}/{task_id}; job: {job_id}.
Workload exit code: {rc}; last stage: {stage}; wall seconds: {wall}.
Selected FV snapshot: {snapshot}.
Missing files: {missing}.

The selected FV snapshot must match the accepted step and physical time in
run_status.json. A failed attempted step is not substituted for that state.
If no matching snapshot exists, launcher_status.json records the reason.
checkpoint_latest.npz is included separately when available and can be older
than the selected snapshot; its own metadata determine the restart time.

The two source files are the exact copies used by this job. run.log includes
module setup, dependency preflight, self-tests and simulation stdout/stderr.
Inspect run_status.json before treating this as a completed 5-second result.
Exit 75 denotes a graceful pause when reported by the solver; it is not full
completion. Other nonzero exit codes also remain nonzero in the Slurm job.

For a pause, submit a NEW job with --array={task_id} and
--export=ALL,CARTESIAN_RESUME=/absolute/path/checkpoint_latest.npz.
Use only the matching case's checkpoint. The new job creates new output and
return directories and never overwrites the previous run.

All full outputs remain in:
{run_root}

Copy this folder or its sibling ZIP back for review. Each array task creates
its own folder and ZIP. Ordinary command failures are packaged; a scheduler
hard kill can prevent packaging. The signal requested 180 seconds before the
time limit allows a graceful checkpoint, but a very long solve may exceed it.
""".format(case=case, array_id=array_id, task_id=task_id, job_id=job_id,
           rc=workload_rc, stage=stage, wall=finished_epoch-started_epoch,
           snapshot=snapshot.name if snapshot else "NONE",
           missing=", ".join(missing) if missing else "none", run_root=run_root)
(return_dir / "README.txt").write_text(readme, encoding="utf-8")
fd, temp_name = tempfile.mkstemp(prefix="." + return_name + "_", suffix=".zip.tmp", dir=str(submit_dir))
os.close(fd)
try:
    with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(return_dir.iterdir()):
            archive.write(str(path), arcname=return_name + "/" + path.name)
    with zipfile.ZipFile(temp_name, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError("ZIP verification failed for " + bad)
    os.link(temp_name, str(return_zip))
finally:
    if os.path.exists(temp_name):
        os.unlink(temp_name)
print("Return folder:", return_dir)
print("Return ZIP:", return_zip)
print("Selected accepted snapshot:", snapshot.name if snapshot else "NONE")
if missing:
    print("Missing files:", ", ".join(missing))
PY
then
    cart_package_rc=0
else
    cart_package_rc=$?
    printf 'Return packaging failed (exit %s); preserved work directory: %s\n' \
        "$cart_package_rc" "$cart_run_root" >&2
fi

# Successful packaging must never conceal an unsuccessful simulation.
if (( cart_workload_rc != 0 )); then
    exit "$cart_workload_rc"
elif (( cart_log_rc != 0 )); then
    exit "$cart_log_rc"
else
    exit "$cart_package_rc"
fi
