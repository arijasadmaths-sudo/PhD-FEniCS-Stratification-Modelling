#!/bin/bash
#SBATCH --job-name=axisym70_fine2s
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=axisym70_fine2s_%j.out
#SBATCH --error=axisym70_fine2s_%j.err

# Submit from the directory containing this file and its Python companion:
#   sbatch run_axisymmetric_70_fine_flux_2s.sh
# One fresh-start fine-mesh diagnostic: 70 cc/min, dt=0.001 s, T=2 s,
# 50,184 triangular cells. No job array and no restart.
# Every command failure is handled explicitly so ordinary solver failures
# still produce a small return folder and ZIP, with the original exit code.
set -uo pipefail
umask 022

axisym_submit_dir="${SLURM_SUBMIT_DIR:?Submit with sbatch from the directory containing both source files.}"
axisym_job_id="${SLURM_JOB_ID:?This launcher requires a Slurm job ID.}"
if [[ ! "$axisym_job_id" =~ ^[0-9]+$ ]]; then
    printf 'Unexpected Slurm job ID: %s\n' "$axisym_job_id" >&2
    exit 2
fi
cd -- "$axisym_submit_dir" || exit $?
axisym_submit_dir="$PWD"
axisym_python_name="axisymmetric_70_fine_flux_2s.py"
axisym_shell_name="run_axisymmetric_70_fine_flux_2s.sh"
for axisym_name in "$axisym_python_name" "$axisym_shell_name"; do
    if [[ ! -f "$axisym_submit_dir/$axisym_name" ]]; then
        printf 'Missing source file: %s\n' "$axisym_submit_dir/$axisym_name" >&2
        exit 2
    fi
done

axisym_run_root="/path/to/work/axisymmetric_70_fine_flux_2s_${axisym_job_id}"
axisym_simulation_dir="$axisym_run_root/simulation"
axisym_return_name="axisymmetric_fine_2s_RETURN_${axisym_job_id}"
axisym_return_dir="$axisym_submit_dir/$axisym_return_name"
axisym_return_zip="$axisym_return_dir.zip"
for axisym_path in "$axisym_run_root" "$axisym_return_dir" "$axisym_return_zip"; do
    if [[ -e "$axisym_path" || -L "$axisym_path" ]]; then
        printf 'Refusing to overwrite an existing path: %s\n' "$axisym_path" >&2
        exit 2
    fi
done
# mkdir, without -p, also protects against a concurrent run using this job ID.
mkdir -- "$axisym_run_root" || exit $?
axisym_run_log="$axisym_run_root/run.log"
axisym_stage="source_snapshot"

axisym_run_workload() {
    printf 'Job ID: %s\nSubmission directory: %s\nWork directory: %s\n' \
        "$axisym_job_id" "$axisym_submit_dir" "$axisym_run_root"
    printf 'Case: balanced_flux_fine_2s; Q=70 cc/min; dt=0.001 s; T=2 s; cells=50184\n'
    printf 'Started (UTC): %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # Run the captured Python file so later edits in the submission directory
    # cannot change the source being executed or archived. Under Slurm,
    # BASH_SOURCE[0] is the actual submitted shell script in the Slurm spool.
    cp -- "$axisym_submit_dir/$axisym_python_name" "$axisym_run_root/$axisym_python_name" || return $?
    cp -- "${BASH_SOURCE[0]}" "$axisym_run_root/$axisym_shell_name" || return $?

    axisym_stage="module_setup"
    module purge || return $?
    module add languages/python/fenics-2019.1.0 || return $?
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export PYTHONUNBUFFERED=1

    axisym_stage="preflight"
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
    raise RuntimeError("This diagnostic must use one MPI rank.")
PY

    axisym_stage="self_test"
    python3 "$axisym_run_root/$axisym_python_name" --self-test || return $?

    # Keep the simulation directory absent until Python creates it: the
    # simulation rejects a nonempty output directory by design.
    axisym_stage="simulation"
    printf 'Simulation output: %s\n' "$axisym_simulation_dir"
    srun --mpi=pmi2 python3 "$axisym_run_root/$axisym_python_name" \
        --output "$axisym_simulation_dir" || return $?
    axisym_stage="completed"
}

# Process substitution leaves module changes in this shell. Wait for tee to
# finish before copying run.log, and check its status separately from the run.
exec 3>&1 4>&2
exec > >(tee --output-error=warn "$axisym_run_log") 2>&1
axisym_tee_pid=$!
if axisym_run_workload; then
    axisym_workload_rc=0
else
    axisym_workload_rc=$?
fi
printf 'Workload exit code: %s; last stage: %s\n' "$axisym_workload_rc" "$axisym_stage"
printf 'Finished (UTC): %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec 1>&3 2>&4 3>&- 4>&-
if wait "$axisym_tee_pid"; then
    axisym_log_rc=0
else
    axisym_log_rc=$?
    printf 'Log capture failed with exit code %s.\n' "$axisym_log_rc" >&2
fi

# Standard-library-only packaging also works after an ordinary solver error.
# The folder is exclusively created; the ZIP is written to a temporary file
# and published with an atomic, non-overwriting hard link in the same directory.
if python3 - "$axisym_run_root" "$axisym_submit_dir" "$axisym_job_id" \
    "$axisym_workload_rc" "$axisym_log_rc" "$axisym_stage" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile

run_root = Path(sys.argv[1])
submit_dir = Path(sys.argv[2])
job_id = sys.argv[3]
workload_rc = int(sys.argv[4])
log_rc = int(sys.argv[5])
stage = sys.argv[6]
simulation_dir = run_root / "simulation"
return_name = "axisymmetric_fine_2s_RETURN_" + job_id
return_dir = submit_dir / return_name
return_zip = submit_dir / (return_name + ".zip")

if os.path.lexists(str(return_dir)) or os.path.lexists(str(return_zip)):
    raise RuntimeError("Refusing to overwrite an existing return folder or ZIP.")
return_dir.mkdir()  # Exclusive creation; any concurrent collision raises.

required = [
    simulation_dir / "verification_configuration.json",
    simulation_dir / "verification_status.json",
    simulation_dir / "verification_budget.csv",
    run_root / "run.log",
    run_root / "axisymmetric_70_fine_flux_2s.py",
    run_root / "run_axisymmetric_70_fine_flux_2s.sh",
]
copied = []
missing = []
for source in required:
    if source.is_file():
        shutil.copy2(str(source), str(return_dir / source.name))
        copied.append(source.name)
    else:
        missing.append(source.name)

snapshot = None
snapshot_kind = None
for name, kind in [
    ("verification_final.npz", "final"),
    ("verification_failure_state.npz", "failure_state"),
]:
    candidate = simulation_dir / name
    if candidate.is_file():
        snapshot, snapshot_kind = candidate, kind
        break
if snapshot is None and simulation_dir.is_dir():
    completed = []
    for candidate in simulation_dir.iterdir():
        match = re.fullmatch(r"verification_step_(\d+)\.npz", candidate.name)
        if match and candidate.is_file():
            completed.append((int(match.group(1)), candidate))
    if completed:
        # These NPZ names are published by atomic rename after snapshot writes.
        snapshot = max(completed, key=lambda item: item[0])[1]
        snapshot_kind = "latest_completed_step"
if snapshot is not None:
    shutil.copy2(str(snapshot), str(return_dir / snapshot.name))
    copied.append(snapshot.name)
else:
    missing.append("final, failure-state, or completed-step NPZ snapshot")

simulation_status = None
status_read_error = None
status_path = return_dir / "verification_status.json"
if status_path.is_file():
    try:
        simulation_status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        status_read_error = str(error)

source_sha256 = {}
for name in ["axisymmetric_70_fine_flux_2s.py", "run_axisymmetric_70_fine_flux_2s.sh"]:
    path = return_dir / name
    if path.is_file():
        source_sha256[name] = hashlib.sha256(path.read_bytes()).hexdigest()

metadata = {
    "case": "balanced_flux_fine_2s",
    "scope": "One fresh-start 2-second fine-mesh diagnostic at 70 cc/min.",
    "Q_cc_min": 70.0,
    "dt_s": 0.001,
    "target_time_s": 2.0,
    "expected_triangular_cells": 50184,
    "slurm_job_id": job_id,
    "work_directory": str(run_root),
    "submission_directory": str(submit_dir),
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "workload_exit_code": workload_rc,
    "log_capture_exit_code": log_rc,
    "last_workload_stage": stage,
    "simulation_status": simulation_status,
    "simulation_status_read_error": status_read_error,
    "selected_snapshot": snapshot.name if snapshot else None,
    "selected_snapshot_kind": snapshot_kind,
    "copied_files": copied,
    "missing_files": missing,
    "source_sha256": source_sha256,
}
(return_dir / "launcher_status.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

sim_state = simulation_status.get("status", "unknown") if isinstance(simulation_status, dict) else "unavailable"
readme = """AXISYMMETRIC FINE-MESH 2-SECOND RETURN PACKAGE

Case: balanced_flux_fine_2s
Requested settings: 70 cc/min; dt=0.001 s; T=2 s; 50,184 triangular cells.
Slurm job ID: {job_id}
Workload exit code: {workload_rc}
Log-capture exit code: {log_rc}
Last workload stage: {stage}
Simulation status: {sim_state}
Selected NPZ: {snapshot_name}
Snapshot selection: {snapshot_kind}
Missing files: {missing}

The selected snapshot is the final NPZ when available, otherwise the failure
state, otherwise the latest atomically completed step snapshot. A failure NPZ
can contain different accepted-scalar and attempted-flow times; use its time
and stage labels. Inspect verification_status.json and launcher_status.json
before treating any output as a completed 2-second result.

run.log combines module setup, dependency preflight, self-tests, and simulation
stdout/stderr. The two source files are the exact copies captured for this job;
the simulation executed the archived Python copy. Source SHA-256 hashes and
the original simulation status are recorded in launcher_status.json.

This is a short fine-mesh diagnostic for comparison at matched physical times.
Further mesh/time studies are needed to establish convergence. Only one NPZ
is included here; all intermediate outputs remain in:
{run_root}

The sibling ZIP contains this complete folder. Copy either the ZIP or this
folder back for review. Packaging runs after normal command completion or
ordinary command failure; a scheduler hard kill can prevent packaging.
""".format(
    job_id=job_id, workload_rc=workload_rc, log_rc=log_rc, stage=stage,
    sim_state=sim_state, snapshot_name=snapshot.name if snapshot else "NONE",
    snapshot_kind=snapshot_kind or "unavailable",
    missing=", ".join(missing) if missing else "none", run_root=run_root,
)
(return_dir / "README.txt").write_text(readme, encoding="utf-8")

fd, temp_name = tempfile.mkstemp(prefix="." + return_name + "_", suffix=".zip.tmp", dir=str(submit_dir))
os.close(fd)
try:
    with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(return_dir.iterdir()):
            archive.write(str(path), arcname=return_name + "/" + path.name)
    with zipfile.ZipFile(temp_name, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError("ZIP verification failed for " + bad_member)
    # Unlike os.replace, link fails if the final archive already exists.
    os.link(temp_name, str(return_zip))
finally:
    if os.path.exists(temp_name):
        os.unlink(temp_name)

print("Return folder:", return_dir)
print("Return ZIP:", return_zip)
print("Selected snapshot:", snapshot.name if snapshot else "NONE")
if missing:
    print("Missing files:", ", ".join(missing))
PY
then
    axisym_package_rc=0
else
    axisym_package_rc=$?
    printf 'Return packaging failed (exit %s); preserved work directory: %s\n' \
        "$axisym_package_rc" "$axisym_run_root" >&2
fi

# A successful archive must never turn a failed solver into a successful job.
if (( axisym_workload_rc != 0 )); then
    exit "$axisym_workload_rc"
elif (( axisym_log_rc != 0 )); then
    exit "$axisym_log_rc"
else
    exit "$axisym_package_rc"
fi
