#!/usr/bin/env python3
"""Collect one case into a return folder and ZIP, including after a hard kill.

python3 pack_return.py --run-root /path/printed/in/job/log --destination .
This helper uses only the standard library, so failed FEniCS setup can be reported.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--log-exit-code", type=int)
    parser.add_argument("--stage", default="manual_recovery")
    parser.add_argument("--label", default="", help="Optional safe suffix if a return ZIP already exists")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    destination = Path(args.destination).resolve()
    info = json.loads((root/"run_info.json").read_text())
    case, job = info["case"], info["job_key"]
    if case not in ("coarse", "fine", "fine_half", "extra_fine", "fine_quarter", "extra_fine_half", "extra_fine_quarter"):
        raise ValueError("Unknown case")
    for value in (job, args.label):
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
            raise ValueError("Unsafe job key or label")
    prefix = "axisymmetric5" if case == "fine_quarter" else "axisymmetric50"
    name = "{}_{}_RETURN_{}".format(prefix, case, job)
    if args.label:
        name += "_" + args.label
    folder, archive = destination/name, destination/(name+".zip")
    if os.path.lexists(str(folder)) or os.path.lexists(str(archive)):
        raise FileExistsError("Return path already exists; use --label recovery1 to create a separate copy")
    folder.mkdir(parents=False)
    simulation = root/"simulation"
    required = [root/"run_info.json", root/"run.log"]
    required += [root/n for n in ("axisymmetric_70_study_50s.py", "study_checkpoint.py", "check_restart.py",
                                 "review_gate.py", "check_underflow.py",
                                 "run_axisymmetric_extra_fine_quarter_50s.sh", "pack_return.py", "README.txt",
                                 "SOURCE_PROVENANCE.json", "restart_check.json", "underflow_check.json")]
    required += [simulation/n for n in ("verification_configuration.json", "verification_status.json",
                                       "verification_budget.csv", "checkpoint_latest.npz")]
    optional = [root/"LOCAL_VALIDATION.json"]
    optional += [simulation/n for n in ("verification_final.npz", "verification_paused.npz",
                                       "verification_failure_state.npz")]
    optional += sorted(simulation.glob("field_step_*.npz"))
    optional += [source for directory in ("provenance", "analysis", "reference_fields")
                 for source in sorted((root/directory).rglob("*"))
                 if source.is_file() and source not in required
                 and "__pycache__" not in source.parts and source.suffix != ".pyc"]
    missing = []
    for source in required+optional:
        # Simulation products keep their established flat return names. Bundled
        # original inputs/provenance retain their directories and exact bytes.
        relative = Path(source.name) if source.parent == simulation else source.relative_to(root)
        if source.is_file() and not source.is_symlink():
            target = folder/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(target))
        elif source in required:
            missing.append(relative.as_posix())
    errors = []
    def read_json(name):
        try:
            return json.loads((folder/name).read_text())
        except (OSError, ValueError) as error:
            errors.append(name+": "+str(error))
            return {}
    status = read_json("verification_status.json")
    config = read_json("verification_configuration.json")
    restart_report = read_json("restart_check.json")
    if restart_report.get("status") == "failed":
        for segment in ("full", "first", "resumed"):
            for filename in ("verification_configuration.json", "verification_status.json",
                             "verification_budget.csv", "checkpoint_latest.npz"):
                source = root/"restart_smoke"/segment/filename
                if source.is_file() and not source.is_symlink():
                    shutil.copy2(str(source), str(folder/("restart_"+segment+"_"+filename)))
    last_row = None
    if (folder/"verification_budget.csv").is_file():
        try:
            with (folder/"verification_budget.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    last_row = row
        except (OSError, ValueError, csv.Error) as error:
            errors.append("Budget read: "+str(error))
    reported_complete = status.get("status") == "completed"
    if reported_complete:
        if restart_report.get("status") != "passed":
            errors.append("No passing checkpoint restart check")
        if not (folder/"verification_final.npz").is_file():
            missing.append("verification_final.npz")
        try:
            dt, target = float(config["dt_s"]), float(config["target_time_s"])
            final_step = int(round(target/dt))
            if (int(status["last_recorded_step"]) != final_step or
                    abs(float(status["last_recorded_time_s"])-target) > 1e-10 or
                    not last_row or int(last_row["step"]) != final_step or
                    abs(float(last_row["time_s"])-target) > 1e-10):
                errors.append("Completion times/steps disagree across status, configuration and budget")
            for time_s in config["comparison_times_s"]:
                filename = "field_step_{:09d}.npz".format(int(round(float(time_s)/dt)))
                if not (folder/filename).is_file():
                    missing.append(filename)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            errors.append("Invalid completion metadata: "+str(error))
    # A review pause at 5 s is intentionally not completion of the 50 s target.
    review_ready = False
    if status.get("status") == "paused" and case == "extra_fine_quarter":
        try:
            if int(status["last_recorded_step"]) == 20000:
                if (abs(float(status["last_recorded_time_s"])-5.0) > 1e-12
                        or not last_row or int(last_row["step"]) != 20000
                        or abs(float(last_row["time_s"])-5.0) > 1e-12
                        or float(config["dt_s"]) != .00025):
                    errors.append("5 s review metadata disagree")
                for filename in ("verification_paused.npz", "field_step_000000000.npz",
                                 "field_step_000008000.npz", "field_step_000020000.npz"):
                    if not (folder/filename).is_file():
                        missing.append(filename)
                if restart_report.get("status") != "passed":
                    errors.append("No passing checkpoint restart check")
                review_ready = not missing and not errors and args.exit_code in (None, 75)
        except (KeyError, TypeError, ValueError) as error:
            errors.append("Invalid 5 s review metadata: "+str(error))
    complete = (reported_complete and not missing and not errors and
                args.exit_code in (None, 0) and args.log_exit_code in (None, 0))
    summary = dict(case=case, job_key=job, workload_exit_code=args.exit_code,
                   log_capture_exit_code=args.log_exit_code,
                   last_launcher_stage=args.stage, simulation_status=status.get("status", "unavailable"),
                   completed_with_required_files=complete, ready_for_5s_review=review_ready, missing_files=missing,
                   metadata_errors=errors, work_directory=str(root),
                   last_recorded_step=status.get("last_recorded_step"),
                   last_recorded_time_s=status.get("last_recorded_time_s"),
                   prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   archive_purpose="Numerical review; no convergence conclusion is implied")
    (folder/"launcher_status.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    (folder/"RETURN_README.txt").write_text(
        "Send this entire ZIP back.\n\nCase: "+case+"\nSimulation status: "+
        str(summary["simulation_status"])+"\nCompleted with required files: "+str(complete)+
        "\n\nThe CSV contains the full history through every carried allocation.\n"
        "field_step files contain raw fresh-fluid fraction at matched physical times.\n"
        "checkpoint_latest.npz is the only restart input. Keep its sibling CSV and\n"
        "field_step files beside it. An attempted failure snapshot is not restartable.\n"
        "provenance/ contains the unchanged baseline source and its repair evidence.\n"
        "analysis/ and reference_fields/ retain the supplied comparison material.\n"
        "Exit 75 means safely paused. A paused/failed ZIP is still useful for review.\n"
        "After a hard kill the CSV may extend beyond the last checkpoint; resume\n"
        "carries only the rows through the accepted checkpoint.\n")
    manifest = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        manifest[path.relative_to(folder).as_posix()] = dict(size_bytes=path.stat().st_size, sha256=digest.hexdigest())
    (folder/"content_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    fd, temp_name = tempfile.mkstemp(prefix="."+name+"_", suffix=".tmp", dir=str(destination))
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_name, "w", zipfile.ZIP_DEFLATED, compresslevel=6,
                             allowZip64=True) as handle:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    handle.write(str(path), arcname=name+"/"+path.relative_to(folder).as_posix())
        with zipfile.ZipFile(temp_name) as handle:
            broken = handle.testzip()
            if broken:
                raise RuntimeError("ZIP CRC failed: "+broken)
        os.link(temp_name, str(archive))  # Publish without replacing another archive.
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print("RETURN ZIP: "+str(archive), flush=True)
    print("Simulation status: {}; complete with required files: {}".format(
          summary["simulation_status"], complete), flush=True)
    if reported_complete and not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
