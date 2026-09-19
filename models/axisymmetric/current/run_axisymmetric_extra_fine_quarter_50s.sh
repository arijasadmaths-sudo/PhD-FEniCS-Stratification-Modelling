#!/bin/bash
#SBATCH --job-name=axisym50_extra_quarter
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=axisymmetric50_extra_fine_quarter_%j.out
#SBATCH --error=axisymmetric50_extra_fine_quarter_%j.err

# Fresh 5 s diagnostic: sbatch run_axisymmetric_extra_fine_quarter_50s.sh
# Default stop remains the absolute 5 s review point; see README.txt.
set -uo pipefail
umask 022
axisym_submit_dir="${SLURM_SUBMIT_DIR:?Submit with sbatch from the bundle directory}"
axisym_job="${SLURM_JOB_ID:?Missing Slurm job ID}"
if [[ ! "$axisym_job" =~ ^[0-9]+$ ]]; then
    echo "Invalid Slurm job identifier" >&2
    exit 2
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "This launcher submits only extra_fine_quarter; omit --array." >&2
    exit 2
fi
axisym_case=extra_fine_quarter
cd -- "$axisym_submit_dir" || exit $?
axisym_submit_dir="$PWD"
axisym_job_key="${axisym_job}"
axisym_work_parent="${AXISYM_WORK_PARENT:-/path/to/work}"
axisym_run_root="${axisym_work_parent}/axisymmetric50_${axisym_case}_${axisym_job_key}"
axisym_return="${axisym_submit_dir}/axisymmetric50_${axisym_case}_RETURN_${axisym_job_key}"
for axisym_path in "$axisym_run_root" "$axisym_return" "$axisym_return.zip"; do
    if [[ -e "$axisym_path" || -L "$axisym_path" ]]; then
        printf 'Refusing to overwrite existing path: %s\n' "$axisym_path" >&2
        exit 2
    fi
done
axisym_files=(axisymmetric_70_study_50s.py study_checkpoint.py check_restart.py review_gate.py check_underflow.py pack_return.py run_axisymmetric_extra_fine_quarter_50s.sh README.txt SOURCE_PROVENANCE.json)
for axisym_name in "${axisym_files[@]}"; do
    if [[ ! -f "$axisym_submit_dir/$axisym_name" || -L "$axisym_submit_dir/$axisym_name" ]]; then
        printf 'Missing regular bundle file: %s\n' "$axisym_name" >&2
        exit 2
    fi
done
mkdir -- "$axisym_run_root" || exit $?
axisym_stage=source_snapshot
axisym_child=""
axisym_stop=0
axisym_signal() {
    axisym_stop=1
    if [[ -n "$axisym_child" ]]; then
        kill -USR1 "$axisym_child" 2>/dev/null || true
    fi
}
trap axisym_signal USR1 TERM INT

axisym_workload() {
    printf 'Case: %s\nJob: %s\nWork directory: %s\n' "$axisym_case" "$axisym_job_key" "$axisym_run_root"
    printf 'Started UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for axisym_name in "${axisym_files[@]}"; do
        if [[ "$axisym_name" == run_axisymmetric_extra_fine_quarter_50s.sh ]]; then
            cp -- "${BASH_SOURCE[0]}" "$axisym_run_root/$axisym_name" || return $?
        else
            cp -- "$axisym_submit_dir/$axisym_name" "$axisym_run_root/$axisym_name" || return $?
        fi
    done
    for axisym_directory in provenance analysis reference_fields; do
        if [[ -d "$axisym_submit_dir/$axisym_directory" ]]; then
            cp -a -- "$axisym_submit_dir/$axisym_directory" "$axisym_run_root/$axisym_directory" || return $?
        fi
    done
    if [[ -f "$axisym_submit_dir/LOCAL_VALIDATION.json" ]]; then
        cp -- "$axisym_submit_dir/LOCAL_VALIDATION.json" "$axisym_run_root/LOCAL_VALIDATION.json" || return $?
    fi
    python3 - "$axisym_run_root" "$axisym_case" "$axisym_job_key" <<'PY' || return $?
import json, sys
from pathlib import Path
root, case, job = sys.argv[1:]
(Path(root)/"run_info.json").write_text(json.dumps(dict(case=case, job_key=job), indent=2)+"\n")
PY
    axisym_stage=module_setup
    module purge || return $?
    module add languages/python/fenics-2019.1.0 || return $?
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export PYTHONUNBUFFERED=1
    axisym_stage=preflight
    python3 - <<'PY' || return $?
import dolfin, numpy, scipy, platform, sys
print("Host:", platform.node(), "Python:", sys.version.replace("\n", " "), flush=True)
print("DOLFIN:", dolfin.__version__, "NumPy:", numpy.__version__, "SciPy:", scipy.__version__, flush=True)
if not dolfin.__version__.startswith("2019.1"):
    raise RuntimeError("Use legacy FEniCS 2019.1")
if dolfin.MPI.size(dolfin.MPI.comm_world) != 1:
    raise RuntimeError("This serial axisymmetric solver requires exactly one MPI rank")
PY
    axisym_stage=self_test
    python3 "$axisym_run_root/axisymmetric_70_study_50s.py" --self-test || return $?
    axisym_stage=underflow_regression
    python3 "$axisym_run_root/check_underflow.py" --output "$axisym_run_root/underflow_check.json" || return $?
    axisym_stage=restart_check
    python3 "$axisym_run_root/check_restart.py" --directory "$axisym_run_root/restart_smoke" || return $?
    if [[ "$axisym_stop" == 1 ]]; then
        axisym_stage=stopped_before_simulation
        return 75
    fi
    axisym_args=(--case "$axisym_case" --end-time 50 --output "$axisym_run_root/simulation")
    axisym_continue="${AXISYM_EXTRA_FINE_QUARTER_CONTINUE_50:-0}"
    if [[ "$axisym_continue" != 0 && "$axisym_continue" != 1 ]]; then
        echo "AXISYM_EXTRA_FINE_QUARTER_CONTINUE_50 must be 0 or 1." >&2
        return 2
    fi
    # Empty/unset is a fresh run from salt water at rest, with zero freshwater.
    axisym_resume="${AXISYM_EXTRA_FINE_QUARTER_RESUME:-}"
    axisym_gate_args=(--source "$axisym_run_root/axisymmetric_70_study_50s.py")
    if [[ -n "$axisym_resume" ]]; then
        axisym_args+=(--resume "$axisym_resume")
        axisym_gate_args+=(--resume "$axisym_resume")
    fi
    if [[ "$axisym_continue" == 1 ]]; then
        axisym_gate_args+=(--continue-50)
    fi
    axisym_stage=review_gate
    axisym_remaining=$(python3 "$axisym_run_root/review_gate.py" "${axisym_gate_args[@]}") || return $?
    if [[ ! "$axisym_remaining" =~ ^[0-9]+$ ]]; then
        echo "Invalid step count from review gate" >&2
        return 2
    fi
    if [[ "$axisym_continue" == 0 ]]; then
        axisym_args+=(--stop-after-steps "$axisym_remaining")
    fi
    axisym_stage=simulation
    # One serial Python process inside the one-task Slurm allocation.
    # Direct launch lets the batch warning signal reach Python reliably.
    python3 -u "$axisym_run_root/axisymmetric_70_study_50s.py" "${axisym_args[@]}" &
    axisym_child=$!
    while true; do
        if wait "$axisym_child"; then axisym_solver_rc=0; else axisym_solver_rc=$?; fi
        # A trapped batch signal interrupts wait before the child has exited.
        if kill -0 "$axisym_child" 2>/dev/null; then continue; fi
        break
    done
    axisym_child=""
    if [[ "$axisym_solver_rc" == 0 ]]; then axisym_stage=completed;
    elif [[ "$axisym_solver_rc" == 75 ]]; then axisym_stage=paused;
    else axisym_stage=solver_failed; fi
    return "$axisym_solver_rc"
}

exec 3>&1 4>&2
exec > >(tee --output-error=warn "$axisym_run_root/run.log") 2>&1
axisym_tee_pid=$!
if axisym_workload; then axisym_workload_rc=0; else axisym_workload_rc=$?; fi
printf 'Exit code: %s; stage: %s\n' "$axisym_workload_rc" "$axisym_stage"
printf 'Finished UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec 1>&3 2>&4 3>&- 4>&-
if wait "$axisym_tee_pid"; then axisym_log_rc=0; else axisym_log_rc=$?; fi
if python3 "$axisym_run_root/pack_return.py" --run-root "$axisym_run_root" \
        --destination "$axisym_submit_dir" --exit-code "$axisym_workload_rc" \
        --log-exit-code "$axisym_log_rc" --stage "$axisym_stage"; then
    axisym_pack_rc=0
else
    axisym_pack_rc=$?
    printf 'Packaging failed; work files remain in %s\n' "$axisym_run_root" >&2
fi
if [[ "$axisym_workload_rc" != 0 ]]; then exit "$axisym_workload_rc"; fi
if [[ "$axisym_log_rc" != 0 ]]; then exit "$axisym_log_rc"; fi
exit "$axisym_pack_rc"
