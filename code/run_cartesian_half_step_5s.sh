#!/bin/bash
#SBATCH --job-name=cart_compact_half
#SBATCH --account=SEMT035584
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#SBATCH --array=0-1
#SBATCH --signal=B:USR1@1800
#SBATCH --output=cart_compact_half_%A_%a.out
#SBATCH --error=cart_compact_half_%A_%a.err

# Both cases start from rest unless CARTESIAN_RESUME explicitly names a new
# half-step RETURN/checkpoint_latest.npz. Resume only one array task at a time.
set -uo pipefail
umask 022
cart_submit_dir="${SLURM_SUBMIT_DIR:?Submit from the folder containing these source files.}"
cart_job_id="${SLURM_JOB_ID:?Submit with sbatch.}"
cart_array_id="${SLURM_ARRAY_JOB_ID:?Use an array, including for a one-case restart.}"
cart_task_id="${SLURM_ARRAY_TASK_ID:?Use an array task.}"
for cart_id in "$cart_job_id" "$cart_array_id"; do
    if [[ ! "$cart_id" =~ ^[0-9]+$ ]]; then
        printf 'Invalid Slurm identifier.\n' >&2
        exit 2
    fi
done
case "$cart_task_id" in
    0) cart_case=compact312p5 ;;
    1) cart_case=compact156p25 ;;
    *) printf 'Only array tasks 0 and 1 are supported.\n' >&2; exit 2 ;;
esac
cart_resume="${CARTESIAN_RESUME:-}"
if [[ -n "$cart_resume" ]]; then
    if [[ "${SLURM_ARRAY_TASK_COUNT:-}" != 1 ]]; then
        printf 'Resume requires a single task: --array=0 or --array=1.\n' >&2
        exit 2
    fi
    if [[ "$cart_resume" != /* || ! -r "$cart_resume" || ! -f "$cart_resume" ]]; then
        printf 'CARTESIAN_RESUME requires an absolute path to a readable half-step RETURN checkpoint.\n' >&2
        exit 2
    fi
fi
cd -- "$cart_submit_dir" || exit $?
cart_submit_dir="$PWD"
cart_python_name=cartesian_half_step_5s.py
cart_shell_name=run_cartesian_half_step_5s.sh
cart_driver_name=cartesian_resume_driver.py
cart_helper_name=cartesian_return.py
for cart_name in "$cart_python_name" "$cart_shell_name" "$cart_driver_name" "$cart_helper_name" README.md; do
    if [[ ! -f "$cart_name" ]]; then
        printf 'Missing required file: %s\n' "$cart_name" >&2
        exit 2
    fi
done
cart_identity="${cart_array_id}_${cart_task_id}"
cart_work_parent="${CARTESIAN_WORK_PARENT:-/user/work/kn22417}"
if [[ "$cart_work_parent" != /* || ! -d "$cart_work_parent" ]]; then
    printf 'Work parent must be an existing absolute directory.\n' >&2
    exit 2
fi
cart_run_root="$cart_work_parent/cartesian_${cart_case}_half_${cart_identity}"
cart_simulation_dir="$cart_run_root/simulation"
cart_return_name="cartesian_${cart_case}_half_RETURN_${cart_identity}"
for cart_path in "$cart_run_root" "$cart_submit_dir/$cart_return_name" "$cart_submit_dir/$cart_return_name.zip"; do
    if [[ -e "$cart_path" || -L "$cart_path" ]]; then
        printf 'Refusing to overwrite: %s\n' "$cart_path" >&2
        exit 2
    fi
done
mkdir -- "$cart_run_root" || exit $?
cart_run_log="$cart_run_root/run.log"
cart_stop_file="$cart_run_root/stop_requested"
cart_request_stop() {
    : > "$cart_stop_file"
    printf 'Batch pause requested; waiting for an accepted checkpoint and return ZIP.\n'
}
trap cart_request_stop USR1
cart_stage=source_snapshot
cart_run_workload() {
    printf 'Case: %s; job: %s; array/task: %s\n' "$cart_case" "$cart_job_id" "$cart_identity"
    printf 'Work directory: %s\nResume: %s\n' "$cart_run_root" "${cart_resume:-NONE; start from rest}"
    printf 'dt=0.0005 s; target=5 s; 10000 steps; one MPI rank.\n'
    cp -- "$cart_submit_dir/$cart_python_name" "$cart_run_root/$cart_python_name" || return $?
    cp -- "${BASH_SOURCE[0]}" "$cart_run_root/$cart_shell_name" || return $?
    for cart_name in "$cart_driver_name" "$cart_helper_name" README.md; do
        cp -- "$cart_submit_dir/$cart_name" "$cart_run_root/$cart_name" || return $?
    done
    if [[ -d "$cart_submit_dir/reference_dt001" ]]; then
        cp -a -- "$cart_submit_dir/reference_dt001" "$cart_run_root/reference_dt001" || return $?
    fi
    cart_stage=module_setup
    module purge || return $?
    module add languages/python/fenics-2019.1.0 || return $?
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
    cart_stage=dependency_preflight
    python3 - <<'PY' || return $?
import platform
import sys
import dolfin
import numpy
import scipy
print('Host:', platform.node(), 'Python:', sys.version.replace('\n', ' '))
print('DOLFIN:', dolfin.__version__, 'NumPy:', numpy.__version__, 'SciPy:', scipy.__version__)
if not dolfin.__version__.startswith('2019.1'):
    raise RuntimeError('Load legacy FEniCS 2019.1')
if dolfin.MPI.size(dolfin.MPI.comm_world) != 1:
    raise RuntimeError('Exactly one MPI rank is required')
PY
    if [[ -n "$cart_resume" ]]; then
        cart_stage=checkpoint_preflight
        python3 "$cart_run_root/$cart_helper_name" prepare --checkpoint "$cart_resume" \
            --root "$cart_run_root" --case "$cart_case" || return $?
    fi
    cart_stage=self_test
    python3 "$cart_run_root/$cart_python_name" --self-test || return $?
    cart_stage=simulation
    cart_arguments=(--case "$cart_case" --output "$cart_simulation_dir")
    if [[ -n "$cart_resume" ]]; then
        cart_arguments+=(--resume "$cart_run_root/checkpoint_input.npz")
    fi
    srun --mpi=pmi2 python3 "$cart_run_root/$cart_driver_name" \
        --stop-file "$cart_stop_file" "$cart_run_root/$cart_python_name" \
        "${cart_arguments[@]}" &
    cart_sim_pid=$!
    while :; do
        if wait "$cart_sim_pid"; then
            cart_sim_rc=0
            break
        else
            cart_sim_rc=$?
        fi
        # USR1 interrupts wait; the MPI step continues while the driver asks the
        # solver to pause. Do not package until the solver has actually exited.
        if (( cart_sim_rc > 128 )) && kill -0 "$cart_sim_pid" 2>/dev/null; then
            continue
        fi
        break
    done
    if (( cart_sim_rc != 0 )); then
        return "$cart_sim_rc"
    fi
    cart_stage=completed
}

exec 3>&1 4>&2
exec > >(tee --output-error=warn "$cart_run_log") 2>&1
cart_tee_pid=$!
if cart_run_workload; then
    cart_workload_rc=0
else
    cart_workload_rc=$?
fi
printf 'Workload exit code: %s; last stage: %s\n' "$cart_workload_rc" "$cart_stage"
exec 1>&3 2>&4 3>&- 4>&-
if wait "$cart_tee_pid"; then
    cart_log_rc=0
else
    cart_log_rc=$?
fi
if python3 "$cart_run_root/$cart_helper_name" pack --root "$cart_run_root" --submit "$cart_submit_dir" \
    --job "$cart_job_id" --array "$cart_array_id" --task "$cart_task_id" --case "$cart_case" \
    --rc "$cart_workload_rc" --log-rc "$cart_log_rc" --stage "$cart_stage" --resume "$cart_resume"; then
    cart_package_rc=0
else
    cart_package_rc=$?
    printf 'Packaging issue (exit %s); retain work directory %s\n' "$cart_package_rc" "$cart_run_root" >&2
fi
if (( cart_workload_rc != 0 )); then
    exit "$cart_workload_rc"
elif (( cart_log_rc != 0 )); then
    exit "$cart_log_rc"
else
    exit "$cart_package_rc"
fi
