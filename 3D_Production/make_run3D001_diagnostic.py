#!/usr/bin/env python3
"""Make a short diagnostic copy of the existing run3D001.py; do not run FEniCS.

Run ONCE in the directory containing your actual production Python file:
    python3 make_run3D001_diagnostic.py

Then use a COPY of your existing Slurm launcher to launch
run3D001_diagnostic.py instead of run3D001.py. Keep the existing environment,
MPI launcher and resource settings; use a short diagnostic allocation.

The original Python file is not changed. The diagnostic copy keeps the same
output root and restart logic, so DO NOT run it alongside another job writing
to that same root. Accepted steps are saved by the original checkpoint code.
By default the diagnostic copy stops after at most ten new steps. A successful
ten-step test does not rule out a failure later in the trajectory.

The extra code exposes PETSc's C-level traceback, identifies the step/time,
records the Python/PETSc/MPI environment, and prints selected KSP residuals.
It does NOT change the mesh, equations, timestep, tolerances or preconditioner.
This is a diagnostic patch, not a claimed fix for PETSc error 76.
"""

import argparse
import ast
import hashlib
import re
import sys
from pathlib import Path

PATCH = r'''    # ERROR76_DIAGNOSTIC_BEGIN -- logging only; solve settings unchanged.
    if step == start_step + 1:
        import dolfin as _diag_dolfin
        import petsc4py as _diag_petsc4py
        root_print("[DIAG] Python executable:", sys.executable)
        root_print("[DIAG] DOLFIN:", _diag_dolfin.__version__, _diag_dolfin.__file__)
        root_print("[DIAG] petsc4py:", _diag_petsc4py.__version__, PETSc.__file__)
        root_print("[DIAG] PETSc version:", PETSc.Sys.getVersion())
        root_print("[DIAG] MPI ranks actually running:", nproc)
        root_print("[DIAG] MPI library:", PY_MPI.Get_library_version().strip())
        root_print("[DIAG] Working directory:", os.getcwd())
        root_print("[DIAG] SLURM job:", os.environ.get("SLURM_JOB_ID", "unset"))
        root_print("[DIAG] CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", "unset"))
        root_print("[DIAG] OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS", "unset"))
        root_print("[DIAG] PETSC_OPTIONS:", os.environ.get("PETSC_OPTIONS", "unset"))
        root_print("[DIAG] Restart file:", restart_path)
        root_print("[DIAG] Restart time:", start_time, "dt:", dt)
        root_print("[DIAG] KSP type / PC type:", ksp.getType(), ksp.getPC().getType())

        def _diag_ksp_monitor(_solver, _iteration, _residual):
            if rank == 0 and (_iteration <= 5 or _iteration % 50 == 0):
                print("[DIAG mono] iteration={} residual={:.16e}".format(
                    _iteration, _residual), flush=True)

        ksp.setMonitor(_diag_ksp_monitor)

    root_print("[DIAG] BEGIN monolithic solve: step={} t={:.12g} dt={:.12g}".format(
        step, t_now, dt))

    # Push immediately before this solve, after DOLFIN startup has finished.
    # Every MPI rank executes this block; no collectives are added on failure.
    PETSc.Sys.pushErrorHandler("traceback")
    try:
        ksp.solve(b_petsc, x_petsc)
    except PETSc.Error as _diag_error:
        print("[DIAG ERROR rank={}] step={} t={:.12g}: {}".format(
            rank, step, t_now, repr(_diag_error)), file=sys.stderr, flush=True)
        print(str(_diag_error), file=sys.stderr, flush=True)
        try:
            print("[DIAG ERROR rank={}] iterations={} reason={} residual={}".format(
                rank, ksp.getIterationNumber(), ksp.getConvergedReason(),
                ksp.getResidualNorm()), file=sys.stderr, flush=True)
        except Exception:
            pass  # Do not replace the original PETSc exception.
        raise  # Never accept a failed step or bypass the original safety checks.
    finally:
        PETSc.Sys.popErrorHandler()

    root_print("[DIAG] END monolithic solve: step={}".format(step))
    # ERROR76_DIAGNOSTIC_END
'''


def build_diagnostic(text: str, max_steps: int) -> str:
    """Validate the known V3 structure and patch only diagnostic behaviour."""
    if "ERROR76_DIAGNOSTIC_BEGIN" in text:
        raise ValueError("This source is already patched; use the original production file.")
    tree = ast.parse(text)
    timestep = [node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "dt" for t in node.targets)]
    if len(timestep) != 1 or abs(float(ast.literal_eval(timestep[0].value)) - 0.001) > 1e-15:
        raise ValueError("Expected a single top-level dt = 0.001. Refusing to patch a test branch.")
    for required in ("from petsc4py import PETSc", "from mpi4py import MPI as PY_MPI",
                     "def root_print(", "start_step", "restart_path", "nproc"):
        if required not in text:
            raise ValueError("Missing expected production-code element: " + required)

    solve_pattern = r"(?m)^    ksp\.solve\(b_petsc, x_petsc\)[ \t]*$"
    if len(re.findall(solve_pattern, text)) != 1:
        raise ValueError("Expected exactly one four-space-indented monolithic ksp.solve call.")
    revised = re.sub(solve_pattern, lambda _: PATCH.rstrip("\n"), text, count=1)

    # Cap the *number of new steps*, leaving dt, T_global and time bookkeeping alone.
    end_pattern = r"(?m)^end_step\s*=\s*min\([^\n]+\)[ \t]*$"
    if len(re.findall(end_pattern, revised)) != 1:
        raise ValueError("Could not uniquely identify the existing end_step calculation.")
    revised = re.sub(end_pattern, lambda m: m.group(0) +
                     "\n# Diagnostic-only cap; no production time/physics settings changed.\n"
                     "end_step = min(end_step, start_step + {})\n".format(max_steps) +
                     "root_print('[DIAG] This job advances at most {} new steps.')".format(max_steps),
                     revised, count=1)
    compile(revised, "run3D001_diagnostic.py", "exec")
    return revised


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default="run3D001.py")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Maximum new timesteps in the diagnostic copy (default: 10).")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    source = Path(args.source).resolve()
    destination = source.with_name("run3D001_diagnostic.py")
    if source == destination:
        parser.error("Use the original run3D001.py as source, not the diagnostic copy.")
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8-sig").replace("\r\n", "\n")
        revised = build_diagnostic(text, args.max_steps)
        # Exclusive creation protects an existing diagnostic file from overwrite.
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(revised)
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError) as exc:
        print("No production file changed. Diagnostic copy not created: {}".format(exc),
              file=sys.stderr)
        return 1
    print("Created: {}".format(destination))
    print("Original unchanged: {}".format(source))
    print("Original SHA256: {}".format(hashlib.sha256(raw).hexdigest()))
    print("Maximum new steps: {} (dt unchanged at 0.001)".format(args.max_steps))
    print("Use your existing MPI/Slurm launcher with run3D001_diagnostic.py.")
    print("Do not run alongside another writer to the production output directory.")
    print("This reports the failure; it does not claim to repair it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
