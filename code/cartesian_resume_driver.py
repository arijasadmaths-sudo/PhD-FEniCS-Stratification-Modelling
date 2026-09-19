#!/usr/bin/env python3
"""Run the unchanged Cartesian solver and relay a batch stop-file request.

The solver already has an accepted-step stop handler. Calling that handler
avoids sending TERM through the MPI runtime. No numerical state is changed.
"""
import argparse
from pathlib import Path
import runpy
import signal
import sys
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("solver")
    parser.add_argument("solver_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    stop_file = Path(args.stop_file)
    finished = threading.Event()

    def watch():
        while not finished.wait(1.0):
            if not stop_file.exists():
                continue
            handler = signal.getsignal(signal.SIGUSR1)
            # Defer a request during imports/initialisation until the solver
            # installs its own hook. Do not invoke an MPI signal handler.
            if callable(handler) and getattr(handler, "__name__", "") == "request_stop":
                handler(signal.SIGUSR1, None)
                return

    observer = threading.Thread(target=watch, name="batch-stop-observer", daemon=True)
    observer.start()
    sys.argv = [args.solver] + args.solver_arguments
    try:
        runpy.run_path(args.solver, run_name="__main__")
    finally:
        finished.set()
        observer.join(timeout=2.0)


if __name__ == "__main__":
    main()
