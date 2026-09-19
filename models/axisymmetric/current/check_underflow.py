#!/usr/bin/env python3
"""Small regression for the repaired scalar residual guard; no FEniCS required."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    source = Path(__file__).with_name("axisymmetric_70_study_50s.py")
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == "transport_residual_roundoff"]
    assert len(nodes) == 1
    namespace = {"np": np}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    check = namespace["transport_residual_roundoff"]
    tiny = np.nextafter(np.float64(0), np.float64(1))
    matrix = sparse.eye(4, format="csr")
    residual = np.array([tiny, tiny, 1e-8, 0.])
    scales = np.array([3.48620637e-314, 0., 1., 0.])
    residual_copy, scale_copy = residual.copy(), scales.copy()
    relative, raw, allowance = check(matrix, residual, scales)
    assert raw[0] > 5e-11 and relative[0] == 0
    assert np.isinf(raw[1]) and relative[1] == 0
    assert relative[2] > 5e-11 and relative[3] == 0
    np.testing.assert_array_equal(residual, residual_copy)
    np.testing.assert_array_equal(scales, scale_copy)
    # A tail error beyond the operation-count allowance must remain visible.
    beyond = np.array([100*tiny, 100*tiny, 1e-8, 0.])
    relative_bad, unused, unused_allowance = check(matrix, beyond, scales)
    assert relative_bad[0] > 5e-11 and np.isinf(relative_bad[1])
    report = dict(status="passed", underflow_only_residual_accepted=True,
                  meaningful_and_tiny_tail_errors_rejected=True,
                  inputs_unchanged=True, normal_relative_tolerance=5e-11,
                  allowance_smallest_subnormal_units=float(allowance[0]/tiny),
                  coupled_fenics_run=False)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
