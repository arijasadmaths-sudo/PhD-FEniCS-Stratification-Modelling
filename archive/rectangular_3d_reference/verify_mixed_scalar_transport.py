"""Small numerical acceptance checks for the production mixed scalar class.

Run with the same DOLFIN/MPI environment as the production solver, for example
``srun -n 4 python3 verify_mixed_scalar_transport.py``.  A nonzero exit prevents
the production submission from proceeding. These test the discretization and
implementation, not convergence of the full experimental plume simulation.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from dolfin import (
    BoxMesh, CellType, Constant, Expression, FacetNormal, Function,
    FunctionSpace, LocalSolver, LogLevel, MPI, Measure, MeshFunction, Point,
    TestFunction, TrialFunction, assemble, dot, facets, inner, parameters,
    set_log_level,
)
from mpi4py import MPI as PY_MPI

from mixed_scalar_transport import MixedScalarTransport


parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["quadrature_degree"] = 6
set_log_level(LogLevel.WARNING)
COMM = MPI.comm_world
RANK = MPI.rank(COMM)


def graded_box(n):
    """Nested, non-orthogonal tetrahedral grids with 4:1 vertical grading."""
    mesh = BoxMesh.create(COMM, [Point(0, 0, 0), Point(1, 1, 1)],
                          [n, n, n], CellType.Type.tetrahedron)
    z = mesh.coordinates()[:, 2].copy()
    mesh.coordinates()[:, 2] = np.where(z <= 0.5, 1.6 * z, 0.8 + 0.4 * (z - 0.5))
    mesh.bounding_box_tree().build(mesh)
    markers = MeshFunction("size_t", mesh, 2, 0)
    mesh.init(2, 3)
    for facet in facets(mesh):
        if not facet.exterior():
            continue
        p = facet.midpoint()
        if abs(p.x()) < 1e-12:
            markers[facet] = 3
        elif abs(p.x() - 1) < 1e-12:
            markers[facet] = 5
        elif abs(p.y()) < 1e-12:
            markers[facet] = 1
        elif abs(p.y() - 1) < 1e-12:
            markers[facet] = 2
        elif abs(p.z()) < 1e-12:
            markers[facet] = 4
        elif abs(p.z() - 1) < 1e-12:
            markers[facet] = 6
        else:
            raise RuntimeError("Verification mesh has an unmarked exterior face.")
    volume = assemble(Constant(1.0) * Measure("dx", domain=mesh))
    if abs(volume - 1.0) > 1e-11:
        raise RuntimeError("Graded verification box has incorrect volume.")
    return mesh, markers


def cell_average(expr, C):
    out = Function(C)
    v = TestFunction(C)
    u = TrialFunction(C)
    dx = Measure("dx", domain=C.mesh())
    local = LocalSolver(u * v * dx, expr * v * dx)
    local.factorize()
    local.solve_local_rhs(out)
    out.vector().apply("insert")
    return out


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def affine_patch():
    mesh, markers = graded_box(4)
    C = FunctionSpace(mesh, "DG", 0)
    exact = Expression("x[0]", degree=1)
    old = cell_average(exact, C)
    initial = Function(C)
    initial.assign(old)
    out = Function(C)
    D, dt = 0.03, 0.02
    transport = MixedScalarTransport(
        mesh, C, old, Constant((0., 0., 0.)), D, dt, markers,
        noflux_markers=(1, 2, 4, 6),
        dirichlet_values={3: exact, 5: exact},
    )
    iterations = transport.solve(out)
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=markers)
    n = FacetNormal(mesh)
    c_error = math.sqrt(abs(assemble((out - initial)**2 * dx)))
    q_error = math.sqrt(abs(assemble(inner(
        transport.diffusive_flux - Constant((-D, 0., 0.)),
        transport.diffusive_flux - Constant((-D, 0., 0.))) * dx)))
    outward = assemble(dot(transport.diffusive_flux, n) * ds)
    balance = assemble((out - initial) * dx) + dt * outward
    require(c_error < 2e-7, "Affine diffusion patch: cell means are incorrect.")
    require(q_error < 2e-7, "Affine diffusion patch: diffusive flux/sign is incorrect.")
    require(abs(balance) < 2e-9, "Affine diffusion patch: mass budget does not close.")
    return dict(cell_mean_l2_error=c_error, flux_l2_error=q_error,
                mass_balance_error=balance, iterations=iterations,
                true_residual=transport.last_true_residual)


def constant_advection():
    mesh, markers = graded_box(4)
    C = FunctionSpace(mesh, "DG", 0)
    value = 0.37
    old = Function(C)
    old.assign(Constant(value))
    out = Function(C)
    velocity = Constant((0.2, 0., 0.))
    dt = 0.02
    transport = MixedScalarTransport(
        mesh, C, old, velocity, 0.01, dt, markers,
        noflux_markers=(1, 2, 4, 6),
        dirichlet_values={3: Constant(value), 5: Constant(value)},
        inflow_values={3: Constant(value), 5: Constant(value)},
    )
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=markers)
    n = FacetNormal(mesh)
    un = dot(velocity, n)
    max_balance = 0.0
    for _ in range(5):
        previous_mass = assemble(old * dx)
        transport.solve(out)
        advective_outward = assemble(
            (0.5 * (un + abs(un)) * out + 0.5 * (un - abs(un)) * value) * ds
        )
        diffusive_outward = assemble(dot(transport.diffusive_flux, n) * ds)
        balance = assemble(out * dx) - previous_mass + dt * (advective_outward + diffusive_outward)
        max_balance = max(max_balance, abs(balance))
        old.assign(out)
    error = math.sqrt(abs(assemble((out - value)**2 * dx)))
    require(error < 2e-7, "Matching-inflow advection does not preserve a constant concentration.")
    require(max_balance < 2e-9, "Constant advection mass budget does not close.")
    return dict(l2_error=error, max_mass_balance_error=max_balance,
                true_residual=transport.last_true_residual)


def independent_small_diffusivities():
    """Exercise the density/dye use case with two states sharing the FE space."""
    mesh, markers = graded_box(4)
    C = FunctionSpace(mesh, "DG", 0)
    first_exact = Expression("x[0]", degree=1)
    second_exact = Expression("0.25 + 0.5*x[0]", degree=1)
    first_old = cell_average(first_exact, C)
    second_old = cell_average(second_exact, C)
    first_out, second_out = Function(C), Function(C)
    first_D, second_D, dt = 1.5e-9, 4.14e-10, 0.001
    first = MixedScalarTransport(
        mesh, C, first_old, Constant((0., 0., 0.)), first_D, dt, markers,
        noflux_markers=(1, 2, 4, 6), dirichlet_values={3:first_exact, 5:first_exact},
    )
    second = MixedScalarTransport(
        mesh, C, second_old, Constant((0., 0., 0.)), second_D, dt, markers,
        noflux_markers=(1, 2, 4, 6), mixed_space=first.W,
        dirichlet_values={3:second_exact, 5:second_exact},
    )
    first.solve(first_out)
    first_state = first.state.vector().copy()
    second.solve(second_out)
    difference = first.state.vector().copy()
    difference.axpy(-1.0, first_state)
    require(difference.norm("linf") == 0.0, "Scalar instances sharing a space share mutable state.")
    dx = Measure("dx", domain=mesh)
    results = {}
    for label, transport, out, old, exact_flux in (
        ("density", first, first_out, first_old, -first_D),
        ("dye", second, second_out, second_old, -0.5*second_D),
    ):
        field_error = math.sqrt(abs(assemble((out-old)**2 * dx)))
        flux_error = math.sqrt(abs(assemble(inner(
            transport.diffusive_flux-Constant((exact_flux,0.,0.)),
            transport.diffusive_flux-Constant((exact_flux,0.,0.))) * dx)))
        relative_flux_error = flux_error / abs(exact_flux)
        require(field_error < 2e-7, "Small-diffusivity affine concentration is incorrect.")
        require(relative_flux_error < 1e-3, "Small-diffusivity flux scaling is inaccurate.")
        results[label] = dict(diffusivity=transport.D, l2_error=field_error,
                              relative_flux_error=relative_flux_error,
                              true_residual=transport.last_true_residual)
    return results


def cosine_decay(ncells, dt):
    mesh, markers = graded_box(ncells)
    C = FunctionSpace(mesh, "DG", 0)
    D, final_time = 0.05, 0.1
    exact = Expression(
        "0.5 + 0.2*exp(-3*pi*pi*D*t)*cos(pi*x[0])*cos(pi*x[1])*cos(pi*x[2])",
        degree=6, pi=math.pi, D=D, t=0.0,
    )
    old = cell_average(exact, C)
    out = Function(C)
    transport = MixedScalarTransport(
        mesh, C, old, Constant((0., 0., 0.)), D, dt, markers,
        noflux_markers=(1, 2, 3, 4, 5, 6), dirichlet_values={},
    )
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=markers)
    normal = FacetNormal(mesh)
    mass_initial = assemble(old * dx)
    max_balance = 0.0
    iterations = []
    for step in range(1, int(round(final_time / dt)) + 1):
        previous_mass = assemble(old * dx)
        iterations.append(transport.solve(out))
        outward = assemble(dot(transport.diffusive_flux, normal) * ds)
        balance = assemble(out * dx) - previous_mass + dt * outward
        max_balance = max(max_balance, abs(balance))
        old.assign(out)
    exact.t = final_time
    error = math.sqrt(abs(assemble((out - exact)**2 * dx)))
    mass_drift = assemble(out * dx) - mass_initial
    require(abs(mass_drift) < 2e-8, "Neumann diffusion does not preserve total scalar mass.")
    require(max_balance < 2e-9, "Transient diffusion step budget does not close.")
    require(out.vector().min() > 0.25 and out.vector().max() < 0.75,
            "Smooth diffusion produces a large concentration excursion.")
    return dict(n=ncells, dt=dt, time=final_time, l2_error=error,
                mass_drift=mass_drift, max_mass_balance_error=max_balance,
                max_iterations=max(iterations), true_residual=transport.last_true_residual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    args = parser.parse_args()
    started = time.time()
    results = {}
    checks = (
        ("affine_patch", affine_patch, ()),
        ("constant_advection", constant_advection, ()),
        ("independent_small_diffusivities", independent_small_diffusivities, ()),
        ("cosine_coarse", cosine_decay, (4, 0.005)),
        ("cosine_fine", cosine_decay, (8, 0.00125)),
    )
    for name, check, check_args in checks:
        if RANK == 0:
            print("VERIFY {}: START".format(name), flush=True)
        results[name] = check(*check_args)
        if RANK == 0:
            print("VERIFY {}: PASSED {}".format(
                name, json.dumps(results[name], sort_keys=True)), flush=True)
    coarse = results["cosine_coarse"]["l2_error"]
    fine = results["cosine_fine"]["l2_error"]
    require(math.isfinite(fine) and fine < 0.75 * coarse,
            "Refinement did not reduce the manufactured diffusion error sufficiently.")
    results.update(passed=True, refinement_error_ratio=fine/coarse,
                   elapsed_seconds=time.time()-started, mpi_ranks=MPI.size(COMM),
                   scope="Mixed scalar implementation verification; not plume validation")
    error = None
    if RANK == 0:
        try:
            if args.report:
                parent = os.path.dirname(os.path.abspath(args.report))
                os.makedirs(parent, exist_ok=True)
                temporary = args.report + ".tmp"
                with open(temporary, "w") as stream:
                    json.dump(results, stream, indent=2, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, args.report)
            print(json.dumps(results, indent=2, sort_keys=True), flush=True)
            print("MIXED SCALAR VERIFICATION PASSED", flush=True)
        except Exception as exc:
            error = str(exc)
    error = PY_MPI.COMM_WORLD.bcast(error, root=0)
    if error is not None:
        raise RuntimeError("Could not write verification report: " + error)


if __name__ == "__main__":
    main()
