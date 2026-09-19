#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded 50-second verification of the archived 70 cc/min axisymmetric run.

Fresh starts only; no production results or checkpoints are read or overwritten.
The production Oseen IPCS weak forms and DG0 upwind transport are preserved.
Default checks change only timestep or mesh. The optional uncorrected case is
an explicitly unbounded diagnostic control, not a production replacement.
A 50-second check does not establish convergence of the 1000-second record.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time as wallclock
import numpy as np

PRODUCTION_SOURCE_SHA256 = "274102538bede1243b140ebecfd9b6fe74bf49f36712af9cba72f974129ccce3"
CASE_SETTINGS = {
    "timestep": {"dt_s": 0.0005, "refinement": 1, "redistribute": True},
    "mesh": {"dt_s": 0.001, "refinement": 2, "redistribute": True},
    "uncorrected": {"dt_s": 0.001, "refinement": 1, "redistribute": False},
    "baseline": {"dt_s": 0.001, "refinement": 1, "redistribute": True},
}


def _verification_refine_nodes(nodes, factor):
    if factor == 1:
        return np.asarray(nodes, dtype=float)
    if factor != 2:
        raise ValueError("Only refinement factors 1 and 2 are supported.")
    nodes = np.asarray(nodes, dtype=float)
    refined = np.empty(2 * len(nodes) - 1)
    refined[0::2] = nodes
    refined[1::2] = 0.5 * (nodes[:-1] + nodes[1:])
    return refined


def _verification_self_test():
    # Execute the exact copied production limiter in isolation, without FEniCS.
    module = ast.parse(Path(__file__).read_text())
    function = next(n for n in ast.walk(module)
                    if isinstance(n, ast.FunctionDef)
                    and n.name == "bound_and_preserve_mass")
    isolated = ast.Module(body=[function], type_ignores=[])
    ns = {"np": np}
    exec(compile(ast.fix_missing_locations(isolated), __file__, "exec"), ns)
    limiter = ns["bound_and_preserve_mass"]
    tests = [
        (np.array([0.0, 0.2, 1.0]), np.array([0.0, 0.1, 1.0])),
        (np.array([1.03, 0.3, 0.01]), np.array([1.0, 0.3, 0.01])),
        (np.array([-0.01, 0.3, 0.5]), np.array([0.0, 0.3, 0.5])),
        (np.array([1.03, -0.01, 0.4]), np.array([1.0, 0.0, 0.4])),
    ]
    volumes = np.array([1.0, 2.0, 3.0]) * 1.0e-6
    for raw, previous in tests:
        result = limiter(raw, previous, volumes)
        bounded = result[0]
        assert np.all((bounded >= 0.0) & (bounded <= 1.0))
        assert abs(np.dot(volumes, bounded - raw)) < 1.0e-18
        assert abs(result[4] - np.dot(volumes, abs(bounded - raw))) < 1e-18
    r = np.r_[np.linspace(0, .001, 3), np.linspace(.001, .010, 10)[1:],
              np.linspace(.010, .30, 31)[1:]]
    z = np.r_[np.linspace(0, .030, 16), np.linspace(.030, .30, 37)[1:]]
    for factor, expected in [(1, 12546), (2, 50184)]:
        rr = _verification_refine_nodes(r, factor)
        zz = _verification_refine_nodes(z, factor)
        assert 6 * (len(rr)-1) * (len(zz)-1) == expected
        assert np.isclose(rr, .001, atol=1e-15).any()
        assert np.all(np.diff(rr) > 0) and np.all(np.diff(zz) > 0)
    assert np.array_equal(_verification_refine_nodes(r, 1), r)
    for values in CASE_SETTINGS.values():
        assert abs(round(50.0 / values["dt_s"]) * values["dt_s"] - 50.0) < 1e-12
    print("PASS: production limiter bounds/mass, physical mesh refinement, case durations.")


def _verification_json(name, data):
    target = Path(ARGS.output) / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(str(temporary), str(target))


def _verification_write_snapshot(name, field=None):
    if field is None:
        field = fresh_n.vector().get_local()
    raw = globals().get("raw_arr", field)
    np.savez_compressed(
        str(Path(ARGS.output) / name),
        case=np.asarray(ARGS.case), dt_s=np.asarray(dt_value),
        Q_cc_min=np.asarray(Q_target_cc_per_min),
        source_base_sha256=np.asarray(PRODUCTION_SOURCE_SHA256),
        geometry_s_z=mesh.coordinates().copy(), topology=mesh.cells().copy(),
        fresh=np.asarray(field)[_verification_cell_dofs],
        fresh_raw=np.asarray(raw)[_verification_cell_dofs],
        cell_volumes_m3=cell_volumes[_verification_cell_dofs],
        time_s=np.asarray(globals().get("_verification_field_time", 0.0)),
        step=np.asarray(globals().get("_verification_field_step", 0)),
        raw_time_s=np.asarray(globals().get("_verification_raw_time", 0.0)),
        raw_step=np.asarray(globals().get("_verification_raw_step", 0)),
    )


def _verification_close_budget():
    handle = globals().get("_verification_csv_handle")
    if handle is not None and not handle.closed:
        handle.flush()
        handle.close()


PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--case", choices=tuple(CASE_SETTINGS), default="timestep")
PARSER.add_argument("--output", help="New or empty output directory; required for a run.")
PARSER.add_argument("--self-test", action="store_true", help="Test scalar/refinement helpers without FEniCS.")
ARGS = PARSER.parse_args()
if ARGS.self_test:
    _verification_self_test()
    sys.exit(0)
if not ARGS.output:
    PARSER.error("--output is required for a simulation")
ARGS.output = str(Path(ARGS.output).expanduser().resolve())
_output_path = Path(ARGS.output)
if _output_path.exists() and (not _output_path.is_dir() or any(_output_path.iterdir())):
    PARSER.error("Output must be a new or empty directory. Existing data are protected.")
_output_path.mkdir(parents=True, exist_ok=True)
CASE = CASE_SETTINGS[ARGS.case]
_verification_last_row = None
_verification_completed = False
_verification_json("verification_status.json", {
    "status": "starting", "case": ARGS.case, "target_time_s": 50.0,
    "note": "Fresh-start 50 s verification; uncorrected case is diagnostic only.",
})

try:
    from fenics import *
    import numpy as np
    import os
    import json
    import glob
    import time as wallclock



    
    # MPI / BASIC SETTINGS
    

    comm = MPI.comm_world
    rank = MPI.rank(comm)
    size = MPI.size(comm)

    parameters["std_out_all_processes"] = False
    parameters["ghost_mode"] = "shared_facet"
    parameters["linear_algebra_backend"] = "PETSc"

    if size != 1:
        raise RuntimeError(
            "This simplified IPCS baseline is intentionally serial. "
            "Run with exactly one MPI task first."
        )


    
    # USER SETTINGS
    

    # Completely separate from the 20 cc/min calculation.
    OUTPUT_FOLDER = ARGS.output

    dt_value = CASE["dt_s"]

    # Finite 50 s fresh-start verification interval.
    T_final = 50.0
    num_steps = int(round(T_final / dt_value))

    # Save visualisation every 10 s.
    write_interval = 10.0
    write_every = int(round(write_interval / dt_value))

    # Print diagnostics every 1 s.
    diagnostic_interval = 1.0
    diagnostic_every = int(round(diagnostic_interval / dt_value))

    ramp_time = 2.0

    # Restart only from checkpoints made by THIS 70 cc/min case.
    AUTO_RESTART = False

    # Full finite-element checkpoint every 10 s.
    checkpoint_interval = 10.0
    checkpoint_every = int(round(checkpoint_interval / dt_value))
    KEEP_CHECKPOINTS = 2

    CHECKPOINT_FOLDER = os.path.join(
        OUTPUT_FOLDER,
        "checkpoints"
    )

    SEGMENTS_FOLDER = os.path.join(
        OUTPUT_FOLDER,
        "segments"
    )

    
    # PHYSICAL GEOMETRY
    

    R = 0.30
    H = 0.30

    nozzle_diameter = 0.002
    nozzle_radius = nozzle_diameter / 2.0

    S = 0.5 * R**2
    S_nozzle = 0.5 * nozzle_radius**2


    
    # PHYSICAL PROPERTIES
    

    rho_salt = 1005.6
    rho_fresh = 999.3
    rho_ref = rho_salt

    drho = rho_salt - rho_fresh
    beta_value = drho / rho_ref

    beta = Constant(beta_value)
    g = Constant(9.81)
    nu = Constant(1.0e-6)
    dt = Constant(dt_value)


    
    # SOURCE
    

    # High-flow experimental injection rate:
    
    #     70 cc/min
    
    # 1 cc = 1e-6 m^3.
    Q_target_cc_per_min = 70.0

    Q_analytic = (
        Q_target_cc_per_min
        * 1.0e-6
        / 60.0
    )

    # Mean velocity through the physical circular nozzle.
    U_mean = (
        Q_analytic
        / (
            np.pi
            * nozzle_radius**2
        )
    )

    # Circular Poiseuille profile:
    #     Umax = 2 Umean
    Umax = 2.0 * U_mean


    
    # EXPERIMENT-MATCHED BARYCENTRIC MESH
    
    # The old validation mesh has its nozzle edge at r = 5 mm and
    # cannot represent the physical r = 1 mm nozzle edge.
    # Build the correct mesh directly in memory.
    

    def make_barycentric_mesh():

        # Physical radial nodes, then mapped to s = r^2/2.
        
        # 0--1 mm    : 0.5 mm spacing
        # 1--10 mm   : 1.0 mm spacing
        # 10--300 mm : ~9.7 mm spacing

        r_inner = np.linspace(
            0.0,
            nozzle_radius,
            3
        )

        r_near = np.linspace(
            nozzle_radius,
            0.010,
            10
        )

        r_outer = np.linspace(
            0.010,
            R,
            31
        )

        r_nodes = np.concatenate(
            (
                r_inner,
                r_near[1:],
                r_outer[1:]
            )
        )

        # Refine physical r before mapping to s; preserve the 1 mm nozzle edge.
        r_nodes = _verification_refine_nodes(r_nodes, CASE["refinement"])
        s_nodes = 0.5 * r_nodes**2

        # Extra vertical refinement near the inlet:
        
        # bottom 30 mm : 2 mm spacing
        # remainder    : 7.5 mm spacing

        z_near = np.linspace(
            0.0,
            0.030,
            16
        )

        z_outer = np.linspace(
            0.030,
            H,
            37
        )

        z_nodes = np.concatenate(
            (
                z_near,
                z_outer[1:]
            )
        )

        z_nodes = _verification_refine_nodes(z_nodes, CASE["refinement"])
        Ns = len(s_nodes)
        Nz = len(z_nodes)

        base_vertices = []

        for j in range(Nz):
            for i in range(Ns):
                base_vertices.append(
                    (
                        s_nodes[i],
                        z_nodes[j]
                    )
                )

        base_vertices = np.array(
            base_vertices,
            dtype=float
        )

        def vid(i, j):
            return j * Ns + i

        base_triangles = []

        for j in range(Nz - 1):
            for i in range(Ns - 1):

                v00 = vid(i, j)
                v10 = vid(i + 1, j)
                v01 = vid(i, j + 1)
                v11 = vid(i + 1, j + 1)

                if (i + j) % 2 == 0:

                    base_triangles.append(
                        (v00, v10, v11)
                    )

                    base_triangles.append(
                        (v00, v11, v01)
                    )

                else:

                    base_triangles.append(
                        (v00, v10, v01)
                    )

                    base_triangles.append(
                        (v10, v11, v01)
                    )

        # Barycentric refinement
        vertices = list(
            map(tuple, base_vertices)
        )

        triangles = []

        for tri in base_triangles:

            a, b, c = tri

            centre = (
                base_vertices[a]
                + base_vertices[b]
                + base_vertices[c]
            ) / 3.0

            centre_index = len(vertices)

            vertices.append(
                tuple(centre)
            )

            triangles.append(
                (a, b, centre_index)
            )

            triangles.append(
                (b, c, centre_index)
            )

            triangles.append(
                (c, a, centre_index)
            )

        vertices = np.array(
            vertices,
            dtype=float
        )

        triangles = np.array(
            triangles,
            dtype=np.uintp
        )

        mesh_local = Mesh()
        editor = MeshEditor()

        editor.open(
            mesh_local,
            "triangle",
            2,
            2
        )

        editor.init_vertices(
            len(vertices)
        )

        editor.init_cells(
            len(triangles)
        )

        for i, point in enumerate(vertices):

            editor.add_vertex(
                i,
                Point(
                    float(point[0]),
                    float(point[1])
                )
            )

        for i, tri in enumerate(triangles):

            editor.add_cell(
                i,
                tri
            )

        editor.close()

        return (
            mesh_local,
            r_nodes,
            z_nodes
        )


    mesh, r_nodes_used, z_nodes_used = make_barycentric_mesh()

    if rank == 0:

        print("")
        print("==============================================")
        print("EXPERIMENT-MATCHED AXISYMMETRIC IPCS")
        print("==============================================")
        print("MPI tasks =", size)
        print("Physical nozzle diameter =", nozzle_diameter, "m")
        print("Target source flow =", Q_analytic, "m^3/s")
        print("Target source flow =", Q_target_cc_per_min, "cc/min")
        print("Mean inlet velocity =", U_mean, "m/s")
        print("Peak inlet velocity =", Umax, "m/s")
        print("vertices =", mesh.num_vertices())
        print("cells    =", mesh.num_cells())
        print("radial base nodes =", len(r_nodes_used))
        print("vertical base nodes =", len(z_nodes_used))
        print("smallest base dr =", np.min(np.diff(r_nodes_used)), "m")
        print("smallest base dz =", np.min(np.diff(z_nodes_used)), "m")


    
    # COORDINATES
    

    x = SpatialCoordinate(mesh)
    s = x[0]
    z = x[1]

    r2_safe = 2.0 * s + Constant(1.0e-14)
    r_phys = sqrt(r2_safe)


    
    # MEASURES
    

    dx_flow = Measure(
        "dx", domain=mesh,
        metadata={"quadrature_degree": 6}
    )

    dx_scalar = Measure(
        "dx", domain=mesh,
        metadata={"quadrature_degree": 2}
    )


    
    # BOUNDARIES
    # 1 nozzle, 2 return, 3 top, 4 outer wall, 5 axis
    

    boundary_tol = 1.0e-9


    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0, boundary_tol)


    class Nozzle(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and near(x[1], 0.0, boundary_tol)
                and x[0] <= S_nozzle + boundary_tol
            )


    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], H, boundary_tol)


    class OuterWall(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], S, boundary_tol)


    class Axis(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0, boundary_tol)


    boundaries = MeshFunction(
        "size_t", mesh, mesh.topology().dim() - 1, 0
    )

    Bottom().mark(boundaries, 2)
    Top().mark(boundaries, 3)
    OuterWall().mark(boundaries, 4)
    Axis().mark(boundaries, 5)
    Nozzle().mark(boundaries, 1)  # nozzle last

    ds_sub = Measure(
        "ds", domain=mesh, subdomain_data=boundaries,
        metadata={"quadrature_degree": 4}
    )

    dS_int = Measure(
        "dS", domain=mesh,
        metadata={"quadrature_degree": 4}
    )

    if rank == 0:
        print("")
        print("Boundary lengths in transformed coordinates:")
        print("  nozzle     =", assemble(Constant(1.0) * ds_sub(1)))
        print("  return     =", assemble(Constant(1.0) * ds_sub(2)))
        print("  top        =", assemble(Constant(1.0) * ds_sub(3)))
        print("  outer wall =", assemble(Constant(1.0) * ds_sub(4)))
        print("  axis       =", assemble(Constant(1.0) * ds_sub(5)))


    
    # BALANCED INLET / RETURN FLOW
    

    inlet_profile_full = (
        Constant(Umax)
        * (Constant(1.0) - s / Constant(S_nozzle))
    )

    Q_nozzle_mesh = assemble(
        Constant(2.0 * np.pi)
        * inlet_profile_full
        * ds_sub(1)
    )

    xi_return = (
        (s - Constant(S_nozzle))
        / Constant(S - S_nozzle)
    )

    return_shape = (
        Constant(6.0)
        * xi_return
        * (Constant(1.0) - xi_return)
    )

    return_shape_flux = assemble(
        Constant(2.0 * np.pi)
        * return_shape
        * ds_sub(2)
    )

    U_return_scale = Q_nozzle_mesh / return_shape_flux

    if rank == 0:
        print("")
        print("Target nozzle Q      =", Q_analytic)
        print("Discrete nozzle Q    =", Q_nozzle_mesh)
        print("Relative Q error     =", (Q_nozzle_mesh - Q_analytic) / Q_analytic)
        print("Mean inlet velocity  =", U_mean)
        print("Peak inlet velocity  =", Umax)
        print("Return amplitude      =", U_return_scale)


    
    # FUNCTION SPACES
    

    V = VectorFunctionSpace(mesh, "CG", 2)
    P = FunctionSpace(mesh, "CG", 1)
    F = FunctionSpace(mesh, "DG", 0)

    # Diagnostic space only.
    SpeedSpace = FunctionSpace(mesh, "CG", 1)

    U_trial = TrialFunction(V)
    Vtest = TestFunction(V)

    p_trial = TrialFunction(P)
    q = TestFunction(P)

    f_trial = TrialFunction(F)
    psi = TestFunction(F)

    U_n = Function(V)
    U_star = Function(V)
    U_new = Function(V)

    p_n = Function(P)
    dp_new = Function(P)
    p_new = Function(P)

    fresh_n = Function(F)
    fresh_new = Function(F)

    U_n.assign(Constant((0.0, 0.0)))
    U_star.assign(Constant((0.0, 0.0)))
    U_new.assign(Constant((0.0, 0.0)))

    p_n.assign(Constant(0.0))
    dp_new.assign(Constant(0.0))
    p_new.assign(Constant(0.0))

    fresh_n.assign(Constant(0.0))
    fresh_new.assign(Constant(0.0))


    
    # VELOCITY BOUNDARY DATA
    

    U_in = Expression(
        ("0.0", "ramp*Umax*(1.0-x[0]/Snoz)"),
        degree=2,
        ramp=0.0,
        Umax=Umax,
        Snoz=S_nozzle,
    )

    U_ret = Expression(
        (
            "0.0",
            "-ramp*A*6.0*"
            "((x[0]-Snoz)/(Smax-Snoz))*"
            "(1.0-(x[0]-Snoz)/(Smax-Snoz))",
        ),
        degree=2,
        ramp=0.0,
        A=U_return_scale,
        Snoz=S_nozzle,
        Smax=S,
    )

    zero_vec = Constant((0.0, 0.0))
    zero = Constant(0.0)

    bcu = [
        DirichletBC(V, U_in, boundaries, 1),
        DirichletBC(V, U_ret, boundaries, 2),
        DirichletBC(V, zero_vec, boundaries, 3),
        DirichletBC(V, zero_vec, boundaries, 4),
        DirichletBC(V.sub(0), zero, boundaries, 5),
    ]


    
    # PRESSURE-INCREMENT PIN
    

    class PressurePoint(SubDomain):
        def inside(self, x, on_boundary):
            return (
                near(x[0], S, boundary_tol)
                and near(x[1], H, boundary_tol)
            )


    bc_dp = DirichletBC(
        P, Constant(0.0), PressurePoint(), method="pointwise"
    )


    
    
    

    def physical_velocity(A):
        return as_vector((A[0] / r_phys, A[1]))


    def physical_pressure_gradient(phi):
        # dp/dr = r dp/ds
        return as_vector((r_phys * phi.dx(0), phi.dx(1)))


    def epsilon_axi_transformed(A):
        ur = A[0] / r_phys
        uz = A[1]

        e_rr = r_phys * ur.dx(0)
        e_zz = uz.dx(1)
        e_tt = ur / r_phys
        e_rz = Constant(0.5) * (
            ur.dx(1) + r_phys * uz.dx(0)
        )

        return as_tensor((
            (e_rr, e_rz, 0.0),
            (e_rz, e_zz, 0.0),
            (0.0, 0.0, e_tt),
        ))


    
    # BUOYANCY
    

    buoyancy_n = as_vector((
        Constant(0.0),
        g * beta * fresh_n,
    ))


    
    # IPCS STAGE 1: TENTATIVE VELOCITY
    

    u_trial_phys = physical_velocity(U_trial)
    v_test_phys = physical_velocity(Vtest)

    # Semi-implicit Oseen convection.
    
    # The advecting velocity U_n is known, while the advected physical
    # velocity is the tentative velocity.  In the transformed (s,z)
    # coordinates the physical convective derivative is
    
    #     (u_n . grad) u_star
    #       = grad(u_star_phys) * U_n,
    
    # because U_s = r*u_r accounts for ds/dr = r.
    
    # This is more robust at 70 cc/min than treating convection fully
    # explicitly.  The consequence is that the Stage-1 matrix depends
    # on U_n and must be rebuilt/factorised every time step.
    convective_trial = dot(
        grad(u_trial_phys),
        U_n,
    )

    a1 = (
        (Constant(1.0) / dt)
        * inner(u_trial_phys, v_test_phys)
        * dx_flow

        + inner(convective_trial, v_test_phys)
        * dx_flow

        + Constant(2.0) * nu
        * inner(
            epsilon_axi_transformed(U_trial),
            epsilon_axi_transformed(Vtest),
        )
        * dx_flow
    )

    L1 = (
        (Constant(1.0) / dt)
        * inner(physical_velocity(U_n), v_test_phys)
        * dx_flow

        + p_n * div(Vtest) * dx_flow

        + inner(buoyancy_n, v_test_phys) * dx_flow
    )


    
    # IPCS STAGE 2: PRESSURE INCREMENT
    

    # From U_new = U_star - dt*(2s dp/ds, dp/dz)
    a2 = (
        (
            Constant(2.0) * s * p_trial.dx(0) * q.dx(0)
            + p_trial.dx(1) * q.dx(1)
        )
        * dx_flow
    )

    L2 = (
        -(Constant(1.0) / dt)
        * div(U_star)
        * q
        * dx_flow
    )


    
    # IPCS STAGE 3: VELOCITY CORRECTION
    

    a3 = (
        inner(physical_velocity(U_trial), v_test_phys)
        * dx_flow
    )

    L3 = (
        inner(physical_velocity(U_star), v_test_phys)
        * dx_flow

        - dt
        * inner(
            physical_pressure_gradient(dp_new),
            v_test_phys,
        )
        * dx_flow
    )


    
    # DG0 CONSERVATIVE FRESH-WATER TRANSPORT
    

    normal = FacetNormal(mesh)
    fresh_in = Constant(1.0)

    Un = dot(U_new, normal)
    Un_plus = dot(U_new("+"), normal("+"))

    f_upwind = conditional(
        gt(Un_plus, Constant(0.0)),
        f_trial("+"),
        f_trial("-"),
    )


    def positive_part(a):
        return (a + abs(a)) / Constant(2.0)


    def negative_part(a):
        return (a - abs(a)) / Constant(2.0)


    # Conservative DG0 upwind transport:
    
    #     df/dt + div(U f) = 0
    
    # This form conserves the integrated fresh-water volume through
    # the boundary fluxes.  Because the IPCS velocity is not exactly
    # pointwise divergence-free, the raw DG0 solution can have small
    # excursions outside [0,1].  Those cell averages are corrected
    # AFTER the conservative solve by a mass-preserving bounds
    # correction below.  We do NOT add -f*div(U), because that was
    # the source of the long-time fresh-water mass creation.
    a_fresh = (
        (Constant(1.0) / dt)
        * f_trial * psi * dx_scalar

        - f_trial
        * dot(U_new, grad(psi))
        * dx_scalar

        + Un_plus
        * f_upwind
        * (psi("+") - psi("-"))
        * dS_int

        + positive_part(Un)
        * f_trial
        * psi
        * ds_sub(2)
    )

    L_fresh = (
        (Constant(1.0) / dt)
        * fresh_n * psi * dx_scalar

        - negative_part(Un)
        * fresh_in
        * psi
        * ds_sub(1)
    )


    
    # DG0 PHYSICAL CELL VOLUMES
    
    # In transformed coordinates dV = 2*pi*ds*dz.
    # For DG0 there is one degree of freedom per cell, so assembling
    # the test basis gives the physical volume associated with each
    # scalar DOF.
    

    cell_volume_vector = assemble(
        Constant(2.0 * np.pi)
        * psi
        * dx_scalar
    )

    cell_volumes = cell_volume_vector.get_local()

    total_tank_volume_discrete = float(
        np.sum(cell_volumes)
    )


    
    # BOUNDED, MASS-PRESERVING DG0 CELL-AVERAGE CORRECTION
    
    # 1. Clip raw DG0 cell averages to [0,1].
    # 2. Compute the mass changed by clipping.
    # 3. Put exactly that mass back into / remove it from cells that
    #    already belong to the tracer support and still have capacity.
    
    # This preserves the TOTAL scalar mass of the conservative solve.
    # The L1 correction is reported so we can verify that the limiter
    # is only making a small boundedness correction rather than
    # materially changing the solution.
    

    def bound_and_preserve_mass(
        raw_values,
        previous_values,
        volumes,
        lower=0.0,
        upper=1.0,
        support_tol=1.0e-14,
        mass_tol=1.0e-18,
    ):

        raw = np.array(
            raw_values,
            dtype=float,
            copy=True
        )

        previous = np.array(
            previous_values,
            dtype=float,
            copy=False
        )

        volumes = np.array(
            volumes,
            dtype=float,
            copy=False
        )

        raw_min = float(np.min(raw))
        raw_max = float(np.max(raw))

        target_mass = float(
            np.dot(
                volumes,
                raw
            )
        )

        total_volume = float(
            np.sum(volumes)
        )

        if (
            target_mass < -mass_tol
            or target_mass > total_volume + mass_tol
        ):
            previous_mass = float(
                np.dot(
                    volumes,
                    previous
                )
            )

            raise RuntimeError(
                "Conservative scalar solve produced an impossible "
                "total fresh-water mass. "
                "target_mass={:.16e}, previous_mass={:.16e}, "
                "tank_volume={:.16e}, raw_min={:.16e}, raw_max={:.16e}"
                .format(
                    target_mass,
                    previous_mass,
                    total_volume,
                    raw_min,
                    raw_max,
                )
            )

        limited = np.clip(
            raw,
            lower,
            upper
        )

        clipped_mass = float(
            np.dot(
                volumes,
                limited
            )
        )

        # Positive delta: clipping removed mass (usually f > 1),
        # so mass must be ADDED back.
        
        # Negative delta: clipping added mass (usually f < 0),
        # so mass must be REMOVED.
        delta_mass = (
            target_mass
            -
            clipped_mass
        )

        # Keep redistribution inside the existing tracer support
        # whenever possible, rather than adding tiny fresh values
        # throughout untouched salt water.
        support = (
            (raw > support_tol)
            |
            (previous > support_tol)
        )

        if delta_mass > mass_tol:

            capacity_fraction = (
                upper
                -
                limited
            )

            mask = (
                support
                &
                (
                    capacity_fraction
                    >
                    support_tol
                )
            )

            capacity_mass = float(
                np.dot(
                    volumes[mask],
                    capacity_fraction[mask]
                )
            )

            # Fallback only if the existing tracer support does not
            # contain enough capacity.
            if capacity_mass < delta_mass - mass_tol:

                mask = (
                    capacity_fraction
                    >
                    support_tol
                )

                capacity_mass = float(
                    np.dot(
                        volumes[mask],
                        capacity_fraction[mask]
                    )
                )

            if capacity_mass < delta_mass - mass_tol:

                raise RuntimeError(
                    "Bound limiter cannot restore clipped fresh mass "
                    "without exceeding fresh=1."
                )

            alpha = (
                delta_mass
                /
                capacity_mass
            )

            limited[mask] += (
                alpha
                *
                capacity_fraction[mask]
            )

        elif delta_mass < -mass_tol:

            mass_to_remove = (
                -delta_mass
            )

            removable_fraction = (
                limited
                -
                lower
            )

            mask = (
                support
                &
                (
                    removable_fraction
                    >
                    support_tol
                )
            )

            removable_mass = float(
                np.dot(
                    volumes[mask],
                    removable_fraction[mask]
                )
            )

            if removable_mass < mass_to_remove - mass_tol:

                mask = (
                    removable_fraction
                    >
                    support_tol
                )

                removable_mass = float(
                    np.dot(
                        volumes[mask],
                        removable_fraction[mask]
                    )
                )

            if removable_mass < mass_to_remove - mass_tol:

                raise RuntimeError(
                    "Bound limiter cannot remove clipped fresh mass "
                    "without going below fresh=0."
                )

            alpha = (
                mass_to_remove
                /
                removable_mass
            )

            limited[mask] -= (
                alpha
                *
                removable_fraction[mask]
            )

        # Final round-off cleanup.
        limited = np.clip(
            limited,
            lower,
            upper
        )

        limited_mass = float(
            np.dot(
                volumes,
                limited
            )
        )

        mass_error = (
            limited_mass
            -
            target_mass
        )

        if abs(mass_error) > 1.0e-13 * max(
            total_volume,
            abs(target_mass),
            1.0
        ):
            raise RuntimeError(
                "Bound limiter failed to preserve scalar mass."
            )

        correction_L1_volume = float(
            np.dot(
                volumes,
                np.abs(
                    limited
                    -
                    raw
                )
            )
        )

        return (
            limited,
            raw_min,
            raw_max,
            target_mass,
            correction_L1_volume,
            mass_error,
        )


    
    # DIRECT SOLVERS + PREASSEMBLY
    
    # Stage 1 is now an Oseen problem and its matrix depends on the
    # previous-step velocity U_n.  It is therefore assembled and
    # factorised inside the time loop.
    
    # The pressure-increment and velocity-correction matrices remain
    # constant and are factorised once.
    
    # The DG0 scalar matrix also changes because its upwind flux
    # depends on the current corrected velocity U_new.
    

    solver_Ustar = LUSolver()
    solver_fresh = LUSolver()

    if rank == 0:
        print("")
        print("Preassembling constant IPCS Stage-2/Stage-3 matrices...")

    # Pressure-increment matrix: constant
    A2_const = assemble(a2)
    bc_dp.apply(A2_const)

    solver_dp = LUSolver()
    solver_dp.set_operator(A2_const)

    # Velocity-correction mass matrix: constant
    A3_const = assemble(a3)

    for bc in bcu:
        bc.apply(A3_const)

    solver_Ucorr = LUSolver()
    solver_Ucorr.set_operator(A3_const)

    if rank == 0:
        print("Constant Stage-2/Stage-3 matrices ready.")
        print("Stage-1 Oseen matrix will be rebuilt every time step.")


    
    # RESTART / CHECKPOINT SUPPORT
    
    # The ordinary XDMF files below are kept for ParaView output.
    # Restart state is stored separately using HDF5File so that the
    # actual CG2 velocity, CG1 pressure and DG0 scalar Functions are
    # written and read directly.
    
    # Each checkpoint is first written to temporary files and only
    # renamed after the HDF5 file is closed.  The previous checkpoint
    # is therefore left intact if a job is killed during a write.
    

    if rank == 0:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
        os.makedirs(SEGMENTS_FOLDER, exist_ok=True)

    MPI.barrier(comm)


    def checkpoint_base(step):
        return os.path.join(
            CHECKPOINT_FOLDER,
            "checkpoint_step_{:09d}".format(int(step))
        )


    def checkpoint_metadata_files():
        return sorted(
            glob.glob(
                os.path.join(
                    CHECKPOINT_FOLDER,
                    "checkpoint_step_*.json"
                )
            ),
            reverse=True,
        )


    def save_checkpoint(
        step,
        time_value,
        cumulative_in,
        cumulative_out,
    ):
        """
        Save the state needed to start the NEXT time step exactly:
            U_n, p_n, fresh_n,
            global step and physical simulation time,
            cumulative scalar inlet/outlet accounting.
        """

        base = checkpoint_base(step)

        h5_final = base + ".h5"
        json_final = base + ".json"

        h5_tmp = base + ".tmp.h5"
        json_tmp = base + ".tmp.json"

        # Remove stale temporary files from an interrupted old write.
        if rank == 0:
            for path in (h5_tmp, json_tmp):
                if os.path.exists(path):
                    os.remove(path)

        MPI.barrier(comm)

        h5 = HDF5File(
            mesh.mpi_comm(),
            h5_tmp,
            "w"
        )

        h5.write(U_n, "/U_n")
        h5.write(p_n, "/p_n")
        h5.write(fresh_n, "/fresh_n")
        h5.close()

        MPI.barrier(comm)

        metadata = {
            "verification_case": ARGS.case,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "step": int(step),
            "simulation_time": float(time_value),
            "cumulative_fresh_in": float(cumulative_in),
            "cumulative_fresh_out": float(cumulative_out),
            "dt_value": float(dt_value),
            "R": float(R),
            "H": float(H),
            "nozzle_diameter": float(nozzle_diameter),
            "Q_analytic": float(Q_analytic),
            "Q_target_cc_per_min": float(Q_target_cc_per_min),
            "mesh_vertices": int(mesh.num_vertices()),
            "mesh_cells": int(mesh.num_cells()),
        }

        if rank == 0:
            with open(json_tmp, "w") as f:
                json.dump(
                    metadata,
                    f,
                    indent=2,
                    sort_keys=True,
                )

            # Atomic rename on the same filesystem after complete writes.
            os.replace(h5_tmp, h5_final)
            os.replace(json_tmp, json_final)

            print("")
            print(
                "CHECKPOINT SAVED: step =",
                step,
                " t =",
                time_value,
                "s"
            )
            print("  ", h5_final)

            # Keep only the newest complete checkpoint pairs.
            complete = []

            for meta_file in checkpoint_metadata_files():
                h5_file = meta_file[:-5] + ".h5"

                if os.path.exists(h5_file):
                    complete.append(
                        (meta_file, h5_file)
                    )

            for meta_file, h5_file in complete[KEEP_CHECKPOINTS:]:
                try:
                    os.remove(meta_file)
                except OSError:
                    pass

                try:
                    os.remove(h5_file)
                except OSError:
                    pass

        MPI.barrier(comm)


    def find_latest_complete_checkpoint():

        if not AUTO_RESTART:
            return None

        for meta_file in checkpoint_metadata_files():

            h5_file = meta_file[:-5] + ".h5"

            if not os.path.exists(h5_file):
                continue

            try:
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
            except Exception:
                continue

            return (
                h5_file,
                meta_file,
                metadata,
            )

        return None


    def validate_restart_metadata(metadata):

        checks = [
            (
                abs(float(metadata["dt_value"]) - dt_value)
                <= 1.0e-15,
                "dt_value"
            ),
            (
                abs(float(metadata["R"]) - R)
                <= 1.0e-15,
                "R"
            ),
            (
                abs(float(metadata["H"]) - H)
                <= 1.0e-15,
                "H"
            ),
            (
                abs(
                    float(metadata["nozzle_diameter"])
                    - nozzle_diameter
                )
                <= 1.0e-15,
                "nozzle_diameter"
            ),
            (
                abs(
                    float(metadata["Q_target_cc_per_min"])
                    - Q_target_cc_per_min
                )
                <= 1.0e-12,
                "Q_target_cc_per_min"
            ),
            (
                abs(float(metadata["Q_analytic"]) - Q_analytic)
                <= 1.0e-15,
                "Q_analytic"
            ),
            (
                int(metadata["mesh_vertices"])
                == mesh.num_vertices(),
                "mesh vertex count"
            ),
            (
                int(metadata["mesh_cells"])
                == mesh.num_cells(),
                "mesh cell count"
            ),
        ]

        failed = [
            name
            for ok, name in checks
            if not ok
        ]

        if failed:
            raise RuntimeError(
                "Checkpoint is incompatible with this code. "
                "Failed checks: "
                + ", ".join(failed)
            )


    
    # STATE INITIALISATION
    
    # Priority:
    #   1. Restart from the newest complete checkpoint in THIS
    #      70 cc/min output folder, if one exists.
    #   2. Otherwise start from quiescent salt water at t = 0.
    
    # No state from the old 20 cc/min calculation is read.
    

    restart_step = 0
    simulation_time = 0.0
    cumulative_fresh_in = 0.0
    cumulative_fresh_out = 0.0

    latest_checkpoint = find_latest_complete_checkpoint()

    if latest_checkpoint is not None:

        (
            restart_h5,
            restart_json,
            restart_metadata,
        ) = latest_checkpoint

        validate_restart_metadata(
            restart_metadata
        )

        h5 = HDF5File(
            mesh.mpi_comm(),
            restart_h5,
            "r"
        )

        h5.read(U_n, "/U_n")
        h5.read(p_n, "/p_n")
        h5.read(fresh_n, "/fresh_n")
        h5.close()

        restart_step = int(
            restart_metadata["step"]
        )

        simulation_time = float(
            restart_metadata["simulation_time"]
        )

        cumulative_fresh_in = float(
            restart_metadata["cumulative_fresh_in"]
        )

        cumulative_fresh_out = float(
            restart_metadata["cumulative_fresh_out"]
        )

        if restart_step >= num_steps:
            raise RuntimeError(
                "Latest checkpoint is already at or beyond "
                "the requested final step."
            )

        U_new.assign(U_n)
        U_star.assign(U_n)

        p_new.assign(p_n)
        dp_new.assign(Constant(0.0))

        fresh_new.assign(fresh_n)

        if rank == 0:
            print("")
            print("==============================================")
            print("RESTARTING 70 CC/MIN CASE FROM CHECKPOINT")
            print("==============================================")
            print("checkpoint =", restart_h5)
            print("step       =", restart_step)
            print("time       =", simulation_time, "s")
            print(
                "fresh in cumulative  =",
                cumulative_fresh_in
            )
            print(
                "fresh out cumulative =",
                cumulative_fresh_out
            )

    else:

        # Fresh initial state: quiescent salt water.
        restart_step = 0
        simulation_time = 0.0
        cumulative_fresh_in = 0.0
        cumulative_fresh_out = 0.0

        U_n.assign(Constant((0.0, 0.0)))
        U_star.assign(Constant((0.0, 0.0)))
        U_new.assign(Constant((0.0, 0.0)))

        p_n.assign(Constant(0.0))
        dp_new.assign(Constant(0.0))
        p_new.assign(Constant(0.0))

        fresh_n.assign(Constant(0.0))
        fresh_new.assign(Constant(0.0))

        if rank == 0:
            print("")
            print("==============================================")
            print("STARTING NEW 70 CC/MIN AXISYMMETRIC CASE")
            print("==============================================")
            print("Q target   =", Q_target_cc_per_min, "cc/min")
            print("Q physical =", Q_analytic, "m^3/s")
            print("U_mean     =", U_mean, "m/s")
            print("U_max      =", Umax, "m/s")
            print("dt         =", dt_value, "s")
            print("T_final    =", T_final, "s")
            print("num_steps  =", num_steps)
            print("Verification case:", ARGS.case, "(50 s only)")

    # Each scheduler job writes visualisation output into a new
    # segment directory, so restarting never overwrites an earlier
    # XDMF time series.
    job_id = os.environ.get(
        "SLURM_JOB_ID",
        wallclock.strftime("%Y%m%d_%H%M%S")
    )

    segment_name = (
        "segment_from_step_{:09d}_t_{:.3f}s_job_{}"
        .format(
            restart_step,
            simulation_time,
            job_id,
        )
        .replace(".", "p")
    )

    SEGMENT_OUTPUT_FOLDER = os.path.join(
        SEGMENTS_FOLDER,
        segment_name
    )

    if rank == 0:
        os.makedirs(
            SEGMENT_OUTPUT_FOLDER,
            exist_ok=True
        )

    MPI.barrier(comm)


    
    
    

    if rank == 0:
        os.makedirs(SEGMENT_OUTPUT_FOLDER, exist_ok=True)

    MPI.barrier(comm)

    xdmf_U = XDMFFile(mesh.mpi_comm(), f"{SEGMENT_OUTPUT_FOLDER}/U_transformed.xdmf")
    xdmf_u = XDMFFile(mesh.mpi_comm(), f"{SEGMENT_OUTPUT_FOLDER}/u_physical.xdmf")
    xdmf_p = XDMFFile(mesh.mpi_comm(), f"{SEGMENT_OUTPUT_FOLDER}/p.xdmf")
    xdmf_fresh = XDMFFile(mesh.mpi_comm(), f"{SEGMENT_OUTPUT_FOLDER}/fresh.xdmf")
    xdmf_c = XDMFFile(mesh.mpi_comm(), f"{SEGMENT_OUTPUT_FOLDER}/c.xdmf")

    for output_file in (xdmf_U, xdmf_u, xdmf_p, xdmf_fresh, xdmf_c):
        output_file.parameters["flush_output"] = True
        output_file.parameters["functions_share_mesh"] = True
        output_file.parameters["rewrite_function_mesh"] = False


    def write_output(time_value):
        U_new.rename("U_transformed", "")
        p_new.rename("p", "")
        fresh_n.rename("fresh", "")

        u_out = project(
            physical_velocity(U_new),
            V,
            solver_type="lu",
        )
        u_out.rename("u", "")

        c_out = Function(F)
        c_values = 1.0 - fresh_n.vector().get_local()
        c_out.vector().set_local(c_values)
        c_out.vector().apply("insert")
        c_out.rename("c", "")

        xdmf_U.write(U_new, time_value)
        xdmf_u.write(u_out, time_value)
        xdmf_p.write(p_new, time_value)
        xdmf_fresh.write(fresh_n, time_value)
        xdmf_c.write(c_out, time_value)


    
    
    

    def physical_volume_flux(marker):
        return assemble(
            Constant(2.0 * np.pi)
            * dot(U_new, normal)
            * ds_sub(marker)
        )


    def diagnostics(step, time_value, wall_elapsed):
        div_star = np.sqrt(
            assemble(div(U_star)**2 * dx_flow)
        )

        div_new = np.sqrt(
            assemble(div(U_new)**2 * dx_flow)
        )

        # IPCS with CG2 velocity / CG1 pressure is not pointwise
        # divergence-free.  The pressure equation acts in the CG1
        # pressure space, so also monitor the CG1 projection of
        # div(U_new).  This is a DIAGNOSTIC ONLY and does not alter
        # the numerical solution.
        div_pressure_space = project(
            div(U_new),
            P,
            solver_type="lu",
        )

        div_pressure_L2 = np.sqrt(
            assemble(div_pressure_space**2 * dx_flow)
        )

        # Global transformed divergence.  By the divergence theorem
        # this corresponds to the net boundary flux (without 2*pi).
        div_global = assemble(
            div(U_new) * dx_flow
        )

        q_nozzle = physical_volume_flux(1)
        q_return = physical_volume_flux(2)
        q_net = q_nozzle + q_return

        fmin = fresh_n.vector().min()
        fmax = fresh_n.vector().max()

        # Retain transformed norm as a numerical diagnostic only.
        # It is not a physical velocity magnitude.
        Uinf = U_new.vector().norm("linf")

        # Genuine physical speed:
        #     |u| = sqrt(u_r^2 + u_z^2)
        u_phys_expr = physical_velocity(U_new)

        speed_phys = project(
            sqrt(
                inner(
                    u_phys_expr,
                    u_phys_expr
                )
            ),
            SpeedSpace,
            solver_type="lu",
        )

        max_physical_speed = speed_phys.vector().max()

        fresh_volume = assemble(
            Constant(2.0 * np.pi)
            * fresh_n
            * dx_scalar
        )

        expected_fresh_volume = (
            cumulative_fresh_in
            -
            cumulative_fresh_out
        )

        fresh_mass_error = (
            fresh_volume
            -
            expected_fresh_volume
        )

        if abs(expected_fresh_volume) > 1.0e-20:

            relative_fresh_mass_error = (
                fresh_mass_error
                /
                expected_fresh_volume
            )

            limiter_relative_L1 = (
                last_limiter_L1
                /
                expected_fresh_volume
            )

        else:

            relative_fresh_mass_error = 0.0
            limiter_relative_L1 = 0.0

        if rank == 0:
            print("")
            print("step =", step, " t =", time_value)
            print("wall elapsed =", wall_elapsed, "s")
            print("||U_transformed||_inf =", Uinf, "(NOT a physical speed)")
            print("max physical |u| =", max_physical_speed, "m/s")
            print("||div(U_star)||_L2 =", div_star)
            print("||div(U_new)||_L2  =", div_new)
            print("||Proj_P div(U_new)||_L2 =", div_pressure_L2)
            print("integral div(U_new) =", div_global)
            print("Q nozzle =", q_nozzle)
            print("Q return =", q_return)
            print("Q net    =", q_net)
            print("raw min(fresh) =", last_raw_fmin)
            print("raw max(fresh) =", last_raw_fmax)
            print("limited min(fresh) =", fmin)
            print("limited max(fresh) =", fmax)
            print("fresh in cumulative =", cumulative_fresh_in)
            print("fresh out cumulative =", cumulative_fresh_out)
            print("fresh expected =", expected_fresh_volume)
            print("fresh stored   =", fresh_volume)
            print("fresh mass error =", fresh_mass_error)
            print("relative fresh mass error =", relative_fresh_mass_error)
            print("limiter L1 correction volume =", last_limiter_L1)
            print("limiter correction / expected =", limiter_relative_L1)
            print("limiter mass error =", last_limiter_mass_error)

        return (
            div_new,
            fmin,
            fmax,
            Uinf,
            max_physical_speed,
            relative_fresh_mass_error,
            limiter_relative_L1,
        )


    
    # INITIAL OUTPUT
    

    write_output(simulation_time)


    
    
    

    wall_start = wallclock.time()

    # Cumulative wall-clock timings.  These let us see exactly
    # which stage is expensive before making any further solver change.
    time_stage1 = 0.0
    time_stage2 = 0.0
    time_stage3 = 0.0
    time_scalar = 0.0
    time_output = 0.0

    # Scalar mass accounting.
    # cumulative_fresh_in/out were either loaded from the checkpoint
    # or initialised to zero above.
    last_raw_fmin = fresh_n.vector().min()
    last_raw_fmax = fresh_n.vector().max()
    last_limiter_L1 = 0.0
    last_limiter_mass_error = 0.0

    # Verification output uses an explicit DOF-to-cell map: DG0 vector order
    # need not be the mesh-cell order used by topology and geometry.
    _verification_cell_dofs = np.array([
        F.dofmap().cell_dofs(cell_index)[0]
        for cell_index in range(mesh.num_cells())
    ], dtype=np.int64)
    if len(np.unique(_verification_cell_dofs)) != mesh.num_cells():
        raise RuntimeError("DG0 cell mapping is not one-to-one.")
    _verification_json("verification_configuration.json", {
        "case": ARGS.case, "target_time_s": T_final, "dt_s": dt_value,
        "num_steps": num_steps, "mesh_refinement": CASE["refinement"],
        "mesh_cells": mesh.num_cells(), "mesh_vertices": mesh.num_vertices(),
        "radial_nodes_m": r_nodes_used.tolist(), "vertical_nodes_m": z_nodes_used.tolist(),
        "R_m": R, "H_m": H, "nozzle_diameter_m": nozzle_diameter,
        "Q_cc_per_min": Q_target_cc_per_min, "Q_cc_min": Q_target_cc_per_min, "Q_m3_s": Q_analytic,
        "rho_salt_kg_m3": rho_salt, "rho_fresh_kg_m3": rho_fresh,
        "nu_m2_s": float(nu), "g_m_s2": float(g), "ramp_time_s": ramp_time,
        "scalar_diffusion": "No molecular diffusion operator; production DG0 upwind advection.",
        "momentum": "Production semi-implicit Oseen IPCS; Stage-1 matrix rebuilt each step.",
        "scalar_correction": "Original mass-preserving redistribution" if CASE["redistribute"] else "NONE: diagnostic control",
        "scalar_guard_lower": -1e-10 if CASE["redistribute"] else -0.05,
        "scalar_guard_upper": 1.0 + 1e-10 if CASE["redistribute"] else 1.25,
        "fresh_start": True, "auto_restart": False, "mpi_ranks": size,
        "production_source_sha256": PRODUCTION_SOURCE_SHA256,
        "source_base_sha256": PRODUCTION_SOURCE_SHA256,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "cumulative_limiter_definition": "sum over steps of integral(abs(f_limited-f_raw)) dV; no dt multiplier; not final solution error",
        "npz_cell_order": "mesh cell order via DG0 dofmap; dV=2*pi*ds*dz",
        "claim_scope": "Matched-time early transient verification only, not 1000 s convergence.",
    })
    _verification_columns = [
        "step", "time_s", "dt_s", "fresh_in_rate_m3_s", "fresh_out_rate_raw_m3_s",
        "fresh_in_cumulative_m3", "fresh_out_cumulative_m3", "expected_fresh_m3",
        "fresh_raw_m3", "fresh_stored_m3", "budget_residual_m3",
        "raw_fresh_min", "raw_fresh_max", "stored_fresh_min", "stored_fresh_max",
        "limiter_L1_step_m3", "limiter_L1_cumulative_m3", "limiter_L1_step_max_m3",
        "limiter_signed_mass_error_step_m3", "limiter_abs_mass_error_cumulative_m3",
        "wall_elapsed_s",
    ]
    _verification_csv_handle = open(str(Path(ARGS.output) / "verification_budget.csv"), "w", newline="")
    _verification_csv = csv.DictWriter(_verification_csv_handle, fieldnames=_verification_columns)
    _verification_csv.writeheader()
    _verification_field_step = 0
    _verification_field_time = 0.0
    _verification_raw_step = 0
    _verification_raw_time = 0.0
    _verification_cumulative_L1 = 0.0
    _verification_max_L1 = 0.0
    _verification_cumulative_abs_mass_error = 0.0
    _verification_write_snapshot("verification_step_000000000.npz")
    _verification_json("verification_status.json", {
        "status": "running", "case": ARGS.case, "last_recorded_step": 0,
        "last_recorded_time_s": 0.0, "target_time_s": T_final,
    })

    for n_step in range(restart_step + 1, num_steps + 1):
        simulation_time += dt_value

        if simulation_time < ramp_time:
            ramp_value = 0.5 * (
                1.0 - np.cos(np.pi * simulation_time / ramp_time)
            )
        else:
            ramp_value = 1.0

        U_in.ramp = ramp_value
        U_ret.ramp = ramp_value

        
        # Stage 1: tentative velocity
        
        # Semi-implicit Oseen convection means A1 depends on U_n.
        # Reassemble and refactorise this matrix every time step.
        # Apply the current ramped Dirichlet data to matrix and RHS
        # together.
        
        stage_start = wallclock.time()

        A1 = assemble(a1)
        b1 = assemble(L1)

        for bc in bcu:
            bc.apply(A1, b1)

        solver_Ustar.solve(
            A1,
            U_star.vector(),
            b1,
        )

        time_stage1 += wallclock.time() - stage_start

        
        # Stage 2: pressure increment
        
        # Reuse the preassembled/factorised constant matrix.
        
        stage_start = wallclock.time()

        b2 = assemble(L2)
        bc_dp.apply(b2)

        solver_dp.solve(dp_new.vector(), b2)

        p_new.vector().zero()
        p_new.vector().axpy(1.0, p_n.vector())
        p_new.vector().axpy(1.0, dp_new.vector())
        p_new.vector().apply("insert")

        time_stage2 += wallclock.time() - stage_start

        
        # Stage 3: velocity correction
        
        # Reuse the constant mass matrix.  Apply the CURRENT
        # time-dependent inlet/return values to the RHS only.
        
        stage_start = wallclock.time()

        b3 = assemble(L3)

        for bc in bcu:
            bc.apply(b3)

        solver_Ucorr.solve(U_new.vector(), b3)

        time_stage3 += wallclock.time() - stage_start

        
        # Fail early if the FLOW solution has become non-finite.
        
        # Previously the first visible failure was often the scalar
        # solve returning inf.  This check identifies whether the
        # velocity/pressure system is actually responsible.
        
        Ustar_local = U_star.vector().get_local()
        Unew_local = U_new.vector().get_local()
        pnew_local = p_new.vector().get_local()

        if (
            not np.all(np.isfinite(Ustar_local))
            or not np.all(np.isfinite(Unew_local))
            or not np.all(np.isfinite(pnew_local))
        ):
            raise RuntimeError(
                "FLOW SOLVER PRODUCED NON-FINITE VALUES at "
                "step={} t={:.16e} s. "
                "Ustar_finite={}, Unew_finite={}, pnew_finite={}"
                .format(
                    n_step,
                    simulation_time,
                    bool(np.all(np.isfinite(Ustar_local))),
                    bool(np.all(np.isfinite(Unew_local))),
                    bool(np.all(np.isfinite(pnew_local))),
                )
            )

        
        # Fresh-water transport
        
        # This matrix genuinely changes with U_new because the
        # upwind direction and flux change with the velocity.
        
        stage_start = wallclock.time()

        Af = assemble(a_fresh)
        bf = assemble(L_fresh)

        solver_fresh.solve(Af, fresh_new.vector(), bf)

        time_scalar += wallclock.time() - stage_start

        
        # Scalar boundary mass balance BEFORE limiting
        
        # The conservative DG solve uses fresh_new on return outflow,
        # so compute the discrete outflow rate from the RAW solution.
        

        fresh_rate_in = assemble(
            Constant(2.0 * np.pi)
            *
            (
                -negative_part(Un)
            )
            *
            fresh_in
            *
            ds_sub(1)
        )

        fresh_rate_out_raw = assemble(
            Constant(2.0 * np.pi)
            *
            positive_part(Un)
            *
            fresh_new
            *
            ds_sub(2)
        )

        cumulative_fresh_in += (
            fresh_rate_in
            *
            dt_value
        )

        cumulative_fresh_out += (
            fresh_rate_out_raw
            *
            dt_value
        )

        
        # Bound correction that preserves the conservative raw mass.
        

        raw_arr = fresh_new.vector().get_local()
        _verification_raw_step = n_step
        _verification_raw_time = simulation_time
        previous_arr = fresh_n.vector().get_local()

        if not np.all(np.isfinite(raw_arr)):
            raise RuntimeError("Non-finite raw scalar at step {}".format(n_step))

        if CASE["redistribute"]:
            (
                limited_arr,
                last_raw_fmin,
                last_raw_fmax,
                raw_fresh_mass,
                last_limiter_L1,
                last_limiter_mass_error,
            ) = bound_and_preserve_mass(
                raw_arr,
                previous_arr,
                cell_volumes,
            )

        else:
            # Diagnostic control: neither clip nor redistribute the raw DG0 scalar.
            limited_arr = raw_arr.copy()
            last_raw_fmin = float(np.min(raw_arr))
            last_raw_fmax = float(np.max(raw_arr))
            raw_fresh_mass = float(np.dot(cell_volumes, raw_arr))
            last_limiter_L1 = 0.0
            last_limiter_mass_error = 0.0

        fresh_new.vector().set_local(
            limited_arr
        )

        fresh_new.vector().apply(
            "insert"
        )

        
        # Update
        
        U_n.assign(U_new)
        p_n.assign(p_new)
        fresh_n.assign(fresh_new)
        _verification_field_step = n_step
        _verification_field_time = simulation_time

        # Record the candidate step before guards, so a failed control leaves
        # its exact raw/accepted scalar and budget available for inspection.
        _verification_cumulative_L1 += last_limiter_L1
        _verification_max_L1 = max(_verification_max_L1, last_limiter_L1)
        _verification_cumulative_abs_mass_error += abs(last_limiter_mass_error)
        _stored_values = fresh_n.vector().get_local()
        _stored_volume = float(np.dot(cell_volumes, _stored_values))
        _expected_volume = cumulative_fresh_in - cumulative_fresh_out
        _verification_last_row = dict(zip(_verification_columns, [
            n_step, simulation_time, dt_value, float(fresh_rate_in), float(fresh_rate_out_raw),
            cumulative_fresh_in, cumulative_fresh_out, _expected_volume,
            raw_fresh_mass, _stored_volume, _stored_volume - _expected_volume,
            last_raw_fmin, last_raw_fmax, float(np.min(_stored_values)), float(np.max(_stored_values)),
            last_limiter_L1, _verification_cumulative_L1, _verification_max_L1,
            last_limiter_mass_error, _verification_cumulative_abs_mass_error,
            wallclock.time() - wall_start,
        ]))
        _verification_csv.writerow(_verification_last_row)
        if n_step % diagnostic_every == 0:
            _verification_csv_handle.flush()
        _lower = -1e-10 if CASE["redistribute"] else -0.05
        _upper = 1.0 + 1e-10 if CASE["redistribute"] else 1.25
        if not np.all(np.isfinite(_stored_values)) or np.min(_stored_values) < _lower or np.max(_stored_values) > _upper:
            raise RuntimeError("Scalar guard failed at step {}: fresh min={}, max={}, allowed [{}, {}]".format(
                n_step, np.min(_stored_values), np.max(_stored_values), _lower, _upper))
        if n_step % write_every == 0:
            _verification_write_snapshot("verification_step_{:09d}.npz".format(n_step))
            _verification_json("verification_status.json", {
                "status": "running", "case": ARGS.case,
                "last_recorded_step": n_step, "last_recorded_time_s": simulation_time,
                "target_time_s": T_final, "last_budget_row": _verification_last_row,
            })

        
        
        
        if n_step <= 5 or n_step % diagnostic_every == 0:
            wall_elapsed = wallclock.time() - wall_start

            (
                div_new,
                fmin,
                fmax,
                Uinf,
                max_physical_speed,
                relative_fresh_mass_error,
                limiter_relative_L1,
            ) = diagnostics(
                n_step,
                simulation_time,
                wall_elapsed,
            )

            if rank == 0:
                print("cumulative stage timings:")
                print("  tentative velocity =", time_stage1, "s")
                print("  pressure increment =", time_stage2, "s")
                print("  velocity correction =", time_stage3, "s")
                print("  scalar transport =", time_scalar, "s")
                print("  output =", time_output, "s")

            if (
                not np.isfinite(Uinf)
                or not np.isfinite(max_physical_speed)
            ):
                raise RuntimeError("Velocity became NaN/Inf.")

            if CASE["redistribute"] and (fmin < -1.0e-10 or fmax > 1.0 + 1.0e-10):
                raise RuntimeError(
                    "Limited fresh-water fraction left [0,1]."
                )

            if abs(relative_fresh_mass_error) > 1.0e-8:
                raise RuntimeError(
                    "Fresh-water mass balance failed."
                )

            # A large limiter correction means the underlying
            # conservative scalar solution is not sufficiently well
            # behaved, even if the final values are bounded.
            if (
                simulation_time > 0.1
                and limiter_relative_L1 > 5.0e-2
            ):
                raise RuntimeError(
                    "Bound limiter is making a large correction; "
                    "scalar transport needs further attention."
                )

        
        
        
        if n_step % write_every == 0:
            output_start = wallclock.time()
            write_output(simulation_time)
            time_output += wallclock.time() - output_start

        
        # Restart checkpoint
        
        if n_step % checkpoint_every == 0:
            save_checkpoint(
                n_step,
                simulation_time,
                cumulative_fresh_in,
                cumulative_fresh_out,
            )


    # Always save the final completed state if it was not already
    # written on the normal checkpoint interval.
    if num_steps % checkpoint_every != 0:
        save_checkpoint(
            num_steps,
            simulation_time,
            cumulative_fresh_in,
            cumulative_fresh_out,
        )


    
    
    

    xdmf_U.close()
    xdmf_u.close()
    xdmf_p.close()
    xdmf_fresh.close()
    xdmf_c.close()


    
    # FINISH
    

    wall_total = wallclock.time() - wall_start

    if rank == 0:
        print("")
        print("==============================================")
        print("70 CC/MIN AXISYMMETRIC IPCS FINISHED")
        print("restart step =", restart_step)
        print("final step   =", num_steps)
        print("simulated time =", simulation_time, "s")
        print("wall time =", wall_total, "s")
        print("visualisation segment =", SEGMENT_OUTPUT_FOLDER)
        print("checkpoints =", CHECKPOINT_FOLDER)
        print("==============================================")
    _verification_write_snapshot("verification_final.npz")
    _verification_close_budget()
    _verification_json("verification_status.json", {
        "status": "completed", "case": ARGS.case,
        "last_recorded_step": n_step, "last_recorded_time_s": simulation_time,
        "target_time_s": T_final, "last_budget_row": _verification_last_row,
        "wall_elapsed_s": wallclock.time() - wall_start,
        "scope": "50 s verification; uncorrected result is diagnostic only.",
    })
    _verification_completed = True

except BaseException as error:
    _verification_close_budget()
    failure = {
        "status": "failed", "case": ARGS.case, "target_time_s": 50.0,
        "exception_type": type(error).__name__, "exception": str(error),
        "attempted_step": int(globals().get("n_step", 0)),
        "attempted_time_s": float(globals().get("simulation_time", 0.0)),
        "last_budget_row": _verification_last_row,
        "note": "Not a completed verification; inspect failure state and original traceback.",
    }
    try:
        if "_verification_cell_dofs" in globals():
            _verification_write_snapshot("verification_failure_state.npz")
            failure["failure_snapshot"] = "verification_failure_state.npz"
    except Exception as snapshot_error:
        failure["snapshot_error"] = str(snapshot_error)
    try:
        _verification_json("verification_status.json", failure)
    except Exception as status_error:
        print("Could not write failure status:", status_error, file=sys.stderr)
    raise
