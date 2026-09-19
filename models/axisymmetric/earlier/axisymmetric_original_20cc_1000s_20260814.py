# -*- coding: utf-8 -*-
"""
Experiment-matched axisymmetric IPCS model for legacy FEniCS 2019.1.0

Axisymmetric transformation:
    s   = r^2 / 2
    U_s = r u_r
    U_z = u_z

so physical incompressibility becomes
    div_s(U) = 0.

Numerics are the validated fast formulation:
    - CG2 transformed velocity
    - CG1 pressure increment
    - IPCS pressure-correction splitting
    - explicit convection
    - DG0 upwind fresh-water transport
    - direct LU solvers with constant IPCS matrices preassembled

Physical source:
    - circular nozzle diameter = 2 mm
    - imposed volume flux      = 20 cc/min
    - parabolic Poiseuille inlet profile

This file generates a mesh matched to the 2 mm nozzle directly in
memory. It is intentionally serial and should be run with one MPI task.

Mass-conservation validation:
    dt = 0.001 s
    T  = 2 s
"""

from fenics import *
import numpy as np
import os
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


OUTPUT_FOLDER = "/path/to/work/output_axisym_massfix_dt001_1000s"

# First experiment-matched validation run
# 100 s experiment-matched run

dt_value = 0.001
num_steps = 1000000          # 1000 s
write_every = 50000          # output every 5 s
diagnostic_every = 5000     # diagnostics every 5 s
ramp_time = 0.5


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


# Experimental injection rate:

#     20 cc/min

# 1 cc = 1e-6 m^3.
Q_target_cc_per_min = 20.0

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

# Explicit convection using the known previous-step velocity.
# This moves convection to the RHS and makes the Stage-1 matrix
# time independent.
convective_known = dot(
    grad(physical_velocity(U_n)),
    U_n,
)

a1 = (
    (Constant(1.0) / dt)
    * inner(u_trial_phys, v_test_phys)
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

    - inner(convective_known, v_test_phys)
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
        raise RuntimeError(
            "Conservative scalar solve produced an impossible "
            "total fresh-water mass."
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



# DIRECT SOLVERS + CONSTANT-MATRIX PREASSEMBLY

# With convection explicit, all three IPCS matrices a1, a2
# and a3 are time independent and are factorised once.

# The DG0 scalar matrix still changes because its upwind flux
# depends on the current corrected velocity U_new.


solver_fresh = LUSolver()

if rank == 0:
    print("")
    print("Preassembling constant IPCS matrices...")

# Tentative-velocity matrix: now constant because convection
# is explicit and appears only in the RHS.
A1_const = assemble(a1)

for bc in bcu:
    bc.apply(A1_const)

solver_Ustar = LUSolver()
solver_Ustar.set_operator(A1_const)

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
    print("Constant IPCS matrices ready.")






if rank == 0:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MPI.barrier(comm)

xdmf_U = XDMFFile(mesh.mpi_comm(), f"{OUTPUT_FOLDER}/U_transformed.xdmf")
xdmf_u = XDMFFile(mesh.mpi_comm(), f"{OUTPUT_FOLDER}/u_physical.xdmf")
xdmf_p = XDMFFile(mesh.mpi_comm(), f"{OUTPUT_FOLDER}/p.xdmf")
xdmf_fresh = XDMFFile(mesh.mpi_comm(), f"{OUTPUT_FOLDER}/fresh.xdmf")
xdmf_c = XDMFFile(mesh.mpi_comm(), f"{OUTPUT_FOLDER}/c.xdmf")

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


write_output(0.0)






simulation_time = 0.0
wall_start = wallclock.time()

# Cumulative wall-clock timings.  These let us see exactly
# which stage is expensive before making any further solver change.
time_stage1 = 0.0
time_stage2 = 0.0
time_stage3 = 0.0
time_scalar = 0.0
time_output = 0.0

# Scalar mass accounting
cumulative_fresh_in = 0.0
cumulative_fresh_out = 0.0
last_raw_fmin = 0.0
last_raw_fmax = 0.0
last_limiter_L1 = 0.0
last_limiter_mass_error = 0.0

for n_step in range(1, num_steps + 1):
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
    
    # Reuse the preassembled/factorised matrix.
    # Only the RHS changes with U_n, pressure and buoyancy.
    
    stage_start = wallclock.time()

    b1 = assemble(L1)

    for bc in bcu:
        bc.apply(b1)

    solver_Ustar.solve(U_star.vector(), b1)

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
    previous_arr = fresh_n.vector().get_local()

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

        if fmin < -1.0e-10 or fmax > 1.0 + 1.0e-10:
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
    print("SIMPLIFIED AXISYMMETRIC IPCS FINISHED")
    print("steps =", num_steps)
    print("simulated time =", simulation_time, "s")
    print("wall time =", wall_total, "s")
    print("output =", OUTPUT_FOLDER)
    print("==============================================")