# -*- coding: utf-8 -*-
from fenics import *
import math
import os
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parameters["std_out_all_processes"] = False


# Geometry

Lx = 0.60
Ly = 0.30

d_nozzle = 0.01
x_centre = Lx / 2.0
xL = x_centre - d_nozzle / 2.0
xR = x_centre + d_nozzle / 2.0

left_return_width = xL
right_return_width = Lx - xR

tol = 1e-10


# Mesh

nx, ny = 120, 60
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)


# Boundary markers

# 1 = central nozzle
# 2 = broad left return flow
# 3 = broad right return flow
# 4 = stationary side walls and glass ceiling

class Nozzle(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
            and near(x[1], 0.0, tol)
            and xL - tol <= x[0] <= xR + tol
        )


class LeftReturn(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
            and near(x[1], 0.0, tol)
            and x[0] <= xL + tol
        )


class RightReturn(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
            and near(x[1], 0.0, tol)
            and x[0] >= xR - tol
        )


class SolidWalls(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
            and (
                near(x[0], 0.0, tol)
                or near(x[0], Lx, tol)
                or near(x[1], Ly, tol)
            )
        )


class PressurePin(SubDomain):
    def inside(self, x, on_boundary):
        # Fix only the pressure gauge at the upper-left mesh vertex.
        return near(x[0], 0.0, tol) and near(x[1], Ly, tol)


boundaries = MeshFunction(
    "size_t", mesh, mesh.topology().dim() - 1, 0
)

# Mark walls first, then overwrite the bottom openings.
SolidWalls().mark(boundaries, 4)
LeftReturn().mark(boundaries, 2)
RightReturn().mark(boundaries, 3)
Nozzle().mark(boundaries, 1)

File("boundary_markers_ipcs_broad_return.pvd") << boundaries

ds_sub = Measure("ds", domain=mesh, subdomain_data=boundaries)
normal = FacetNormal(mesh)


# Boundary-length diagnostics

nozzle_length = assemble(Constant(1.0) * ds_sub(1))
left_return_length = assemble(Constant(1.0) * ds_sub(2))
right_return_length = assemble(Constant(1.0) * ds_sub(3))
solid_wall_length = assemble(Constant(1.0) * ds_sub(4))

if rank == 0:
    print("Boundary lengths:", flush=True)
    print(
        "  Nozzle       =", nozzle_length,
        "(expected 0.01)", flush=True
    )
    print(
        "  Left return  =", left_return_length,
        "(expected 0.295)", flush=True
    )
    print(
        "  Right return =", right_return_length,
        "(expected 0.295)", flush=True
    )
    print(
        "  Solid walls  =", solid_wall_length,
        "(expected 1.20)", flush=True
    )


# Function spaces

V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
C = FunctionSpace(mesh, "CG", 1)

# Velocity trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Pressure trial and test functions
p = TrialFunction(P)
q = TestFunction(P)

# Concentration trial and test functions
c_trial = TrialFunction(C)
c_test = TestFunction(C)


# Time-dependent solution fields

u_n = Function(V)
u_star = Function(V)
u_new = Function(V)

p_n = Function(P)
p_new = Function(P)

c_n = Function(C)
c_new = Function(C)


# Parameters

dt_value = 0.001

# First validation run: this crosses the previous failure time.
# Change to 20000 only after all 12000 steps remain stable.
num_steps = 60000
write_every = 1000

dt = Constant(dt_value)
nu_visc = Constant(1e-6)
D = Constant(1e-6)

g = Constant(9.81)
beta = Constant(-1e-3)
c0 = Constant(1.0)


# Initial conditions

u_n.assign(Constant((0.0, 0.0)))
u_star.assign(Constant((0.0, 0.0)))
u_new.assign(Constant((0.0, 0.0)))

p_n.assign(Constant(0.0))
p_new.assign(Constant(0.0))

# Ambient salt water
c_n.assign(Constant(1.0))
c_new.assign(Constant(1.0))


# Balanced inlet and return-flow profiles

U_in = 0.02

# For a parabolic profile, Q = (2/3) U_max width.
# Two broad return regions remove exactly the nozzle flux.
U_return = U_in * d_nozzle / (2.0 * left_return_width)

u_in = Expression(
    (
        "0.0",
        "4.0*amp*U*(x[0]-xL)*(xR-x[0])/pow(xR-xL,2)"
    ),
    degree=2,
    amp=0.0,
    U=U_in,
    xL=xL,
    xR=xR
)

u_return_left = Expression(
    (
        "0.0",
        "-4.0*amp*U*(x[0]-xA)*(xB-x[0])/pow(xB-xA,2)"
    ),
    degree=2,
    amp=0.0,
    U=U_return,
    xA=0.0,
    xB=xL
)

u_return_right = Expression(
    (
        "0.0",
        "-4.0*amp*U*(x[0]-xA)*(xB-x[0])/pow(xB-xA,2)"
    ),
    degree=2,
    amp=0.0,
    U=U_return,
    xA=xR,
    xB=Lx
)

if rank == 0:
    print("Velocity settings:", flush=True)
    print("  Maximum nozzle velocity =", U_in, flush=True)
    print("  Maximum return velocity =", U_return, flush=True)
    print(
        "  Return/nozzle ratio      =", U_return / U_in,
        flush=True
    )


# Boundary conditions

bcu = [
    DirichletBC(V, u_in, boundaries, 1),
    DirichletBC(V, u_return_left, boundaries, 2),
    DirichletBC(V, u_return_right, boundaries, 3),
    DirichletBC(V, Constant((0.0, 0.0)), boundaries, 4)
]

# Pressure is physically determined only up to an additive constant.
# Pinning one point supplies a gauge without imposing pressure on a wall.
bcp = [
    DirichletBC(
        P,
        Constant(0.0),
        PressurePin(),
        method="pointwise"
    )
]

# Fresh water enters through the nozzle. No scalar value is imposed
# on the return boundaries, so fluid leaves with its local concentration.
bcc = [
    DirichletBC(C, Constant(0.0), boundaries, 1)
]


# IPCS step 1: tentative velocity

# Semi-implicit Oseen advection is retained from the working base:
# u_n convects the new tentative velocity.

f_buoy = as_vector((0.0, g * beta * (c_n - c0)))

F1 = (
    (1.0 / dt) * inner(u - u_n, v) * dx
    + inner(dot(u_n, nabla_grad(u)), v) * dx
    + nu_visc * inner(grad(u), grad(v)) * dx
    - p_n * div(v) * dx
    - inner(f_buoy, v) * dx
)

a1 = lhs(F1)
L1 = rhs(F1)


# IPCS step 2: pressure correction

# grad(p_new - p_n) enforces the divergence correction.

a2 = inner(grad(p), grad(q)) * dx
L2 = (
    inner(grad(p_n), grad(q)) * dx
    - (1.0 / dt) * div(u_star) * q * dx
)


# IPCS step 3: velocity correction

a3 = inner(u, v) * dx
L3 = (
    inner(u_star, v) * dx
    - dt * inner(grad(p_new - p_n), v) * dx
)

# The pressure matrix is constant; the velocity-correction matrix is
# reassembled with the current nonzero boundary values.
A2 = assemble(a2)
for bc in bcp:
    bc.apply(A2)


# Scalar transport with SUPG, using the corrected velocity

h = CellDiameter(mesh)
u_mag = sqrt(dot(u_new, u_new) + DOLFIN_EPS)

tau = 1.0 / sqrt(
    (2.0 / dt) ** 2
    + (2.0 * u_mag / h) ** 2
    + (4.0 * D / (h * h)) ** 2
)

r_c = (
    (1.0 / dt) * (c_trial - c_n)
    + dot(u_new, grad(c_trial))
    - div(D * grad(c_trial))
)

F_c = (
    (1.0 / dt) * (c_trial - c_n) * c_test * dx
    + dot(u_new, grad(c_trial)) * c_test * dx
    + D * dot(grad(c_trial), grad(c_test)) * dx
    + tau * r_c * dot(u_new, grad(c_test)) * dx
)

a_c = lhs(F_c)
L_c = rhs(F_c)


# Solvers

tentative_solver = LUSolver()
pressure_solver = LUSolver(A2)
correction_solver = LUSolver()
scalar_solver = LUSolver()




folder = "/user/work/kn22417/output_physical_plume_ipcs"

if rank == 0:
    os.makedirs(folder, exist_ok=True)

comm.Barrier()

xdmf_u = XDMFFile(mesh.mpi_comm(), f"{folder}/u.xdmf")
xdmf_p = XDMFFile(mesh.mpi_comm(), f"{folder}/p.xdmf")
xdmf_c = XDMFFile(mesh.mpi_comm(), f"{folder}/c.xdmf")

for output_file in (xdmf_u, xdmf_p, xdmf_c):
    output_file.parameters["flush_output"] = True
    output_file.parameters["functions_share_mesh"] = True
    output_file.parameters["rewrite_function_mesh"] = False




time = 0.0
ramp_time = 0.5

for n in range(num_steps):
    time += dt_value

    # Smooth half-cosine startup. The same amplitude is applied to
    # the nozzle and both returns, preserving exact flux balance.
    if time < ramp_time:
        ramp = 0.5 * (
            1.0 - math.cos(math.pi * time / ramp_time)
        )
    else:
        ramp = 1.0

    u_in.amp = ramp
    u_return_left.amp = ramp
    u_return_right.amp = ramp

    if rank == 0 and ((n + 1) % 100 == 0 or n < 5):
        print(
            f"Step {n + 1}/{num_steps}, "
            f"t = {time:.3f} s, ramp = {ramp:.6f}",
            flush=True
        )

    
    # Step 1: tentative velocity
    
    A1 = assemble(a1)
    b1 = assemble(L1)

    for bc in bcu:
        bc.apply(A1, b1)

    tentative_solver.solve(A1, u_star.vector(), b1)

    
    # Step 2: pressure correction
    
    b2 = assemble(L2)

    for bc in bcp:
        bc.apply(b2)

    pressure_solver.solve(p_new.vector(), b2)

    
    # Step 3: velocity correction
    
    A3 = assemble(a3)
    b3 = assemble(L3)

    for bc in bcu:
        bc.apply(A3, b3)

    correction_solver.solve(A3, u_new.vector(), b3)

    
    # Scalar transport using the corrected velocity
    
    A_c = assemble(a_c)
    b_c = assemble(L_c)

    for bc in bcc:
        bc.apply(A_c, b_c)

    scalar_solver.solve(A_c, c_new.vector(), b_c)

    
    
    
    velocity_norm = u_new.vector().norm("linf")

    c_min_before_clip = c_new.vector().min()
    c_max_before_clip = c_new.vector().max()

    if (n + 1) % 100 == 0 or n < 5:
        flux_nozzle = assemble(dot(u_new, normal) * ds_sub(1))
        flux_left = assemble(dot(u_new, normal) * ds_sub(2))
        flux_right = assemble(dot(u_new, normal) * ds_sub(3))
        net_flux = flux_nozzle + flux_left + flux_right

        tentative_divergence = sqrt(
            assemble(div(u_star) * div(u_star) * dx)
        )
        corrected_divergence = sqrt(
            assemble(div(u_new) * div(u_new) * dx)
        )

        if rank == 0:
            print("    nozzle flux       =", flux_nozzle, flush=True)
            print("    left return flux  =", flux_left, flush=True)
            print("    right return flux =", flux_right, flush=True)
            print("    net flux          =", net_flux, flush=True)
            print(
                "    ||div(u*)||_L2    =", tentative_divergence,
                flush=True
            )
            print(
                "    ||div(u)||_L2     =", corrected_divergence,
                flush=True
            )
            print("    min(c) before clip=", c_min_before_clip, flush=True)
            print("    max(c) before clip=", c_max_before_clip, flush=True)
            print("    ||u||_inf         =", velocity_norm, flush=True)

    
    # Blow-up guards
    
    c_arr = c_new.vector().get_local()

    if (
        not np.isfinite(velocity_norm)
        or velocity_norm > 5.0
        or not np.all(np.isfinite(c_arr))
    ):
        if rank == 0:
            print(
                "ERROR: instability detected at "
                f"step {n + 1}, t = {time:.6f} s.",
                flush=True
            )
        break

    # Clip scalar for robustness, as in the working base code.
    c_arr = np.clip(c_arr, 0.0, 1.0)
    c_new.vector().set_local(c_arr)
    c_new.vector().apply("insert")

    
    # Update previous fields
    
    u_n.assign(u_new)
    p_n.assign(p_new)
    c_n.assign(c_new)

    
    
    
    if (n + 1) % write_every == 0 or n == 0:
        u_new.rename("u", "")
        p_new.rename("p", "")
        c_new.rename("c", "")

        xdmf_u.write(u_new, time)
        xdmf_p.write(p_new, time)
        xdmf_c.write(c_new, time)


# Close files

xdmf_u.close()
xdmf_p.close()
xdmf_c.close()

if rank == 0:
    print("Run finished.", flush=True)