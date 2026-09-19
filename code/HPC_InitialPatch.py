# -*- coding: utf-8 -*-
from fenics import *
import numpy as np
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parameters["std_out_all_processes"] = False


# Geometry: inverted tank inside larger salt-water tank

Lx = 0.60          # tank width, m
Ly = 0.15          # shortened tank height, m

# Coarse mesh for fast diagnostic run
nx, ny = 50, 15
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)


# Nozzle geometry

x_centre = Lx / 2.0

# BRUTAL VISIBILITY TEST:
# This is deliberately large. Not physical.
# It is just to check whether the plume/output is visible.
nozzle_diameter = 0.04
nozzle_width = nozzle_diameter

xL = x_centre - nozzle_width / 2.0
xR = x_centre + nozzle_width / 2.0

tol = 1e-8


# Experimental fluid properties

rho_salt = 1005.6      # kg/m^3
rho_fresh = 999.3      # kg/m^3
rho_ref = rho_salt

drho = rho_salt - rho_fresh

# c = 1 is salt
# c = 0 is fresh
# rho(c) = rho_fresh + c*(rho_salt-rho_fresh)
beta = Constant(drho / rho_ref)

g = Constant(9.81)


# Flow rate and inlet speed

Q_cc_min = 20.0
Q_m3_s = Q_cc_min * 1e-6 / 60.0

nozzle_radius = nozzle_diameter / 2.0
A_nozzle_3D = np.pi * nozzle_radius**2

# BRUTAL VISIBILITY TEST:
# Increased deliberately to make a visible plume.
# Not final physical value.
U_mean = 0.05

# Parabolic inlet profile has mean velocity 2Umax/3.
Umax = 1.5 * U_mean

if rank == 0:
    print("Nozzle diameter =", nozzle_diameter, "m", flush=True)
    print("Q =", Q_m3_s, "m^3/s", flush=True)
    print("Diagnostic mean inlet speed =", U_mean, "m/s", flush=True)
    print("Parabolic Umax =", Umax, "m/s", flush=True)
    print("xL =", xL, "xR =", xR, flush=True)
    print("density difference =", drho, "kg/m^3", flush=True)


# Local refinement near plume/nozzle

# Only one refinement level for speed.
for k in range(1):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim(), False)

    for cell in cells(mesh):
        mp = cell.midpoint()
        x = mp.x()
        y = mp.y()

        plume_width = 0.012 + 0.04 * (y / Ly)
        near_plume = abs(x - x_centre) <= plume_width

        near_nozzle = abs(x - x_centre) <= 0.06 and y <= 0.045

        if near_plume or near_nozzle:
            cell_markers[cell] = True

    mesh = refine(mesh, cell_markers)


# Boundary markers

class Nozzle(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
            and near(x[1], 0.0, tol)
            and (xL - tol <= x[0] <= xR + tol)
        )

class BottomOpen(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
            and near(x[1], 0.0, tol)
            and not (xL - tol <= x[0] <= xR + tol)
        )

class TopClosed(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], Ly, tol)

class Sides(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
            and (near(x[0], 0.0, tol) or near(x[0], Lx, tol))
        )

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

Nozzle().mark(boundaries, 1)
BottomOpen().mark(boundaries, 2)
TopClosed().mark(boundaries, 3)
Sides().mark(boundaries, 4)

File("boundary_markers_brutal_visibility_test.pvd") << boundaries

ds_sub = Measure("ds", domain=mesh, subdomain_data=boundaries)

if rank == 0:
    print("Boundary lengths:", flush=True)
    print("  Nozzle      =", assemble(Constant(1.0) * ds_sub(1)), flush=True)
    print("  Bottom open =", assemble(Constant(1.0) * ds_sub(2)), flush=True)
    print("  Top closed  =", assemble(Constant(1.0) * ds_sub(3)), flush=True)
    print("  Sides       =", assemble(Constant(1.0) * ds_sub(4)), flush=True)


# Function spaces

V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
C = FunctionSpace(mesh, "CG", 1)

TH = MixedElement([V.ufl_element(), P.ufl_element()])
W = FunctionSpace(mesh, TH)

up = Function(W)
(u, p) = split(up)
(v, q) = TestFunctions(W)

c = Function(C)
c_n = Function(C)

c_trial = TrialFunction(C)
c_test = TestFunction(C)

u_n = Function(V)
p_n = Function(P)


# Numerical parameters

dt_value = 0.0005

# Short brutal diagnostic run
num_steps = 2000      # 1 second physical time
write_every = 100     # frequent output

dt = Constant(dt_value)

nu = Constant(1e-6)

# Physical salt diffusivity is closer to 1e-9.
# This is slightly larger for numerical stability.
D = Constant(1e-8)


# Initial conditions

u_n.assign(Constant((0.0, 0.0)))
p_n.assign(Constant(0.0))

# Initially all salt water except for a visible fresh patch above the nozzle.
# This is diagnostic only. It proves whether the output/ParaView pipeline works.
class InitialConcentration(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0   # salt everywhere

        # obvious fresh patch above nozzle
        if abs(x[0] - x_centre) <= 0.04 and x[1] <= 0.04:
            values[0] = 0.0

    def value_shape(self):
        return ()

c_init = InitialConcentration(degree=1)
c_n.interpolate(c_init)
c.assign(c_n)

assign(up.sub(0), u_n)
assign(up.sub(1), p_n)


# Inlet velocity and scalar

u_in = Expression(
    (
        "0.0",
        "4.0*Umax*(x[0]-xL)*(xR-x[0])/pow(xR-xL,2)"
    ),
    degree=2,
    Umax=Umax,
    xL=xL,
    xR=xR
)

# Fresh inlet concentration
c_fresh = Constant(0.0)


# Prescribed bottom outflow

# 2D inlet flux per unit depth.
# Since U_mean is the mean velocity of the parabolic inlet:
Q_in_2D = U_mean * nozzle_width

bottom_open_width = Lx - nozzle_width

# Weak downward outflow through the open bottom.
U_bottom_out = Q_in_2D / bottom_open_width

u_bottom_out = Expression(
    ("0.0", "-Uout"),
    degree=1,
    Uout=U_bottom_out
)

if rank == 0:
    print("2D inlet flux =", Q_in_2D, flush=True)
    print("Bottom outflow speed =", U_bottom_out, flush=True)


# Pressure pin

# Since we are prescribing velocity on the open bottom rather than pressure,
# we pin pressure at one point to remove the arbitrary pressure constant.
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0, tol) and near(x[1], Ly, tol)


# Boundary conditions

bcu_mixed = [
    # Central fresh-water inlet
    DirichletBC(W.sub(0), u_in, boundaries, 1),

    # Open bottom: displaced salt water leaves downward
    DirichletBC(W.sub(0), u_bottom_out, boundaries, 2),

    # Sides are solid tank walls
    DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 4),

    # Closed top: no vertical penetration, free horizontal slip
    DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 3),

    # Pressure pin
    DirichletBC(W.sub(1), Constant(0.0), PressurePoint(), method="pointwise"),
]

# Scalar:
# Fresh enters through nozzle.
# No scalar Dirichlet condition is imposed on the open bottom,
# because we are prescribing outflow there.
bcc = [
    DirichletBC(C, c_fresh, boundaries, 1)
]


# Buoyancy force

# c = 1 salt, c = 0 fresh
# rho-rho_ref = drho*(c-1)
# force is upward when c < 1.
f_buoy = as_vector((0.0, -g * beta * (c_n - Constant(1.0))))


# Flow problem

F_flow = (
    (1.0 / dt) * inner(u - u_n, v) * dx
    + inner(dot(u_n, nabla_grad(u)), v) * dx
    + nu * inner(grad(u), grad(v)) * dx
    - p * div(v) * dx
    - q * div(u) * dx
    - inner(f_buoy, v) * dx
)


# Scalar transport with SUPG

h = CellDiameter(mesh)
u_mag = sqrt(dot(u_n, u_n) + DOLFIN_EPS)

tau = 1.0 / sqrt(
    (2.0 / dt)**2
    + (2.0 * u_mag / h)**2
    + (4.0 * D / (h * h))**2
)

r_c = (
    (1.0 / dt) * (c_trial - c_n)
    + dot(u_n, grad(c_trial))
    - div(D * grad(c_trial))
)

F_c = (
    (1.0 / dt) * (c_trial - c_n) * c_test * dx
    + dot(u_n, grad(c_trial)) * c_test * dx
    + D * dot(grad(c_trial), grad(c_test)) * dx
    + tau * r_c * dot(u_n, grad(c_test)) * dx
)

a_c = lhs(F_c)
L_c = rhs(F_c)


# Solver

scalar_solver = LUSolver()




folder = "/user/work/kn22417/output_21_brutal_visibility_test"

if rank == 0:
    os.makedirs(folder, exist_ok=True)

xdmf_u = XDMFFile(mesh.mpi_comm(), f"{folder}/u.xdmf")
xdmf_p = XDMFFile(mesh.mpi_comm(), f"{folder}/p.xdmf")
xdmf_c = XDMFFile(mesh.mpi_comm(), f"{folder}/c.xdmf")
xdmf_fresh = XDMFFile(mesh.mpi_comm(), f"{folder}/fresh.xdmf")

for f in (xdmf_u, xdmf_p, xdmf_c, xdmf_fresh):
    f.parameters["flush_output"] = True
    f.parameters["functions_share_mesh"] = True
    f.parameters["rewrite_function_mesh"] = False


# Helper function for output

def write_output(time_value, u_value, p_value, c_value):
    u_file = Function(V)
    p_file = Function(P)
    c_file = Function(C)
    fresh_file = Function(C)

    u_file.assign(u_value)
    p_file.assign(p_value)
    c_file.assign(c_value)

    fresh_arr = 1.0 - c_value.vector().get_local()
    fresh_file.vector().set_local(fresh_arr)
    fresh_file.vector().apply("insert")

    u_file.rename("u", "")
    p_file.rename("p", "")
    c_file.rename("c", "")
    fresh_file.rename("fresh", "")

    xdmf_u.write(u_file, time_value)
    xdmf_p.write(p_file, time_value)
    xdmf_c.write(c_file, time_value)
    xdmf_fresh.write(fresh_file, time_value)

# Write initial condition immediately so the fresh patch must appear at t = 0
write_output(0.0, u_n, p_n, c_n)




time = 0.0
wall_start = MPI.Wtime()

for n in range(num_steps):
    time += dt_value

    if rank == 0 and ((n + 1) % 100 == 0 or n < 5):
        elapsed = MPI.Wtime() - wall_start
        sec_per_step = elapsed / (n + 1)
        remaining = sec_per_step * (num_steps - n - 1)

        print(f"Step {n+1}/{num_steps}, t = {time:.4f} s", flush=True)
        print(f"    wall elapsed = {elapsed/60:.2f} min", flush=True)
        print(f"    sec/step     = {sec_per_step:.3f}", flush=True)
        print(f"    est left     = {remaining/60:.2f} min", flush=True)

    # --- Flow solve
    solve(F_flow == 0, up, bcu_mixed)
    u_sol, p_sol = up.split(deepcopy=True)

    # --- Scalar solve
    A_c = assemble(a_c)
    b_c = assemble(L_c)

    for bc in bcc:
        bc.apply(A_c, b_c)

    scalar_solver.solve(A_c, c.vector(), b_c)

    # --- Clip scalar
    c_arr = c.vector().get_local()
    c_arr = np.clip(c_arr, 0.0, 1.0)
    c.vector().set_local(c_arr)
    c.vector().apply("insert")

    if rank == 0 and ((n + 1) % 100 == 0 or n < 5):
        cmin = c.vector().min()
        cmax = c.vector().max()
        fresh_max = 1.0 - cmin
        fresh_mass = assemble((Constant(1.0) - c) * dx)

        print("    min(c) =", cmin, flush=True)
        print("    max(c) =", cmax, flush=True)
        print("    max(fresh) =", fresh_max, flush=True)
        print("    total fresh mass =", fresh_mass, flush=True)
        print("    ||u||_inf =", u_sol.vector().norm("linf"), flush=True)

    # --- Update
    u_n.assign(u_sol)
    p_n.assign(p_sol)
    c_n.assign(c)

    # --- Output
    if (n + 1) % write_every == 0:
        write_output(time, u_sol, p_sol, c)

if rank == 0:
    print("Done.", flush=True)