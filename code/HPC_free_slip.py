# -*- coding: utf-8 -*-
from fenics import *
import numpy as np
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parameters["std_out_all_processes"] = False


# Geometry + mesh

Lx = 0.60
Ly = 0.30

nx, ny = 100, 50
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)


# Align nozzle with mesh

dx_base = Lx / nx
x_centre = Lx / 2.0
i_centre = int(round(x_centre / dx_base))

# 2 base cells wide
xL = (i_centre - 1) * dx_base
xR = (i_centre + 1) * dx_base

tol = 1e-6

if rank == 0:
    print("xL =", xL, "xR =", xR, flush=True)


# Local refinement (wedge)

for k in range(2):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim(), False)
    for cell in cells(mesh):
        mp = cell.midpoint()
        y = mp.y()
        half_width = 0.01 + 0.04 * (y / Ly)
        if abs(mp.x() - x_centre) <= half_width:
            cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)


# Boundary markers

class Nozzle(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol) and (xL - tol <= x[0] <= xR + tol)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol) and not (xL - tol <= x[0] <= xR + tol)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], Ly, tol)

class Sides(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0, tol) or near(x[0], Lx, tol))

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
Nozzle().mark(boundaries, 1)
Bottom().mark(boundaries, 2)
Top().mark(boundaries, 3)
Sides().mark(boundaries, 4)

File("boundary_markers_physical_plume.pvd") << boundaries

ds_sub = Measure("ds", domain=mesh, subdomain_data=boundaries)

if rank == 0:
    print("Boundary lengths:", flush=True)
    print("  Nozzle =", assemble(Constant(1.0) * ds_sub(1)), flush=True)
    print("  Bottom =", assemble(Constant(1.0) * ds_sub(2)), flush=True)
    print("  Top    =", assemble(Constant(1.0) * ds_sub(3)), flush=True)
    print("  Sides  =", assemble(Constant(1.0) * ds_sub(4)), flush=True)


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


# Parameters

dt_value = 0.005
num_steps = 30000
write_every = 1000

dt = Constant(dt_value)
nu = Constant(1e-6)
D = Constant(1e-6)
g = Constant(9.81)
beta = Constant(-5e-3)

c0 = Constant(1.0)


# Initial conditions

u_n.assign(Constant((0.0, 0.0)))
p_n.assign(Constant(0.0))
c_n.assign(Constant(1.0))

assign(up.sub(0), u_n)
assign(up.sub(1), p_n)


# Inflow

U_in = 0.02

u_in = Expression(
    ("0.0",
     "4.0*U*(x[0]-xL)*(xR-x[0])/pow(xR-xL,2)"),
    degree=2, U=U_in, xL=xL, xR=xR
)


# Boundary conditions

bcu_mixed = [
    DirichletBC(W.sub(0), u_in, boundaries, 1),                    # nozzle
    DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 2),   # bottom except nozzle
    DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 4),   # side walls
    DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 3),   # top free-slip: uy = 0
]

bcc = [
    DirichletBC(C, Constant(0.0), boundaries, 1)
]


# Buoyancy

f_buoy = as_vector((0.0, g * beta * (c_n - c0)))


# Flow problem

F_flow = (
    (1.0/dt) * inner(u - u_n, v) * dx
    + inner(dot(u_n, nabla_grad(u)), v) * dx
    + nu * inner(grad(u), grad(v)) * dx
    - p * div(v) * dx
    - q * div(u) * dx
    - inner(f_buoy, v) * dx
)


# Scalar transport with SUPG

h = CellDiameter(mesh)
u_mag = sqrt(dot(u_n, u_n) + DOLFIN_EPS)

tau = 1.0 / sqrt((2.0 / dt)**2 + (2.0 * u_mag / h)**2 + (4.0 * D / (h * h))**2)

r_c = (1.0/dt) * (c_trial - c_n) + dot(u_n, grad(c_trial)) - div(D * grad(c_trial))

F_c = (
    (1.0/dt) * (c_trial - c_n) * c_test * dx
    + dot(u_n, grad(c_trial)) * c_test * dx
    + D * dot(grad(c_trial), grad(c_test)) * dx
    + tau * r_c * dot(u_n, grad(c_test)) * dx
)

a_c = lhs(F_c)
L_c = rhs(F_c)


# Solver

scalar_solver = LUSolver()




folder = "/user/work/kn22417/output_physical_plume"
if rank == 0:
    os.makedirs(folder, exist_ok=True)

xdmf_u = XDMFFile(mesh.mpi_comm(), f"{folder}/u.xdmf")
xdmf_p = XDMFFile(mesh.mpi_comm(), f"{folder}/p.xdmf")
xdmf_c = XDMFFile(mesh.mpi_comm(), f"{folder}/c.xdmf")

for f in (xdmf_u, xdmf_p, xdmf_c):
    f.parameters["flush_output"] = True
    f.parameters["functions_share_mesh"] = True
    f.parameters["rewrite_function_mesh"] = False




time = 0.0

for n in range(num_steps):
    time += dt_value

    if rank == 0 and ((n + 1) % 500 == 0 or n < 5):
        print(f"Step {n+1}/{num_steps}, t = {time:.3f} s", flush=True)

    # --- Flow
    solve(F_flow == 0, up, bcu_mixed)
    u_sol, p_sol = up.split(deepcopy=True)

    # --- Scalar
    A_c = assemble(a_c)
    b_c = assemble(L_c)
    for bc in bcc:
        bc.apply(A_c, b_c)
    scalar_solver.solve(A_c, c.vector(), b_c)

    if rank == 0 and ((n + 1) % 100 == 0 or n < 5):
        print("    min(c) =", c.vector().min(), flush=True)
        print("    max(c) =", c.vector().max(), flush=True)
        print("    ||u||_inf =", u_sol.vector().norm("linf"), flush=True)

    # Clip scalar
    c_arr = c.vector().get_local()
    c_arr = np.clip(c_arr, 0.0, 1.0)
    c.vector().set_local(c_arr)
    c.vector().apply("insert")

    # Update
    u_n.assign(u_sol)
    p_n.assign(p_sol)
    c_n.assign(c)

    
    if (n + 1) % write_every == 0 or n == 0:
        u_out = Function(V)
        p_out = Function(P)
        c_out = Function(C)

        u_out.assign(u_sol)
        p_out.assign(p_sol)
        c_out.assign(c)

        u_out.rename("u", "")
        p_out.rename("p", "")
        c_out.rename("c", "")

        xdmf_u.write(u_out, time)
        xdmf_p.write(p_out, time)
        xdmf_c.write(c_out, time)

if rank == 0:
    print("Done.", flush=True)