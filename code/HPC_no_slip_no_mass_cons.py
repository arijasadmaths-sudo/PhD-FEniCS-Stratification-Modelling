# -*- coding: utf-8 -*-
from fenics import *
import os
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

tol = 1e-10


# Mesh

nx, ny = 120, 60
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)


# Boundary markers

class Nozzle(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol) and (xL - tol <= x[0] <= xR + tol)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and not (
            near(x[1], 0.0, tol) and (xL - tol <= x[0] <= xR + tol)
        )

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
Nozzle().mark(boundaries, 1)
Walls().mark(boundaries, 2)

File("boundary_markers_physical_plume.pvd") << boundaries

ds_sub = Measure("ds", domain=mesh, subdomain_data=boundaries)
if rank == 0:
    print("Boundary lengths:", flush=True)
    print("  Nozzle =", assemble(Constant(1.0) * ds_sub(1)), flush=True)
    print("  Walls  =", assemble(Constant(1.0) * ds_sub(2)), flush=True)


# Function spaces
# Taylor-Hood: P2-P1

V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
C = FunctionSpace(mesh, "CG", 1)

TH = MixedElement([V.ufl_element(), P.ufl_element()])
W = FunctionSpace(mesh, TH)

# Mixed unknowns/tests
up = Function(W)
(u_trial, p_trial) = TrialFunctions(W)
(v_test, q_test) = TestFunctions(W)

# Scalar unknowns/tests
c = Function(C)
c_n = Function(C)
c_trial = TrialFunction(C)
c_test = TestFunction(C)

# For split fields after solve
u_n = Function(V)
p_n = Function(P)


# Parameters

dt_value = 0.001
num_steps = 20000
write_every = 1000

dt = Constant(dt_value)
nu = Constant(1e-6)
D = Constant(1e-6)
g = Constant(9.81)
beta = Constant(-1e-3)

c0 = Constant(1.0)


# Initial conditions

u_n.assign(Constant((0.0, 0.0)))
p_n.assign(Constant(0.0))
c_n.assign(Constant(1.0))


# Inflow profile

U_in = 0.02

u_in = Expression(
    ("0.0",
     "4.0*U*(x[0]-xL)*(xR-x[0])/pow(xR-xL,2)"),
    degree=2, U=U_in, xL=xL, xR=xR
)


# Boundary conditions

# Mixed velocity BCs apply to W.sub(0)
bcu_mixed = [
    DirichletBC(W.sub(0), u_in, boundaries, 1),
    DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 2)
]

bcc = [
    DirichletBC(C, Constant(0.0), boundaries, 1)
]


# Buoyancy

f_buoy = as_vector((0.0, g * beta * (c_n - c0)))


# Mixed flow problem: unsteady Stokes + buoyancy

# (u - u_n)/dt - nu ?u + ?p = f
# div(u) = 0

F_flow = (
    (1/dt)*inner(u_trial - u_n, v_test)*dx
    + inner(dot(u_n, nabla_grad(u_trial)), v_test)*dx
    + nu*inner(grad(u_trial), grad(v_test))*dx
    - inner(p_trial, div(v_test))*dx
    - inner(div(u_trial), q_test)*dx
    - inner(f_buoy, v_test)*dx
)

a_flow = lhs(F_flow)
L_flow = rhs(F_flow)


# Scalar transport with SUPG stabilisation
# c_t + u · ?c = D ?c

# Note: this uses u_n from previous step, which is safer
h = CellDiameter(mesh)
u_mag = sqrt(dot(u_n, u_n) + DOLFIN_EPS)
tau = 1.0 / sqrt((2.0 / dt)**2 + (2.0 * u_mag / h)**2 + (4.0 * D / (h * h))**2)

r_c = (1.0 / dt) * (c_trial - c_n) + dot(u_n, grad(c_trial)) - div(D * grad(c_trial))

F_c = (
    (1.0 / dt) * (c_trial - c_n) * c_test * dx
    + dot(u_n, grad(c_trial)) * c_test * dx
    + D * dot(grad(c_trial), grad(c_test)) * dx
    + tau * r_c * dot(u_n, grad(c_test)) * dx
)

a_c = lhs(F_c)
L_c = rhs(F_c)


# Solvers

flow_solver = LUSolver()
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

    # --- Flow solve
    A_flow = assemble(a_flow)
    b_flow = assemble(L_flow)
    for bc in bcu_mixed:
        bc.apply(A_flow, b_flow)
    flow_solver.solve(A_flow, up.vector(), b_flow)

    u_sol, p_sol = up.split(deepcopy=True)

    # --- Scalar solve
    A_c = assemble(a_c)
    b_c = assemble(L_c)
    for bc in bcc:
        bc.apply(A_c, b_c)
    scalar_solver.solve(A_c, c.vector(), b_c)

    if rank == 0 and ((n + 1) % 100 == 0 or n < 5):
        print("    min(c) before clip =", c.vector().min(), flush=True)
        print("    max(c) before clip =", c.vector().max(), flush=True)
        print("    ||u||_inf =", u_sol.vector().norm("linf"), flush=True)

    # Clip scalar for robustness
    c_arr = c.vector().get_local()
    c_arr[c_arr < 0.0] = 0.0
    c_arr[c_arr > 1.0] = 1.0
    c.vector().set_local(c_arr)
    c.vector().apply("insert")

    # Update states
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