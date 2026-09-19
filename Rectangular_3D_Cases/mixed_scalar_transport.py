"""Conservative RT0/DG0 advection--diffusion for legacy DOLFIN 2019.1.

The auxiliary variable is r = -sqrt(dt*D)*grad(c), so the physical outward
diffusive flux is q = sqrt(D/dt)*r.  RT0 means FEniCS's ``RT`` degree 1.
Unlike a two-point cell-diameter penalty, this mixed formulation represents
isotropic Fickian diffusion on general tetrahedral meshes.  It does not imply
a discrete maximum principle: callers must diagnose concentration extrema.
"""

import math

from petsc4py import PETSc

from dolfin import (
    Constant, DirichletBC, FacetNormal, FiniteElement, Function,
    FunctionAssigner, FunctionSpace, Measure, MixedElement, PETScKrylovSolver,
    PETScMatrix, PETScVector, SystemAssembler, TestFunctions, TrialFunctions,
    avg, div, dot, grad, inner, jump, split,
)


class MixedScalarTransport:
    """One implicit conservative scalar step with a mixed diffusive flux.

    ``c_old`` and ``velocity`` are live coefficients updated by the caller.
    ``mixed_space`` may reuse an existing RT1 x DG0 space on this mesh.

    ``dirichlet_values`` maps boundary markers to imposed concentrations for
    diffusion.  Its default is fresh concentration zero on ``nozzle_marker``.
    An explicitly empty mapping requests no diffusive Dirichlet boundaries.
    ``noflux_markers`` impose q.n=0; these markers must not overlap the
    diffusive Dirichlet markers.

    ``inflow_values`` maps markers to exterior concentration for advection;
    every omitted marker has exterior concentration zero.  ``source`` is an
    optional concentration source per unit time, intended for verification.
    """

    def __init__(self, mesh, C, c_old, velocity, diffusivity, dt, boundaries,
                 nozzle_marker=3, noflux_markers=(1, 2, 4), mixed_space=None,
                 dirichlet_values=None, inflow_values=None, source=None):
        self.D = float(diffusivity)
        self.dt = float(dt)
        if not math.isfinite(self.D) or self.D <= 0.0:
            raise ValueError("Mixed scalar diffusivity must be finite and positive.")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("Mixed scalar timestep must be finite and positive.")
        if mesh.geometry().dim() != 3:
            raise ValueError("This production scalar module requires a 3D mesh.")
        if C.ufl_element().degree() != 0:
            raise ValueError("MixedScalarTransport requires DG0 concentration.")

        self.mesh = mesh
        self.C = C
        self.c_old = c_old
        self.velocity = velocity
        if mixed_space is None:
            flux_element = FiniteElement("RT", mesh.ufl_cell(), 1)
            scalar_element = FiniteElement("DG", mesh.ufl_cell(), 0)
            self.W = FunctionSpace(mesh, MixedElement([flux_element, scalar_element]))
        else:
            self.W = mixed_space
        elements = self.W.ufl_element().sub_elements()
        if (len(elements) != 2 or elements[0].degree() != 1
                or elements[1].degree() != 0
                or elements[0].value_shape() != (3,)):
            raise ValueError("mixed_space must contain RT degree 1 and DG degree 0.")

        if dirichlet_values is None:
            dirichlet_values = {int(nozzle_marker): Constant(0.0)}
        if inflow_values is None:
            inflow_values = {}
        self.dirichlet_values = dict(dirichlet_values)
        self.inflow_values = dict(inflow_values)
        self.noflux_markers = tuple(int(marker) for marker in noflux_markers)
        if set(self.noflux_markers).intersection(self.dirichlet_values):
            raise ValueError("A boundary cannot impose both concentration and zero diffusion flux.")
        if source is None:
            source = Constant(0.0)

        dx = Measure("dx", domain=mesh)
        dS = Measure("dS", domain=mesh)
        ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
        n = FacetNormal(mesh)
        scale = Constant(math.sqrt(self.dt * self.D))
        timestep = Constant(self.dt)
        r, c = TrialFunctions(self.W)
        v, s = TestFunctions(self.W)

        un_int = dot(avg(velocity), n("+"))
        advective_face_flux = un_int * avg(c) + 0.5 * abs(un_int) * jump(c)
        un = dot(velocity, n)
        outflow = 0.5 * (un + abs(un))
        inflow = 0.5 * (un - abs(un))

        self.a = (
            inner(r, v) * dx
            - scale * c * div(v) * dx
            + c * s * dx
            + scale * div(r) * s * dx
            - timestep * c * dot(velocity, grad(s)) * dx
            + timestep * advective_face_flux * jump(s) * dS
            + timestep * outflow * c * s * ds
        )
        self.L = c_old * s * dx + timestep * source * s * dx
        for marker, value in self.dirichlet_values.items():
            # r = -sqrt(dt*D) grad(c): prescribed c appears with this minus sign.
            self.L -= scale * value * dot(v, n) * ds(int(marker))
        for marker, value in self.inflow_values.items():
            self.L -= timestep * inflow * value * s * ds(int(marker))

        zero_flux = Constant((0.0, 0.0, 0.0))
        self.bcs = [DirichletBC(self.W.sub(0), zero_flux, boundaries, marker)
                    for marker in self.noflux_markers]
        self.assembler = SystemAssembler(self.a, self.L, self.bcs)
        self.state = Function(self.W)
        self.state.vector().zero()
        self.state.vector().apply("insert")
        self._copy_scalar = FunctionAssigner(C, self.W.sub(1))
        r_state, self.concentration = split(self.state)
        self.diffusive_flux = Constant(math.sqrt(self.D / self.dt)) * r_state

        # Some legacy Python builds expose the PETSc KSP overload but not the
        # C++ (communicator, method, preconditioner) constructor. Build the KSP
        # on this mesh's communicator and keep it alive alongside its wrapper.
        self._ksp = PETSc.KSP().create(comm=mesh.mpi_comm())
        self.solver = PETScKrylovSolver(self._ksp)
        self._ksp.setType("gmres")
        self._ksp.getPC().setType("jacobi")
        # Stop on the unpreconditioned residual of the original system.
        # Left Jacobi scaling can hide flux-block error at small diffusivity.
        # GMRES supports this norm with right preconditioning.
        self._ksp.setPCSide(PETSc.PC.Side.RIGHT)
        self._ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        self.solver.parameters["relative_tolerance"] = 1.0e-10
        self.solver.parameters["absolute_tolerance"] = 1.0e-14
        self.solver.parameters["maximum_iterations"] = 500
        self.solver.parameters["error_on_nonconvergence"] = True
        self.solver.parameters["nonzero_initial_guess"] = True
        self.A = PETScMatrix(mesh.mpi_comm())
        self.b = PETScVector(mesh.mpi_comm())
        self.last_iterations = 0
        self.last_true_residual = float("nan")
        self.last_relative_residual = float("nan")
        self.solve_calls = 0

    def solve(self, c_out):
        """Solve one step, copy c into ``c_out``, and return Krylov iterations.

        The previous auxiliary solution is only a linear-solver initial guess;
        restarting with an initially zero auxiliary state is valid.  No limiter
        or concentration clipping is applied.
        """
        self.assembler.assemble(self.A, self.b)
        self.solver.set_operator(self.A)
        iterations = int(self.solver.solve(self.state.vector(), self.b))
        self.state.vector().apply("insert")

        residual = self.b.copy()
        self.A.mult(self.state.vector(), residual)
        residual.axpy(-1.0, self.b)
        self.last_true_residual = float(residual.norm("l2"))
        rhs_norm = float(self.b.norm("l2"))
        self.last_relative_residual = self.last_true_residual / max(rhs_norm, 1.0e-30)
        self.last_iterations = iterations
        self.solve_calls += 1
        # Recompute the original-system residual independently: the Krylov
        # residual estimate can differ because of recurrence and roundoff.
        true_tolerance = max(1.0e-13, 1.0e-8 * rhs_norm)
        if (not math.isfinite(self.last_true_residual)
                or self.last_true_residual > true_tolerance):
            raise RuntimeError(
                "Mixed scalar true residual is too large: {:.6e} (relative {:.6e}); "
                "required <= {:.6e}, after {} iterations.".format(
                    self.last_true_residual, self.last_relative_residual,
                    true_tolerance, iterations
                )
            )
        self._copy_scalar.assign(c_out, self.state.sub(1))
        c_out.vector().apply("insert")
        return iterations
