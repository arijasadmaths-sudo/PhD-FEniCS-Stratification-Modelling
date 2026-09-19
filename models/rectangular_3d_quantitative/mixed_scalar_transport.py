"""Conservative RT0/DG0 advection--diffusion for legacy DOLFIN 2019.1.

The auxiliary variable is r = -sqrt(dt*D)*grad(c), so the physical outward
diffusive flux is q = sqrt(D/dt)*r.  RT0 means FEniCS's ``RT`` degree 1.
Unlike a two-point cell-diameter penalty, this mixed formulation represents
isotropic Fickian diffusion on general tetrahedral meshes.  It does not imply
a discrete maximum principle: callers must diagnose concentration extrema.
"""

import math

import numpy as np
from petsc4py import PETSc

from dolfin import (
    Constant, DirichletBC, FacetNormal, FiniteElement, Function,
    FunctionAssigner, FunctionSpace, Measure, MixedElement, MPI, PETScKrylovSolver,
    PETScMatrix, PETScVector, SystemAssembler, TestFunctions, TrialFunctions,
    as_backend_type, avg, div, dot, grad, inner, jump, split,
)


SCALAR_SOLVER_REVISION = "equilibrated-2026-09-17"


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
        # Keep these options separate from the flow and H(div) solvers.
        # Reorthogonalization reduces loss of the GMRES basis at small residuals.
        self._ksp.setOptionsPrefix("rect3dq_scalar_")
        options = PETSc.Options()
        options["rect3dq_scalar_ksp_gmres_cgs_refinement_type"] = "refine_always"
        self._ksp.setFromOptions()
        # The Krylov solve below operates on the symmetrically equilibrated
        # system. Its unpreconditioned norm is not the original-system norm;
        # the latter is recomputed and checked independently after unscaling.
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
        self.last_recovery_used = False
        self.recovery_count = 0
        self.solve_calls = 0
        self._scale = None
        self._inverse_scale = None
        self._minimum_scale = None
        if MPI.rank(self.mesh.mpi_comm()) == 0:
            print("MIXED SCALAR SOLVER: revision={}; D={:.6e}; "
                  "symmetric diagonal equilibration; original residual guard enabled.".format(
                      SCALAR_SOLVER_REVISION, self.D), flush=True)

    def _prepare_equilibration(self):
        """Form S_ii = 1/sqrt(A_ii) from this timestep's positive diagonal.

        RT face-flux and DG cell-mean coefficients have different dimensions.
        At sub-millimetre cell sizes, one-sided Jacobi leaves very unbalanced
        off-diagonal blocks. Solving (S A S)y = S b with x = S y balances both
        blocks without changing the finite-element equations.
        """
        if self._scale is None:
            self._scale = self.b.copy()
            self._inverse_scale = self.b.copy()
        self.A.get_diagonal(self._inverse_scale)
        diagonal = self._inverse_scale.get_local()
        valid = int(np.all(np.isfinite(diagonal)) and np.all(diagonal > 0.0))
        if not MPI.min(self.mesh.mpi_comm(), valid):
            raise RuntimeError("Mixed scalar matrix has a nonpositive or nonfinite diagonal.")
        root_diagonal = np.sqrt(diagonal)
        self._inverse_scale.set_local(root_diagonal)
        self._inverse_scale.apply("insert")
        self._scale.set_local(1.0 / root_diagonal)
        self._scale.apply("insert")
        largest_root = float(self._inverse_scale.norm("linf"))
        self._minimum_scale = 1.0 / largest_root

    def _solve_equilibrated(self, solution, rhs):
        """Temporarily scale A, solve, then restore A and the physical unknown.

        The RHS is copied, and only vectors are allocated in addition to A.
        Both the original matrix and the physical iterate are restored even
        when DOLFIN reports nonconvergence. Each call sets the operator after
        scaling, so Jacobi is rebuilt for the matrix actually being solved.
        """
        scale = as_backend_type(self._scale).vec()
        inverse = as_backend_type(self._inverse_scale).vec()
        scaled_rhs = rhs.copy()
        scaled_solution = solution.copy()
        as_backend_type(scaled_rhs).vec().pointwiseMult(
            scale, as_backend_type(rhs).vec())
        as_backend_type(scaled_solution).vec().pointwiseMult(
            inverse, as_backend_type(solution).vec())
        matrix = self.A.mat()
        matrix.diagonalScale(scale, scale)
        try:
            self.solver.set_operator(self.A)
            iterations = int(self.solver.solve(scaled_solution, scaled_rhs))
        finally:
            matrix.diagonalScale(inverse, inverse)
            as_backend_type(solution).vec().pointwiseMult(
                scale, as_backend_type(scaled_solution).vec())
            solution.apply("insert")
        if int(self._ksp.getConvergedReason()) <= 0:
            raise RuntimeError("Mixed scalar equilibrated solve did not converge.")
        return iterations

    def _solve_with_breakdown_recovery(self):
        """Use at most one equilibrated correction of the original residual.

        A correction is required after a GMRES breakdown or when a converged
        scaled solve has not yet met the original-system accuracy threshold.
        The correction must converge. The unchanged final residual guard is
        applied by solve() before copying concentration to the caller.
        """
        self.last_recovery_used = False
        breakdown = False
        try:
            first_iterations = self._solve_equilibrated(self.state.vector(), self.b)
        except RuntimeError:
            reason = int(self._ksp.getConvergedReason())
            if reason != int(PETSc.KSP.ConvergedReason.DIVERGED_BREAKDOWN):
                raise
            breakdown = True
            first_iterations = int(self._ksp.getIterationNumber())

        # Form b - A*x independently, including the latest failed iterate.
        # A is back in its original units and b has never been modified.
        self.state.vector().apply("insert")
        correction_rhs = self.b.copy()
        self.A.mult(self.state.vector(), correction_rhs)
        correction_rhs *= -1.0
        correction_rhs.axpy(1.0, self.b)
        before = float(correction_rhs.norm("l2"))
        if (not math.isfinite(before)
                or not math.isfinite(float(self.state.vector().norm("l2")))):
            raise RuntimeError("Mixed scalar solve has a non-finite iterate or residual.")
        true_tolerance = max(1.0e-13, 1.0e-8 * float(self.b.norm("l2")))
        if not breakdown and before <= true_tolerance:
            return first_iterations
        if breakdown and MPI.rank(self.mesh.mpi_comm()) == 0:
            print("MIXED SCALAR RECOVERY: D={:.6e}; call={}; original true "
                  "residual={:.6e}; attempting one equilibrated correction.".format(
                      self.D, self.solve_calls + 1, before), flush=True)

        correction = self.state.vector().copy()
        correction.zero()
        correction.apply("insert")
        previous_guess = self.solver.parameters["nonzero_initial_guess"]
        previous_atol = self.solver.parameters["absolute_tolerance"]
        previous_rtol = self.solver.parameters["relative_tolerance"]
        self.solver.parameters["nonzero_initial_guess"] = False
        # ||r_original|| <= ||S^-1|| * ||r_scaled||. Use this bound to ensure
        # both stopping tolerances can meet the original residual guard, with
        # a factor-ten margin. Neither tolerance is relaxed.
        scaled_target = 0.1 * true_tolerance * self._minimum_scale
        scaled_correction_rhs = correction_rhs.copy()
        as_backend_type(scaled_correction_rhs).vec().pointwiseMult(
            as_backend_type(self._scale).vec(), as_backend_type(correction_rhs).vec())
        scaled_rhs_norm = float(scaled_correction_rhs.norm("l2"))
        self.solver.parameters["absolute_tolerance"] = min(
            float(previous_atol), scaled_target)
        self.solver.parameters["relative_tolerance"] = min(
            float(previous_rtol), scaled_target / max(scaled_rhs_norm, 1.0e-30))
        try:
            extra_iterations = self._solve_equilibrated(correction, correction_rhs)
        finally:
            self.solver.parameters["nonzero_initial_guess"] = previous_guess
            self.solver.parameters["absolute_tolerance"] = previous_atol
            self.solver.parameters["relative_tolerance"] = previous_rtol

        self.state.vector().axpy(1.0, correction)
        self.state.vector().apply("insert")
        self.last_recovery_used = True
        return first_iterations + extra_iterations

    def solve(self, c_out):
        """Solve one step, copy c into ``c_out``, and return Krylov iterations.

        The previous auxiliary solution is only a linear-solver initial guess;
        restarting with an initially zero auxiliary state is valid.  No limiter
        or concentration clipping is applied.
        """
        self.assembler.assemble(self.A, self.b)
        self._prepare_equilibration()
        iterations = self._solve_with_breakdown_recovery()
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
        if self.last_recovery_used:
            self.recovery_count += 1
            if self.recovery_count <= 2 and MPI.rank(self.mesh.mpi_comm()) == 0:
                print("MIXED SCALAR EQUILIBRATED CORRECTION PASSED: D={:.6e}; original true "
                      "residual={:.6e}; required<={:.6e}; iterations={}.".format(
                          self.D, self.last_true_residual, true_tolerance,
                          iterations), flush=True)
        self._copy_scalar.assign(c_out, self.state.sub(1))
        c_out.vector().apply("insert")
        return iterations
