#!/usr/bin/env python3
"""Axisymmetric 70 cc/min: a 2-second local scalar-flux diagnostic.

Run: python3 axisymmetric_70_flux_2s.py --output NEW_EMPTY_DIRECTORY
Tests without FEniCS: python3 axisymmetric_70_flux_2s.py --self-test

Legacy FEniCS 2019.1, NumPy and SciPy; exactly one MPI rank.
The original mesh, physical parameters, 2 s inlet ramp, dt=0.001 s and
semi-implicit Oseen IPCS equations are retained. No molecular scalar
diffusion, clipping or scalar redistribution is applied.

The scalar now uses one signed, integrated normal volume flux per facet.
CG2 facet integrals are evaluated exactly by Simpson's rule in (s,z),
where s=r^2/2 and U=(r*u_r,u_z). All boundary fluxes remain fixed.
Interior fluxes are projected onto cellwise balanced fluxes by minimizing
the unweighted sum of squared facet-flux changes. This is a diagnostic
projection choice, not a claim of spatial convergence or an RT mass-norm
projection. Its magnitude and sign changes are recorded.

Backward-Euler upwind finite-volume transport of the DG0 cell averages
then preserves the scalar budget and bounds to solver tolerance. The
physical volume of each transformed triangle is 2*pi*area(s,z).
The velocity used in the momentum equations is NOT replaced by this
scalar-transport flux field. The resulting scalar feeds the next step's
buoyancy in the usual way.

A raw-flux shadow solve starts from the SAME previous scalar every step.
It isolates the flux projection's effect within this integrated-flux
scheme; it is not a separately evolved run, and it never feeds buoyancy.
The archived scheme used pointwise quadrature upwinding: on facets where
U.n changes sign, net-flux upwinding can differ even before projection.

Outputs: verification_configuration.json, verification_status.json,
verification_budget.csv and NPZ snapshots every 0.1 s, at completion,
and on a caught failure. Snapshots contain actual CG2 velocity DOFs,
pressure DOFs, cell averages and raw/corrected facet fluxes with their
time labels. This program starts from zero; it does not resume old runs.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time as wallclock
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import cKDTree

PRODUCTION_SOURCE_SHA256 = "274102538bede1243b140ebecfd9b6fe74bf49f36712af9cba72f974129ccce3"


class CG2FacetReader:
    """Serial CG2 DOF lookup, constructed once; no point searches in the loop."""

    def __init__(self, coordinates, component_dofs, transport):
        self.coordinates = np.asarray(coordinates, dtype=float).reshape((-1, 2))
        self.component_dofs = np.asarray(component_dofs, dtype=np.int64)
        self.transport = transport
        xy = transport.vertices
        edge_xy = xy[transport.edges]
        self.points = np.concatenate((edge_xy[:, 0], edge_xy.mean(axis=1), edge_xy[:, 1]))
        scale = np.ptp(xy, axis=0)
        if np.any(scale <= 0):
            raise ValueError("Degenerate coordinate extent")
        maps = []
        for ids in self.component_dofs:
            coords = self.coordinates[ids] / scale
            if len(np.unique(coords, axis=0)) != len(ids):
                raise ValueError("Duplicate component CG2 coordinates")
            distance, index = cKDTree(coords).query(self.points / scale)
            if np.max(distance) > 5e-12:
                raise RuntimeError("Could not match facet quadrature nodes to CG2 DOFs")
            maps.append(ids[index])
        self.maps = np.asarray(maps)

    def read(self, dofs):
        dofs = np.asarray(dofs)
        values = np.stack([dofs[ids] for ids in self.maps], axis=-1)
        n = len(self.transport.edges)
        v0, vm, v1 = values[:n], values[n:2*n], values[2*n:]
        means = (v0 + 4.0*vm + v1)/6.0
        flux = 2.0*np.pi*np.einsum("ij,ij->i", means, self.transport.normal_length)
        # Exact extrema of the quadratic normal trace on each facet.
        n0 = np.einsum("ij,ij->i", v0, self.transport.normal_length)
        nm = np.einsum("ij,ij->i", vm, self.transport.normal_length)
        n1 = np.einsum("ij,ij->i", v1, self.transport.normal_length)
        a = 2.0*(n0+n1-2.0*nm)
        b = n1-n0-a
        root = np.zeros_like(a)
        np.divide(-b, 2.0*a, out=root, where=a != 0)
        inner = (a != 0) & (root > 0) & (root < 1)
        ext = a*root*root+b*root+n0
        low = np.minimum(n0, n1)
        high = np.maximum(n0, n1)
        low[inner] = np.minimum(low[inner], ext[inner])
        high[inner] = np.maximum(high[inner], ext[inner])
        tol = 1e-12*max(float(np.max(np.abs(values))), 1e-30)*np.linalg.norm(
            self.transport.normal_length, axis=1)
        sign_changes = (low < -tol) & (high > tol)
        return flux, sign_changes


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(output, name, value):
    path = output / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(str(temp), str(path))


def _verification_refine_nodes(nodes, factor):
    if factor != 1:
        raise ValueError("This short diagnostic deliberately uses the original mesh")
    return np.asarray(nodes).copy()


def bridge_numpy_test():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    tr = BalancedFacetTransport(xy, np.array([[0, 1, 2], [0, 2, 3]]))
    scalar_nodes = np.vstack([xy, xy[tr.edges].mean(axis=1)])
    # Deliberately scrambled/interleaved vector DOF order.
    coords = np.repeat(scalar_nodes, 2, axis=0)
    ids = np.arange(len(coords)).reshape((-1, 2)).T
    rng = np.random.RandomState(42)
    order = rng.permutation(len(coords))
    inverse = np.argsort(order)
    coords = coords[order]
    ids = inverse[ids]
    reader = CG2FacetReader(coords, ids, tr)
    field = np.zeros(len(coords))
    field[ids[0]] = scalar_nodes[:, 0]**2 + 2*scalar_nodes[:, 1]
    field[ids[1]] = scalar_nodes[:, 0] - scalar_nodes[:, 1]**2
    flux, _ = reader.read(field)
    cell_net = np.bincount(tr.owner, weights=flux, minlength=2)
    ii = tr.interior
    cell_net -= np.bincount(tr.neighbor[ii], weights=flux[ii], minlength=2)
    # div(U)=2s-2z is linear: centroid quadrature is exact.
    expected = tr.cell_volumes*(2*tr.centroids[:, 0]-2*tr.centroids[:, 1])
    np.testing.assert_allclose(cell_net, expected, rtol=1e-12, atol=1e-12)
    print("PASS: scrambled CG2 component DOFs and exact quadratic facet integration")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", help="New/empty output directory; old files are protected")
    p.add_argument("--self-test", action="store_true", help="Run NumPy/SciPy tests without FEniCS")
    args = p.parse_args()
    if args.self_test:
        print("PASS: flux/transport tests", json.dumps(kernel_self_test(), sort_keys=True))
        bridge_numpy_test()
        sys.exit(0)
    if not args.output:
        p.error("--output is required for a run")
    output = Path(args.output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        p.error("Output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    args.output = output
    return args


"""NumPy/SciPy kernel for a conservative axisymmetric DG0 diagnostic.

Coordinates are (s, z), where s = r**2 / 2.  Physical cell volumes are
2*pi times triangle area.  A facet flux is an integrated physical volume
rate [m^3/s], positive out of its owner cell (into its neighbour, if any).
The caller supplies these rates; this module does not interpolate velocity.

The projection deliberately uses UNIT facet weights: it minimizes the sum
of squared changes to interior integrated volume fluxes.  It is a diagnostic
choice, not a claim of a physically optimal velocity reconstruction.  ALL
boundary rates remain bitwise unchanged.  No scalar redistribution, clipping,
or extra diffusion is applied.  First-order implicit upwinding itself has its
usual numerical diffusion.
"""

import math

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu


class BalancedFacetTransport:
    """Build topology and factor a component-anchored graph Laplacian once.

    ``interior`` and ``boundary`` are boolean facet masks. ``normal_length``
    is the outward owner normal times edge length in the (s,z) plane, without
    a 2*pi multiplier. ``boundary_fresh`` in the transport methods can be a
    scalar, an all-facet vector, or a boundary-only vector in mask order.
    Only its values at inflowing boundary facets are used.
    """

    def __init__(self, vertices, cells):
        self.vertices = np.asarray(vertices, dtype=float).copy()
        cell_input = np.asarray(cells)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (Nv, 2) in (s,z) coordinates")
        if not np.all(np.isfinite(self.vertices)):
            raise ValueError("vertices must be finite")
        if cell_input.ndim != 2 or cell_input.shape[1] != 3 or not len(cell_input):
            raise ValueError("cells must have nonempty shape (Nc, 3)")
        if not np.issubdtype(cell_input.dtype, np.integer):
            raise ValueError("cells must contain integer vertex indices")
        self.cells = cell_input.astype(np.int64, copy=True)
        if np.min(self.cells) < 0 or np.max(self.cells) >= len(self.vertices):
            raise ValueError("cell vertex index is out of range")
        points = self.vertices[self.cells]
        side1, side2 = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
        twice_area = np.abs(side1[:, 0] * side2[:, 1] - side1[:, 1] * side2[:, 0])
        if np.any(twice_area <= 0) or not np.all(np.isfinite(twice_area)):
            raise ValueError("mesh contains a degenerate triangle")
        self.cell_volumes = np.pi * twice_area
        self.centroids = np.mean(points, axis=1)
        self.n_cells = len(self.cells)

        edge_ids, edges, owner, neighbor = {}, [], [], []
        for ci, triangle in enumerate(self.cells):
            for local_a, local_b in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((int(triangle[local_a]), int(triangle[local_b]))))
                if edge not in edge_ids:
                    edge_ids[edge] = len(edges)
                    edges.append(edge)
                    owner.append(ci)
                    neighbor.append(-1)
                else:
                    fi = edge_ids[edge]
                    if neighbor[fi] != -1:
                        raise ValueError("mesh contains a nonmanifold edge")
                    neighbor[fi] = ci
        self.edges = np.asarray(edges, dtype=np.int64)
        self.owner = np.asarray(owner, dtype=np.int64)
        self.neighbor = np.asarray(neighbor, dtype=np.int64)
        self.interior = self.neighbor >= 0
        self.boundary = ~self.interior
        self.n_facets = len(self.edges)
        self._interior_ids = np.flatnonzero(self.interior)
        self._boundary_ids = np.flatnonzero(self.boundary)
        edge_points = self.vertices[self.edges]
        tangents = edge_points[:, 1] - edge_points[:, 0]
        normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
        midpoints = np.mean(edge_points, axis=1)
        dot = np.einsum("ij,ij->i", normals, midpoints - self.centroids[self.owner])
        normals[dot < 0] *= -1
        self.normal_length = normals
        if np.any(dot == 0):
            raise ValueError("cannot orient a degenerate facet")
        if np.any(np.einsum(
                "ij,ij->i", normals[self.interior],
                self.centroids[self.neighbor[self.interior]] - midpoints[self.interior]) <= 0):
            raise ValueError("adjacent triangles overlap or have inconsistent topology")

        ni = len(self._interior_ids)
        ii = np.arange(ni)
        oi, nj = self.owner[self.interior], self.neighbor[self.interior]
        self._B = sparse.csc_matrix(
            (np.concatenate((np.ones(ni), -np.ones(ni))),
             (np.concatenate((oi, nj)), np.concatenate((ii, ii)))),
            shape=(self.n_cells, ni))
        laplacian = (self._B @ self._B.T).tocsc()
        self.n_components, self._component = csgraph.connected_components(
            laplacian, directed=False, return_labels=True)
        # Put unavoidable floating-point compatibility error in a large cell.
        self.anchors = np.asarray([
            group[np.argmax(self.cell_volumes[group])]
            for group in (np.flatnonzero(self._component == k)
                          for k in range(self.n_components))], dtype=np.int64)
        free_mask = np.ones(self.n_cells, dtype=bool)
        free_mask[self.anchors] = False
        self._free = np.flatnonzero(free_mask)
        self._laplace_lu = (splu(laplacian[self._free][:, self._free].tocsc())
                            if len(self._free) else None)
        boundary_components = self._component[self.owner[self.boundary]]
        self._component_boundary_ids = [
            self._boundary_ids[boundary_components == k]
            for k in range(self.n_components)]
        self.balance_rtol = 2.0e-11
        self.compatibility_rtol = 512.0 * np.finfo(float).eps
        self.linear_rtol = 5.0e-11
        self.budget_rtol = 5.0e-11
        self.bounds_atol = 2.0e-10
        self.last_candidate = None

    def _flux_vector(self, flux):
        flux = np.asarray(flux, dtype=float)
        if flux.shape != (self.n_facets,) or not np.all(np.isfinite(flux)):
            raise ValueError("flux must be a finite all-facet vector in m^3/s")
        return flux

    def cell_net_flux(self, flux):
        """Return each cell's signed outward integrated volume rate [m^3/s]."""
        flux = self._flux_vector(flux)
        net = np.bincount(self.owner, weights=flux, minlength=self.n_cells)
        net -= np.bincount(self.neighbor[self.interior],
                           weights=flux[self.interior], minlength=self.n_cells)
        return net

    def _flux_scale(self, flux):
        incident = np.bincount(self.owner, weights=np.abs(flux), minlength=self.n_cells)
        incident += np.bincount(self.neighbor[self.interior],
                                weights=np.abs(flux[self.interior]), minlength=self.n_cells)
        return float(np.max(incident, initial=0.0))

    @staticmethod
    def _ratio(value, scale):
        return float(value / scale) if scale > 0 else (0.0 if value == 0 else float("inf"))

    def _graph_potential(self, rhs):
        potential = np.zeros(self.n_cells)
        if self._laplace_lu is not None:
            potential[self._free] = self._laplace_lu.solve(rhs[self._free])
        return potential

    def balance(self, raw_flux):
        """Project interior rates onto cell balance; leave boundary rates fixed.

        An incompatible prescribed boundary budget raises ValueError.  The
        compatibility tolerance is relative to the sum of absolute boundary
        rates within each connected component and has no dimensional floor.
        Returned correction metrics expose the size of this diagnostic change.
        """
        raw = self._flux_vector(raw_flux)
        component_net, component_tol = [], []
        for k, ids in enumerate(self._component_boundary_ids):
            net = math.fsum(float(value) for value in raw[ids])
            boundary_l1 = math.fsum(float(value) for value in np.abs(raw[ids]))
            tolerance = self.compatibility_rtol * boundary_l1
            component_net.append(net)
            component_tol.append(tolerance)
            if abs(net) > tolerance:
                raise ValueError(
                    "Fixed boundary fluxes are incompatible in connected component "
                    f"{k}: net outward {net:.17g} m^3/s; tolerance "
                    f"{tolerance:.6g} m^3/s. Cannot balance every cell without "
                    "changing boundary data.")
        balanced = raw.copy()
        raw_net = self.cell_net_flux(raw)
        iterations = 0
        # Each correction is in range(B.T), so the final correction retains
        # the same unit-weight minimum-L2 projection to solver accuracy.
        for iteration in range(8):
            net = self.cell_net_flux(balanced)
            scale = self._flux_scale(balanced)
            net_inf = float(np.max(np.abs(net), initial=0.0))
            if net_inf <= self.balance_rtol * scale:
                break
            potential = self._graph_potential(net)
            balanced[self.interior] -= np.asarray(self._B.T @ potential).ravel()
            iterations += 1
        net = self.cell_net_flux(balanced)
        scale = self._flux_scale(balanced)
        residual_inf = float(np.max(np.abs(net), initial=0.0))
        if not np.all(np.isfinite(balanced)) or residual_inf > self.balance_rtol * scale:
            raise RuntimeError(
                "Interior flux projection failed cell balance with fixed boundaries: "
                f"max cell net {residual_inf:.6g} m^3/s, scale {scale:.6g} m^3/s")
        if not np.array_equal(balanced[self.boundary], raw[self.boundary]):
            raise RuntimeError("internal error: projection modified a boundary rate")
        correction = balanced[self.interior] - raw[self.interior]
        correction_l2 = float(np.linalg.norm(correction))
        raw_l2 = float(np.linalg.norm(raw[self.interior]))
        correction_max = float(np.max(np.abs(correction), initial=0.0))
        raw_max = float(np.max(np.abs(raw[self.interior]), initial=0.0))
        metrics = {
            "projection": "unit_weight_minimum_L2_interior_integrated_flux",
            "graph_components": int(self.n_components),
            "projection_solve_count": int(iterations),
            "raw_cell_net_max_m3_s": float(np.max(np.abs(raw_net), initial=0.0)),
            "balanced_cell_net_max_m3_s": residual_inf,
            "balanced_cell_net_scaled": self._ratio(residual_inf, scale),
            "flux_scale_m3_s": scale,
            "correction_l2_m3_s": correction_l2,
            "correction_l2_relative": self._ratio(correction_l2, raw_l2),
            "correction_max_m3_s": correction_max,
            "correction_max_relative": self._ratio(correction_max, raw_max),
            "interior_sign_flip_count": int(np.count_nonzero(
                ((raw[self.interior] > 0) & (balanced[self.interior] < 0)) |
                ((raw[self.interior] < 0) & (balanced[self.interior] > 0)))),
            "boundary_flux_net_m3_s": math.fsum(component_net),
            "component_boundary_flux_net_m3_s": component_net,
            "component_boundary_compatibility_tolerance_m3_s": component_tol,
            "boundary_flux_unchanged": True,
        }
        return balanced, metrics

    def _boundary_values(self, boundary_fresh, flux):
        value = np.asarray(boundary_fresh, dtype=float)
        if value.ndim == 0:
            value = np.full(len(self._boundary_ids), float(value))
        elif value.shape == (self.n_facets,):
            value = value[self.boundary].copy()
        elif value.shape == (len(self._boundary_ids),):
            value = value.copy()
        else:
            raise ValueError("boundary_fresh must be scalar, all-facet, or boundary-only")
        inflow = flux[self.boundary] < 0
        if not np.all(np.isfinite(value[inflow])):
            raise ValueError("inflow boundary fresh fractions must be finite")
        if np.any(value[inflow] < 0) or np.any(value[inflow] > 1):
            raise ValueError("inflow boundary fresh fractions must lie in [0,1]")
        # Unused outflow values do not participate, even if supplied as NaN.
        value[~inflow] = 0.0
        return value

    def advance(self, f_old, flux, dt, boundary_fresh):
        """One implicit upwind step with cell-balance and maximum-principle guards."""
        return self._advance(f_old, flux, dt, boundary_fresh, enforce_balance=True)

    def shadow_advance(self, f_old, raw_flux, dt, boundary_fresh):
        """One independent raw-flux diagnostic; NEVER feed its field forward.

        This is the identical integrated-flux scheme used by ``advance``.
        Only local volume-balance and maximum-principle guards are disabled.
        Finite-solution, linear-residual, and conservative scalar-budget checks
        remain active.  The caller may catch/report a shadow failure while
        retaining a successful balanced primary update.
        """
        return self._advance(f_old, raw_flux, dt, boundary_fresh, enforce_balance=False)

    def _advance(self, f_old, flux, dt, boundary_fresh, enforce_balance):
        self.last_candidate = None
        old = np.asarray(f_old, dtype=float)
        if old.shape != (self.n_cells,) or not np.all(np.isfinite(old)):
            raise ValueError("f_old must be a finite cell vector")
        if np.min(old) < -self.bounds_atol or np.max(old) > 1 + self.bounds_atol:
            raise ValueError("f_old is outside fresh-fraction bounds")
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        dt = float(dt)
        flux = self._flux_vector(flux)
        boundary_value = self._boundary_values(boundary_fresh, flux)
        cell_net = self.cell_net_flux(flux)
        net_inf = float(np.max(np.abs(cell_net), initial=0.0))
        flux_scale = self._flux_scale(flux)
        if enforce_balance and net_inf > self.balance_rtol * flux_scale:
            raise ValueError(
                f"advance requires locally balanced flux: max cell net {net_inf:.6g} m^3/s")

        # M = diag(V) + dt*A; A is conservative upwind advection [m^3/s].
        # An interior donor contributes +|q| to its diagonal and -|q| to
        # the receiver's row in the same donor column.
        q = flux[self.interior]
        oi, nj = self.owner[self.interior], self.neighbor[self.interior]
        donor = np.where(q >= 0, oi, nj)
        receiver = np.where(q >= 0, nj, oi)
        rates = np.abs(q)
        qb = flux[self.boundary]
        ob = self.owner[self.boundary]
        incoming = qb < 0
        outgoing = qb > 0
        diagonal_rates = np.bincount(donor, weights=rates, minlength=self.n_cells)
        diagonal_rates += np.bincount(ob[outgoing], weights=qb[outgoing], minlength=self.n_cells)
        inflow_source = np.bincount(
            ob[incoming], weights=-qb[incoming] * boundary_value[incoming],
            minlength=self.n_cells)
        matrix = (sparse.diags(self.cell_volumes + dt * diagonal_rates, format="csc") +
                  sparse.csc_matrix((-dt * rates, (receiver, donor)),
                                    shape=(self.n_cells, self.n_cells)))
        rhs = self.cell_volumes * old + dt * inflow_source
        if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(rhs)):
            raise RuntimeError("implicit transport matrix or RHS overflowed")
        if dt == 0:
            new = old.copy()
        else:
            factor = splu(matrix)
            new = factor.solve(rhs)
            # Iterative refinement improves the conservative solve without
            # clipping, scalar redistribution, or changes to its equation.
            for _ in range(3):
                residual = matrix @ new - rhs
                row_scale = abs(matrix) @ np.abs(new) + np.abs(rhs)
                relative = np.divide(np.abs(residual), row_scale,
                                     out=np.zeros_like(residual), where=row_scale > 0)
                if float(np.max(relative, initial=0.0)) <= 32 * np.finfo(float).eps:
                    break
                new -= factor.solve(residual)
        # Save the exact candidate before every post-solve guard, including
        # when the caller needs to preserve an attempted failed scalar step.
        self.last_candidate = new.copy()
        if not np.all(np.isfinite(new)):
            raise RuntimeError("implicit transport produced a nonfinite scalar")
        residual = matrix @ new - rhs
        row_scale = abs(matrix) @ np.abs(new) + np.abs(rhs)
        relative = np.divide(np.abs(residual), row_scale,
                             out=np.zeros_like(residual), where=row_scale > 0)
        linear_scaled = float(np.max(relative, initial=0.0))
        if linear_scaled > self.linear_rtol:
            raise RuntimeError(f"transport linear residual failed: {linear_scaled:.6g}")

        old_inventory = math.fsum(float(v) * float(f) for v, f in zip(self.cell_volumes, old))
        new_inventory = math.fsum(float(v) * float(f) for v, f in zip(self.cell_volumes, new))
        inflow_rate = math.fsum(float(-qv) * float(fv)
                               for qv, fv in zip(qb[incoming], boundary_value[incoming]))
        outflow_rate = math.fsum(float(qv) * float(fv)
                                for qv, fv in zip(qb[outgoing], new[ob[outgoing]]))
        # Sum local inventory changes directly to reduce cancellation in
        # nearly steady inventories; report this exact algebraic budget.
        inventory_change = math.fsum(float(v) * float(fn - fo)
                                    for v, fn, fo in zip(self.cell_volumes, new, old))
        budget_residual = math.fsum((inventory_change, dt * outflow_rate, -dt * inflow_rate))
        budget_scale = abs(old_inventory) + abs(new_inventory) + dt * (abs(inflow_rate) + abs(outflow_rate))
        budget_scaled = self._ratio(abs(budget_residual), budget_scale)
        if budget_scaled > self.budget_rtol:
            raise RuntimeError(
                f"transport scalar budget failed: residual {budget_residual:.6g} m^3, "
                f"scaled {budget_scaled:.6g}")
        lower = min(float(np.min(old)), float(np.min(boundary_value[incoming], initial=np.inf)))
        upper = max(float(np.max(old)), float(np.max(boundary_value[incoming], initial=-np.inf)))
        new_min, new_max = float(np.min(new)), float(np.max(new))
        lower_violation = max(0.0, lower - new_min)
        upper_violation = max(0.0, new_max - upper)
        if enforce_balance and max(lower_violation, upper_violation) > self.bounds_atol:
            raise RuntimeError(
                "transport maximum principle failed without clipping: "
                f"new range [{new_min:.17g}, {new_max:.17g}], "
                f"allowed [{lower:.17g}, {upper:.17g}]")
        metrics = {
            "scheme": "implicit_upwind_integrated_facet_flux",
            "balanced_guards_enabled": bool(enforce_balance),
            "dt_s": dt,
            "fresh_inventory_old_m3": old_inventory,
            "fresh_inventory_new_m3": new_inventory,
            "fresh_inventory_change_m3": inventory_change,
            "fresh_boundary_inflow_m3_s": inflow_rate,
            "fresh_boundary_outflow_m3_s": outflow_rate,
            "fresh_boundary_net_outflow_m3_s": outflow_rate - inflow_rate,
            "fresh_budget_residual_m3": budget_residual,
            "fresh_budget_residual_scaled": budget_scaled,
            "linear_residual_max_m3": float(np.max(np.abs(residual), initial=0.0)),
            "linear_residual_scaled": linear_scaled,
            "cell_net_max_m3_s": net_inf,
            "cell_net_scaled": self._ratio(net_inf, flux_scale),
            "fresh_min": new_min,
            "fresh_max": new_max,
            "maximum_principle_lower": lower,
            "maximum_principle_upper": upper,
            "maximum_principle_lower_violation": lower_violation,
            "maximum_principle_upper_violation": upper_violation,
        }
        return new, metrics


def kernel_self_test():
    """Exercise geometry, projection, budgets, bounds and shadow diagnostics.

    Requires only NumPy/SciPy.  Raises AssertionError on failure and returns
    concise test counts/maximum errors on success.
    """
    vertices = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    kernel = BalancedFacetTransport(vertices, cells)
    assert np.allclose(kernel.cell_volumes, np.pi)
    assert np.allclose(kernel.normal_length.sum(axis=0),
                       kernel.normal_length[kernel.interior].sum(axis=0))
    exact = 2 * np.pi * (kernel.normal_length @ np.array([0.7, -0.2]))
    raw = exact.copy()
    raw[kernel.interior] += 9.3
    corrected, projection_metrics = kernel.balance(raw)
    assert np.array_equal(corrected[kernel.boundary], raw[kernel.boundary])
    np.testing.assert_allclose(corrected, exact, rtol=2e-13, atol=2e-13)
    assert projection_metrics["projection_solve_count"] > 0
    bad = raw.copy()
    bad[np.flatnonzero(kernel.boundary)[0]] += 0.1
    try:
        kernel.balance(bad)
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("incompatible fixed boundary data was accepted")

    constant_error = 0.0
    budget_error = 0.0
    step_count = 0
    for value in (0.0, 0.1, 0.9, 1.0):
        for dt in (0.0, 1e-9, 0.3, 20.0, 1e6):
            new, info = kernel.advance(np.full(2, value), corrected, dt, value)
            constant_error = max(constant_error, float(np.max(np.abs(new - value))))
            np.testing.assert_allclose(new, value, rtol=5e-12, atol=5e-12)
            budget_error = max(budget_error, info["fresh_budget_residual_scaled"])
            step_count += 1
    for dt in (1e-9, 0.07, 12.0, 1e6):
        new, info = kernel.advance(np.array([0.1, 0.9]), corrected, dt, 1.0)
        assert new.min() >= 0.1 - 1e-12 and new.max() <= 1.0 + 1e-12
        budget_error = max(budget_error, info["fresh_budget_residual_scaled"])
        step_count += 1
    # The shadow uses the same algebra but can expose overshoot caused by
    # raw cell volume imbalance; the next primary step starts from old.
    old = np.full(2, 0.9)
    shadow, sinfo = kernel.shadow_advance(old, raw, 3.0, 0.9)
    assert not sinfo["balanced_guards_enabled"]
    assert np.max(np.abs(shadow - 0.9)) > 0.01
    primary, _ = kernel.advance(old, corrected, 3.0, 0.9)
    np.testing.assert_allclose(primary, 0.9, atol=2e-13)

    # A closed four-cell circulation retains its cycle component under the
    # unit-weight graph projection and conserves inventory at every dt.
    cycle_vertices = np.vstack((vertices, [[0.5, 0.5]]))
    cycle_cells = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    cycle = BalancedFacetTransport(cycle_vertices, cycle_cells)
    circulation = np.zeros(cycle.n_facets)
    for fi in np.flatnonzero(cycle.interior):
        a, b = cycle.owner[fi], cycle.neighbor[fi]
        circulation[fi] = 2.0 if (b - a) % 4 == 1 else -2.0
    np.testing.assert_allclose(cycle.cell_net_flux(circulation), 0, atol=1e-14)
    potential = np.array([0.2, -0.3, 0.7, -0.1])
    distorted = circulation.copy()
    distorted[cycle.interior] += np.asarray(cycle._B.T @ potential).ravel()
    recovered, _ = cycle.balance(distorted)
    np.testing.assert_allclose(recovered, circulation, atol=5e-13)
    for dt in (1e-9, 0.3, 300.0, 1000.0):
        new, info = cycle.advance(np.array([0., 0.1, 0.9, 1.]), recovered, dt, 0.0)
        np.testing.assert_allclose(np.dot(cycle.cell_volumes, new),
                                   np.dot(cycle.cell_volumes, [0., 0.1, 0.9, 1.]),
                                   rtol=4e-10, atol=1e-12)
        assert new.min() >= -1e-12 and new.max() <= 1 + 1e-12
        budget_error = max(budget_error, info["fresh_budget_residual_scaled"])
        step_count += 1

    # Vertex ordering and owner changes must not change physical transport.
    reference, _ = kernel.advance(np.array([0.1, 0.9]), exact, 0.7, 1.0)
    orientation_count = 0
    for permutation in ((0, 1, 2), (0, 2, 1), (2, 0, 1), (1, 0, 2)):
        for reverse_cells in (False, True):
            test_cells = cells[:, permutation]
            if reverse_cells:
                test_cells = test_cells[::-1]
            test = BalancedFacetTransport(vertices, test_cells)
            q = 2 * np.pi * (test.normal_length @ np.array([0.7, -0.2]))
            q, _ = test.balance(q)
            test_old = np.array([0.1, 0.9])[::-1] if reverse_cells else np.array([0.1, 0.9])
            new, _ = test.advance(test_old, q, 0.7, 1.0)
            if reverse_cells:
                new = new[::-1]
            np.testing.assert_allclose(new, reference, rtol=2e-13, atol=2e-13)
            orientation_count += 1

    # Disconnected domains require compatibility in each component, even
    # when opposite incompatible component budgets cancel globally.
    disconnected_vertices = np.vstack((vertices, vertices + [3., 0.]))
    disconnected_cells = np.vstack((cells, cells + len(vertices)))
    disconnected = BalancedFacetTransport(disconnected_vertices, disconnected_cells)
    q = np.zeros(disconnected.n_facets)
    q[disconnected._component_boundary_ids[0][0]] = 1.0
    q[disconnected._component_boundary_ids[1][0]] = -1.0
    try:
        disconnected.balance(q)
    except ValueError:
        pass
    else:
        raise AssertionError("disconnected component incompatibility was accepted")
    zero, _ = disconnected.balance(np.zeros(disconnected.n_facets))
    unchanged, _ = disconnected.advance(np.array([0., 0.1, 0.9, 1.]), zero, 100., 0.)
    np.testing.assert_array_equal(unchanged, [0., 0.1, 0.9, 1.])
    return {
        "passed": True,
        "implicit_steps_checked": step_count,
        "orientation_variants_checked": orientation_count,
        "constant_state_max_error": constant_error,
        "budget_scaled_max_error": budget_error,
        "projection_choice": "unit_weight_minimum_L2_interior_integrated_flux",
    }




ARGS = parse_args()
OUTPUT = ARGS.output
state = {"status": "starting", "target_time_s": 2.0, "case": "balanced_flux_2s"}
write_json(OUTPUT, "verification_status.json", state)
budget_handle = None
snapshot_ready = False
try:
    import dolfin
    from fenics import *
    import numpy as np
    if not dolfin.__version__.startswith("2019.1"):
        raise RuntimeError("Use legacy FEniCS/DOLFIN 2019.1")
    comm = MPI.comm_world
    rank = MPI.rank(comm)
    if MPI.size(comm) != 1:
        raise RuntimeError("Run with exactly one MPI task")
    size = 1
    parameters["std_out_all_processes"] = False
    parameters["ghost_mode"] = "shared_facet"
    parameters["linear_algebra_backend"] = "PETSc"
    CASE = {"refinement": 1}
    dt_value = 0.001
    T_final = 2.0
    num_steps = 2000
    ramp_time = 2.0
    wall_start = wallclock.time()

    
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




    transport = BalancedFacetTransport(mesh.coordinates(), mesh.cells())
    ncells = mesh.num_cells()
    cell_dofs = np.asarray([F.dofmap().cell_dofs(i)[0] for i in range(ncells)], dtype=np.int64)
    if len(np.unique(cell_dofs)) != ncells:
        raise RuntimeError("DG0 cell/DOF map is not a permutation")
    volume_dofs = assemble(Constant(2.0*np.pi)*psi*dx_scalar).get_local()
    np.testing.assert_allclose(volume_dofs[cell_dofs], transport.cell_volumes, rtol=1e-12, atol=0)
    if ncells != 12546:
        raise RuntimeError("Unexpected mesh: this diagnostic must use 12546 cells")
    velocity_coords = V.tabulate_dof_coordinates().reshape((-1, 2))
    velocity_ids = np.asarray([V.sub(k).dofmap().dofs() for k in range(2)], dtype=np.int64)
    pressure_coords = P.tabulate_dof_coordinates().reshape((-1, 2))
    facet_reader = CG2FacetReader(velocity_coords, velocity_ids, transport)
    edge_xy = mesh.coordinates()[transport.edges]
    facet_tags = np.zeros(len(transport.edges), dtype=np.int64)
    eps_boundary = 1e-13
    bottom_edge = transport.boundary & np.all(np.abs(edge_xy[:, :, 1]) < eps_boundary, axis=1)
    nozzle_edge = bottom_edge & np.all(edge_xy[:, :, 0] <= S_nozzle+eps_boundary, axis=1)
    facet_tags[bottom_edge] = 2
    facet_tags[nozzle_edge] = 1
    facet_tags[transport.boundary & np.all(np.abs(edge_xy[:, :, 1]-H) < eps_boundary, axis=1)] = 3
    facet_tags[transport.boundary & np.all(np.abs(edge_xy[:, :, 0]-S) < eps_boundary, axis=1)] = 4
    facet_tags[transport.boundary & np.all(np.abs(edge_xy[:, :, 0]) < eps_boundary, axis=1)] = 5
    if np.any(facet_tags[transport.boundary] == 0):
        raise RuntimeError("Unrecognized boundary facet")
    boundary_fresh = np.zeros(len(transport.edges))
    boundary_fresh[facet_tags == 1] = 1.0

    def cell_net(flux):
        out = np.bincount(transport.owner, weights=flux, minlength=ncells)
        ii = transport.interior
        out -= np.bincount(transport.neighbor[ii], weights=flux[ii], minlength=ncells)
        return out

    # Test the live legacy DOF map and integration, not just the NumPy kernel.
    U_check = interpolate(Expression(("x[0]*x[0]+2*x[1]", "x[0]-x[1]*x[1]"), degree=2), V)
    check_flux, _ = facet_reader.read(U_check.vector().get_local())
    check_net = cell_net(check_flux)
    assembled_net = assemble(Constant(2.0*np.pi)*div(U_check)*psi*dx_scalar).get_local()[cell_dofs]
    np.testing.assert_allclose(check_net, assembled_net, rtol=2e-8, atol=2e-16)
    nnormal = FacetNormal(mesh)
    for tag in range(1, 6):
        assembled_boundary = assemble(Constant(2.0*np.pi)*dot(U_check, nnormal)*ds_sub(tag))
        np.testing.assert_allclose(check_flux[facet_tags == tag].sum(), assembled_boundary, rtol=1e-11, atol=1e-14)
    del U_check
    print("PASS: actual FEniCS CG2 mapping, DG0 cell volumes and facet/cell integration", flush=True)

    solver_Ustar = LUSolver()
    A2_const = assemble(a2)
    bc_dp.apply(A2_const)
    solver_dp = LUSolver()
    solver_dp.set_operator(A2_const)
    A3_const = assemble(a3)
    for bc in bcu:
        bc.apply(A3_const)
    solver_Ucorr = LUSolver()
    solver_Ucorr.set_operator(A3_const)

    config = {
        "case": "balanced_flux_2s", "target_time_s": T_final, "dt_s": dt_value,
        "num_steps": num_steps, "Q_cc_min": Q_target_cc_per_min, "Q_m3_s": Q_analytic,
        "R_m": R, "H_m": H, "nozzle_diameter_m": nozzle_diameter,
        "rho_fresh_kg_m3": rho_fresh, "rho_salt_kg_m3": rho_salt,
        "g_m_s2": 9.81, "nu_m2_s": 1e-6, "ramp_time_s": ramp_time,
        "radial_nodes_m": r_nodes_used, "vertical_nodes_m": z_nodes_used,
        "mesh_cells": ncells, "mesh_vertices": mesh.num_vertices(), "mpi_ranks": 1,
        "mesh_refinement": 1, "fresh_start": True, "auto_restart": False,
        "momentum": "Unchanged production Oseen IPCS, Stage-1 matrix rebuilt each step",
        "scalar": "Backward-Euler upwind DG0 cell averages using one integrated flux per facet",
        "flux_projection": "Minimum unweighted Euclidean norm interior-flux correction; all boundary fluxes fixed",
        "scalar_correction": "NONE: neither clipping nor scalar redistribution",
        "scalar_diffusion": "No molecular diffusion operator",
        "shadow": "One-step unbalanced integrated-flux solve from the same previous scalar; not independently evolved and never fed back",
        "facet_orientation": "Interior positive owner-to-neighbor; boundary positive outward from owner",
        "volume_measure": "dV=2*pi*ds*dz with s=r^2/2; fluxes in m3/s",
        "source_base_sha256": PRODUCTION_SOURCE_SHA256,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dolfin_version": dolfin.__version__, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "claim_scope": "Two-second startup diagnostic, not mesh convergence or validation of 50/1000 s results",
    }
    write_json(OUTPUT, "verification_configuration.json", config)
    accepted_step = 0
    accepted_time = 0.0
    attempted_step = 0
    attempted_time = 0.0
    velocity_time = 0.0
    flux_time = 0.0
    raw_flux = np.zeros(len(transport.edges))
    balanced_flux = raw_flux.copy()
    previous_fresh = np.zeros(ncells)
    shadow_fresh = previous_fresh.copy()
    shadow_success = True
    candidate_fresh = None
    sign_change_faces = np.zeros(len(transport.edges), dtype=bool)
    facet_trace_available = True
    flow_stages = {key: {"time_s": 0.0, "status": "initial"}
                   for key in ("tentative_velocity", "pressure", "corrected_velocity")}
    last_row = None
    cumulative_in = 0.0
    cumulative_out = 0.0
    max_budget_relative = 0.0
    max_flux_change_relative = 0.0

    def save_snapshot(name):
        # Explicit time labels prevent an attempted flow state being confused
        # with the last accepted scalar if a step fails before scalar acceptance.
        fresh = fresh_n.vector().get_local()[cell_dofs]
        last_candidate = getattr(transport, "last_candidate", None)
        candidate = candidate_fresh if candidate_fresh is not None else last_candidate
        if candidate is None:
            candidate = np.full(ncells, np.nan)
        def stage_dofs(function, stage):
            values = function.vector().get_local()
            return values if flow_stages[stage]["status"] in ("initial", "complete") else np.full_like(values, np.nan)
        target = OUTPUT/name
        temp = target.with_suffix(".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle,
                case=np.asarray("balanced_flux_2s"), time_s=np.asarray(accepted_time), step=np.asarray(accepted_step),
                attempted_step=np.asarray(attempted_step), attempted_time_s=np.asarray(attempted_time),
                dt_s=np.asarray(dt_value), Q_cc_min=np.asarray(Q_target_cc_per_min),
                geometry_s_z=mesh.coordinates().copy(), topology=mesh.cells().copy(),
                cell_volumes_m3=transport.cell_volumes, cell_dofs=cell_dofs,
                fresh=fresh, fresh_candidate=np.asarray(candidate), fresh_previous=previous_fresh,
                fresh_shadow=shadow_fresh, shadow_success=np.asarray(shadow_success),
                shadow_start_time_s=np.asarray(max(0.0, attempted_time-dt_value)),
                shadow_candidate_time_s=np.asarray(attempted_time),
                velocity_dof_coordinates_s_z=velocity_coords,
                velocity_component_dofs=velocity_ids,
                velocity_star_dofs=stage_dofs(U_star, "tentative_velocity"),
                velocity_new_dofs=stage_dofs(U_new, "corrected_velocity"),
                velocity_time_s=np.asarray(velocity_time),
                velocity_star_time_s=np.asarray(flow_stages["tentative_velocity"]["time_s"]),
                velocity_new_time_s=np.asarray(flow_stages["corrected_velocity"]["time_s"]),
                pressure_time_s=np.asarray(flow_stages["pressure"]["time_s"]),
                flow_stage_status_json=np.asarray(json.dumps(flow_stages, sort_keys=True)),
                pressure_dof_coordinates_s_z=pressure_coords,
                pressure_dofs=stage_dofs(p_new, "pressure"),
                pressure_increment_dofs=stage_dofs(dp_new, "pressure"),
                facet_vertices=transport.edges, facet_owner=transport.owner, facet_neighbor=transport.neighbor,
                facet_boundary_tag=facet_tags, facet_normal_length_s_z=transport.normal_length,
                facet_flux_raw_m3_s=raw_flux, facet_flux_balanced_m3_s=balanced_flux,
                facet_raw_trace_changes_sign=sign_change_faces, facet_trace_available=np.asarray(facet_trace_available),
                flux_time_s=np.asarray(flux_time),
                cell_net_raw_m3_s=cell_net(raw_flux), cell_net_balanced_m3_s=cell_net(balanced_flux),
                cell_div_raw_s_inv=cell_net(raw_flux)/transport.cell_volumes,
                cell_div_balanced_s_inv=cell_net(balanced_flux)/transport.cell_volumes,
            )
        os.replace(str(temp), str(target))

    snapshot_ready = True
    save_snapshot("verification_step_000000000.npz")
    budget_handle = (OUTPUT/"verification_budget.csv").open("w", newline="", buffering=1)
    budget_writer = None
    state.update(status="running", last_recorded_step=0, last_recorded_time_s=0.0)
    write_json(OUTPUT, "verification_status.json", state)

    print("Starting 2000 steps, dt=0.001 s, target=2 s. Output:", OUTPUT, flush=True)
    for n_step in range(1, num_steps+1):
        attempted_step = n_step
        attempted_time = n_step*dt_value
        state.update(attempted_step=n_step, attempted_time_s=attempted_time)
        candidate_fresh = None
        transport.last_candidate = None
        shadow_fresh = np.full(ncells, np.nan)
        shadow_success = False
        shadow_error = ""
        sign_change_faces = np.zeros(len(transport.edges), dtype=bool)
        facet_trace_available = False
        previous_fresh = fresh_n.vector().get_local()[cell_dofs].copy()
        # Missing fields on failure stay visibly missing, never stale.
        raw_flux = np.full(len(transport.edges), np.nan)
        balanced_flux = raw_flux.copy()
        flux_time = attempted_time
        ramp_value = 0.5*(1.0-np.cos(np.pi*attempted_time/ramp_time)) if attempted_time < ramp_time else 1.0
        U_in.ramp = ramp_value
        U_ret.ramp = ramp_value

        flow_stages["tentative_velocity"] = {"time_s": attempted_time, "status": "solving"}
        A1 = assemble(a1)
        b1 = assemble(L1)
        for bc in bcu:
            bc.apply(A1, b1)
        solver_Ustar.solve(A1, U_star.vector(), b1)
        flow_stages["tentative_velocity"]["status"] = "complete"

        flow_stages["pressure"] = {"time_s": attempted_time, "status": "solving"}
        b2 = assemble(L2)
        bc_dp.apply(b2)
        solver_dp.solve(dp_new.vector(), b2)
        p_new.vector().zero()
        p_new.vector().axpy(1.0, p_n.vector())
        p_new.vector().axpy(1.0, dp_new.vector())
        p_new.vector().apply("insert")
        flow_stages["pressure"]["status"] = "complete"

        flow_stages["corrected_velocity"] = {"time_s": attempted_time, "status": "solving"}
        b3 = assemble(L3)
        for bc in bcu:
            bc.apply(b3)
        solver_Ucorr.solve(U_new.vector(), b3)
        flow_stages["corrected_velocity"]["status"] = "complete"
        velocity_time = attempted_time
        if not all(np.all(np.isfinite(a.vector().get_local())) for a in (U_star, U_new, p_new)):
            raise RuntimeError("Non-finite flow at step {}".format(n_step))

        raw_flux, sign_change_faces = facet_reader.read(U_new.vector().get_local())
        facet_trace_available = True
        flux_scale = max(Q_analytic*ramp_value, 1e-30)
        boundary_tolerance = 2e-10*flux_scale
        if np.any(raw_flux[facet_tags == 1] > boundary_tolerance):
            raise RuntimeError("Nozzle unexpectedly has outward flow")
        if np.any(raw_flux[facet_tags == 2] < -boundary_tolerance):
            raise RuntimeError("Return unexpectedly has inward flow")
        if np.any(np.abs(raw_flux[facet_tags >= 3]) > boundary_tolerance):
            raise RuntimeError("Nonzero wall/axis boundary flux")
        nozzle_in = -float(np.sum(raw_flux[facet_tags == 1]))
        if abs(nozzle_in-Q_analytic*ramp_value) > boundary_tolerance:
            raise RuntimeError("Facet inlet differs from the prescribed ramped Q")

        balanced_flux, projection_metrics = transport.balance(raw_flux)
        if not np.array_equal(balanced_flux[transport.boundary], raw_flux[transport.boundary]):
            raise RuntimeError("Flux projection changed a boundary flux")

        # Diagnostic only. A failed shadow is recorded, never used to alter
        # or bypass the guards on the accepted balanced transport.
        try:
            shadow_fresh, shadow_metrics = transport.shadow_advance(previous_fresh, raw_flux, dt_value, boundary_fresh)
            shadow_success = True
        except (RuntimeError, ValueError, FloatingPointError) as shadow_exception:
            shadow_error = str(shadow_exception)
        transport.last_candidate = None
        candidate_fresh, scalar_metrics = transport.advance(previous_fresh, balanced_flux, dt_value, boundary_fresh)
        values = np.empty(F.dim())
        values[cell_dofs] = candidate_fresh
        fresh_n.vector().set_local(values)
        fresh_n.vector().apply("insert")
        U_n.assign(U_new)
        p_n.assign(p_new)
        accepted_step = n_step
        accepted_time = attempted_time

        boundary = transport.boundary
        bf = balanced_flux[boundary]
        in_rate = float(-np.dot(np.minimum(bf, 0.0), boundary_fresh[boundary]))
        out_rate = float(np.dot(np.maximum(bf, 0.0), candidate_fresh[transport.owner[boundary]]))
        cumulative_in += dt_value*in_rate
        cumulative_out += dt_value*out_rate
        mass = float(np.dot(transport.cell_volumes, candidate_fresh))
        expected = cumulative_in-cumulative_out
        residual = mass-expected
        relative_residual = residual/max(cumulative_in, 1e-30)
        raw_net = cell_net(raw_flux)
        corrected_net = cell_net(balanced_flux)
        ii = transport.interior
        delta = balanced_flux[ii]-raw_flux[ii]
        change_relative = float(np.linalg.norm(delta)/max(np.linalg.norm(raw_flux[ii]), 1e-30))
        max_budget_relative = max(max_budget_relative, abs(relative_residual))
        max_flux_change_relative = max(max_flux_change_relative, change_relative)
        imax = int(np.argmax(candidate_fresh))
        velocity_values = U_new.vector().get_local()
        physical_r = np.sqrt(2.0*np.maximum(velocity_coords[velocity_ids[0], 0], 0.0))
        # Each vector component is independently ordered; calculate component
        # maxima separately, avoiding an invalid cross-component index pairing.
        radial_speed = np.zeros(len(physical_r))
        np.divide(velocity_values[velocity_ids[0]], physical_r, out=radial_speed, where=physical_r > 0)
        last_row = {
            "step": n_step, "time_s": accepted_time, "dt_s": dt_value, "ramp": ramp_value,
            "fresh_in_rate_m3_s": in_rate, "fresh_out_rate_m3_s": out_rate,
            "fresh_in_cumulative_m3": cumulative_in, "fresh_out_cumulative_m3": cumulative_out,
            "expected_fresh_m3": expected, "fresh_stored_m3": mass,
            "budget_residual_m3": residual, "budget_residual_relative": relative_residual,
            "fresh_min": float(candidate_fresh.min()), "fresh_max": float(candidate_fresh.max()),
            "fresh_max_cell": imax, "fresh_max_r_m": float(np.sqrt(2*transport.centroids[imax, 0])),
            "fresh_max_z_m": float(transport.centroids[imax, 1]),
            "raw_cell_net_max_abs_m3_s": float(np.max(np.abs(raw_net))),
            "balanced_cell_net_max_abs_m3_s": float(np.max(np.abs(corrected_net))),
            "raw_cell_div_max_abs_s_inv": float(np.max(np.abs(raw_net)/transport.cell_volumes)),
            "balanced_cell_div_max_abs_s_inv": float(np.max(np.abs(corrected_net)/transport.cell_volumes)),
            "boundary_net_m3_s": float(raw_flux[boundary].sum()),
            "flux_correction_L2_m3_s": float(np.linalg.norm(delta)),
            "flux_correction_relative_L2": change_relative,
            "flux_correction_max_abs_m3_s": float(np.max(np.abs(delta))),
            "interior_flux_sign_flips": int(np.sum(raw_flux[ii]*balanced_flux[ii] < 0)),
            "raw_interior_trace_sign_changes": int(np.sum(sign_change_faces[ii])),
            "constant_state_relative_residual": float(np.max(dt_value*np.abs(corrected_net)/transport.cell_volumes)),
            "one_step_shadow_success": int(shadow_success),
            "one_step_shadow_min": float(np.min(shadow_fresh)) if shadow_success else "",
            "one_step_shadow_max": float(np.max(shadow_fresh)) if shadow_success else "",
            "one_step_shadow_inventory_m3": float(np.dot(transport.cell_volumes, shadow_fresh)) if shadow_success else "",
            "one_step_shadow_budget_residual_m3": shadow_metrics["fresh_budget_residual_m3"] if shadow_success else "",
            "one_step_shadow_minus_balanced_L1_m3": float(np.dot(transport.cell_volumes, np.abs(shadow_fresh-candidate_fresh))) if shadow_success else "",
            "one_step_shadow_excess_above_one_m3": float(np.dot(transport.cell_volumes, np.maximum(shadow_fresh-1, 0))) if shadow_success else "",
            "one_step_shadow_error": shadow_error,
            "max_abs_radial_speed_at_CG2_nodes_m_s": float(np.max(np.abs(radial_speed))),
            "max_abs_vertical_speed_at_CG2_nodes_m_s": float(np.max(np.abs(velocity_values[velocity_ids[1]]))),
            "wall_elapsed_s": wallclock.time()-wall_start,
        }
        state.update(last_recorded_step=n_step, last_recorded_time_s=accepted_time, last_budget_row=last_row,
                     projection_metrics=projection_metrics, scalar_metrics=scalar_metrics,
                     max_budget_relative=max_budget_relative, max_flux_change_relative=max_flux_change_relative)
        if budget_writer is None:
            budget_writer = csv.DictWriter(budget_handle, fieldnames=list(last_row))
            budget_writer.writeheader()
        budget_writer.writerow(last_row)
        if abs(relative_residual) > 1e-8:
            raise RuntimeError("Cumulative scalar budget failed: relative residual {}".format(relative_residual))
        if not (candidate_fresh.min() >= -1e-10 and candidate_fresh.max() <= 1+1e-10):
            raise RuntimeError("Balanced scalar left physical bounds")
        if n_step <= 5 or n_step % 100 == 0:
            budget_handle.flush()
            print("step={} t={:.3f} fresh=[{:.6g},{:.6g}] raw-shadow-max={} net-raw={:.3e} net-balanced={:.3e} correction={:.3%} budget={:.3e}".format(
                n_step, accepted_time, candidate_fresh.min(), candidate_fresh.max(),
                last_row["one_step_shadow_max"], last_row["raw_cell_net_max_abs_m3_s"],
                last_row["balanced_cell_net_max_abs_m3_s"], change_relative, relative_residual), flush=True)
            write_json(OUTPUT, "verification_status.json", state)
        if n_step % 100 == 0:
            save_snapshot("verification_step_{:09d}.npz".format(n_step))

    save_snapshot("verification_final.npz")
    budget_handle.flush()
    budget_handle.close()
    state.update(status="completed", wall_elapsed_s=wallclock.time()-wall_start)
    write_json(OUTPUT, "verification_status.json", state)
    print("Completed the 2-second diagnostic. Send configuration, status, budget, final NPZ and the Slurm .out file.", flush=True)


except BaseException as error:
    if budget_handle is not None and not budget_handle.closed:
        budget_handle.flush()
        budget_handle.close()
    state.update(status="failed", exception_type=type(error).__name__, exception=str(error))
    if snapshot_ready:
        try:
            save_snapshot("verification_failure_state.npz")
            state["failure_snapshot"] = "verification_failure_state.npz"
        except Exception as snapshot_error:
            state["snapshot_error"] = str(snapshot_error)
    write_json(OUTPUT, "verification_status.json", state)
    raise
