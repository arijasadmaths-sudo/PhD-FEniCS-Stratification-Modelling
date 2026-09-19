#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched five-second Cartesian mesh and time-step verification runs.

Run with legacy FEniCS/DOLFIN 2019.1, NumPy, SciPy and mpi4py, ONE MPI rank:
    python3 cartesian_ipcs_verification_5s.py --case mesh --output NEW_EMPTY_DIRECTORY
    python3 cartesian_ipcs_verification_5s.py --case timestep --output NEW_EMPTY_DIRECTORY
Self-tests (NumPy/SciPy only):
    python3 cartesian_ipcs_verification_5s.py --self-test

Cases (each starts from rest and ends at exactly t=5 seconds):
    baseline: 120 x 60 cells, dt=0.001 s, 5000 steps (optional reproduction).
    mesh:     240 x 120 cells, dt=0.001 s, 5000 steps.
    timestep: 120 x 60 cells, dt=0.0005 s, 10000 steps.
All retain the verified pilot's physical parameters, IPCS equations, scalar
FV update and internal-face flux projection. The baseline source SHA256 is
recorded below and in the run configuration. No clipping or mass rescaling.

Physical setup: 0.60 x 0.30 m; 0.01 m parabolic slot at Umax=0.001 m/s;
flux-balanced broad returns; nu=1e-6; D=1e-7 m2/s; beta=-0.00627;
half-cosine 0.5 s velocity ramp; initially c=1, nozzle c=0.
Scalar: rectangular cell averages, implicit first-order upwind advection,
orthogonal two-point diffusion, mapped to paired DG0 triangles for buoyancy.
P2 velocity face integrals are corrected on INTERNAL faces to local volume
balance. External prescribed fluxes remain unchanged.

Outputs are raw salt fraction c (1 ambient, 0 fresh), never min-max normalised.
Every 0.25 s (also step 1), fv_step_*.npz saves c, coordinates, physical time,
raw_fx_m2_s/raw_fy_m2_s from the IPCS velocity, and the corrected
transport_fx_m2_s/transport_fy_m2_s. Flux units are m2/s per unit depth;
divide by face length to obtain face-normal velocity. Snapshot step numbers
therefore differ between time-step cases; compare the saved time_s value.
The last snapshot is also saved as fv_final.npz for a direct t=5 s comparison.
XDMF files retain u, p and the DG0 scalar fields at the same times.

The scalar budget records the actual numerical advection and nozzle diffusion.
Other boundaries have zero diffusive flux. First-order numerical diffusion
and changes in transport flux still need assessment against these paired runs.
Pure scalar/flux tests are embedded. Full coupled FEniCS runs require HPC.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import sys
import unittest
import scipy
from scipy import sparse

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized, spsolve


VERIFIED_BASELINE_SHA256 = "793220fab7b3740f4ecdce0b9e773122e9d7bcbe4f96d3eeb5eb2123a0bd5b8c"
VERIFICATION_CASES = {
    "baseline": {"nx": 120, "ny": 60, "dt_s": 0.001},
    "mesh": {"nx": 240, "ny": 120, "dt_s": 0.001},
    "timestep": {"nx": 120, "ny": 60, "dt_s": 0.0005},
}


def exact_step_count(duration, dt):
    """Reject a time interval that does not contain a whole number of steps."""
    if not (np.isfinite(duration) and np.isfinite(dt)) or duration <= 0. or dt <= 0.:
        raise ValueError("Time intervals must be finite and positive")
    count = int(round(duration / dt))
    if count < 1 or not math.isclose(count * dt, duration, rel_tol=0., abs_tol=1.e-12):
        raise ValueError("Requested interval is not an integer multiple of the time step")
    return count


def verification_settings(case):
    if case not in VERIFICATION_CASES:
        raise ValueError("Unknown verification case: " + str(case))
    settings = dict(VERIFICATION_CASES[case])
    settings.update(duration_s=5.0, field_interval_s=0.25, log_interval_s=0.1)
    for key, interval in (("steps", "duration_s"),
                          ("write_every", "field_interval_s"),
                          ("log_every", "log_interval_s")):
        settings[key] = exact_step_count(settings[interval], settings["dt_s"])
    return settings


def aligned_nozzle_mask(nx, Lx, d_nozzle):
    """Require both nozzle edges to lie on grid faces, with full coverage."""
    if nx < 1 or not (0. < d_nozzle < Lx):
        raise ValueError("Invalid nozzle/grid geometry")
    hx = float(Lx) / nx
    xL, xR = (Lx-d_nozzle)/2., (Lx+d_nozzle)/2.
    edge_indices = np.array([xL/hx, xR/hx])
    rounded = np.rint(edge_indices).astype(int)
    if np.max(np.abs(edge_indices-rounded)) > 1.e-9:
        raise ValueError("Both nozzle endpoints must coincide with FV grid faces")
    expected_count = int(round(d_nozzle/hx))
    if expected_count < 1 or abs(expected_count*hx-d_nozzle) > 1.e-12:
        raise ValueError("Nozzle width must contain an integer number of full faces")
    edges = np.linspace(0., Lx, nx+1)
    mask = (edges[:-1] >= xL-1.e-10) & (edges[1:] <= xR+1.e-10)
    if (np.count_nonzero(mask) != expected_count or
            rounded[1]-rounded[0] != expected_count or
            abs(np.sum(mask)*hx-d_nozzle) > 1.e-12):
        raise ValueError("Nozzle mask does not cover the complete source interval")
    return mask


class BalancedFluxGrid(object):
    def __init__(self, nx, ny, Lx, Ly):
        self.nx, self.ny = int(nx), int(ny)
        if self.nx < 1 or self.ny < 1 or Lx <= 0 or Ly <= 0:
            raise ValueError("Grid counts and lengths must be positive.")
        self.dx, self.dy = float(Lx) / self.nx, float(Ly) / self.ny
        self.volume = self.dx * self.dy
        self.n = self.nx * self.ny
        ids = np.arange(self.n).reshape(self.ny, self.nx)
        self.ids = ids
        self.u = np.concatenate((ids[:, :-1].ravel(), ids[:-1, :].ravel()))
        self.v = np.concatenate((ids[:, 1:].ravel(), ids[1:, :].ravel()))
        self.n_xinterior = self.ny * (self.nx - 1)
        self.weights = np.concatenate((
            np.full(self.n_xinterior, self.dy / self.dx),
            np.full((self.ny - 1) * self.nx, self.dx / self.dy)))
        self.edge_rows = np.concatenate((self.u, self.v, self.u, self.v))
        self.edge_cols = np.concatenate((self.u, self.v, self.v, self.u))
        w = self.weights
        lap = coo_matrix((np.concatenate((w, w, -w, -w)),
                          (self.edge_rows, self.edge_cols)),
                         shape=(self.n, self.n)).tocsc()
        # Fix one potential only. Exterior volume fluxes are never adjusted.
        self._project_solve = factorized(lap[1:, 1:]) if self.n > 1 else None
        self.boundary_ids = np.concatenate((ids[:, 0], ids[:, -1],
                                             ids[0, :], ids[-1, :]))

    def _check_flux(self, fx, fy):
        fx, fy = np.asarray(fx, dtype=float), np.asarray(fy, dtype=float)
        if fx.shape != (self.ny, self.nx + 1):
            raise ValueError("fx must have shape (ny, nx+1).")
        if fy.shape != (self.ny + 1, self.nx):
            raise ValueError("fy must have shape (ny+1, nx).")
        if not (np.all(np.isfinite(fx)) and np.all(np.isfinite(fy))):
            raise ValueError("Non-finite face volume flux.")
        return fx, fy

    def divergence(self, fx, fy):
        """Integrated outward volume flux in each cell [m^2/s]."""
        return fx[:, 1:] - fx[:, :-1] + fy[1:, :] - fy[:-1, :]

    def _exterior_outward(self, fx, fy):
        return np.concatenate((-fx[:, 0], fx[:, -1], -fy[0, :], fy[-1, :]))

    def _flux_scale(self, fx, fy):
        return max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))),
                   float(np.sum(np.abs(self._exterior_outward(fx, fy)))), 1.e-15)

    def balance(self, fx, fy):
        """Project interior face fluxes to zero cell divergence.

        The correction minimizes sum((q_new-q_old)^2 / face_weight) with
        fixed exterior fluxes; face_weight is face length / centre distance.
        One graph-Laplacian factorization is reused for every time step.
        """
        fx, fy = self._check_flux(fx, fy)
        scale = self._flux_scale(fx, fy)
        exterior = self._exterior_outward(fx, fy)
        net = float(np.sum(exterior))
        if abs(net) > 1.e-11 * scale:
            raise ValueError("Incompatible exterior volume flux: {:.6e} m^2/s".format(net))
        before = self.divergence(fx, fy).ravel()
        potential = np.zeros(self.n)
        if self.n > 1:
            potential[1:] = self._project_solve(before[1:])
        correction = self.weights * (potential[self.u] - potential[self.v])
        new_fx, new_fy = fx.copy(), fy.copy()
        new_fx[:, 1:-1] -= correction[:self.n_xinterior].reshape(self.ny, self.nx - 1)
        new_fy[1:-1, :] -= correction[self.n_xinterior:].reshape(self.ny - 1, self.nx)
        after = self.divergence(new_fx, new_fy).ravel()
        max_after = float(np.max(np.abs(after)))
        if max_after > 1.e-10 * max(scale, self._flux_scale(new_fx, new_fy)):
            raise RuntimeError("Face-flux projection did not reach local volume balance.")
        old_internal = np.concatenate((fx[:, 1:-1].ravel(), fy[1:-1, :].ravel()))
        diagnostics = {
            'boundary_net_volume_flux_m2_s': net,
            'cell_flux_residual_before_max_m2_s': float(np.max(np.abs(before))),
            'cell_flux_residual_after_max_m2_s': max_after,
            'cell_divergence_before_l2_sinv': float(np.linalg.norm(before) / np.sqrt(self.volume)),
            'cell_divergence_after_l2_sinv': float(np.linalg.norm(after) / np.sqrt(self.volume)),
            'face_flux_correction_l2_m2_s': float(np.linalg.norm(correction)),
            'face_flux_correction_relative_l2': float(np.linalg.norm(correction) /
                max(np.linalg.norm(old_internal), np.finfo(float).tiny)),
        }
        return new_fx, new_fy, diagnostics

    def _boundary_values(self, boundary_c):
        if isinstance(boundary_c, dict):
            values = [np.broadcast_to(np.asarray(boundary_c.get(name, 0.), dtype=float), (size,))
                      for name, size in [('left', self.ny), ('right', self.ny),
                                         ('bottom', self.nx), ('top', self.nx)]]
        else:
            value = float(boundary_c)
            values = [np.full(self.ny, value), np.full(self.ny, value),
                      np.full(self.nx, value), np.full(self.nx, value)]
        result = np.concatenate(values)
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite scalar boundary data.")
        return result

    def step(self, c, fx, fy, dt, D, nozzle_mask, boundary_c=0.):
        """One implicit upwind + orthogonal-diffusion scalar step.

        Nozzle bottom faces use a physical Dirichlet diffusion flux with
        half-cell distance. All other boundaries have zero diffusive flux.
        Inward advective flux uses boundary_c; outward flux uses the new
        interior value. boundary_c can be a scalar or a dict with left,
        right, bottom and top scalar/array values.
        """
        fx, fy = self._check_flux(fx, fy)
        c = np.asarray(c, dtype=float)
        mask = np.asarray(nozzle_mask, dtype=bool)
        if c.shape != (self.ny, self.nx) or not np.all(np.isfinite(c)):
            raise ValueError("c must be a finite (ny,nx) array.")
        if mask.shape != (self.nx,):
            raise ValueError("nozzle_mask must have shape (nx,).")
        if not np.isfinite(dt) or not np.isfinite(D) or dt <= 0 or D < 0:
            raise ValueError("dt must be positive and D non-negative.")
        div = self.divergence(fx, fy)
        if np.max(np.abs(div)) > 1.e-10 * self._flux_scale(fx, fy):
            raise ValueError("Scalar transport requires locally balanced face fluxes.")
        q = np.concatenate((fx[:, 1:-1].ravel(), fy[1:-1, :].ravel()))
        qp, qm = np.maximum(q, 0.), np.minimum(q, 0.)
        T = float(D) * self.weights
        # Every interior numerical flux enters its two cells with opposite signs.
        data = np.concatenate((qp + T, -qm + T, qm - T, -qp - T))
        qout = self._exterior_outward(fx, fy)
        cb = self._boundary_values(boundary_c)
        diag = np.full(self.n, self.volume / dt)
        rhs = c.ravel() * (self.volume / dt)
        np.add.at(diag, self.boundary_ids, np.maximum(qout, 0.))
        np.add.at(rhs, self.boundary_ids, -np.minimum(qout, 0.) * cb)
        nozzle_ids = self.ids[0, mask]
        nozzle_cb = cb[2 * self.ny:2 * self.ny + self.nx][mask]
        nozzle_T = 2. * float(D) * self.dx / self.dy
        np.add.at(diag, nozzle_ids, nozzle_T)
        np.add.at(rhs, nozzle_ids, nozzle_T * nozzle_cb)
        all_ids = np.arange(self.n)
        matrix = coo_matrix((np.concatenate((data, diag)),
                             (np.concatenate((self.edge_rows, all_ids)),
                              np.concatenate((self.edge_cols, all_ids)))),
                            shape=(self.n, self.n)).tocsc()
        solved = spsolve(matrix, rhs)
        if not np.all(np.isfinite(solved)):
            raise RuntimeError("Non-finite scalar solve.")
        cnew = solved.reshape(self.ny, self.nx)
        cup = np.where(qout >= 0., solved[self.boundary_ids], cb)
        fresh_adv_outward = qout * (1. - cup)
        fresh_adv_in = -float(np.sum(fresh_adv_outward[qout < 0.]))
        fresh_adv_out = float(np.sum(fresh_adv_outward[qout >= 0.]))
        # For f=1-c, inward diffusive fresh flux is outward diffusive salt flux.
        fresh_diff_in = float(nozzle_T * np.sum(solved[nozzle_ids] - nozzle_cb))
        vf_old = float(self.volume * np.sum(1. - c))
        vf_new = float(self.volume * np.sum(1. - cnew))
        fresh_balance = vf_new - vf_old - dt * (fresh_adv_in - fresh_adv_out + fresh_diff_in)
        row_residual = matrix.dot(solved) - rhs
        diagnostics = {
            'vf_previous_m2': vf_old,
            'vf_m2': vf_new,
            'fresh_adv_in_m2_s': fresh_adv_in,
            'fresh_adv_out_m2_s': fresh_adv_out,
            'fresh_diff_in_m2_s': fresh_diff_in,
            'step_budget_residual_m2': float(fresh_balance),
            'scalar_row_residual_max_m2_s': float(np.max(np.abs(row_residual))),
            'cell_volume_residual_max_m2_s': float(np.max(np.abs(div))),
            'volume_in_m2_s': -float(np.sum(qout[qout < 0.])),
            'volume_out_m2_s': float(np.sum(qout[qout >= 0.])),
            'c_min': float(np.min(cnew)),
            'c_max': float(np.max(cnew)),
        }
        return cnew, diagnostics

def production_boundary_fluxes(grid, U=0.001, d=0.01):
    fx = np.zeros((grid.ny, grid.nx + 1))
    fy = np.zeros((grid.ny + 1, grid.nx))
    Lx = grid.nx * grid.dx
    a, b = 0.5 * (Lx - d), 0.5 * (Lx + d)
    centre = (np.arange(grid.nx) + 0.5) * grid.dx
    nodes = [centre - grid.dx / (2. * np.sqrt(3.)),
             centre + grid.dx / (2. * np.sqrt(3.))]
    Ur = U * d / (Lx - d)
    for x in nodes:
        vel = np.where(x < a, -4. * Ur * x * (a - x) / a**2,
                       np.where(x > b, -4. * Ur * (x - b) * (Lx - x) / (Lx - b)**2,
                                4. * U * (x - a) * (b - x) / d**2))
        fy[0] += 0.5 * grid.dx * vel
    return fx, fy, (centre > a) & (centre < b)


class KernelTests(unittest.TestCase):
    def test_verification_case_time_and_nozzle_alignment(self):
        expected = {"baseline": (120,60,.001,5000,250,100,2),
                    "mesh": (240,120,.001,5000,250,100,4),
                    "timestep": (120,60,.0005,10000,500,200,2)}
        for name, values in expected.items():
            case = verification_settings(name)
            self.assertEqual(tuple(case[k] for k in
                ("nx","ny","dt_s","steps","write_every","log_every")), values[:6])
            self.assertEqual(case["steps"]*case["dt_s"], 5.)
            self.assertEqual(case["write_every"]*case["dt_s"], .25)
            self.assertEqual(case["log_every"]*case["dt_s"], .1)
            mask = aligned_nozzle_mask(case["nx"], .6, .01)
            self.assertEqual(int(mask.sum()), values[6])
        with self.assertRaises(ValueError):
            aligned_nozzle_mask(121, .6, .01)
        with self.assertRaises(ValueError):
            exact_step_count(5., .003)
        with self.assertRaises(ValueError):
            verification_settings("unknown")

    def test_refined_grid_projection_uniform_and_source_budget(self):
        g = BalancedFluxGrid(240, 120, .6, .3)
        fx, fy, mask = production_boundary_fluxes(g)
        np.testing.assert_array_equal(mask, aligned_nozzle_mask(240,.6,.01))
        self.assertEqual(int(mask.sum()), 4)
        self.assertAlmostEqual(float(fy[0,mask].sum()), (2./3.)*.001*.01, places=16)
        rng = np.random.RandomState(170)
        fx[:,1:-1] = rng.normal(scale=1.e-5, size=(120,239))
        fy[1:-1,:] = rng.normal(scale=1.e-5, size=(119,240))
        fx, fy, diagnostics = g.balance(fx,fy)
        uniform, d = g.step(np.full((120,240),.37),fx,fy,.001,1.e-7,mask,boundary_c=.37)
        np.testing.assert_allclose(uniform,.37,atol=1.e-13,rtol=0.)
        c, d = g.step(np.ones((120,240)),fx,fy,.001,1.e-7,mask,
                     boundary_c={"left":1.,"right":1.,"top":1.,
                                 "bottom":np.where(mask,0.,1.)})
        self.assertGreaterEqual(c.min(),-1.e-12)
        self.assertLessEqual(c.max(),1.+1.e-12)
        self.assertLess(abs(d["step_budget_residual_m2"]),1.e-13)
        self.assertGreater(d["fresh_adv_in_m2_s"],0.)
        self.assertGreater(d["fresh_diff_in_m2_s"],0.)

    def test_projection_preserves_boundary_and_closes_every_cell(self):
        g = BalancedFluxGrid(12, 8, .6, .4)
        rng = np.random.RandomState(35)
        fx = rng.normal(scale=.01, size=(8, 13))
        fy = rng.normal(scale=.01, size=(9, 12))
        fx[:, 0], fx[:, -1] = .002, .002
        fy[0], fy[-1] = 0., 0.
        xn, yn, d = g.balance(fx, fy)
        np.testing.assert_array_equal(xn[:, 0], fx[:, 0])
        np.testing.assert_array_equal(xn[:, -1], fx[:, -1])
        np.testing.assert_array_equal(yn[0], fy[0])
        np.testing.assert_array_equal(yn[-1], fy[-1])
        self.assertLess(np.max(np.abs(g.divergence(xn, yn))), 2.e-15)
        self.assertGreater(d['face_flux_correction_l2_m2_s'], 0.)
        # A second projection should leave an already-balanced field alone.
        x2, y2, unused = g.balance(xn, yn)
        np.testing.assert_allclose(x2, xn, atol=2.e-15, rtol=0.)
        np.testing.assert_allclose(y2, yn, atol=2.e-15, rtol=0.)

    def test_incompatible_boundary_rejected(self):
        g = BalancedFluxGrid(3, 2, 1., 1.)
        fx, fy = np.zeros((2, 4)), np.zeros((3, 3))
        fy[0, 1] = .01
        with self.assertRaises(ValueError):
            g.balance(fx, fy)

    def test_step_rejects_unbalanced_flux(self):
        g = BalancedFluxGrid(3, 2, 1., 1.)
        fx, fy = np.zeros((2, 4)), np.zeros((3, 3))
        fx[0, 1] = .01
        with self.assertRaises(ValueError):
            g.step(np.ones((2, 3)), fx, fy, .1, 0., np.zeros(3, bool))

    def test_single_cell_analytical_update(self):
        g = BalancedFluxGrid(1, 1, 1., 1.)
        fx, fy = np.zeros((1, 2)), np.full((2, 1), .4)
        c, d = g.step(np.ones((1, 1)), fx, fy, .3, .2, np.array([True]))
        expected = 1. / (1. + .3 * (.4 + 2. * .2))
        self.assertAlmostEqual(float(c[0, 0]), expected, places=15)
        self.assertLess(abs(d['step_budget_residual_m2']), 1.e-15)
        self.assertAlmostEqual(d['fresh_diff_in_m2_s'], .4 * expected, places=15)

    def test_closed_diffusion_preserves_mass_and_bounds(self):
        g = BalancedFluxGrid(13, 7, .65, .35)
        fx, fy = np.zeros((7, 14)), np.zeros((8, 13))
        old = np.random.RandomState(4).uniform(size=(7, 13))
        new, d = g.step(old, fx, fy, 10., 1.e-3, np.zeros(13, bool))
        self.assertAlmostEqual(float(np.sum(old)), float(np.sum(new)), places=11)
        self.assertGreaterEqual(new.min(), old.min())
        self.assertLessEqual(new.max(), old.max())
        self.assertLess(abs(d['step_budget_residual_m2']), 1.e-14)

    def test_diffusive_flux_sign_reverses(self):
        g = BalancedFluxGrid(1, 1, 1., 1.)
        fx, fy = np.zeros((1, 2)), np.zeros((2, 1))
        c, d = g.step(np.zeros((1, 1)), fx, fy, .3, .2, np.ones(1, bool), boundary_c=1.)
        self.assertGreater(c[0, 0], 0.)
        self.assertLess(d['fresh_diff_in_m2_s'], 0.)
        self.assertLess(abs(d['step_budget_residual_m2']), 1.e-15)

    def test_upwind_direction_and_boundary_values(self):
        # A right-to-left throughflow must select the right inflow value.
        g = BalancedFluxGrid(2, 1, 2., 1.)
        fx, fy = -np.ones((1, 3)), np.zeros((2, 2))
        c, d = g.step(np.ones((1, 2)), fx, fy, 1., 0., np.zeros(2, bool),
                      boundary_c={'left': 1., 'right': 0.})
        np.testing.assert_allclose(c, [[.75, .5]], rtol=0., atol=1.e-15)
        self.assertLess(abs(d['step_budget_residual_m2']), 1.e-15)

    def test_full_size_uniform_preservation_and_bounds(self):
        g = BalancedFluxGrid(120, 60, .6, .3)
        fx, fy, mask = production_boundary_fluxes(g)
        self.assertEqual(int(mask.sum()), 2)
        self.assertAlmostEqual(float(fy[0, mask].sum()), 2. / 3. * .001 * .01, places=16)
        rng = np.random.RandomState(47)
        # Add interior velocity-like noise; projection must remove its compression.
        fx[:, 1:-1] = rng.normal(scale=2.e-5, size=(60, 119))
        fy[1:-1] = rng.normal(scale=2.e-5, size=(59, 120))
        fx, fy, proj = g.balance(fx, fy)
        value = .37
        uniform, d = g.step(np.full((60, 120), value), fx, fy, .001, 1.e-7, mask,
                            boundary_c=value)
        np.testing.assert_allclose(uniform, value, atol=3.e-14, rtol=0.)
        self.assertLess(abs(d['step_budget_residual_m2']), 1.e-14)
        c = rng.uniform(size=(60, 120))
        start = g.volume * np.sum(1. - c)
        integral = 0.
        maximum_budget_error = 0.
        for dt in [.001, .1, 1., 10.]:
            c, d = g.step(c, fx, fy, dt, 1.e-7, mask)
            self.assertGreaterEqual(c.min(), -1.e-12)
            self.assertLessEqual(c.max(), 1. + 1.e-12)
            self.assertLess(abs(d['step_budget_residual_m2']), 1.e-13)
            integral += dt * (d['fresh_adv_in_m2_s'] - d['fresh_adv_out_m2_s'] + d['fresh_diff_in_m2_s'])
            maximum_budget_error = max(maximum_budget_error, abs(d['step_budget_residual_m2']))
        self.assertLess(abs(d['vf_m2'] - start - integral), 2.e-13)
        print(json.dumps({'full_size_uniform_max_error': float(np.max(np.abs(uniform-value))),
                          'full_size_cell_volume_residual_max_m2_s': proj['cell_flux_residual_after_max_m2_s'],
                          'full_size_step_budget_error_max_m2': maximum_budget_error,
                          'full_size_c_min': float(c.min()), 'full_size_c_max': float(c.max())}, sort_keys=True))



def run_self_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(KernelTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Conservative scalar self-tests failed")
    print("All scalar/flux self-tests passed. These do not exercise the coupled FEniCS solver.", flush=True)

def velocity_flux_maps(V, nx, ny, Lx, Ly):
    """Exact edge integrals of continuous P2 velocity on the aligned mesh.

    Simpson weights integrate each quadratic edge trace exactly. Parent-space
    DOF indices are used; the standalone implementation is deliberately serial.
    """
    hx, hy = Lx / nx, Ly / ny
    xy = np.asarray(V.tabulate_dof_coordinates()).reshape((-1, 2))
    if xy.shape[0] != V.dim():
        raise RuntimeError("Unexpected vector-space coordinate layout")
    matrices = []
    for component in (0, 1):
        lookup = {}
        for dof in V.sub(component).dofmap().dofs():
            dof = int(dof)
            scaled = xy[dof] / np.array([hx / 2.0, hy / 2.0])
            key = tuple(np.rint(scaled).astype(int))
            if np.max(np.abs(scaled - np.array(key))) > 1e-7:
                raise RuntimeError("Velocity DOF is not on the expected P2 half-grid")
            if key in lookup:
                raise RuntimeError("Duplicate coordinate in a velocity component")
            lookup[key] = dof
        rows, cols, data = [], [], []
        if component == 0:
            keys = [((2*i, 2*j), (2*i, 2*j+1), (2*i, 2*j+2))
                    for j in range(ny) for i in range(nx+1)]
            length = hy
        else:
            keys = [((2*i, 2*j), (2*i+1, 2*j), (2*i+2, 2*j))
                    for j in range(ny+1) for i in range(nx)]
            length = hx
        for row, triple in enumerate(keys):
            for key, weight in zip(triple, (1.0, 4.0, 1.0)):
                if key not in lookup:
                    raise RuntimeError("Missing P2 edge DOF at {}".format(key))
                rows.append(row)
                cols.append(lookup[key])
                data.append(length * weight / 6.0)
        matrices.append(sparse.csr_matrix((data, (rows, cols)),
                                         shape=(len(keys), V.dim())))
    return tuple(matrices)


def run_pilot(args):
    import dolfin as df
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("This pilot requires exactly one MPI process; use the supplied .sh file")
    df.parameters["std_out_all_processes"] = False
    df.set_log_level(30)
    folder = os.path.abspath(args.output)
    if os.path.exists(folder) and (not os.path.isdir(folder) or os.listdir(folder)):
        raise RuntimeError("Output directory must be new or empty: " + folder)
    os.makedirs(folder, exist_ok=True)
    status = {"status": "starting", "last_completed_step": 0,
              "last_completed_time_s": 0.0, "requested_steps": args.steps,
              "requested_duration_s": 5.0, "verification_case": args.case}

    def write_status():
        tmp = os.path.join(folder, "run_status.json.tmp")
        with open(tmp, "w") as f:
            json.dump(status, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, os.path.join(folder, "run_status.json"))

    outputs = []
    budget_file = None
    write_status()
    try:
        # Same physical parameters; change only the selected mesh or time step.
        settings = verification_settings(args.case)
        Lx, Ly, nx, ny = 0.60, 0.30, settings["nx"], settings["ny"]
        hx, hy = Lx / nx, Ly / ny
        d_nozzle, U_in = 0.01, 0.001
        xL, xR = (Lx-d_nozzle)/2.0, (Lx+d_nozzle)/2.0
        U_return = U_in * d_nozzle / (2.0*xL)
        dt_value, ramp_time, D_value = settings["dt_s"], 0.5, 1e-7
        write_every, log_every = settings["write_every"], settings["log_every"]
        aligned_nozzle_mask(nx, Lx, d_nozzle)
        mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly), nx, ny)
        tol = 1e-10

        class Nozzle(df.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and df.near(x[1], 0.0, tol) and xL-tol <= x[0] <= xR+tol

        class LeftReturn(df.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and df.near(x[1], 0.0, tol) and x[0] <= xL+tol

        class RightReturn(df.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and df.near(x[1], 0.0, tol) and x[0] >= xR-tol

        class SolidWalls(df.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (df.near(x[0], 0.0, tol) or df.near(x[0], Lx, tol)
                                        or df.near(x[1], Ly, tol))

        class PressurePin(df.SubDomain):
            def inside(self, x, on_boundary):
                return df.near(x[0], 0.0, tol) and df.near(x[1], Ly, tol)

        boundaries = df.MeshFunction("size_t", mesh, 1, 0)
        SolidWalls().mark(boundaries, 4)
        LeftReturn().mark(boundaries, 2)
        RightReturn().mark(boundaries, 3)
        Nozzle().mark(boundaries, 1)
        ds_sub = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
        normal = df.FacetNormal(mesh)
        for marker, length in ((1,d_nozzle), (2,xL), (3,Lx-xR), (4,Lx+2*Ly)):
            actual = float(df.assemble(df.Constant(1.0)*ds_sub(marker)))
            if abs(actual-length) > 1e-10:
                raise RuntimeError("Boundary length does not match the original geometry")

        V = df.VectorFunctionSpace(mesh, "CG", 2)
        P = df.FunctionSpace(mesh, "CG", 1)
        C = df.FunctionSpace(mesh, "DG", 0)
        u, v = df.TrialFunction(V), df.TestFunction(V)
        p, q = df.TrialFunction(P), df.TestFunction(P)
        u_n, u_star, u_new = df.Function(V), df.Function(V), df.Function(V)
        p_n, p_new = df.Function(P), df.Function(P)
        for value in (u_n, u_star, u_new):
            value.assign(df.Constant((0.0, 0.0)))
        for value in (p_n, p_new):
            value.assign(df.Constant(0.0))
        c_n = df.Function(C)
        c_n.assign(df.Constant(1.0))
        c_n.rename("c", "Salt fraction: 1 ambient, 0 source")
        u_new.rename("u", "IPCS velocity")
        p_new.rename("p", "IPCS pressure")
        dt, nu_visc = df.Constant(dt_value), df.Constant(1e-6)
        g, beta, c0 = df.Constant(9.81), df.Constant(-6.27e-3), df.Constant(1.0)
        u_in = df.Expression(("0.0", "4.0*amp*U*(x[0]-xL)*(xR-x[0])/pow(xR-xL,2)"),
                             degree=2, amp=0.0, U=U_in, xL=xL, xR=xR)
        u_return_left = df.Expression(("0.0", "-4.0*amp*U*(x[0]-xA)*(xB-x[0])/pow(xB-xA,2)"),
                                      degree=2, amp=0.0, U=U_return, xA=0.0, xB=xL)
        u_return_right = df.Expression(("0.0", "-4.0*amp*U*(x[0]-xA)*(xB-x[0])/pow(xB-xA,2)"),
                                       degree=2, amp=0.0, U=U_return, xA=xR, xB=Lx)
        bcu = [df.DirichletBC(V,u_in,boundaries,1),
               df.DirichletBC(V,u_return_left,boundaries,2),
               df.DirichletBC(V,u_return_right,boundaries,3),
               df.DirichletBC(V,df.Constant((0.0,0.0)),boundaries,4)]
        bcp = [df.DirichletBC(P,df.Constant(0.0),PressurePin(),method="pointwise")]

        # Oseen IPCS forms and solver order retained from the diagnostic baseline.
        # Only the scalar representation in buoyancy changes from CG1 to cell averages.
        f_buoy = df.as_vector((0.0, g*beta*(c_n-c0)))
        F1 = ((1.0/dt)*df.inner(u-u_n,v)*df.dx
              + df.inner(df.dot(u_n,df.nabla_grad(u)),v)*df.dx
              + nu_visc*df.inner(df.grad(u),df.grad(v))*df.dx
              - p_n*df.div(v)*df.dx - df.inner(f_buoy,v)*df.dx)
        a1, L1 = df.lhs(F1), df.rhs(F1)
        a2 = df.inner(df.grad(p),df.grad(q))*df.dx
        L2 = df.inner(df.grad(p_n),df.grad(q))*df.dx-(1.0/dt)*df.div(u_star)*q*df.dx
        a3 = df.inner(u,v)*df.dx
        L3 = df.inner(u_star,v)*df.dx-dt*df.inner(df.grad(p_new-p_n),v)*df.dx
        A2 = df.assemble(a2)
        for bc in bcp:
            bc.apply(A2)
        tentative_solver = df.LUSolver()
        pressure_solver = df.LUSolver(A2)
        correction_solver = df.LUSolver()

        grid = BalancedFluxGrid(nx, ny, Lx, Ly)
        map_x, map_y = velocity_flux_maps(V, nx, ny, Lx, Ly)
        # Runtime check of the P2 component/coordinate mapping using a polynomial.
        probe = df.interpolate(df.Expression(("1+x[0]*x[0]+2*x[1]*x[1]",
                                               "2+3*x[0]*x[0]+x[1]*x[1]"), degree=2), V)
        x_edges, y_edges = np.linspace(0,Lx,nx+1), np.linspace(0,Ly,ny+1)
        px = np.asarray(map_x.dot(probe.vector().get_local())).reshape(ny,nx+1)
        py = np.asarray(map_y.dot(probe.vector().get_local())).reshape(ny+1,nx)
        ex = hy*(1+x_edges[None,:]**2)+(2.0/3.0)*np.diff(y_edges**3)[:,None]
        ey = hx*(2+y_edges[:,None]**2)+np.diff(x_edges**3)[None,:]
        if max(np.max(np.abs(px-ex)),np.max(np.abs(py-ey))) > 1e-12:
            raise RuntimeError("P2 face-integration startup test failed")
        del probe

        # Rectangular FV cell -> two DG0 triangle values, with no projection/smoothing.
        dg_to_rect = np.full(C.dim(),-1,dtype=int)
        triangle_counts = np.zeros(nx*ny,dtype=int)
        for cell in df.cells(mesh):
            point = cell.midpoint()
            i, j = int(math.floor(point.x()/hx)), int(math.floor(point.y()/hy))
            if not (0 <= i < nx and 0 <= j < ny):
                raise RuntimeError("Unexpected triangle midpoint")
            if abs(cell.volume()-hx*hy/2.0) > 1e-14:
                raise RuntimeError("Velocity mesh does not have two equal triangles per FV cell")
            dofs = C.dofmap().cell_dofs(cell.index())
            if len(dofs) != 1 or dg_to_rect[int(dofs[0])] != -1:
                raise RuntimeError("Invalid DG0 cell map")
            dg_to_rect[int(dofs[0])] = j*nx+i
            triangle_counts[j*nx+i] += 1
        if np.any(dg_to_rect < 0) or np.any(triangle_counts != 2):
            raise RuntimeError("Incomplete rectangular-cell to DG0 mapping")

        def set_scalar(values):
            c_n.vector().set_local(values.ravel()[dg_to_rect])
            c_n.vector().apply("insert")

        # Verify mapping with a nonuniform field, then restore ambient initial state.
        map_test = np.linspace(0.0,1.0,nx*ny).reshape(ny,nx)
        set_scalar(map_test)
        if abs(float(df.assemble(c_n*df.dx))-hx*hy*np.sum(map_test)) > 1e-11:
            raise RuntimeError("FV to DG0 mapping does not preserve the scalar integral")
        c = np.ones((ny,nx))
        set_scalar(c)
        nozzle = aligned_nozzle_mask(nx, Lx, d_nozzle)
        centres = 0.5*(x_edges[:-1]+x_edges[1:])
        left, right = centres < xL, centres > xR
        # Exact prescribed bottom face integrals: Simpson is exact for each parabola.
        bottom_x = np.stack((x_edges[:-1], centres, x_edges[1:]), axis=1)
        bottom_values = np.zeros_like(bottom_x)
        bottom_values[nozzle] = 4*U_in*(bottom_x[nozzle]-xL)*(xR-bottom_x[nozzle])/d_nozzle**2
        bottom_values[left] = -4*U_return*bottom_x[left]*(xL-bottom_x[left])/xL**2
        bottom_values[right] = -4*U_return*(bottom_x[right]-xR)*(Lx-bottom_x[right])/(Lx-xR)**2
        prescribed_bottom = hx*np.sum(bottom_values*np.array([1,4,1])[None,:],axis=1)/6.0

        config = {"method":"Oseen IPCS + conservative rectangular FV scalar",
                  "source_baseline":"IPSC_0.001.py", "fresh_start":True,
                  "verified_pilot_source":"cartesian_ipcs_conservative_5s.py",
                  "verified_pilot_sha256":VERIFIED_BASELINE_SHA256,
                  "verification_case":args.case,
                  "script_sha256":hashlib.sha256(open(__file__,"rb").read()).hexdigest(),
                  "dolfin_version":df.__version__, "numpy_version":np.__version__,
                  "scipy_version":scipy.__version__, "mpi_processes":1,
                  "domain_m":[Lx,Ly],"velocity_mesh_divisions":[nx,ny],
                  "scalar_grid_cells":[nx,ny],"scalar_representation":"rectangular cell averages, DG0 on paired triangles for buoyancy/output",
                  "maximum_inlet_velocity_m_s":U_in,"maximum_return_velocity_m_s":U_return,
                  "nozzle_width_m":d_nozzle,"nu_m2_s":float(nu_visc),"D_m2_s":D_value,
                  "g_m_s2":float(g),"beta":float(beta),"ambient_c":1.0,"inlet_c":0.0,
                  "dt_s":dt_value,"requested_steps":args.steps,"requested_duration_s":dt_value*args.steps,
                  "ramp_s":ramp_time,"ramp_type":"half cosine, velocity only",
                  "field_write_every_steps":write_every,"budget_write_every_steps":1,
                  "field_interval_s":0.25,"log_every_steps":log_every,"log_interval_s":0.1,
                  "nozzle_bottom_face_count":int(np.sum(nozzle)),
                  "snapshot_flux_fields":"raw_fx_m2_s/raw_fy_m2_s are IPCS P2 integrals; transport_fx_m2_s/transport_fy_m2_s are locally balanced fluxes",
                  "scalar_time_scheme":"backward Euler","scalar_advection":"first-order conservative upwind",
                  "scalar_diffusion":"orthogonal two-point flux; nozzle Dirichlet, all other boundaries zero diffusive flux",
                  "transport_velocity":"P2 face integrals, interior-only weighted graph projection for local volume balance",
                  "exterior_fluxes":"taken from IPCS velocity, checked against prescribed profiles, unchanged by projection",
                  "scalar_clipping_enabled":False,"global_scalar_mass_rescaling":False,
                  "bounds_stop_tolerance":1e-10,
                  "interpretation":"Matched five-second verification case. Compare raw fields at equal physical times with the baseline; one comparison does not establish full convergence. First-order upwind introduces numerical diffusion.",
                  "budget_signs":"fresh inventory change = cumulative advective input - output + signed diffusive input; no clipping or divergence source",
                  "references":["https://arxiv.org/abs/2204.07480",
                                "https://fenicsproject.discourse.group/t/how-to-obtain-coordinates-of-nodes-how-to-work-with-dofmap/8507"]}
        with open(os.path.join(folder,"budget_configuration.json"),"x") as f:
            json.dump(config,f,indent=2,allow_nan=False)
            f.write("\n")
        for field in ("u","p","c"):
            out = df.XDMFFile(mesh.mpi_comm(),os.path.join(folder,field+".xdmf"))
            out.parameters["flush_output"] = True
            out.parameters["functions_share_mesh"] = True
            out.parameters["rewrite_function_mesh"] = False
            outputs.append(out)
        def save_fields(step, time, fx=None, fy=None, fx_raw=None, fy_raw=None):
            for out, value in zip(outputs,(u_new,p_new,c_n)):
                out.write(value,time)
            data = dict(c=c, time_s=time, step=step, x_m=centres,
                        y_m=0.5*(y_edges[:-1]+y_edges[1:]))
            if fx is not None:
                data.update(transport_fx_m2_s=fx,transport_fy_m2_s=fy,
                            raw_fx_m2_s=fx_raw,raw_fy_m2_s=fy_raw)
            np.savez_compressed(os.path.join(folder,"fv_step_{:07d}.npz".format(step)),**data)
            if step == args.steps:
                np.savez_compressed(os.path.join(folder,"fv_final.npz"),**data)
        zero_fx = np.zeros((ny,nx+1))
        zero_fy = np.zeros((ny+1,nx))
        save_fields(0,0.0,zero_fx,zero_fy,zero_fx,zero_fy)
        cumulative_in = cumulative_out = cumulative_diff = 0.0
        budget_writer = None
        budget_file = open(os.path.join(folder,"scalar_budget.csv"),"x",newline="")
        status["status"] = "running"
        write_status()
        print("Conservative Cartesian pilot: NO CLIPPING; {} steps = {:.3f} s".format(args.steps,args.steps*dt_value),flush=True)
        print("Output: "+folder,flush=True)
        for step in range(1,args.steps+1):
            time = step*dt_value
            status.update(attempted_step=step,attempted_time_s=time)
            ramp = 0.5*(1.0-math.cos(math.pi*time/ramp_time)) if time < ramp_time else 1.0
            u_in.amp = u_return_left.amp = u_return_right.amp = ramp
            A1, b1 = df.assemble(a1), df.assemble(L1)
            for bc in bcu:
                bc.apply(A1,b1)
            tentative_solver.solve(A1,u_star.vector(),b1)
            b2 = df.assemble(L2)
            for bc in bcp:
                bc.apply(b2)
            pressure_solver.solve(p_new.vector(),b2)
            A3, b3 = df.assemble(a3), df.assemble(L3)
            for bc in bcu:
                bc.apply(A3,b3)
            correction_solver.solve(A3,u_new.vector(),b3)
            uv = u_new.vector().get_local()
            velocity_norm = float(u_new.vector().norm("linf"))
            if not np.all(np.isfinite(uv)) or velocity_norm > 5.0:
                raise RuntimeError("Nonfinite velocity or velocity infinity norm above 5 m/s")
            fx_raw = np.asarray(map_x.dot(uv)).reshape(ny,nx+1)
            fy_raw = np.asarray(map_y.dot(uv)).reshape(ny+1,nx)
            boundary_error = max(np.max(np.abs(fy_raw[0]-ramp*prescribed_bottom)),
                                 np.max(np.abs(fy_raw[-1])),np.max(np.abs(fx_raw[:,0])),
                                 np.max(np.abs(fx_raw[:,-1])))
            if boundary_error > 1e-12:
                raise RuntimeError("IPCS boundary face fluxes differ from the prescribed parabolas/walls")
            fx, fy, balance_info = grid.balance(fx_raw,fy_raw)
            candidate, scalar_info = grid.step(c,fx,fy,dt_value,D_value,nozzle,boundary_c={"left":1.0,"right":1.0,"top":1.0,
                "bottom":np.where(nozzle,0.0,1.0)})
            # Diagnostics are written before any failed candidate can enter buoyancy.
            cumulative_in += dt_value*scalar_info["fresh_adv_in_m2_s"]
            cumulative_out += dt_value*scalar_info["fresh_adv_out_m2_s"]
            cumulative_diff += dt_value*scalar_info["fresh_diff_in_m2_s"]
            inventory = float(hx*hy*np.sum(1.0-candidate))
            cum_residual = inventory-(cumulative_in-cumulative_out+cumulative_diff)
            row = dict(step=step,time_s=time,ramp=ramp,
                       vf_m2=inventory,c_min=float(np.min(candidate)),c_max=float(np.max(candidate)),
                       velocity_linf_m_s=velocity_norm,boundary_flux_error_max_m2_s=float(boundary_error),
                       volume_nozzle_in_m2_s=float(np.sum(fy[0,nozzle])),
                       volume_left_return_out_m2_s=float(-np.sum(fy[0,left])),
                       volume_right_return_out_m2_s=float(-np.sum(fy[0,right])),
                       cum_fresh_adv_in_m2=cumulative_in,cum_fresh_adv_out_m2=cumulative_out,
                       cum_fresh_diff_in_m2=cumulative_diff,cumulative_budget_residual_m2=cum_residual,
                       clipping_change_m2=0.0,cum_clipping_change_m2=0.0)
            row.update(balance_info)
            row.update(scalar_info)
            # L2 divergence of the ORIGINAL CG2 velocity is diagnostic only; the FV
            # solver uses locally balanced normal fluxes. Empty cells mark unsampled rows.
            sampled = step <= 5 or step % log_every == 0 or step == args.steps
            row["ipcs_divergence_l2"] = float(df.assemble(df.div(u_new)**2*df.dx))**0.5 if sampled else ""
            row["tentative_divergence_l2"] = float(df.assemble(df.div(u_star)**2*df.dx))**0.5 if sampled else ""
            if budget_writer is None:
                budget_writer = csv.DictWriter(budget_file,fieldnames=list(row))
                budget_writer.writeheader()
            budget_writer.writerow(row)
            failure = None
            if not np.all(np.isfinite(candidate)):
                failure = "Nonfinite scalar"
            elif np.min(candidate) < -1e-10 or np.max(candidate) > 1+1e-10:
                failure = "Scalar bounds failed; values were NOT clipped"
            elif abs(scalar_info["step_budget_residual_m2"]) > 5e-13:
                failure = "Single-step scalar budget failed"
            elif abs(cum_residual) > 2e-10+1e-7*(abs(cumulative_in)+abs(cumulative_diff)):
                failure = "Cumulative scalar budget failed"
            if failure:
                budget_file.flush()
                np.savez_compressed(os.path.join(folder,"failed_candidate.npz"),c=candidate,
                                    time_s=time,fx=fx,fy=fy,c_previous=c)
                raise RuntimeError(failure)
            c = candidate
            set_scalar(c)
            u_n.assign(u_new)
            p_n.assign(p_new)
            status.update(last_completed_step=step,last_completed_time_s=time)
            if sampled:
                budget_file.flush()
                write_status()
                print("Step {}/{} t={:.3f} s: c=[{:.9g}, {:.9g}], V_f={:.9g} m2, budget R={:.3e} m2".format(
                      step,args.steps,time,np.min(c),np.max(c),inventory,cum_residual),flush=True)
                print("    IPCS div L2={:.3e}; |u|inf={:.3e} m/s".format(
                      row["ipcs_divergence_l2"],velocity_norm),flush=True)
            if step == 1 or step % write_every == 0 or step == args.steps:
                mapped_inventory = float(df.assemble((1.0-c_n)*df.dx))
                if abs(mapped_inventory-inventory) > 1e-11:
                    raise RuntimeError("DG0 output/buoyancy scalar integral differs from FV inventory")
                save_fields(step,time,fx,fy,fx_raw,fy_raw)
        status.update(status="completed",final_c_min=float(np.min(c)),final_c_max=float(np.max(c)),
                      final_vf_m2=inventory,final_budget_residual_m2=cum_residual)
        print("Pilot completed. Send scalar_budget.csv, budget_configuration.json and run_status.json for review.",flush=True)
    except Exception as exc:
        status.update(status="failed",error=str(exc))
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        close_errors = []
        if budget_file is not None:
            try:
                budget_file.close()
            except Exception as exc:
                close_errors.append(str(exc))
        for out in outputs:
            try:
                out.close()
            except Exception as exc:
                close_errors.append(str(exc))
        if close_errors:
            status.update(status="failed", output_close_errors=close_errors)
        write_status()
        if close_errors and not active_error:
            raise RuntimeError("Output close failed: " + "; ".join(close_errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default=None,help="New or empty output directory")
    parser.add_argument("--case",choices=("baseline","mesh","timestep"),default="baseline",
                        help="Matched five-second verification case (default: baseline)")
    parser.add_argument("--self-test",action="store_true",help="Run scalar and flux tests without FEniCS")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    args.steps = verification_settings(args.case)["steps"]
    if args.output is None:
        args.output = os.environ.get("CARTESIAN_FV_OUTPUT",
            "cartesian_"+args.case+"_check_5s_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_pilot(args)


if __name__ == "__main__":
    main()
