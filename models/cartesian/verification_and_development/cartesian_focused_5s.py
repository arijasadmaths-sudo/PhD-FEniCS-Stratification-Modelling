#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two short Cartesian checks using graded rectangular finite-volume cells.

focus1250: 1.25 mm core, 178 x 101 FV cells (compare uniform480 at 5 s).
focus625:  0.625 mm core, 252 x 154 FV cells (then compare local refinement).
Fine core x=[.26,.34] m, z=[0,.06] m; full domain [.0,.6] x [.0,.3] m.
The outer spacing grows smoothly to <=5 mm; nozzle edges stay aligned.

Same physical parameters and Oseen-IPCS equations as the completed480 run;
same conservative implicit upwind scalar with molecular diffusion D=1e-7.
Variable cell areas and face-length/centre-distance transmissibilities are
used consistently. Interior face fluxes are locally balanced; prescribed
exterior fluxes are unchanged. NO clipping or global scalar mass rescaling.
Pressure and velocity-correction matrices are constant and reused; the Oseen
matrix is assembled and factorised at every step.

Each case starts from rest, dt=.001 s, T=5 s. Legacy DOLFIN2019.1, one MPIrank.
  python3 cartesian_focused_5s.py --case focus1250 --output NEW_DIRECTORY
  python3 cartesian_focused_5s.py --case focus625 --output NEW_DIRECTORY
  python3 cartesian_focused_5s.py --self-test
  python3 cartesian_focused_5s.py --case focus625 --output NEW_DIRECTORY \
      --resume /absolute/path/checkpoint_latest.npz

Snapshots every .25 s contain raw c, cell edges/areas and raw/projectedfluxes.
The restart file additionally saves full FEvelocity/pressure DOFs, exact mesh
and DOFcoordinates, cumulative scalar budgets, and the source fingerprint.
Restart requires exactly this script, same case and compatible FE DOFlayout;
it starts a new output segment and never overwrites the old one.
SIGTERM/SIGUSR1 requests a checkpoint after the current completed step.
A scheduler hard kill can prevent that final save; periodic checkpoints remain.

These runs test spatial sensitivity; they do not establish late-time accuracy.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import signal
import sys
import tempfile
import time as clock
import unittest
import warnings
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized, spsolve, MatrixRankWarning

CASE_H = {'focus1250':.00125, 'focus625':.000625}
BASELINE_SHA256='92124a6f7ed3ca6c2e528bff298d6b26a347f9ddd6f5d53c33dd58485871850a'

def atomic_json(path,data):
    path=os.path.abspath(path)
    fd,temp=tempfile.mkstemp(prefix='.write_',suffix='.json',dir=os.path.dirname(path))
    try:
        with os.fdopen(fd,'w') as out:
            json.dump(data,out,indent=2,allow_nan=False); out.write('\n')
            out.flush(); os.fsync(out.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def atomic_npz(path,**data):
    path=os.path.abspath(path)
    fd,temp=tempfile.mkstemp(prefix='.write_',suffix='.npz',dir=os.path.dirname(path))
    try:
        with os.fdopen(fd,'wb') as out:
            np.savez_compressed(out,**data)
            out.flush(); os.fsync(out.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def source_hash():
    with open(__file__,'rb') as f: return hashlib.sha256(f.read()).hexdigest()

def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def graded_outer_widths(length, first_width, max_width=0.005,
                        max_ratio=1.15):
    """Widths directed from fine core towards an outer boundary.

    Select the smallest number of cells reachable with the specified growth
    bound, then solve for a constant growth factor whose capped widths sum
    to the required length. The first cell matches the core spacing exactly.
    """
    length, first_width, max_width, max_ratio = map(
        float, (length, first_width, max_width, max_ratio))
    if not (length >= first_width > 0 and
            max_width >= first_width and max_ratio > 1):
        raise ValueError("Invalid outer-grid dimensions or growth limit")

    def sequence(n, ratio):
        # The logarithmic construction cannot overflow for a large n.
        return np.exp(np.minimum(
            np.log(first_width) + np.arange(n) * np.log(ratio),
            np.log(max_width)))

    n = max(1, int(np.ceil(length / max_width)))
    while sequence(n, max_ratio).sum() < length - 1.e-14:
        n += 1
    if n * first_width > length + 1.e-14:
        raise ValueError("No grid satisfies the requested first width")
    low, high = 1.0, max_ratio
    for _ in range(90):
        midpoint = (low + high) / 2
        if sequence(n, midpoint).sum() > length:
            high = midpoint
        else:
            low = midpoint
    widths = sequence(n, (low + high) / 2)
    # Keep both the first width and the outer endpoint exact to rounding.
    widths[0] = first_width
    widths[-1] += length - widths.sum()
    if np.max(widths) > max_width * (1 + 1.e-12):
        raise AssertionError("Maximum cell width exceeded")
    if n > 1 and np.max(widths[1:] / widths[:-1]) > max_ratio + 1.e-12:
        raise AssertionError("Maximum growth ratio exceeded")
    return widths


def make_graded_grid(core_h, Lx=0.60, Ly=0.30,
                     core_x=(0.26, 0.34), core_z_top=0.06,
                     max_width=0.005, max_ratio=1.15):
    """Return x,z edge arrays suitable for rectangular finite-volume cells."""
    core_h = float(core_h)
    xl, xr = map(float, core_x)

    def exact_count(length):
        n = int(round(length / core_h))
        if n < 1 or abs(n * core_h - length) > 1.e-12:
            raise ValueError("Fine core dimensions must divide by core_h")
        return n

    left = graded_outer_widths(xl, core_h, max_width, max_ratio)
    right = graded_outer_widths(Lx-xr, core_h, max_width, max_ratio)
    upper = graded_outer_widths(Ly-core_z_top, core_h,
                                max_width, max_ratio)
    xcore = np.linspace(xl, xr, exact_count(xr-xl)+1)
    zcore = np.linspace(0., core_z_top, exact_count(core_z_top)+1)
    xedges = np.concatenate((
        (xl-np.cumsum(left))[::-1],
        xcore,
        xr+np.cumsum(right)))
    zedges = np.concatenate((zcore, core_z_top+np.cumsum(upper)))
    xedges[0], xedges[-1] = 0., Lx
    zedges[0], zedges[-1] = 0., Ly
    return xedges, zedges


class BalancedFluxGrid(object):
    def __init__(self,x_edges,y_edges):
        self.x_edges=np.asarray(x_edges,dtype=float).copy()
        self.y_edges=np.asarray(y_edges,dtype=float).copy()
        for e in (self.x_edges,self.y_edges):
            if e.ndim!=1 or len(e)<2 or not np.all(np.isfinite(e)) or np.any(np.diff(e)<=0):
                raise ValueError('Strictly increasing finite cell edges required')
        self.dx,self.dy=np.diff(self.x_edges),np.diff(self.y_edges)
        self.nx,self.ny=len(self.dx),len(self.dy)
        self.xc=.5*(self.x_edges[1:]+self.x_edges[:-1])
        self.yc=.5*(self.y_edges[1:]+self.y_edges[:-1])
        self.volume=self.dy[:,None]*self.dx[None,:]
        self.n=self.nx*self.ny
        ids=np.arange(self.n).reshape(self.ny,self.nx); self.ids=ids
        self.u=np.concatenate((ids[:,:-1].ravel(),ids[:-1,:].ravel()))
        self.v=np.concatenate((ids[:,1:].ravel(),ids[1:,:].ravel()))
        self.n_xinterior=self.ny*(self.nx-1)
        self.weights=np.concatenate(((self.dy[:,None]/np.diff(self.xc)[None,:]).ravel(),
            (self.dx[None,:]/np.diff(self.yc)[:,None]).ravel()))
        self.edge_rows=np.concatenate((self.u,self.v,self.u,self.v))
        self.edge_cols=np.concatenate((self.u,self.v,self.v,self.u))
        w=self.weights
        lap=coo_matrix((np.concatenate((w,w,-w,-w)),(self.edge_rows,self.edge_cols)),
            shape=(self.n,self.n)).tocsc()
        self._project_solve=factorized(lap[1:,1:]) if self.n>1 else None
        self.boundary_ids=np.concatenate((ids[:,0],ids[:,-1],ids[0,:],ids[-1,:]))

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
            'cell_divergence_before_l2_sinv': float(np.sqrt(np.sum(before**2/self.volume.ravel()))),
            'cell_divergence_after_l2_sinv': float(np.sqrt(np.sum(after**2/self.volume.ravel()))),
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
        diag = self.volume.ravel().copy() / dt
        rhs = c.ravel() * (self.volume.ravel() / dt)
        np.add.at(diag, self.boundary_ids, np.maximum(qout, 0.))
        np.add.at(rhs, self.boundary_ids, -np.minimum(qout, 0.) * cb)
        nozzle_ids = self.ids[0, mask]
        nozzle_cb = cb[2 * self.ny:2 * self.ny + self.nx][mask]
        nozzle_T = 2. * float(D) * self.dx[mask] / self.dy[0]
        np.add.at(diag, nozzle_ids, nozzle_T)
        np.add.at(rhs, nozzle_ids, nozzle_T * nozzle_cb)
        all_ids = np.arange(self.n)
        matrix = coo_matrix((np.concatenate((data, diag)),
                             (np.concatenate((self.edge_rows, all_ids)),
                              np.concatenate((self.edge_cols, all_ids)))),
                            shape=(self.n, self.n)).tocsc()
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            solved = spsolve(matrix, rhs)
        if not np.all(np.isfinite(solved)):
            raise RuntimeError("Non-finite scalar solve.")
        cnew = solved.reshape(self.ny, self.nx)
        cup = np.where(qout >= 0., solved[self.boundary_ids], cb)
        fresh_adv_outward = qout * (1. - cup)
        fresh_adv_in = -float(np.sum(fresh_adv_outward[qout < 0.]))
        fresh_adv_out = float(np.sum(fresh_adv_outward[qout >= 0.]))
        # For f=1-c, inward diffusive fresh flux is outward diffusive salt flux.
        fresh_diff_in = float(np.sum(nozzle_T * (solved[nozzle_ids] - nozzle_cb)))
        vf_old = float(np.sum(self.volume * (1. - c)))
        vf_new = float(np.sum(self.volume * (1. - cnew)))
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


def nozzle_mask(x_edges):
    x_edges=np.asarray(x_edges)
    for v in (.295,.305):
        if np.min(np.abs(x_edges-v))>1e-12: raise ValueError('Nozzle edges must align')
    mask=(x_edges[:-1]>=.295-1e-12)&(x_edges[1:]<=.305+1e-12)
    if abs(np.diff(x_edges)[mask].sum()-.01)>1e-12: raise ValueError('Nozzle coverage incorrect')
    return mask

def prescribed_bottom_flux(x_edges):
    x_edges=np.asarray(x_edges); dx=np.diff(x_edges)
    centre=.5*(x_edges[:-1]+x_edges[1:])
    nozz=nozzle_mask(x_edges)
    left,right=centre<.295,centre>.305
    nodes=np.stack((x_edges[:-1],centre,x_edges[1:]),axis=1)
    vals=np.zeros_like(nodes); U=.001; Ur=U*.01/.59
    vals[nozz]=4*U*(nodes[nozz]-.295)*(.305-nodes[nozz])/.01**2
    vals[left]=-4*Ur*nodes[left]*(.295-nodes[left])/.295**2
    vals[right]=-4*Ur*(nodes[right]-.305)*(.6-nodes[right])/.295**2
    return dx*np.sum(vals*np.array([1,4,1])[None,:],axis=1)/6

def half_edges(e):
    a=np.empty(2*len(e)-1); a[::2]=e; a[1::2]=.5*(e[:-1]+e[1:]); return a

def match_axis(values, nodes):
    values=np.asarray(values); nodes=np.asarray(nodes)
    k=np.searchsorted(nodes,values); k=np.minimum(k,len(nodes)-1)
    prev=np.maximum(k-1,0)
    k=np.where(np.abs(nodes[prev]-values)<np.abs(nodes[k]-values),prev,k)
    if np.max(np.abs(nodes[k]-values))>1e-11: raise RuntimeError('FE DOF is off tensor half-grid')
    return k

def velocity_flux_maps(V,x_edges,y_edges):
    # Simpson's rule integrates each physical P2 edge trace exactly.
    nx,ny=len(x_edges)-1,len(y_edges)-1
    dx,dy=np.diff(x_edges),np.diff(y_edges)
    xy=np.asarray(V.tabulate_dof_coordinates()).reshape((-1,2))
    if xy.shape[0]!=V.dim(): raise RuntimeError('Unexpected vector DOF layout')
    ix,iy=match_axis(xy[:,0],half_edges(x_edges)),match_axis(xy[:,1],half_edges(y_edges))
    matrices=[]
    for comp in (0,1):
        dofs=np.asarray(V.sub(comp).dofmap().dofs(),dtype=int)
        lookup={ (int(ix[d]),int(iy[d])):int(d) for d in dofs }
        if len(lookup)!=len(dofs): raise RuntimeError('Duplicate P2 component coordinate')
        rows=[]; cols=[]; data=[]
        if comp==0:
            triples=[((2*i,2*j),(2*i,2*j+1),(2*i,2*j+2)) for j in range(ny) for i in range(nx+1)]
            lengths=np.repeat(dy,nx+1)
        else:
            triples=[((2*i,2*j),(2*i+1,2*j),(2*i+2,2*j)) for j in range(ny+1) for i in range(nx)]
            lengths=np.tile(dx,ny+1)
        for row,triple in enumerate(triples):
            for key,weight in zip(triple,(1.,4.,1.)):
                if key not in lookup: raise RuntimeError('Missing P2 edge DOF: '+str(key))
                rows.append(row); cols.append(lookup[key]); data.append(lengths[row]*weight/6)
        matrices.append(sparse.csr_matrix((data,(rows,cols)),shape=(len(triples),V.dim())))
    return tuple(matrices)

def validate_restart(a,case,fingerprint,grid,velocity_xy,pressure_xy,mesh_xy,mesh_cells):
    meta=json.loads(str(a['metadata_json']))
    if meta.get('format_version')!=1 or meta.get('case')!=case or meta.get('fingerprint')!=fingerprint:
        raise ValueError('Checkpoint does not match this script and case; do not bypass this guard')
    if not np.array_equal(a['x_edges_m'],grid.x_edges) or not np.array_equal(a['y_edges_m'],grid.y_edges):
        raise ValueError('Checkpoint mesh differs')
    for key,expected in [('velocity_dof_xy_m',velocity_xy),('pressure_dof_xy_m',pressure_xy),
                         ('mesh_coordinates_m',mesh_xy),('mesh_cells',mesh_cells)]:
        if not np.array_equal(a[key],expected): raise ValueError('Incompatible restart DOF/mesh ordering: '+key)
    step=int(a['step'])
    if step<0 or step>=5000 or abs(float(a['time_s'])-.001*step)>1e-12:
        raise ValueError('Checkpoint is already complete or has an invalid time')
    if a['c'].shape!=(grid.ny,grid.nx) or a['u_dofs'].shape!=(len(velocity_xy),) or a['p_dofs'].shape!=(len(pressure_xy),):
        raise ValueError('Checkpoint field shapes do not match')
    for key in ('c','u_dofs','p_dofs'):
        if not np.all(np.isfinite(a[key])): raise ValueError('Nonfinite checkpoint field')
    if a['c'].min() < -1e-10 or a['c'].max() > 1+1e-10: raise ValueError('Checkpoint scalar out of bounds')
    inventory=float(np.sum(grid.volume*(1-a['c'])))
    cum=np.asarray(a['cumulative_budget_m2'],dtype=float)
    if cum.shape!=(3,) or not np.all(np.isfinite(cum)): raise ValueError('Invalid cumulative checkpoint budget')
    if abs(inventory-float(a['vf_m2']))>1e-13 or abs(inventory-(cum[0]-cum[1]+cum[2]))>2e-10:
        raise ValueError('Checkpoint inventory/budget mismatch')
    return step,cum

def run_case(args):
    import dolfin as df
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size()!=1:
        raise RuntimeError('Use exactly one MPI rank with the supplied launcher')
    if not df.__version__.startswith('2019.1'):
        raise RuntimeError('This programme requires legacy DOLFIN2019.1')
    df.parameters['std_out_all_processes']=False
    df.set_log_level(30)
    folder=os.path.abspath(args.output)
    if os.path.exists(folder) and (not os.path.isdir(folder) or os.listdir(folder)):
        raise RuntimeError('Output directory must be new or empty: '+folder)
    os.makedirs(folder,exist_ok=True)
    wall_start=clock.perf_counter(); prior_wall=0.
    phases={k:0. for k in ('tentative_s','pressure_s','correction_s','flux_projection_s','scalar_s','output_s')}
    status={'status':'starting','verification_case':args.case,'requested_steps':5000,
            'requested_duration_s':5.,'last_completed_step':0,'last_completed_time_s':0.,
            'started_utc':utcnow(),'resume_from':os.path.abspath(args.resume) if args.resume else None}
    def write_status():
        status.update(updated_utc=utcnow(),elapsed_wall_s=clock.perf_counter()-wall_start,
                      cumulative_wall_s=prior_wall+clock.perf_counter()-wall_start,phase_wall_s=phases.copy())
        atomic_json(os.path.join(folder,'run_status.json'),status)
    write_status()
    stop_requested=[False]
    def request_stop(signum,frame):
        stop_requested[0]=True
        print('Stop requested: finishing the current step and saving a restart checkpoint.',flush=True)
    previous_handlers={s:signal.signal(s,request_stop) for s in (signal.SIGTERM,signal.SIGUSR1)}
    outputs=[]; budget_file=None; save_checkpoint=None; accepted_ready=False
    try:
        x_edges,y_edges=make_graded_grid(CASE_H[args.case])
        grid=BalancedFluxGrid(x_edges,y_edges)
        nx,ny=grid.nx,grid.ny; Lx,Ly=.6,.3
        dx,dy=grid.dx,grid.dy; centres=grid.xc
        nozz=nozzle_mask(x_edges); left=centres<.295; right=centres>.305
        expected_shape={'focus1250':(178,101),'focus625':(252,154)}[args.case]
        if (nx,ny)!=expected_shape: raise RuntimeError('Unexpected grid shape')
        dt_value=.001; D_value=1e-7; ramp_time=.5
        d_nozzle=.01; U_in=.001; xL,xR=.295,.305; U_return=U_in*d_nozzle/.59
        mesh=df.RectangleMesh(df.Point(0.,0.),df.Point(Lx,Ly),nx,ny)
        xy=mesh.coordinates()
        xi=np.rint(xy[:,0]/Lx*nx).astype(int)
        yi=np.rint(xy[:,1]/Ly*ny).astype(int)
        xy[:,0]=x_edges[xi]; xy[:,1]=y_edges[yi]
        mesh.bounding_box_tree().build(mesh)
        tol=1e-10

        class Nozzle(df.SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and df.near(x[1],0.,tol) and xL-tol<=x[0]<=xR+tol
        class LeftReturn(df.SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and df.near(x[1],0.,tol) and x[0]<=xL+tol
        class RightReturn(df.SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and df.near(x[1],0.,tol) and x[0]>=xR-tol
        class SolidWalls(df.SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and (df.near(x[0],0.,tol) or df.near(x[0],Lx,tol) or df.near(x[1],Ly,tol))
        class PressurePin(df.SubDomain):
            def inside(self,x,on_boundary):
                return df.near(x[0],0.,tol) and df.near(x[1],Ly,tol)

        boundaries=df.MeshFunction('size_t',mesh,1,0)
        SolidWalls().mark(boundaries,4); LeftReturn().mark(boundaries,2)
        RightReturn().mark(boundaries,3); Nozzle().mark(boundaries,1)
        ds_sub=df.Measure('ds',domain=mesh,subdomain_data=boundaries)
        for marker,length in ((1,.01),(2,.295),(3,.295),(4,Lx+2*Ly)):
            actual=float(df.assemble(df.Constant(1.)*ds_sub(marker)))
            if abs(actual-length)>1e-10: raise RuntimeError('Boundary measure mismatch')
        if abs(float(df.assemble(df.Constant(1.)*df.dx(domain=mesh)))-.18)>1e-12:
            raise RuntimeError('Domain area changed')

        V=df.VectorFunctionSpace(mesh,'CG',2); P=df.FunctionSpace(mesh,'CG',1)
        C=df.FunctionSpace(mesh,'DG',0)
        u,v=df.TrialFunction(V),df.TestFunction(V)
        p,q=df.TrialFunction(P),df.TestFunction(P)
        u_n,u_star,u_new=df.Function(V),df.Function(V),df.Function(V)
        p_n,p_new=df.Function(P),df.Function(P)
        for value in (u_n,u_star,u_new): value.assign(df.Constant((0.,0.)))
        for value in (p_n,p_new): value.assign(df.Constant(0.))
        c_n=df.Function(C); c_n.assign(df.Constant(1.))
        c_n.rename('c','Raw salt fraction: 1 ambient, 0 source')
        u_new.rename('u','IPCS velocity'); p_new.rename('p','IPCS pressure')
        dt,nu_visc=df.Constant(dt_value),df.Constant(1e-6)
        g,beta,c0=df.Constant(9.81),df.Constant(-6.27e-3),df.Constant(1.)
        u_in=df.Expression(('0.0','4.0*amp*U*(x[0]-xL)*(xR-x[0])/pow(xR-xL,2)'),
            degree=2,amp=0.,U=U_in,xL=xL,xR=xR)
        u_left=df.Expression(('0.0','-4.0*amp*U*x[0]*(xL-x[0])/pow(xL,2)'),
            degree=2,amp=0.,U=U_return,xL=xL)
        u_right=df.Expression(('0.0','-4.0*amp*U*(x[0]-xR)*(Lx-x[0])/pow(Lx-xR,2)'),
            degree=2,amp=0.,U=U_return,xR=xR,Lx=Lx)
        bcu=[df.DirichletBC(V,u_in,boundaries,1),df.DirichletBC(V,u_left,boundaries,2),
             df.DirichletBC(V,u_right,boundaries,3),df.DirichletBC(V,df.Constant((0.,0.)),boundaries,4)]
        bcp=[df.DirichletBC(P,df.Constant(0.),PressurePin(),method='pointwise')]
        f_buoy=df.as_vector((0.,g*beta*(c_n-c0)))
        F1=((1./dt)*df.inner(u-u_n,v)*df.dx
            +df.inner(df.dot(u_n,df.nabla_grad(u)),v)*df.dx
            +nu_visc*df.inner(df.grad(u),df.grad(v))*df.dx
            -p_n*df.div(v)*df.dx-df.inner(f_buoy,v)*df.dx)
        a1,L1=df.lhs(F1),df.rhs(F1)
        a2=df.inner(df.grad(p),df.grad(q))*df.dx
        L2=df.inner(df.grad(p_n),df.grad(q))*df.dx-(1./dt)*df.div(u_star)*q*df.dx
        a3=df.inner(u,v)*df.dx
        L3=df.inner(u_star,v)*df.dx-dt*df.inner(df.grad(p_new-p_n),v)*df.dx
        A2=df.assemble(a2)
        for bc in bcp: bc.apply(A2)
        A3=df.assemble(a3)
        for bc in bcu: bc.apply(A3)
        tentative_solver=df.LUSolver()
        pressure_solver=df.LUSolver(A2)
        correction_solver=df.LUSolver(A3)
        # The BC locations/matrix rows are constant; prescribed values enter RHS.
        # No reuse flag is assumed: these are the same legacy solver objects.
        map_x,map_y=velocity_flux_maps(V,x_edges,y_edges)
        probe=df.interpolate(df.Expression(('1+x[0]*x[0]+2*x[1]*x[1]',
            '2+3*x[0]*x[0]+x[1]*x[1]'),degree=2),V)
        px=np.asarray(map_x.dot(probe.vector().get_local())).reshape(ny,nx+1)
        py=np.asarray(map_y.dot(probe.vector().get_local())).reshape(ny+1,nx)
        ex=dy[:,None]*(1+x_edges[None,:]**2)+(2./3.)*np.diff(y_edges**3)[:,None]
        ey=dx[None,:]*(2+y_edges[:,None]**2)+np.diff(x_edges**3)[None,:]
        if max(np.max(abs(px-ex)),np.max(abs(py-ey)))>1e-12:
            raise RuntimeError('P2 physical edge integration test failed')
        del probe

        dg_to_rect=np.full(C.dim(),-1,dtype=int); counts=np.zeros(nx*ny,dtype=int)
        for cell in df.cells(mesh):
            point=cell.midpoint()
            i=int(np.searchsorted(x_edges,point.x(),side='right')-1)
            j=int(np.searchsorted(y_edges,point.y(),side='right')-1)
            if not (0<=i<nx and 0<=j<ny): raise RuntimeError('Cell outside FV mesh')
            if abs(cell.volume()-grid.volume[j,i]/2)>1e-14: raise RuntimeError('Triangle area mismatch')
            dofs=C.dofmap().cell_dofs(cell.index())
            if len(dofs)!=1 or dg_to_rect[int(dofs[0])]!=-1: raise RuntimeError('Invalid DG0 map')
            dg_to_rect[int(dofs[0])]=j*nx+i; counts[j*nx+i]+=1
        if np.any(dg_to_rect<0) or np.any(counts!=2): raise RuntimeError('Incomplete DG0 map')
        def set_scalar(values):
            c_n.vector().set_local(values.ravel()[dg_to_rect]); c_n.vector().apply('insert')
        test_values=np.linspace(0.,1.,nx*ny).reshape(ny,nx)
        set_scalar(test_values)
        if abs(float(df.assemble(c_n*df.dx))-float(np.sum(grid.volume*test_values)))>1e-12:
            raise RuntimeError('FV-DG0 map does not preserve the scalar integral')
        c=np.ones((ny,nx)); set_scalar(c)
        velocity_xy=np.asarray(V.tabulate_dof_coordinates()).reshape((-1,2)).copy()
        pressure_xy=np.asarray(P.tabulate_dof_coordinates()).reshape((-1,2)).copy()
        mesh_xy=mesh.coordinates().copy(); mesh_cells=mesh.cells().copy()
        fingerprint=hashlib.sha256((source_hash()+'|'+args.case+'|dt=.001|T=5').encode()).hexdigest()
        cumulative=np.zeros(3); start_step=0
        if args.resume:
            with np.load(args.resume,allow_pickle=False) as f: restart={k:f[k] for k in f.files}
            start_step,cumulative=validate_restart(restart,args.case,fingerprint,grid,
                velocity_xy,pressure_xy,mesh_xy,mesh_cells)
            c=restart['c'].copy(); set_scalar(c)
            for f,key in [(u_n,'u_dofs'),(p_n,'p_dofs')]:
                f.vector().set_local(restart[key]); f.vector().apply('insert')
            u_new.assign(u_n); p_new.assign(p_n)
            prior_wall=float(restart['cumulative_wall_s'])
            status.update(last_completed_step=start_step,last_completed_time_s=start_step*dt_value)
        inventory=float(np.sum(grid.volume*(1-c)))
        prescribed_bottom=prescribed_bottom_flux(x_edges)
        config={'method':'Oseen IPCS + conservative FV on graded rectangular cells',
            'verification_case':args.case,'baseline_uniform480_sha256':BASELINE_SHA256,
            'script_sha256':source_hash(),'checkpoint_fingerprint':fingerprint,
            'dolfin_version':df.__version__,'numpy_version':np.__version__,'scipy_version':scipy.__version__,
            'mpi_processes':1,'fresh_start':not bool(args.resume),'resume_from':status['resume_from'],
            'initial_step':start_step,'initial_time_s':start_step*dt_value,
            'initial_vf_m2':inventory,'initial_cumulative_budget_m2':cumulative.tolist(),
            'domain_m':[Lx,Ly],'velocity_mesh_divisions':[nx,ny],'scalar_grid_cells':[nx,ny],
            'triangular_cells':mesh.num_cells(),'velocity_dofs':V.dim(),'pressure_dofs':P.dim(),
            'x_edges_m':x_edges.tolist(),'y_edges_m':y_edges.tolist(),
            'core_spacing_m':CASE_H[args.case],'core_x_m':[.26,.34],'core_z_m':[0.,.06],
            'maximum_outer_spacing_m':.005,'maximum_adjacent_spacing_ratio':1.15,
            'maximum_inlet_velocity_m_s':U_in,'maximum_return_velocity_m_s':U_return,
            'nozzle_width_m':d_nozzle,'nozzle_bottom_face_count':int(nozz.sum()),
            'nu_m2_s':1e-6,'D_m2_s':D_value,'g_m_s2':9.81,'beta':-.00627,
            'ambient_c':1.,'inlet_c':0.,'dt_s':dt_value,'requested_steps':5000,'requested_duration_s':5.,
            'ramp_s':ramp_time,'ramp_type':'half cosine, velocity only',
            'scalar_time_scheme':'backward Euler','scalar_advection':'first-order conservative upwind',
            'scalar_diffusion':'two-point physical face length / cell-centre distance; Dirichlet nozzle only',
            'transport_velocity':'P2 physical face integrals; interior weighted graph projection',
            'exterior_fluxes':'preserved exactly by projection',
            'scalar_clipping_enabled':False,'global_scalar_mass_rescaling':False,
            'budget_signs':'fresh inventory = cumulative adv input - output + signed diffusive input',
            'restart_budget':'new CSV segment; cumulative columns include accepted previous segment',
            'field_interval_s':.25,'checkpoint_interval_s':.25,'log_interval_s':.1,
            'bounds_stop_tolerance':1e-10,'constant_matrices_reused':['pressure','velocity correction'],
            'interpretation':'Compare focus1250 with uniform480 before interpreting focus625; not a late-time or globally uniform mesh convergence claim'}
        atomic_json(os.path.join(folder,'budget_configuration.json'),config)
        for field in ('u','p','c'):
            out=df.XDMFFile(mesh.mpi_comm(),os.path.join(folder,field+'.xdmf'))
            out.parameters['flush_output']=True; out.parameters['functions_share_mesh']=True
            out.parameters['rewrite_function_mesh']=False; outputs.append(out)
        def snapshot_data(step,fx,fy,rawfx,rawfy):
            return dict(c=c,time_s=step*dt_value,step=step,x_m=grid.xc,y_m=grid.yc,
                x_edges_m=x_edges,y_edges_m=y_edges,cell_area_m2=grid.volume,
                transport_fx_m2_s=fx,transport_fy_m2_s=fy,raw_fx_m2_s=rawfx,raw_fy_m2_s=rawfy)
        def save_fields(step,fx,fy,rawfx,rawfy):
            for out,value in zip(outputs,(u_new,p_new,c_n)): out.write(value,step*dt_value)
            data=snapshot_data(step,fx,fy,rawfx,rawfy)
            atomic_npz(os.path.join(folder,'fv_step_{:07d}.npz'.format(step)),**data)
            if step==5000: atomic_npz(os.path.join(folder,'fv_final.npz'),**data)
        def checkpoint_impl(step):
            metadata={'format_version':1,'case':args.case,'fingerprint':fingerprint,
                      'dt_s':dt_value,'source_sha256':source_hash()}
            atomic_npz(os.path.join(folder,'checkpoint_latest.npz'),
                metadata_json=json.dumps(metadata,sort_keys=True),step=step,time_s=step*dt_value,
                c=c,u_dofs=u_n.vector().get_local(),p_dofs=p_n.vector().get_local(),
                velocity_dof_xy_m=velocity_xy,pressure_dof_xy_m=pressure_xy,
                mesh_coordinates_m=mesh_xy,mesh_cells=mesh_cells,x_edges_m=x_edges,y_edges_m=y_edges,
                cumulative_budget_m2=cumulative,vf_m2=float(np.sum(grid.volume*(1-c))),
                cumulative_wall_s=prior_wall+clock.perf_counter()-wall_start)
        save_checkpoint=checkpoint_impl
        rawfx=np.asarray(map_x.dot(u_n.vector().get_local())).reshape(ny,nx+1)
        rawfy=np.asarray(map_y.dot(u_n.vector().get_local())).reshape(ny+1,nx)
        fx,fy,_=grid.balance(rawfx,rawfy)
        save_fields(start_step,fx,fy,rawfx,rawfy); save_checkpoint(start_step)
        accepted_ready=True
        budget_file=open(os.path.join(folder,'scalar_budget.csv'),'x',newline=''); writer=None
        status['status']='running'; write_status()
        print('Case {}: {} x {} FV cells, {} triangles; dt=.001s, target5s'.format(args.case,nx,ny,mesh.num_cells()),flush=True)
        print('Starting from step {}; output {}'.format(start_step,folder),flush=True)
        for step in range(start_step+1,5001):
            if stop_requested[0]: break
            t=step*dt_value; status.update(attempted_step=step,attempted_time_s=t)
            ramp=.5*(1-math.cos(math.pi*t/ramp_time)) if t<ramp_time else 1.
            u_in.amp=u_left.amp=u_right.amp=ramp
            tick=clock.perf_counter()
            A1,b1=df.assemble(a1),df.assemble(L1)
            for bc in bcu: bc.apply(A1,b1)
            tentative_solver.solve(A1,u_star.vector(),b1)
            phases['tentative_s']+=clock.perf_counter()-tick; tick=clock.perf_counter()
            b2=df.assemble(L2)
            for bc in bcp: bc.apply(b2)
            pressure_solver.solve(p_new.vector(),b2)
            phases['pressure_s']+=clock.perf_counter()-tick; tick=clock.perf_counter()
            b3=df.assemble(L3)
            for bc in bcu: bc.apply(b3)
            correction_solver.solve(u_new.vector(),b3)
            phases['correction_s']+=clock.perf_counter()-tick; tick=clock.perf_counter()
            uv=u_new.vector().get_local(); velocity_norm=float(u_new.vector().norm('linf'))
            if not np.all(np.isfinite(uv)) or velocity_norm>5.: raise RuntimeError('Velocity guard failed')
            rawfx=np.asarray(map_x.dot(uv)).reshape(ny,nx+1)
            rawfy=np.asarray(map_y.dot(uv)).reshape(ny+1,nx)
            boundary_error=max(np.max(abs(rawfy[0]-ramp*prescribed_bottom)),np.max(abs(rawfy[-1])),
                np.max(abs(rawfx[:,0])),np.max(abs(rawfx[:,-1])))
            if boundary_error>1e-12: raise RuntimeError('Boundary flux differs from prescribed profile')
            fx,fy,balance=grid.balance(rawfx,rawfy)
            phases['flux_projection_s']+=clock.perf_counter()-tick; tick=clock.perf_counter()
            candidate,info=grid.step(c,fx,fy,dt_value,D_value,nozz,boundary_c={'left':1.,'right':1.,'top':1.,'bottom':np.where(nozz,0.,1.)})
            phases['scalar_s']+=clock.perf_counter()-tick
            prospective=cumulative+dt_value*np.array([info['fresh_adv_in_m2_s'],info['fresh_adv_out_m2_s'],info['fresh_diff_in_m2_s']])
            inventory=float(np.sum(grid.volume*(1-candidate)))
            residual=inventory-(prospective[0]-prospective[1]+prospective[2])
            row=dict(step=step,time_s=t,ramp=ramp,vf_m2=inventory,c_min=float(candidate.min()),c_max=float(candidate.max()),
                velocity_linf_m_s=velocity_norm,boundary_flux_error_max_m2_s=float(boundary_error),
                volume_nozzle_in_m2_s=float(np.sum(fy[0,nozz])),volume_left_return_out_m2_s=float(-np.sum(fy[0,left])),
                volume_right_return_out_m2_s=float(-np.sum(fy[0,right])),cum_fresh_adv_in_m2=float(prospective[0]),
                cum_fresh_adv_out_m2=float(prospective[1]),cum_fresh_diff_in_m2=float(prospective[2]),
                cumulative_budget_residual_m2=float(residual),clipping_change_m2=0.,cum_clipping_change_m2=0.)
            row.update(balance); row.update(info)
            sampled=step<=5 or step%100==0 or step==5000
            row['ipcs_divergence_l2']=float(df.assemble(df.div(u_new)**2*df.dx))**.5 if sampled else ''
            row['tentative_divergence_l2']=float(df.assemble(df.div(u_star)**2*df.dx))**.5 if sampled else ''
            row['elapsed_wall_s']=clock.perf_counter()-wall_start
            row['cumulative_wall_s']=prior_wall+row['elapsed_wall_s']
            failure=None
            if not np.all(np.isfinite(candidate)): failure='Nonfinite scalar'
            elif candidate.min() < -1e-10 or candidate.max()>1+1e-10: failure='Scalar bounds failed; not clipped'
            elif abs(info['step_budget_residual_m2'])>5e-13: failure='Scalar step budget failed'
            elif abs(residual)>2e-10+1e-7*(abs(prospective[0])+abs(prospective[2])): failure='Cumulative scalar budget failed'
            row['accepted']=0 if failure else 1
            if writer is None: writer=csv.DictWriter(budget_file,fieldnames=list(row)); writer.writeheader()
            writer.writerow(row)
            if failure:
                budget_file.flush()
                atomic_npz(os.path.join(folder,'failed_candidate.npz'),c=candidate,c_previous=c,
                    time_s=t,step=step,previous_time_s=(step-1)*dt_value,x_edges_m=x_edges,y_edges_m=y_edges,
                    transport_fx_m2_s=fx,transport_fy_m2_s=fy,raw_fx_m2_s=rawfx,raw_fy_m2_s=rawfy)
                raise RuntimeError(failure)
            c=candidate; cumulative=prospective; set_scalar(c); u_n.assign(u_new); p_n.assign(p_new)
            status.update(last_completed_step=step,last_completed_time_s=t)
            if sampled:
                budget_file.flush(); write_status()
                print('Step {}/5000 t={:.3f}s c=[{:.9g},{:.9g}] Vf={:.9g} R={:.3e} elapsed={:.1f}min'.format(
                    step,t,c.min(),c.max(),inventory,residual,(clock.perf_counter()-wall_start)/60),flush=True)
            if step==1 or step%250==0 or step==5000 or stop_requested[0]:
                tick=clock.perf_counter()
                if abs(float(df.assemble((1-c_n)*df.dx))-inventory)>1e-11: raise RuntimeError('DG0 inventory mismatch')
                save_fields(step,fx,fy,rawfx,rawfy); save_checkpoint(step)
                phases['output_s']+=clock.perf_counter()-tick; write_status()
        completed=status['last_completed_step']
        if completed==5000:
            status.update(status='completed',final_vf_m2=float(np.sum(grid.volume*(1-c))),
                final_c_min=float(c.min()),final_c_max=float(c.max()),
                final_budget_residual_m2=float(np.sum(grid.volume*(1-c))-(cumulative[0]-cumulative[1]+cumulative[2])))
            print('Completed5s. Return the ZIP created by the launcher.',flush=True)
            return 0
        status['status']='stopped'
        save_fields(completed,fx,fy,rawfx,rawfy); save_checkpoint(completed)
        print('Stopped safely at step {}; restart checkpoint saved.'.format(completed),flush=True)
        return 75
    except Exception as exc:
        status.update(status='failed',error=str(exc))
        if accepted_ready and save_checkpoint is not None:
            try:
                # Attempted u_new/p_new may be ahead; restart always stores accepted u_n/p_n.
                u_new.assign(u_n); p_new.assign(p_n)
                accepted_rawfx=np.asarray(map_x.dot(u_n.vector().get_local())).reshape(ny,nx+1)
                accepted_rawfy=np.asarray(map_y.dot(u_n.vector().get_local())).reshape(ny+1,nx)
                accepted_fx,accepted_fy,_=grid.balance(accepted_rawfx,accepted_rawfy)
                completed=status['last_completed_step']
                save_fields(completed,accepted_fx,accepted_fy,accepted_rawfx,accepted_rawfy)
                save_checkpoint(completed)
            except Exception as checkpoint_error:
                status['emergency_checkpoint_error']=str(checkpoint_error)
        raise
    finally:
        active_error=sys.exc_info()[0] is not None
        close_errors=[]
        if budget_file is not None:
            try: budget_file.close()
            except Exception as e: close_errors.append(str(e))
        for out in outputs:
            try: out.close()
            except Exception as e: close_errors.append(str(e))
        if close_errors: status.update(status='failed',output_close_errors=close_errors)
        write_status()
        for sig,handler in previous_handlers.items(): signal.signal(sig,handler)
        if close_errors and not active_error: raise RuntimeError('Output close failed: '+'; '.join(close_errors))


class KernelTests(unittest.TestCase):
    def test_grid_domain_nozzle_alignment_and_grading(self):
        for case,h in CASE_H.items():
            x,y=make_graded_grid(h)
            self.assertEqual((len(x)-1,len(y)-1),{'focus1250':(178,101),'focus625':(252,154)}[case])
            for widths in (np.diff(x),np.diff(y)):
                self.assertTrue(np.all(widths>0))
                self.assertLessEqual(widths.max(),.005+1e-13)
                self.assertLessEqual(np.max(np.maximum(widths[1:]/widths[:-1],widths[:-1]/widths[1:])),1.15+1e-11)
            self.assertAlmostEqual(np.sum(np.outer(np.diff(y),np.diff(x))),.18,places=14)
            mask=nozzle_mask(x); self.assertEqual(mask.sum(),round(.01/h))
            bottom=prescribed_bottom_flux(x)
            self.assertAlmostEqual(bottom[mask].sum(),2./3*.001*.01,places=16)
            self.assertLess(abs(bottom.sum()),1e-18)

    def test_single_cell_analytic_update_and_diffusive_flux_sign(self):
        g=BalancedFluxGrid([0,.004],[0,.007]); vol=.004*.007; D=1e-7; dt=.2
        for old,bc in [(0.9,.1),(.1,.9)]:
            out,info=g.step(np.array([[old]]),np.zeros((1,2)),np.zeros((2,1)),dt,D,[True],boundary_c=bc)
            trans=2*D*.004/.007
            expected=(vol/dt*old+trans*bc)/(vol/dt+trans)
            self.assertAlmostEqual(out[0,0],expected,places=15)
            self.assertEqual(np.sign(info['fresh_diff_in_m2_s']),np.sign(old-bc))
            self.assertLess(abs(info['step_budget_residual_m2']),1e-18)

    def test_variable_cell_diffusion_closed_mass_and_bounds(self):
        g=BalancedFluxGrid([0,.001,.003,.009],[0,.002,.005])
        c=np.array([[0.,1.,.2],[.9,.1,.6]])
        old=float(np.sum(g.volume*c))
        out,d=g.step(c,np.zeros((2,4)),np.zeros((3,3)),10.,1e-7,[False]*3)
        self.assertLess(abs(np.sum(g.volume*out)-old),1e-17)
        self.assertGreaterEqual(out.min(),0.); self.assertLessEqual(out.max(),1.)
        self.assertLess(abs(d['step_budget_residual_m2']),1e-17)

    def test_variable_transmissibility_exact_linear_gradient(self):
        g=BalancedFluxGrid([0,.002,.003,.009],[0,.001,.004])
        value=2*g.xc[None,:]+3*g.yc[:,None]
        flux=g.weights*(value.ravel()[g.v]-value.ravel()[g.u])
        expected=np.concatenate((np.repeat(2*g.dy,g.nx-1),np.tile(3*g.dx,g.ny-1)))
        np.testing.assert_allclose(flux,expected,rtol=1e-14,atol=1e-17)

    def test_incompatible_boundary_and_unbalanced_transport_rejected(self):
        g=BalancedFluxGrid([0,.002,.005],[0,.003,.01])
        fx=np.zeros((2,3)); fy=np.zeros((3,2)); fx[0,0]=1e-5
        with self.assertRaises(ValueError): g.balance(fx,fy)
        fx[:]=0; fx[0,1]=1e-5
        with self.assertRaises(ValueError): g.step(np.ones((2,2)),fx,fy,.001,0.,[False,False])

    def test_both_full_grids_projection_constant_bounds_and_source_budget(self):
        for case,h in CASE_H.items():
            x,y=make_graded_grid(h); g=BalancedFluxGrid(x,y); mask=nozzle_mask(x)
            rng=np.random.RandomState(418)
            fx=np.zeros((g.ny,g.nx+1)); fy=np.zeros((g.ny+1,g.nx))
            fy[0]=prescribed_bottom_flux(x)
            fx[:,1:-1]=rng.normal(size=(g.ny,g.nx-1))*1e-5
            fy[1:-1]=rng.normal(size=(g.ny-1,g.nx))*1e-5
            outfx,outfy,proj=g.balance(fx,fy)
            np.testing.assert_array_equal(outfy[[0,-1]],fy[[0,-1]])
            np.testing.assert_array_equal(outfx[:,[0,-1]],fx[:,[0,-1]])
            self.assertLess(proj['cell_flux_residual_after_max_m2_s'],1e-16)
            constant,d=g.step(np.full(g.volume.shape,.37),outfx,outfy,.001,1e-7,mask,boundary_c=.37)
            np.testing.assert_allclose(constant,.37,rtol=0,atol=2e-12)
            c=np.ones_like(constant); cumulative=0.
            for dt in (.001,.002,.01):
                c,d=g.step(c,outfx,outfy,dt,1e-7,mask,boundary_c={'left':1.,'right':1.,'top':1.,'bottom':np.where(mask,0.,1.)})
                self.assertGreaterEqual(c.min(),-1e-12); self.assertLessEqual(c.max(),1+1e-12)
                self.assertLess(abs(d['step_budget_residual_m2']),1e-13)
                cumulative+=dt*(d['fresh_adv_in_m2_s']-d['fresh_adv_out_m2_s']+d['fresh_diff_in_m2_s'])
            self.assertLess(abs(np.sum(g.volume*(1-c))-cumulative),2e-13)
            print('Full grid test',case,g.volume.shape,'constant error',np.max(abs(constant-.37)),flush=True)

    def test_coordinate_map_with_nonuniform_edges(self):
        x=np.array([0.,.0007,.002,.008])
        np.testing.assert_array_equal(match_axis(half_edges(x),half_edges(x)),np.arange(7))
        with self.assertRaises(RuntimeError): match_axis([.001],x)

    def test_atomic_checkpoint_restart_and_wrong_case_rejection(self):
        g=BalancedFluxGrid([0,.004],[0,.007]); vel=np.zeros((4,2)); pre=np.zeros((2,2))
        mesh_xy=np.array([[0,0],[.004,0],[0,.007],[.004,.007]])
        mesh_cells=np.array([[0,1,2],[1,2,3]])
        c=np.array([[.9]]); cum=np.array([np.sum(g.volume*(1-c)),0.,0.])
        data=dict(metadata_json=json.dumps({'format_version':1,'case':'focus1250','fingerprint':'test'}),
            x_edges_m=g.x_edges,y_edges_m=g.y_edges,c=c,u_dofs=np.zeros(4),p_dofs=np.zeros(2),
            velocity_dof_xy_m=vel,pressure_dof_xy_m=pre,mesh_coordinates_m=mesh_xy,mesh_cells=mesh_cells,
            step=10,time_s=.01,cumulative_budget_m2=cum,vf_m2=cum[0],cumulative_wall_s=1.)
        with tempfile.TemporaryDirectory() as folder:
            path=os.path.join(folder,'checkpoint_latest.npz'); atomic_npz(path,**data)
            with np.load(path,allow_pickle=False) as f: loaded={k:f[k] for k in f.files}
            step,found=validate_restart(loaded,'focus1250','test',g,vel,pre,mesh_xy,mesh_cells)
            self.assertEqual(step,10); np.testing.assert_array_equal(found,cum)
            with self.assertRaises(ValueError): validate_restart(loaded,'focus625','test',g,vel,pre,mesh_xy,mesh_cells)
            with self.assertRaises(ValueError): validate_restart(loaded,'focus1250','changed',g,vel,pre,mesh_xy,mesh_cells)
            loaded['mesh_cells']=loaded['mesh_cells'][::-1]
            with self.assertRaises(ValueError): validate_restart(loaded,'focus1250','test',g,vel,pre,mesh_xy,mesh_cells)
            self.assertEqual(os.listdir(folder),['checkpoint_latest.npz'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=tuple(CASE_H),default='focus1250')
    p.add_argument('--output',help='New or empty output directory')
    p.add_argument('--resume',help='Absolute path to this programme\'s accepted checkpoint')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--describe',action='store_true')
    args=p.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(KernelTests))
        return 0 if result.wasSuccessful() else 1
    if args.describe:
        for case,h in CASE_H.items():
            x,y=make_graded_grid(h)
            print(json.dumps({'case':case,'nx':len(x)-1,'ny':len(y)-1,'core_spacing_m':h,'dt_s':.001,'T_s':5.}))
        return 0
    if not args.output: p.error('--output is required for a simulation')
    if args.resume and (not os.path.isabs(args.resume) or not os.path.isfile(args.resume)):
        p.error('--resume must be an existing absolute checkpoint path')
    return run_case(args)

if __name__=='__main__':
    sys.exit(main())
