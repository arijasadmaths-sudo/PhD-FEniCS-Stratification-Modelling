#!/usr/bin/env python3
"""Inspect legacy FEniCS DG0 tetrahedral XDMF/HDF5 concentration output.

Requires numpy, scipy, h5py, matplotlib. No DOLFIN installation is needed.
All integral diagnostics use raw concentration, without clipping. Planar
sections show the cellwise constant values on exact tetrahedron intersections.
Horizontal means are area weighted; no nodal interpolation is used.
"""
import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize


EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def section(points, axis, level):
    """Return cell indices, padded polygons, and areas for a plane cut.

    Move the plane by 1e-10 m only when it coincides with a vertex layer;
    this selects a one-sided DG0 trace rather than double-counting a face.
    """
    coord = points[:, :, axis]
    if np.any(np.abs(coord - level) < 1e-13):
        level += 1e-10
    ids = np.flatnonzero((coord.min(1) < level) & (coord.max(1) > level))
    p = points[ids]
    a, b = p[:, EDGES[:, 0]], p[:, EDGES[:, 1]]
    cross = ((a[:, :, axis] < level) != (b[:, :, axis] < level))
    fraction = np.zeros(cross.shape)
    np.divide(level - a[:, :, axis], b[:, :, axis] - a[:, :, axis],
              out=fraction, where=cross)
    q = a + fraction[:, :, None] * (b - a)
    q = q[:, :, [j for j in range(3) if j != axis]]
    q[~cross] = np.nan
    counts = cross.sum(1)
    if not np.all((counts == 3) | (counts == 4)):
        raise RuntimeError("Unexpected tetrahedron-plane intersection.")
    centres = np.nanmean(q, axis=1)
    angle = np.arctan2(q[:, :, 1] - centres[:, None, 1],
                       q[:, :, 0] - centres[:, None, 0])
    angle[~cross] = np.inf
    order = np.argsort(angle, axis=1)
    q = np.take_along_axis(q, order[:, :, None], axis=1)
    q = np.where((np.arange(6)[None, :] < counts[:, None])[:, :, None],
                 q, q[:, :1])
    nxt = np.roll(q, -1, axis=1)
    area = 0.5 * np.abs(np.sum(q[:, :, 0]*nxt[:, :, 1]
                               - q[:, :, 1]*nxt[:, :, 0], axis=1))
    return ids, q, area


def upper_cell_volumes(points, volumes, cutoff):
    """Exact clipped-cell volumes above a horizontal plane."""
    low, high = points[:, :, 2].min(1), points[:, :, 2].max(1)
    out = np.where(low >= cutoff, volumes, 0.0)
    for i in np.flatnonzero((low < cutoff) & (high > cutoff)):
        p = points[i]
        inside = p[:, 2] >= cutoff
        vertices = [v for v in p[inside]]
        for a, b in EDGES:
            if inside[a] != inside[b]:
                alpha = (cutoff - p[a, 2]) / (p[b, 2] - p[a, 2])
                vertices.append(p[a] + alpha * (p[b] - p[a]))
        vertices = np.unique(np.round(vertices, 14), axis=0)
        out[i] = ConvexHull(vertices).volume if len(vertices) >= 4 else 0.0
    if np.any(out < -1e-15) or np.any(out > volumes + 1e-15):
        raise RuntimeError("Clipped volume is outside the parent cell.")
    return out


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(np.interp(quantile*cumulative[-1], cumulative, values[order]))


def parse_xdmf(path):
    snapshots = []
    for grid in ET.parse(path).getroot().iter("Grid"):
        time = grid.find("Time")
        attribute = grid.find("Attribute[@Name='concentration']")
        if time is not None and attribute is not None:
            if attribute.attrib.get("Center") != "Cell":
                raise ValueError("This analysis expects cellwise DG0 concentration.")
            reference = attribute.find("DataItem").text.strip()
            snapshots.append((float(time.attrib["Value"]), reference.split(":", 1)[1]))
    if not snapshots or np.any(np.diff([x[0] for x in snapshots]) <= 0):
        raise ValueError("Missing or non-increasing snapshot times.")
    return snapshots


def log_freshwater(path):
    if path is None:
        return {}
    text = path.read_text()
    result = {}
    for block in re.split(r"(?=^step = \d+\s+t = )", text, flags=re.M):
        t = re.search(r"^step = \d+\s+t = ([\d.eE+-]+)", block, re.M)
        v = re.search(r"^fresh volume total = ([\d.eE+-]+)", block, re.M)
        if t and v:
            result[round(float(t.group(1)), 9)] = float(v.group(1))
    return result


def save_figure(fig, destination, name):
    for extension in ("png", "pdf"):
        fig.savefig(destination / (name + "." + extension), dpi=220,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def draw_section(ax, cut, concentration, vmin, vmax, title, limits, edges=False):
    ids, polygons, area = cut
    artist = PolyCollection(polygons*1000, array=concentration[ids],
                            cmap="viridis", norm=Normalize(vmin, vmax),
                            edgecolors="#777777" if edges else "none",
                            linewidths=0.25 if edges else 0,
                            antialiaseds=edges, rasterized=True)
    ax.add_collection(artist)
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    return artist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--xdmf", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    snapshots = parse_xdmf(args.xdmf)
    times = np.array([s[0] for s in snapshots])
    log_values = log_freshwater(args.log)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.pad_inches": .08})

    with h5py.File(args.h5, "r") as f:
        xyz = f["/Mesh/0/mesh/geometry"][:]
        cells = f["/Mesh/0/mesh/topology"][:]
        points = xyz[cells]
        centres = points.mean(1)
        volumes = np.abs(np.linalg.det(points[:, 1:] - points[:, :1])) / 6
        if np.any(volumes <= 0):
            raise ValueError("Non-positive cell volume.")
        lower, upper = xyz.min(0), xyz.max(0)
        lengths = upper - lower
        cross_area = lengths[0]*lengths[1]
        if not np.isclose(volumes.sum(), np.prod(lengths), rtol=1e-10):
            raise ValueError("Mesh volume differs from bounding box volume.")
        top20 = upper_cell_volumes(points, volumes, upper[2] - .02)
        top50 = upper_cell_volumes(points, volumes, upper[2] - .05)
        assert np.isclose(top20.sum(), cross_area*.02, rtol=1e-9)
        assert np.isclose(top50.sum(), cross_area*.05, rtol=1e-9)
        print("Validated mesh and exact clipped volumes.", flush=True)

        # Exact cross-section area weights at 1 mm vertical spacing.
        heights = np.linspace(lower[2]+.0005, upper[2]-.0005,
                              int(round(lengths[2]/.001)))
        row, column, weight = [], [], []
        for j, height in enumerate(heights):
            ids, poly, area = section(points, 2, height)
            if not np.isclose(area.sum(), cross_area, rtol=1e-8):
                raise ValueError("Horizontal section does not cover the box.")
            row.extend([j]*len(ids)); column.extend(ids); weight.extend(area/cross_area)
        mean_operator = csr_matrix((weight, (row, column)),
                                   shape=(len(heights), len(cells)))
        central = section(points, 1, 0.0)
        transverse = section(points, 0, 0.0)
        plan = section(points, 2, upper[2]-.002)
        assert np.isclose(central[2].sum(), lengths[0]*lengths[2], rtol=1e-8)
        assert np.isclose(transverse[2].sum(), lengths[1]*lengths[2], rtol=1e-8)
        assert np.isclose(plan[2].sum(), cross_area, rtol=1e-8)
        print("Validated planar section areas.", flush=True)

        chosen_indices = [0, len(times)//2, len(times)-1]
        chosen_fields = {}
        profiles = []
        metrics = []
        for i, (time, dataset) in enumerate(snapshots):
            c = np.asarray(f[dataset][:]).reshape(-1)
            if len(c) != len(cells) or not np.all(np.isfinite(c)):
                raise ValueError("Invalid concentration field: " + dataset)
            fresh = 1.0-c
            mass = volumes*fresh
            total = mass.sum()
            centroid = mass @ centres / total
            top20fresh = top20 @ fresh
            section_fresh = fresh[plan[0]]
            field_mean = mean_operator @ fresh
            profiles.append(field_mean)
            high = section_fresh >= .01
            p = plan[1][high]
            log_total = log_values.get(round(time, 9), np.nan)
            entry = dict(time_s=time, freshwater_mL=total*1e6,
                         top20mm_freshwater_mL=top20fresh*1e6,
                         top50mm_freshwater_mL=(top50 @ fresh)*1e6,
                         top20mm_freshwater_fraction=top20fresh/total,
                         x_centroid_m=centroid[0], y_centroid_m=centroid[1],
                         z_centroid_m=centroid[2], c_min=c.min(), c_max=c.max(),
                         mean_freshwater_0p5mm_below_ceiling=field_mean[-1],
                         plan_1percent_area_fraction=plan[2][high].sum()/cross_area,
                         plan_1percent_x_min_m=p[:,:,0].min() if len(p) else np.nan,
                         plan_1percent_x_max_m=p[:,:,0].max() if len(p) else np.nan,
                         plan_1percent_y_min_m=p[:,:,1].min() if len(p) else np.nan,
                         plan_1percent_y_max_m=p[:,:,1].max() if len(p) else np.nan,
                         log_freshwater_mL=log_total*1e6,
                         field_minus_log_mL=(total-log_total)*1e6)
            metrics.append(entry)
            if i in chosen_indices:
                chosen_fields[i] = c
            if i % 100 == 0:
                print("Read snapshot", i, "t=", time, flush=True)

        profiles = np.asarray(profiles)
        final_c = chosen_fields[chosen_indices[-1]]
        vertical_spans = np.ptp(points[:,:,2], axis=1)
        # Positive weights only for mesh-resolution quantiles, not mass budgets.
        resolution_weights = top20*np.maximum(1-final_c, 0)
        last = metrics[-1]
        errors = [abs(m["field_minus_log_mL"]) for m in metrics
                  if np.isfinite(m["field_minus_log_mL"])]
        summary = dict(
            source_h5=args.h5.name, source_xdmf=args.xdmf.name,
            time_start_s=times[0], time_end_s=times[-1], snapshot_count=len(times),
            domain_dimensions_m=lengths.tolist(), cell_count=len(cells),
            vertex_count=len(xyz), domain_volume_m3=float(volumes.sum()),
            first=metrics[0], final=last,
            log_comparison_count=len(errors),
            max_field_log_difference_mL=max(errors) if errors else None,
            top20mm_freshwater_weighted_cell_z_span_median_mm=1000*weighted_quantile(
                vertical_spans,resolution_weights,.5),
            top20mm_freshwater_weighted_cell_z_span_90th_mm=1000*weighted_quantile(
                vertical_spans,resolution_weights,.9),
            notes=["Concentration c=0 fresh, c=1 initial salt water.",
                   "No concentration clipping or per-image normalization in metrics.",
                   "DG0 planar sections and means use tetrahedron intersection areas.",
                   "Top-region integrals use exact tetrahedron clipping.",
                   "The 1% plan footprint is a descriptive threshold at z=198 mm, not a layer-height definition.",
                   "No velocity field or observations before 48 s supplied.",
                   "This analysis does not establish mesh/time convergence or a growth exponent."])

    with open(args.out/'rectangular_3d_metrics.csv', 'w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with open(args.out/'rectangular_3d_profiles.csv', 'w', newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['time_s','height_m','depth_below_ceiling_m','area_mean_freshwater_fraction'])
        for time, profile in zip(times, profiles):
            writer.writerows((time,z,upper[2]-z,chi) for z,chi in zip(heights,profile))
    # Missing log values are null in JSON rather than nonstandard NaN tokens.
    def clean(item):
        if isinstance(item,dict): return {k:clean(v) for k,v in item.items()}
        if isinstance(item,list): return [clean(v) for v in item]
        if isinstance(item,(float,np.floating)): return float(item) if np.isfinite(item) else None
        if isinstance(item,np.integer): return int(item)
        return item
    (args.out/'rectangular_3d_analysis_summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')

    fig, axes=plt.subplots(2,3,figsize=(14,6),layout='constrained')
    for col,i in enumerate(chosen_indices):
        for row,(vmin,vmax) in enumerate(((0,1),(.97,1))):
            artist=draw_section(axes[row,col],central,chosen_fields[i],vmin,vmax,
                                f't = {times[i]:g} s',((-300,300),(0,200)))
            axes[row,col].set_xlabel('x (mm)')
            if col == 0: axes[row,col].set_ylabel('Height z (mm)')
        
    for row,(vmin,vmax) in enumerate(((0,1),(.97,1))):
        sm=plt.cm.ScalarMappable(norm=Normalize(vmin,vmax),cmap='viridis')
        cb=fig.colorbar(sm,ax=axes[row,:],shrink=.83,pad=.02,extend='min' if row else 'neither')
        cb.set_label('Concentration c' if row==0 else 'Concentration c (contrast view)')
    fig.suptitle('Central vertical sections through the 3D rectangular plume',fontsize=16)
    fig.supxlabel('Upper row: full scale 0–1. Lower row: fixed contrast scale 0.97–1; values below 0.97 saturate.\n'
                  'Cellwise DG0 values; y ≈ 0. Freshwater: c = 0. Initial salt water: c = 1.',fontsize=10)
    save_figure(fig,args.out,'rectangular_3d_concentration_slices')

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    art=draw_section(axes[0,0],plan,final_c,.97,1,
                     'Horizontal section: 2 mm below ceiling, t = 78 s',((-300,300),(-150,150)))
    axes[0,0].set_xlabel('x (mm)'); axes[0,0].set_ylabel('y (mm)')
    fig.colorbar(art,ax=axes[0,0],shrink=.75,extend='min',label='Concentration c')
    for i in np.unique(np.linspace(0,len(times)-1,4,dtype=int)):
        axes[0,1].plot(profiles[i]*100,(upper[2]-heights)*1000,label=f'{times[i]:g} s',lw=1.8)
    axes[0,1].set_ylim(80,0); axes[0,1].set_xlim(left=0)
    axes[0,1].set_xlabel('Horizontal mean freshwater fraction (%)')
    axes[0,1].set_ylabel('Depth below ceiling (mm)')
    axes[0,1].set_title('A growing upper-region concentration signal')
    axes[0,1].legend(frameon=False); axes[0,1].grid(alpha=.2)
    axes[1,0].plot(times,[m['freshwater_mL'] for m in metrics],label='Entire tank',color='#222222',lw=2)
    axes[1,0].plot(times,[m['top20mm_freshwater_mL'] for m in metrics],label='Upper 20 mm',color='#176d95',lw=2)
    axes[1,0].plot(times,[m['top50mm_freshwater_mL'] for m in metrics],label='Upper 50 mm',color='#aa6a22',lw=2)
    axes[1,0].set_xlabel('Time (s)'); axes[1,0].set_ylabel('Freshwater-equivalent volume (mL)')
    axes[1,0].set_ylim(bottom=0); axes[1,0].set_title('Freshwater accumulates near the ceiling')
    axes[1,0].legend(frameon=False); axes[1,0].grid(alpha=.2)
    art=draw_section(axes[1,1],central,final_c,.97,1,
                     'Upper-region mesh: central section at 78 s',((-100,100),(120,200)),edges=True)
    axes[1,1].set_xlabel('x (mm)'); axes[1,1].set_ylabel('Height z (mm)')
    fig.colorbar(art,ax=axes[1,1],shrink=.75,extend='min',label='Concentration c')
    fig.suptitle('Upper-layer development and numerical resolution',fontsize=16)
    fig.supxlabel('Raw concentration; fixed display scales. Plane means are area weighted; volume budgets use exact cell volumes.',fontsize=10)
    save_figure(fig,args.out,'rectangular_3d_layer_diagnostics')
    print(json.dumps(clean(summary),indent=2),flush=True)


if __name__ == '__main__':
    main()
