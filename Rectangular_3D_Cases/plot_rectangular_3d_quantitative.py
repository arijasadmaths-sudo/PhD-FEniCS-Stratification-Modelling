#!/usr/bin/env python3
"""Create figures from restarted rectangular 3D DG0 concentration output.

Example (after the run, using one ordinary Python process, not srun):
  python3 plot_rectangular_3d_quantitative.py --results RESULTS_DIRECTORY

Default --field dye plots dye_complement_dg0.xdmf, for comparison with PLIF.
Use --field density for c_dg0.xdmf (the active buoyancy scalar). The two fields
can differ because the molecular diffusivities differ. In dye mode, 1-c is a
dye-derived source fraction and its integral is dye-equivalent source volume.

Requires numpy, h5py and matplotlib; FEniCS is not needed. Each concentration
frame is kept cellwise constant on exact tetrahedron/plane intersections.
The fixed contrast plot uses c in [0.97, 1]; it never rescales individual images.
Horizontal means use section areas, without concentration clipping. These
figures describe this simulation; they do not establish experimental agreement
or convergence. Prefer running after an allocation ends, while output is closed.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize


EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def section(points, axis, level):
    """Return intersecting cell IDs, padded polygons and exact section areas.

    A +1e-10 m displacement at a vertex layer selects the positive-side DG0
    trace and prevents counting both sides of an internal face.
    """
    coordinate = points[:, :, axis]
    if np.any(np.abs(coordinate - level) < 1e-13):
        level += 1e-10
    ids = np.flatnonzero((coordinate.min(1) < level) &
                         (coordinate.max(1) > level))
    if not len(ids):
        raise ValueError("The section plane does not intersect any cells.")
    p = points[ids]
    a, b = p[:, EDGES[:, 0]], p[:, EDGES[:, 1]]
    crossing = ((a[:, :, axis] < level) != (b[:, :, axis] < level))
    fraction = np.zeros(crossing.shape)
    np.divide(level - a[:, :, axis], b[:, :, axis] - a[:, :, axis],
              out=fraction, where=crossing)
    q = a + fraction[:, :, None] * (b - a)
    q = q[:, :, [j for j in range(3) if j != axis]]
    q[~crossing] = np.nan
    counts = crossing.sum(1)
    if not np.all((counts == 3) | (counts == 4)):
        raise ValueError("Unexpected tetrahedron/plane intersection.")
    centre = np.nanmean(q, axis=1)
    angle = np.arctan2(q[:, :, 1] - centre[:, None, 1],
                       q[:, :, 0] - centre[:, None, 0])
    angle[~crossing] = np.inf
    q = np.take_along_axis(q, np.argsort(angle, axis=1)[:, :, None], axis=1)
    q = np.where((np.arange(6)[None, :] < counts[:, None])[:, :, None],
                 q, q[:, :1])
    following = np.roll(q, -1, axis=1)
    areas = 0.5 * np.abs(np.sum(q[:, :, 0] * following[:, :, 1] -
                                q[:, :, 1] * following[:, :, 0], axis=1))
    return ids, q, areas


def hdf_reference(item, xdmf):
    if item is None or item.attrib.get("Format", "XML").upper() != "HDF":
        raise ValueError("Expected a direct HDF DataItem in " + str(xdmf))
    filename, dataset = item.text.strip().rsplit(":", 1)
    return str((xdmf.parent / filename).resolve()), dataset


def read_array(reference):
    filename, dataset = reference
    with h5py.File(filename, "r") as handle:
        return handle[dataset][:]


def read_snapshots(xdmf):
    root = ET.parse(xdmf).getroot()
    geometry = topology = None
    frames = []
    for grid in root.iter("Grid"):
        own_geometry = grid.find("Geometry/DataItem")
        own_topology = grid.find("Topology/DataItem")
        if own_geometry is not None:
            geometry = hdf_reference(own_geometry, xdmf)
        if own_topology is not None:
            topology = hdf_reference(own_topology, xdmf)
        time = grid.find("Time")
        if time is None:
            continue
        attributes = [a for a in grid.findall("Attribute")
                      if a.attrib.get("Center") == "Cell" and
                      a.attrib.get("AttributeType", "Scalar") == "Scalar"]
        preferred = [a for a in attributes if a.attrib.get("Name") in
                     ("concentration", "c", "c_dg0")]
        if preferred:
            attributes = preferred
        if len(attributes) != 1 or geometry is None or topology is None:
            raise ValueError("Expected one DG0 concentration with a mesh: " + str(xdmf))
        value = float(time.attrib["Value"])
        if not math.isfinite(value):
            raise ValueError("Nonfinite time in " + str(xdmf))
        frames.append(dict(time=value, xdmf=str(xdmf),
                           mtime=xdmf.parent.stat().st_mtime_ns,
                           geometry=geometry, topology=topology,
                           concentration=hdf_reference(attributes[0].find("DataItem"), xdmf)))
    return frames


def discover(results, pattern):
    files = sorted(results.glob(pattern))
    if not files:
        raise ValueError("No concentration XDMF files matched " + str(results / pattern))
    # A later segment supersedes its predecessor at a repeated restart time.
    by_time = {}
    for xdmf in files:
        try:
            frames = read_snapshots(xdmf)
        except (ET.ParseError, OSError, ValueError) as exc:
            warnings.warn("Skipping unreadable/incomplete XDMF %s: %s" % (xdmf, exc))
            continue
        for frame in frames:
            key = round(frame["time"], 9)
            old = by_time.get(key)
            if old is None or (frame["mtime"], frame["xdmf"]) > (old["mtime"], old["xdmf"]):
                by_time[key] = frame
    frames = [by_time[k] for k in sorted(by_time)]
    if not frames:
        raise ValueError("No readable concentration frames were found.")
    return frames


def choose_frames(frames, requested, tolerance):
    times = np.array([f["time"] for f in frames])
    if tolerance is None:
        gaps = np.diff(times)
        # The median tolerates occasional off-cadence checkpoint/segment frames.
        tolerance = 0.5 * float(np.median(gaps)) if len(gaps) else 1e-9
    chosen = []
    for target in requested:
        index = int(np.argmin(np.abs(times - target)))
        match = frames[index] if abs(times[index] - target) <= tolerance + 1e-9 else None
        chosen.append(dict(requested=target, frame=match))
    return chosen, tolerance


def prepare_frames(chosen, profile_spacing):
    prepared = {}
    groups = {}
    for choice in chosen:
        frame = choice["frame"]
        if frame is not None:
            key = (frame["geometry"], frame["topology"])
            groups.setdefault(key, {})[frame["time"]] = frame
    for (geometry, topology), group in groups.items():
        xyz = np.asarray(read_array(geometry), dtype=float)
        cells = np.asarray(read_array(topology), dtype=np.int64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or cells.ndim != 2 or cells.shape[1] != 4:
            raise ValueError("Only 3D tetrahedral meshes are supported.")
        points = xyz[cells]
        lower, upper = xyz.min(0), xyz.max(0)
        lengths = upper - lower
        volumes = np.abs(np.linalg.det(points[:, 1:] - points[:, :1])) / 6
        if np.any(volumes <= 0) or not np.isclose(volumes.sum(), np.prod(lengths), rtol=1e-8):
            raise ValueError("Mesh cells do not fill their rectangular bounding box.")
        middle_y = 0.0 if lower[1] < 0 < upper[1] else 0.5 * (lower[1] + upper[1])
        central = section(points, 1, middle_y)
        if not np.isclose(central[2].sum(), lengths[0] * lengths[2], rtol=1e-7):
            raise ValueError("Central section does not cover the rectangular domain.")
        count = max(2, int(np.ceil(lengths[2] / profile_spacing)))
        heights = lower[2] + (np.arange(count) + 0.5) * lengths[2] / count
        fields = {}
        profiles = {}
        for time, frame in group.items():
            c = np.asarray(read_array(frame["concentration"]), dtype=float).reshape(-1)
            if len(c) != len(cells) or not np.all(np.isfinite(c)):
                raise ValueError("Invalid concentration field at t=%g in %s" % (time, frame["xdmf"]))
            fields[time], profiles[time] = c, np.empty(count)
        cross_area = lengths[0] * lengths[1]
        for j, height in enumerate(heights):
            ids, _, area = section(points, 2, height)
            if not np.isclose(area.sum(), cross_area, rtol=1e-7):
                raise ValueError("Horizontal section does not cover the rectangular domain.")
            for time, c in fields.items():
                profiles[time][j] = np.dot(area, 1.0 - c[ids]) / cross_area
        for time, c in fields.items():
            prepared[time] = dict(polygons=central[1], section_c=c[central[0]],
                                  lower=lower, upper=upper, middle_y=middle_y,
                                  heights=heights, profile=profiles[time],
                                  freshwater_mL=float(np.dot(volumes, 1.0 - c) * 1e6),
                                  c_min=float(c.min()), c_max=float(c.max()),
                                  cell_count=len(cells))
        print("Prepared exact sections for times:", ", ".join("%g s" % t for t in group), flush=True)
    return prepared


def save_figure(fig, destination, name):
    for extension in ("png", "pdf"):
        fig.savefig(destination / (name + "." + extension), dpi=250,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_panels(chosen, prepared, destination, vmin, name, field):
    columns = min(3, len(chosen))
    rows = int(math.ceil(len(chosen) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.7 * columns, 2.9 * rows),
                             squeeze=False, constrained_layout=True)
    artist = None
    for ax, choice in zip(axes.flat, chosen):
        frame = choice["frame"]
        if frame is None:
            ax.text(0.5, 0.5, "%g s: not available" % choice["requested"],
                    ha="center", va="center", transform=ax.transAxes, color="0.45")
            ax.set_axis_off()
            continue
        data = prepared[frame["time"]]
        artist = PolyCollection(data["polygons"] * 1000, array=data["section_c"],
                                cmap="viridis", norm=Normalize(vmin, 1),
                                edgecolors="none", antialiaseds=False, rasterized=True)
        ax.add_collection(artist)
        ax.set_xlim(data["lower"][0] * 1000, data["upper"][0] * 1000)
        ax.set_ylim(data["lower"][2] * 1000, data["upper"][2] * 1000)
        ax.set_aspect("equal")
        ax.set_title("$t = %g$ s" % frame["time"])
        ax.set_xlabel("$x$ (mm)")
        ax.set_ylabel("$z$ (mm)")
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_axis_off()
    if artist is not None:
        bar = fig.colorbar(artist, ax=list(axes.flat), shrink=0.8, pad=0.02,
                           extend="min" if vmin > 0 else "neither")
        bar.set_label("Dye complement $c_d$ (0: source, 1: ambient)" if field == "dye" else "Density proxy $c$ (0: fresh, 1: initial salt water)")
    save_figure(fig, destination, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[20, 40, 60, 80, 100, 120])
    parser.add_argument("--field", choices=("dye", "density"), default="dye")
    parser.add_argument("--pattern", help="Glob relative to --results; default follows --field.")
    parser.add_argument("--tolerance", type=float,
                        help="Maximum mismatch in seconds; default: half median saved cadence")
    parser.add_argument("--profile-spacing-mm", type=float, default=1.0)
    args = parser.parse_args()
    if len(args.times) > 12 or any(not math.isfinite(t) or t < 0 for t in args.times):
        parser.error("Supply 1 to 12 finite, nonnegative requested times.")
    if not math.isfinite(args.profile_spacing_mm) or args.profile_spacing_mm < 0.1:
        parser.error("Profile sampling spacing must be at least 0.1 mm.")
    if args.tolerance is not None and (not math.isfinite(args.tolerance) or args.tolerance < 0):
        parser.error("--tolerance must be finite and nonnegative.")
    args.results = args.results.expanduser().resolve()
    args.pattern = args.pattern or ("segments/*/dye_complement_dg0.xdmf" if args.field == "dye" else "segments/*/c_dg0.xdmf")
    frames = discover(args.results, args.pattern)
    chosen, tolerance = choose_frames(frames, args.times, args.tolerance)
    available = [c for c in chosen if c["frame"] is not None]
    print("Available time range: %g to %g s (%d frames)." %
          (frames[0]["time"], frames[-1]["time"], len(frames)))
    print("Selection tolerance: %g s." % tolerance)
    if not available:
        parser.error("None of the requested times are available. Re-run with --times and actual saved times within this range.")
    prepared = prepare_frames(chosen, args.profile_spacing_mm * 1e-3)
    destination = args.results / "figures" / args.field
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.pad_inches": 0.08})
    plot_panels(chosen, prepared, destination, 0, "rectangular_3d_concentration_full", args.field)
    plot_panels(chosen, prepared, destination, 0.97, "rectangular_3d_concentration_contrast_097", args.field)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    rows = []
    for time in sorted(prepared):
        data = prepared[time]
        ax.plot(data["profile"] * 100, data["heights"] * 1000, label="%g s" % time)
        rows.extend((time, float(z), float(data["upper"][2] - z), float(value))
                    for z, value in zip(data["heights"], data["profile"]))
    ax.set_xlabel("Horizontally averaged dye-derived source fraction (%)" if args.field == "dye" else "Horizontally averaged freshwater fraction, $100(1-c)$ (%)")
    ax.set_ylabel("$z$ (mm)")
    ax.legend(title="Time", frameon=False)
    save_figure(fig, destination, "rectangular_3d_horizontal_profiles")
    with (destination / "rectangular_3d_horizontal_profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "z_m", "depth_below_ceiling_m", "area_mean_source_fraction"])
        writer.writerows(rows)
    selected = []
    for choice in chosen:
        frame = choice["frame"]
        entry = dict(requested_time_s=choice["requested"], actual_time_s=None)
        if frame is not None:
            data = prepared[frame["time"]]
            entry.update(actual_time_s=frame["time"], source_xdmf=frame["xdmf"],
                         hdf_file=frame["concentration"][0], dataset=frame["concentration"][1],
                         domain_dimensions_m=(data["upper"] - data["lower"]).tolist(),
                         section_y_m=data["middle_y"], cell_count=data["cell_count"],
                         source_equivalent_volume_mL=data["freshwater_mL"], c_min=data["c_min"], c_max=data["c_max"])
        selected.append(entry)
    summary = dict(field=args.field, available_time_range_s=[frames[0]["time"], frames[-1]["time"]],
                   unique_snapshot_count=len(frames), time_tolerance_s=tolerance,
                   selected=selected,
                   notes=["Exact tetrahedron section polygons; no nodal smoothing.",
                          "Positive-side DG0 trace at a mesh vertex layer (1e-10 m displacement).",
                          "Fixed concentration limits: full [0,1], contrast [0.97,1].",
                          "Profiles and source-volume integrals use raw values without clipping.",
                          "In dye mode, source-volume integrals are dye-equivalent source-fluid volume, not a density measurement.",
                          "1 mm profile sampling does not imply 1 mm mesh resolution.",
                          "These figures do not establish mesh convergence or experimental agreement."])
    with (destination / "rectangular_3d_figure_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Saved PNG/PDF figures, profiles and source/time summary to", destination)


if __name__ == "__main__":
    main()
