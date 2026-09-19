"""Regenerate volume-averaged lower-flow axisymmetric profiles from DG0 output.

Example (run in a Python environment with numpy, h5py and matplotlib):
  python regenerate_axisymmetric_lower_profiles.py --original ORIGINAL_FOLDER \
      --continuation CONTINUATION_FOLDER --output OUTPUT_FOLDER

Both input folders must contain c.xdmf and its referenced HDF5 file. Each of
51 equal-height slabs is integrated by exact triangle/slab intersections in
(s,z), where s=r**2/2. Thus area weights are proportional to physical volume.
No concentration clipping, smoothing, or interpolation is used in the means.
The plotted straight segments connect slab-centre values. The axes extend
over the complete [0,1] concentration and z/H ranges, including the half-slab
gaps between the end samples and the horizontal walls.
"""
from pathlib import Path
import argparse
import csv
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def clip_horizontal(poly, level, keep_above):
    if len(poly) == 0:
        return []
    result = []
    start = poly[-1]
    start_in = start[1] >= level if keep_above else start[1] <= level
    for end in poly:
        end_in = end[1] >= level if keep_above else end[1] <= level
        if start_in != end_in:
            ratio = (level - start[1]) / (end[1] - start[1])
            result.append(start + ratio * (end - start))
        if end_in:
            result.append(end)
        start, start_in = end, end_in
    return result


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly)
    return 0.5 * abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                           - p[:, 1] * np.roll(p[:, 0], -1)))


def read_series(folder):
    root = ET.parse(folder / "c.xdmf").getroot()
    records = []
    for grid in root.findall(".//Grid"):
        time_node = grid.find("Time")
        attributes = grid.findall("Attribute")
        if time_node is None or not attributes:
            continue
        data = attributes[0].find("DataItem")
        if data is None or data.attrib.get("Format") != "HDF":
            raise ValueError("Expected a direct HDF5 scalar DataItem")
        filename, dataset = data.text.strip().split(":", 1)
        records.append((float(time_node.attrib["Value"]),
                        folder / filename, dataset))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    series = read_series(args.original) + read_series(args.continuation)
    selected_times = [100, 250, 500, 650, 850, 1000]
    records = []
    for requested in selected_times:
        matching = [record for record in series if abs(record[0]-requested) < 1e-4]
        if not matching:
            raise ValueError(f"No saved scalar field at {requested} seconds")
        records.append(matching[-1])  # prefer the continuation at the shared time

    with h5py.File(records[0][1], "r") as file:
        xy = file["/Mesh/0/mesh/geometry"][:]
        cells = file["/Mesh/0/mesh/topology"][:]
    height = float(np.max(xy[:, 1]))
    smax = float(np.max(xy[:, 0]))
    edges = np.linspace(0.0, height, 52)
    centres = 0.5 * (edges[:-1] + edges[1:])
    weights = np.zeros((51, len(cells)))
    for cell_index, triangle in enumerate(xy[cells]):
        low = max(0, np.searchsorted(edges, np.min(triangle[:, 1]), side="right") - 1)
        high = min(50, np.searchsorted(edges, np.max(triangle[:, 1]), side="left"))
        for slab in range(low, high + 1):
            poly = clip_horizontal(list(triangle), edges[slab], True)
            poly = clip_horizontal(poly, edges[slab+1], False)
            weights[slab, cell_index] = polygon_area(poly)
    slab_area = weights.sum(axis=1)
    if not np.allclose(slab_area, smax*np.diff(edges), rtol=1e-10, atol=1e-15):
        raise AssertionError("Triangle/slab overlaps do not cover the domain")

    rows, means, profiles = [], [], []
    for requested, (actual, filename, dataset) in zip(selected_times, records):
        with h5py.File(filename, "r") as file:
            if not np.array_equal(file["/Mesh/0/mesh/geometry"][:], xy):
                raise ValueError("Mesh geometry changed between snapshots")
            if not np.array_equal(file["/Mesh/0/mesh/topology"][:], cells):
                raise ValueError("Mesh topology changed between snapshots")
            scalar = file[dataset][:].reshape(-1)
        mean = weights @ scalar / slab_area
        cmin, cmax = float(np.min(mean)), float(np.max(mean))
        if cmax <= cmin:
            raise ValueError("Cannot min-max normalise a constant profile")
        phi = (mean-cmin)/(cmax-cmin)
        means.append(mean)
        profiles.append(phi)
        for i in range(51):
            rows.append([requested, actual, i+1, edges[i], edges[i+1], centres[i],
                         centres[i]/height, slab_area[i]*2*np.pi, mean[i],
                         cmin, cmax, phi[i]])
        fresh = 2*np.pi*np.sum((1-mean)*slab_area)
        print(f"t={requested:4d} s: slab-mean min={cmin:.12g}, max={cmax:.12g}, "
              f"fresh volume={fresh:.12g} m3")
    csv_file = args.output / "axisymmetric_lower_profiles_verified.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "stored_time_s", "slab", "z_bottom_m", "z_top_m",
                         "z_centre_m", "z_over_H", "slab_volume_m3", "c_volume_mean",
                         "profile_min_c", "profile_max_c", "c_profile_normalised"])
        writer.writerows(rows)

    fig = plt.figure(figsize=(5629/600, 8635/600), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    colours = np.column_stack([np.ones(len(records)),
                              np.linspace(0.82, 0, len(records)),
                              np.linspace(0.82, 0, len(records))])
    for phi, colour in zip(profiles, colours):
        ax.plot(phi, centres/height, color=colour, lw=2.5, solid_capstyle="butt")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.savefig(args.output / "axisymmetric_lower_profiles_verified.png",
                dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()