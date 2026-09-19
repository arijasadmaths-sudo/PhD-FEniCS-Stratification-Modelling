#!/usr/bin/env python3
"""Quantify rectangular DG0 plume/dye output and compare cases at shared times.

Requires numpy, h5py, matplotlib and the accompanying
plot_rectangular_3d_quantitative.py. Run with an ordinary Python process:

  python3 analyse_rectangular_3d_quantitative.py --results RUN_DIRECTORY
  python3 analyse_rectangular_3d_quantitative.py --results RUN_A --results RUN_B \
      --labels coarse fine --comparison-out comparison

Default --field dye reads dye_complement_dg0.xdmf for comparison with PLIF.
Here 1-c_dye is dye-derived source fraction; its integral is dye-equivalent
source-fluid volume, not a direct freshwater/density measurement. --field
density reads c_dg0.xdmf and its 1-c field is the density-proxy source fraction.
Different molecular diffusivities mean these two tracers need not coincide.

The central laser-sheet analogue is averaged horizontally within 1 mm vertical
bands, including the plume. Full 3D profiles average complete horizontal slabs
of the same thickness. Exact cell/region intersections supply both weights.
The reference profile is subtracted first: b(t)=mean_c(t)-mean_c(reference),
then phi=(b-min(b))/(max(b)-min(b)). This reproduces experimental background
subtraction before normalization, including the plume present in the reference.
Raw profiles, raw amplitudes and raw budgets are retained separately. The height
rule scans retained rows from bottom upwards for the FIRST phi<1-epsilon.
It is intentionally not replaced with a ceiling-connected threshold rule.

Default ROI is the whole central plane. It is NOT the experimental image crop:
camera pixels must first be calibrated into --roi-x MIN MAX --roi-z MIN MAX
(metres). Pixel indices alone do not provide this physical calibration.

Optional experimental CSV schema: time_s,h_m (height below ceiling in metres,
measured with epsilon=0.08 and a matching ROI). Its time zero must match the
simulation comparison origin. When supplying this CSV, explicitly supply one
--time-origin value per case, even if zero; use each simulation's observed first
ceiling-contact time when experiment time starts at impingement. No onset time
is inferred. Without --time-origin, comparison times start at injection.
--background-time accepts one requested physical time per case. If omitted,
the reference is the --time-origin frame when supplied, otherwise the earliest
available frame. A match within half the saved cadence is required. Both
requested and actual reference times are recorded; earliest available is not
automatically physical onset.
Experiment interpolation is restricted to the supplied range: no extrapolation.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_rectangular_3d_quantitative import discover, read_array, section, save_figure


EPSILONS = (0.04, 0.08, 0.12)
PHYSICS_KEYS = (
    "dimensions_m", "nozzle_diameter_m", "flow_rate_m3_per_s",
    "rho_fresh_kg_per_m3", "rho_salt_kg_per_m3", "nu_m2_per_s",
    "scalar_diffusivity_m2_per_s", "dye_diffusivity_m2_per_s", "gravity_m_per_s2", "ramp_time_s",
    "inlet_perturbation", "perturbation_period_x_s", "perturbation_period_y_s",
    "numerical_return", "scalar_discretization",
)


def eps_tag(eps):
    return "eps" + ("%.2f" % eps).replace(".", "p")


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def normalized_height(profile, heights, ceiling, amplitude_floor):
    """Retained-row bottom-up FIRST threshold crossing; return raw flags."""
    profile = np.asarray(profile, dtype=float)
    amplitude = float(np.ptp(profile))
    phi = np.full(profile.shape, np.nan)
    results = {}
    if amplitude <= amplitude_floor or not np.all(np.isfinite(profile)):
        for eps in EPSILONS:
            results[eps] = (None, "tiny_amplitude", None)
        return phi, amplitude, results
    phi = (profile - profile.min()) / amplitude
    for eps in EPSILONS:
        crossed = phi < 1.0 - eps
        indices = np.flatnonzero(crossed)
        if not len(indices):
            results[eps] = (None, "no_crossing", None)
            continue
        row = int(indices[0])
        flags = []
        if row == 0:
            flags.append("lower_roi_hit")
        if not crossed[-1]:
            flags.append("not_at_retained_top")
        if np.any(~crossed[row:]):
            flags.append("disconnected_features")
        results[eps] = (float(ceiling - heights[row]), ";".join(flags) or "ok", row)
    return phi, amplitude, results


def tetra_volume(a, b, c, d):
    return np.abs(np.linalg.det(np.stack((b - a, c - a, d - a), axis=1))) / 6.0


def volume_below(sorted_points, volumes, levels):
    """Exact tetrahedron volume below one horizontal level per tetrahedron.

    Vertices must be sorted by z. One- and three-vertex cases are similar
    tetrahedra; the two-vertex clipped wedge is decomposed into three tetrahedra.
    Equal vertex heights are safe because division only joins opposite sides.
    """
    levels = np.broadcast_to(np.asarray(levels), (len(volumes),))
    counts = np.sum(sorted_points[:, :, 2] < levels[:, None], axis=1)
    answer = np.zeros(len(volumes))
    answer[counts == 4] = volumes[counts == 4]
    for inside_count in (1, 3):
        ids = np.flatnonzero(counts == inside_count)
        if not len(ids):
            continue
        p = sorted_points[ids]
        if inside_count == 1:
            factors = (levels[ids, None] - p[:, :1, 2]) / (p[:, 1:, 2] - p[:, :1, 2])
            answer[ids] = volumes[ids] * np.prod(factors, axis=1)
        else:
            factors = (p[:, 3:, 2] - levels[ids, None]) / (p[:, 3:, 2] - p[:, :3, 2])
            answer[ids] = volumes[ids] * (1.0 - np.prod(factors, axis=1))
    ids = np.flatnonzero(counts == 2)
    if len(ids):
        p = sorted_points[ids]
        a, b, c, d = (p[:, j] for j in range(4))
        level = levels[ids]
        def cross(start, end):
            ratio = (level - start[:, 2]) / (end[:, 2] - start[:, 2])
            return start + ratio[:, None] * (end - start)
        ac, ad, bc, bd = cross(a, c), cross(a, d), cross(b, c), cross(b, d)
        answer[ids] = (tetra_volume(a, b, ac, ad) +
                       tetra_volume(b, ac, ad, bd) +
                       tetra_volume(b, ac, bc, bd))
    tolerance = 1e-10 * np.maximum(volumes, 1e-30)
    if np.any(answer < -tolerance) or np.any(answer > volumes + tolerance):
        raise ValueError("Clipped volume is outside its tetrahedron.")
    return np.minimum(np.maximum(answer, 0.0), volumes)


def clip_polygons(vertices, counts, axis, bound, keep_above):
    """Vectorized Sutherland-Hodgman clipping of padded convex polygons."""
    n, capacity, _ = vertices.shape
    if n == 0:
        return vertices, counts
    edge = np.arange(capacity)[None, :]
    active = edge < counts[:, None]
    previous_index = np.mod(edge - 1, np.maximum(counts[:, None], 1))
    previous = np.take_along_axis(vertices, previous_index[:, :, None], axis=1)
    inside = (vertices[:, :, axis] >= bound[:, None]) if keep_above else (vertices[:, :, axis] <= bound[:, None])
    previous_inside = (previous[:, :, axis] >= bound[:, None]) if keep_above else (previous[:, :, axis] <= bound[:, None])
    crossing = active & (inside != previous_inside)
    ratio = np.zeros((n, capacity))
    np.divide(bound[:, None] - previous[:, :, axis],
              vertices[:, :, axis] - previous[:, :, axis], out=ratio, where=crossing)
    intersection = previous + ratio[:, :, None] * (vertices - previous)
    emitted = np.stack((crossing, active & inside), axis=2).reshape(n, -1)
    candidates = np.stack((intersection, vertices), axis=2).reshape(n, -1, 2)
    count_out = emitted.sum(1)
    if np.any(count_out > capacity):
        raise ValueError("Polygon clipping exceeded its vertex capacity.")
    offsets = np.cumsum(emitted, axis=1) - 1
    row, column = np.nonzero(emitted)
    output = np.zeros_like(vertices)
    output[row, offsets[row, column]] = candidates[row, column]
    return output, count_out


def polygon_areas(vertices, counts):
    edge = np.arange(vertices.shape[1])[None, :]
    next_index = np.mod(edge + 1, np.maximum(counts[:, None], 1))
    following = np.take_along_axis(vertices, next_index[:, :, None], axis=1)
    cross = vertices[:, :, 0] * following[:, :, 1] - vertices[:, :, 1] * following[:, :, 0]
    return 0.5 * np.abs(np.sum(np.where(edge < counts[:, None], cross, 0), axis=1))


def candidate_rows(low, high, edges):
    """COO ownership for regions with nonzero overlap in uniform z bands."""
    dz = edges[1] - edges[0]
    start = np.floor((low - edges[0]) / dz).astype(int)
    stop = np.ceil((high - edges[0]) / dz).astype(int) - 1
    start = np.maximum(start, 0)
    stop = np.minimum(stop, len(edges) - 2)
    number = np.maximum(stop - start + 1, 0)
    owner = np.repeat(np.arange(len(low)), number)
    offset = np.repeat(np.cumsum(number) - number, number)
    row = np.repeat(start, number) + np.arange(number.sum()) - offset
    return owner, row


def plane_band_weights(points, y, edges, x_roi):
    cells, polygons, _ = section(points, 1, y)
    counts = 1 + np.sum(np.any(polygons != polygons[:, :1], axis=2), axis=1)
    vertices = np.zeros((len(polygons), 12, 2))
    vertices[:, :6] = polygons
    for bound, keep_above in ((x_roi[0], True), (x_roi[1], False)):
        vertices, counts = clip_polygons(vertices, counts, 0,
                                         np.full(len(vertices), bound), keep_above)
    valid = counts >= 3
    cells, vertices, counts = cells[valid], vertices[valid], counts[valid]
    mask = np.arange(vertices.shape[1])[None, :] < counts[:, None]
    low = np.where(mask, vertices[:, :, 1], np.inf).min(1)
    high = np.where(mask, vertices[:, :, 1], -np.inf).max(1)
    owner, row = candidate_rows(low, high, edges)
    area = np.empty(len(owner))
    for begin in range(0, len(owner), 10000):
        sl = slice(begin, begin + 10000)
        polygon, n = vertices[owner[sl]], counts[owner[sl]]
        polygon, n = clip_polygons(polygon, n, 1, edges[row[sl]], True)
        polygon, n = clip_polygons(polygon, n, 1, edges[row[sl] + 1], False)
        area[sl] = polygon_areas(polygon, n)
    expected = (x_roi[1] - x_roi[0]) * np.diff(edges)
    if not np.allclose(np.bincount(row, weights=area, minlength=len(expected)), expected, rtol=2e-7, atol=1e-14):
        raise ValueError("Central plane bands do not cover the requested rectangular ROI.")
    keep = area > 0
    return cells[owner[keep]], row[keep], area[keep] / expected[row[keep]]


def slab_weights(sorted_points, volumes, edges, horizontal_area):
    owner, row = candidate_rows(sorted_points[:, 0, 2], sorted_points[:, 3, 2], edges)
    weights = np.empty(len(owner))
    for begin in range(0, len(owner), 50000):
        sl = slice(begin, begin + 50000)
        p, v = sorted_points[owner[sl]], volumes[owner[sl]]
        weights[sl] = (volume_below(p, v, edges[row[sl] + 1]) -
                       volume_below(p, v, edges[row[sl]]))
    expected = horizontal_area * np.diff(edges)
    if not np.allclose(np.bincount(row, weights=weights, minlength=len(expected)), expected, rtol=2e-7, atol=1e-14):
        raise ValueError("Tetrahedron slab volumes do not fill the horizontal bands.")
    keep = weights > 0
    return owner[keep], row[keep], weights[keep] / expected[row[keep]]


def apply_weights(operator, c, bands):
    cells, rows, weights = operator
    return np.bincount(rows, weights=weights * c[cells], minlength=bands)


def build_operators(xyz, cells, args):
    points = xyz[cells]
    lower, upper = xyz.min(0), xyz.max(0)
    lengths = upper - lower
    volumes = tetra_volume(points[:, 0], points[:, 1], points[:, 2], points[:, 3])
    if np.any(volumes <= 0) or not np.isclose(volumes.sum(), np.prod(lengths), rtol=1e-8):
        raise ValueError("The tetrahedra do not fill their rectangular bounding box.")
    x_roi = np.array(args.roi_x if args.roi_x is not None else [lower[0], upper[0]])
    z_roi = np.array(args.roi_z if args.roi_z is not None else [lower[2], upper[2]])
    if x_roi[0] < lower[0] - 1e-12 or x_roi[1] > upper[0] + 1e-12 or z_roi[0] < lower[2] - 1e-12 or z_roi[1] > upper[2] + 1e-12:
        raise ValueError("Requested ROI extends outside the mesh.")
    y = float(args.sheet_y) if args.sheet_y is not None else float(0.5 * (lower[1] + upper[1]))
    if not lower[1] < y < upper[1]:
        raise ValueError("The laser-sheet y plane must lie strictly inside the box.")
    bands = max(2, int(np.ceil((z_roi[1] - z_roi[0]) / (args.band_mm * 1e-3))))
    edges = np.linspace(z_roi[0], z_roi[1], bands + 1)
    order = np.argsort(points[:, :, 2], axis=1)
    sorted_points = np.take_along_axis(points, order[:, :, None], axis=1)
    operators = dict(plane=plane_band_weights(points, y, edges, x_roi),
                     full3d=slab_weights(sorted_points, volumes, edges, lengths[0] * lengths[1]))
    top_volumes = {}
    for thickness_mm in (20, 40, 60):
        cutoff = max(lower[2], upper[2] - thickness_mm * 1e-3)
        clipped = volumes - volume_below(sorted_points, volumes, cutoff)
        expected = lengths[0] * lengths[1] * (upper[2] - cutoff)
        if not np.isclose(clipped.sum(), expected, rtol=2e-7):
            raise ValueError("Incorrect clipped upper-layer volume.")
        top_volumes[thickness_mm] = clipped
    return dict(operators=operators, volumes=volumes, top_volumes=top_volumes,
                heights=0.5 * (edges[:-1] + edges[1:]), ceiling=float(upper[2]),
                dimensions=lengths.tolist(), x_roi=x_roi.tolist(), z_roi=z_roi.tolist(),
                sheet_y=y, actual_band_mm=float(np.diff(edges)[0] * 1000),
                cell_count=len(cells))


def analyse_case(root, label, origin, background_request, args):
    root = root.expanduser().resolve()
    frames = discover(root, args.pattern)
    destination = root / "quantitative_analysis" / args.field
    destination.mkdir(parents=True, exist_ok=True)
    config_path = root / "configuration.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    operator_cache, reference_cache = {}, {}
    records, profiles, profile_rows = [], {}, []
    def mesh_data(frame):
        reference = (frame["geometry"], frame["topology"])
        if reference not in reference_cache:
            xyz = np.asarray(read_array(frame["geometry"]), dtype=float)
            cells = np.asarray(read_array(frame["topology"]), dtype=np.int64)
            if xyz.ndim != 2 or xyz.shape[1] != 3 or cells.ndim != 2 or cells.shape[1] != 4:
                raise ValueError("Expected a three-dimensional tetrahedral mesh.")
            digest = hashlib.sha256(xyz.tobytes() + cells.tobytes()).hexdigest()
            reference_cache[reference] = digest
            if digest not in operator_cache:
                print("%s: precomputing exact plane/slab weights (%d cells)." % (label, len(cells)), flush=True)
                operator_cache[digest] = build_operators(xyz, cells, args)
            del xyz, cells
        return operator_cache[reference_cache[reference]]

    times = np.array([frame["time"] for frame in frames])
    tolerance = 0.5 * float(np.median(np.diff(times))) if len(times) > 1 else 1e-9
    if background_request is not None:
        requested_reference = background_request
        reference_rule = "explicit_background_time"
    elif args.time_origin is not None:
        requested_reference = origin
        reference_rule = "explicit_comparison_origin_frame"
    else:
        requested_reference = frames[0]["time"]
        reference_rule = "earliest_available_NOT_verified_physical_onset"
    reference_index = int(np.argmin(abs(times - requested_reference)))
    reference_frame = frames[reference_index]
    if abs(reference_frame["time"] - requested_reference) > tolerance + 1e-9:
        raise ValueError("%s: requested background time %g s has no saved frame within %g s. Supply an actual --background-time; no reference is silently substituted." % (label, requested_reference, tolerance))
    reference_data = mesh_data(reference_frame)
    reference_c = np.asarray(read_array(reference_frame["concentration"]), dtype=float).reshape(-1)
    if len(reference_c) != reference_data["cell_count"] or not np.all(np.isfinite(reference_c)):
        raise ValueError("Invalid background-reference concentration.")
    backgrounds = {kind: apply_weights(reference_data["operators"][kind], reference_c, len(reference_data["heights"]))
                   for kind in ("plane", "full3d")}
    del reference_c
    expected_geometry = (reference_data["dimensions"], reference_data["x_roi"], reference_data["z_roi"], reference_data["sheet_y"])
    print("%s: background requested=%g s, actual=%g s (%s)." %
          (label, requested_reference, reference_frame["time"], reference_rule), flush=True)
    for frame_index, frame in enumerate(frames):
        data = mesh_data(frame)
        geometry = (data["dimensions"], data["x_roi"], data["z_roi"], data["sheet_y"])
        if geometry != expected_geometry:
            raise ValueError("Domain or ROI changed between segments of one case.")
        c = np.asarray(read_array(frame["concentration"]), dtype=float).reshape(-1)
        if len(c) != data["cell_count"] or not np.all(np.isfinite(c)):
            raise ValueError("Invalid concentration at t=%g in %s" % (frame["time"], frame["xdmf"]))
        fresh = 1.0 - c
        comparison_time = float(frame["time"] - origin)
        record = dict(time_s=frame["time"], comparison_time_s=comparison_time, source_xdmf=frame["xdmf"],
                      c_field_min=float(c.min()), c_field_max=float(c.max()),
                      source_equivalent_volume_mL=float(np.dot(data["volumes"], fresh) * 1e6),
                      volume_mean_c=float(np.dot(data["volumes"], c) / data["volumes"].sum()))
        for thickness, volumes in data["top_volumes"].items():
            record["source_equivalent_top%dmm_mL" % thickness] = float(np.dot(volumes, fresh) * 1e6)
        frame_profiles = dict(heights=data["heights"], ceiling=data["ceiling"], injection_time_s=frame["time"])
        for kind in ("plane", "full3d"):
            raw = apply_weights(data["operators"][kind], c, len(data["heights"]))
            corrected = raw - backgrounds[kind]
            phi, amplitude, heights = normalized_height(corrected, data["heights"], data["ceiling"], args.amplitude_floor)
            frame_profiles[kind] = dict(raw=raw, corrected=corrected, phi=phi)
            record[kind + "_profile_c_min"] = float(raw.min())
            record[kind + "_profile_c_max"] = float(raw.max())
            record[kind + "_profile_c_mean"] = float(raw.mean())
            record[kind + "_raw_amplitude_c"] = float(np.ptp(raw))
            record[kind + "_corrected_profile_c_min"] = float(corrected.min())
            record[kind + "_corrected_profile_c_max"] = float(corrected.max())
            record[kind + "_corrected_profile_c_mean"] = float(corrected.mean())
            record[kind + "_corrected_amplitude_c"] = amplitude
            rho_f, rho_s = config.get("rho_fresh_kg_per_m3"), config.get("rho_salt_kg_per_m3")
            record[kind + "_corrected_amplitude_kg_per_m3"] = float(amplitude * (rho_s - rho_f)) if args.field == "density" and rho_f is not None and rho_s is not None else None
            for eps, (height, status, row) in heights.items():
                tag = kind + "_" + eps_tag(eps)
                record[tag + "_h_m"] = height
                record[tag + "_status"] = ("before_background_reference;" if frame["time"] < reference_frame["time"] - 1e-9 else "") + status
                record[tag + "_first_row"] = row
            for z, value, background, difference, normalized in zip(data["heights"], raw, backgrounds[kind], corrected, phi):
                profile_rows.append(dict(time_s=frame["time"], comparison_time_s=comparison_time, profile_kind=kind,
                                         z_m=float(z), depth_below_ceiling_m=float(data["ceiling"] - z),
                                         mean_c=float(value), mean_source_fraction=float(1 - value),
                                         background_mean_c=float(background), background_subtracted_mean_c=float(difference),
                                         phi=finite_or_none(normalized)))
        records.append(record)
        profiles[round(comparison_time, 9)] = frame_profiles
        if frame_index % 25 == 0 or frame_index + 1 == len(frames):
            print("%s: analysed %d/%d saved frames, t=%g s." % (label, frame_index + 1, len(frames), frame["time"]), flush=True)
    write_csv(destination / "time_metrics.csv", records, list(records[0]))
    write_csv(destination / "profiles_all_times.csv", profile_rows, list(profile_rows[0]))
    data = next(iter(operator_cache.values()))
    if "dimensions_m" in config and not np.allclose(config["dimensions_m"], data["dimensions"], rtol=1e-8, atol=1e-12):
        raise ValueError("configuration.json dimensions disagree with saved mesh.")
    roi_status = ("user_supplied_physical_roi_calibration_not_verified" if args.roi_x is not None and args.roi_z is not None
                  else "NOT_experiment_cropped_missing_complete_physical_ROI")
    summary = dict(case_label=label, source_results=str(root), snapshot_count=len(records),
                   time_range_s=[records[0]["time_s"], records[-1]["time_s"]],
                   field=args.field, comparison_time_origin_from_injection_s=origin,
                   time_origin_explicitly_supplied=args.time_origin is not None,
                   background_reference=dict(selection_rule=reference_rule, requested_time_s=requested_reference,
                                             actual_time_s=reference_frame["time"], matching_tolerance_s=tolerance,
                                             source_xdmf=reference_frame["xdmf"],
                                             note="First available output is not assumed to be first ceiling contact. Reference profiles are subtracted before normalization; budgets remain raw."),
                   source_equivalent_volume_definition=("Integral of 1-c_dye: dye-equivalent source-fluid volume, not density-derived freshwater volume."
                                                        if args.field == "dye" else "Integral of 1-c_density: density-proxy freshwater-equivalent volume."),
                   dimensions_m=data["dimensions"], central_sheet_y_m=data["sheet_y"],
                   roi_x_m=data["x_roi"], roi_z_m=data["z_roi"], roi_status=roi_status,
                   actual_vertical_band_mm=data["actual_band_mm"], amplitude_floor_c=args.amplitude_floor,
                   distinct_saved_meshes=len(operator_cache), epsilon_values=list(EPSILONS),
                   height_rule="First retained row from bottom with phi<1-epsilon; h=ceiling-z_row_centre.",
                   normalization="b=mean_c(t)-mean_c(reference); phi=(b-min(b))/(max(b)-min(b)), separately for each time/profile kind.",
                   profile_definitions={"plane": "Central sheet: exact cell/plane polygons integrated over x-ROI and vertical row band, including plume.",
                                        "full3d": "Complete horizontal slabs over same retained z-ROI; exact tetrahedron/slab volume means."},
                   configuration=config,
                   notes=["No raw-field clipping or per-image normalization.",
                          "Threshold heights with lower_roi_hit/disconnected_features are retained and flagged, not replaced.",
                          "Band-centre threshold locations have finite vertical sampling uncertainty; 1mm sampling does not imply 1mm mesh resolution.",
                          "The inlet/outlet and experiment ROI must be physically matched before claiming experimental validation.",
                          "Mesh/time differences from two cases are sensitivity observations, not a Richardson extrapolation or formal convergence order."])
    write_json(destination / "analysis_summary.json", summary)
    plot_case(records, destination, label, args.field)
    return dict(root=root, label=label, records=records, profiles=profiles, summary=summary, config=config)


def plot_case(records, destination, label, field):
    time = [r["comparison_time_s"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for kind, name in (("plane", "Central sheet"), ("full3d", "Full 3D slabs")):
        for eps in EPSILONS:
            key = kind + "_" + eps_tag(eps)
            values = [np.nan if r[key + "_h_m"] is None else 1000 * r[key + "_h_m"] for r in records]
            axes[0].plot(time, values, linestyle="-" if kind == "plane" else "--",
                         label="%s, epsilon=%g" % (name, eps), alpha=1 if kind == "plane" else 0.65)
        key = kind + "_" + eps_tag(0.08)
        flagged = [r for r in records if r[key + "_h_m"] is not None and r[key + "_status"] != "ok"]
        axes[0].scatter([r["comparison_time_s"] for r in flagged], [1000 * r[key + "_h_m"] for r in flagged],
                        marker="x", s=12, color="black", zorder=4)
    axes[0].set_xlabel("Time from selected comparison origin (s)")
    axes[0].set_ylabel("First-crossing height below ceiling (mm)")
    axes[0].set_title("%s; crosses flag epsilon=0.08 estimates" % label, fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].plot(time, [r["source_equivalent_volume_mL"] for r in records], label="Entire tank")
    for depth in (20, 40, 60):
        axes[1].plot(time, [r["source_equivalent_top%dmm_mL" % depth] for r in records], label="Upper %d mm" % depth)
    axes[1].set_xlabel("Time from selected comparison origin (s)")
    axes[1].set_ylabel("Dye-equivalent source volume (mL)" if field == "dye" else "Density-proxy freshwater volume (mL)")
    axes[1].legend(frameon=False)
    save_figure(fig, destination, "height_and_source_volume_all_times")


def physics_comparison(cases):
    missing, mismatches = [], []
    reference = cases[0]
    for key in PHYSICS_KEYS:
        values = [case["config"].get(key) for case in cases]
        if any(value is None for value in values):
            missing.append(key)
            continue
        first = values[0]
        for case, value in zip(cases[1:], values[1:]):
            try:
                a, b = np.asarray(first, dtype=float), np.asarray(value, dtype=float)
                equal = a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-10, atol=1e-14))
            except (TypeError, ValueError):
                equal = value == first
            if not equal:
                mismatches.append(dict(field=key, reference_case=reference["label"], reference_value=first,
                                       case=case["label"], value=value))
    status = "physics_mismatch" if mismatches else ("physics_unverified_missing_metadata" if missing else "matched_recorded_physics")
    pair_types = []
    for case in cases[1:]:
        a, b = reference["config"], case["config"]
        mesh_known = all("mesh_recipe" in cfg and "mesh_cells" in cfg for cfg in (a, b))
        dt_known = "dt_s" in a and "dt_s" in b
        mesh_differs = mesh_known and (a["mesh_recipe"] != b["mesh_recipe"] or a["mesh_cells"] != b["mesh_cells"])
        dt_differs = dt_known and not np.isclose(a["dt_s"], b["dt_s"], rtol=1e-12, atol=1e-15)
        if status != "matched_recorded_physics" or not mesh_known or not dt_known:
            kind = "descriptive_comparison_only"
        elif mesh_differs and dt_differs:
            kind = "combined_mesh_and_time_sensitivity"
        elif mesh_differs:
            kind = "mesh_sensitivity_at_fixed_dt"
        elif dt_differs:
            kind = "time_step_sensitivity_at_fixed_mesh"
        else:
            kind = "same_recorded_mesh_and_dt"
        pair_types.append(dict(reference=reference["label"], case=case["label"], interpretation=kind))
    return dict(status=status, missing_fields=missing, mismatches=mismatches, pairs=pair_types,
                warning="Two-case differences do not establish convergence order or support Richardson extrapolation.")


def load_experiment(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"time_s", "h_m"}.issubset(reader.fieldnames or []):
            raise ValueError("Experiment CSV must contain time_s,h_m columns.")
        rows = [(float(r["time_s"]), float(r["h_m"])) for r in reader]
    if not rows or not np.all(np.isfinite(rows)) or any(t < 0 or h < 0 for t, h in rows):
        raise ValueError("Experiment CSV requires finite, nonnegative time_s and h_m.")
    rows.sort()
    time, height = np.asarray(rows).T
    if np.any(np.diff(time) <= 0):
        raise ValueError("Experiment CSV contains duplicate times.")
    return time, height


def compare_cases(cases, destination, args):
    destination.mkdir(parents=True, exist_ok=True)
    check = physics_comparison(cases)
    common = sorted(set.intersection(*(set(case["profiles"]) for case in cases)))
    if len(cases) > 1 and check["status"] != "matched_recorded_physics":
        warnings.warn("Case comparison is descriptive only: " + check["status"])
    summary = dict(physics_check=check, common_saved_time_count=len(common),
                   common_time_range_s=[common[0], common[-1]] if common else None,
                   field=args.field,
                   case_time_origins_s={case["label"]: case["summary"]["comparison_time_origin_from_injection_s"] for case in cases},
                   profile_comparison="Same saved comparison times after explicit case time shifts; no interpolation of simulation time.")
    indices = [{round(r["comparison_time_s"], 9): r for r in case["records"]} for case in cases]
    differences = []
    for case_index in range(1, len(cases)):
        for time in common:
            a, b = indices[0][time], indices[case_index][time]
            key = "plane_" + eps_tag(0.08)
            first, second = a[key + "_h_m"], b[key + "_h_m"]
            entry = dict(reference_case=cases[0]["label"], case=cases[case_index]["label"],
                         comparison_time_s=time, reference_injection_time_s=a["time_s"], case_injection_time_s=b["time_s"],
                         reference_plane_h_m=first, case_plane_h_m=second,
                         height_difference_m=second - first if first is not None and second is not None else None,
                         both_thresholds_unflagged=a[key + "_status"] == b[key + "_status"] == "ok",
                         reference_status=a[key + "_status"], case_status=b[key + "_status"],
                         source_equivalent_volume_difference_mL=b["source_equivalent_volume_mL"] - a["source_equivalent_volume_mL"])
            pa, pb = cases[0]["profiles"][time], cases[case_index]["profiles"][time]
            same_rows = pa["heights"].shape == pb["heights"].shape and np.allclose(pa["heights"], pb["heights"], rtol=0, atol=1e-12)
            entry["profile_difference_status"] = "same_physical_row_bands" if same_rows else "unmatched_vertical_bands_no_interpolation"
            for kind in ("plane", "full3d"):
                for value_kind in ("raw", "corrected", "phi"):
                    delta = pb[kind][value_kind] - pa[kind][value_kind] if same_rows else None
                    valid = delta is not None and np.all(np.isfinite(delta))
                    entry[kind + "_" + value_kind + "_rms_difference"] = float(np.sqrt(np.mean(delta ** 2))) if valid else None
                    entry[kind + "_" + value_kind + "_max_absolute_difference"] = float(np.max(np.abs(delta))) if valid else None
            differences.append(entry)
    if differences:
        write_csv(destination / "case_differences_at_common_times.csv", differences, list(differences[0]))
    fig, ax = plt.subplots(figsize=(7, 4.7), constrained_layout=True)
    for case in cases:
        key = "plane_" + eps_tag(0.08)
        ax.plot([r["comparison_time_s"] for r in case["records"]],
                [np.nan if r[key + "_h_m"] is None else r[key + "_h_m"] * 1000 for r in case["records"]],
                label=case["label"])
        flagged = [r for r in case["records"] if r[key + "_h_m"] is not None and r[key + "_status"] != "ok"]
        ax.scatter([r["comparison_time_s"] for r in flagged], [r[key + "_h_m"] * 1000 for r in flagged], marker="x", s=13)
    if args.experiment is not None:
        ex_time, ex_height = load_experiment(args.experiment)
        ax.scatter(ex_time, ex_height * 1000, color="black", s=20, label="Experiment (epsilon=0.08)", zorder=5)
        experiment_rows, error_summaries = [], []
        for case in cases:
            errors = []
            key = "plane_" + eps_tag(0.08)
            for record in case["records"]:
                t, height = record["comparison_time_s"], record[key + "_h_m"]
                if t < ex_time[0] or t > ex_time[-1]:
                    continue
                interpolated = float(np.interp(t, ex_time, ex_height))
                valid = height is not None and record[key + "_status"] == "ok"
                difference = height - interpolated if height is not None else None
                experiment_rows.append(dict(case=case["label"], comparison_time_s=t, injection_time_s=record["time_s"], simulation_h_m=height,
                                            experiment_interpolated_h_m=interpolated, difference_m=difference,
                                            included_in_error_summary=valid, threshold_status=record[key + "_status"]))
                if valid:
                    errors.append(difference)
            error_summaries.append(dict(case=case["label"], unflagged_overlap_count=len(errors),
                                        mean_absolute_difference_m=float(np.mean(np.abs(errors))) if errors else None,
                                        root_mean_square_difference_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                                        roi_status=case["summary"]["roi_status"]))
        if experiment_rows:
            write_csv(destination / "experiment_comparison_within_supplied_range.csv", experiment_rows, list(experiment_rows[0]))
        summary["experiment"] = dict(source=str(args.experiment.resolve()), error_summaries=error_summaries,
                                     notes="No extrapolation. Errors use unflagged epsilon=0.08 central-sheet heights only. Matching ROI, time origin and experiment settings remain required.")
    ax.set_xlabel("Time from selected comparison origin (s)")
    ax.set_ylabel("Central-sheet first-crossing height (mm)")
    ax.set_title("epsilon=0.08; crosses mark flagged thresholds", fontsize=10)
    ax.legend(frameon=False)
    save_figure(fig, destination, "case_height_comparison")
    selected = [round(t, 9) for t in args.times if round(t, 9) in common]
    if not selected and common:
        selected = [common[-1]]
    for value_kind in ("raw", "phi") if selected else ():
        columns = min(3, len(selected))
        fig, axes = plt.subplots(int(math.ceil(len(selected) / columns)), columns,
                                 figsize=(4 * columns, 3.4 * math.ceil(len(selected) / columns)),
                                 squeeze=False, constrained_layout=True)
        for ax, time in zip(axes.flat, selected):
            for case in cases:
                profile = case["profiles"][time]
                values = 100 * (1 - profile["plane"]["raw"]) if value_kind == "raw" else profile["plane"]["phi"]
                ax.plot(values, profile["heights"] * 1000, label=case["label"])
            ax.set_title("Common comparison time: %g s" % time)
            ax.set_xlabel("Raw central-sheet source fraction (%)" if value_kind == "raw" else "Background-subtracted normalized profile, phi")
            ax.set_ylabel("Height z (mm)")
            ax.legend(fontsize=8, frameon=False)
        for ax in list(axes.flat)[len(selected):]:
            ax.set_axis_off()
        save_figure(fig, destination, "central_raw_profiles_at_common_times" if value_kind == "raw" else "central_profiles_at_common_times")
    summary["profile_figure_actual_times_s"] = selected
    write_json(destination / "comparison_summary.json", summary)


def self_test():
    # Six tetrahedra exactly tile a unit cube; three clipping configurations and
    # vertex-equal planes exercise the exact clipped-volume formula.
    xyz = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)], dtype=float)
    cells = np.array([(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7),
                      (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)])
    points = xyz[cells]
    sorted_points = np.take_along_axis(points, np.argsort(points[:, :, 2], axis=1)[:, :, None], axis=1)
    volumes = tetra_volume(points[:, 0], points[:, 1], points[:, 2], points[:, 3])
    for height in (0, 0.1, 0.25, 0.5, 0.8, 1):
        assert np.isclose(volume_below(sorted_points, volumes, height).sum(), height, atol=1e-13)
    edges = np.linspace(0.1, 0.9, 9)
    for operator in (slab_weights(sorted_points, volumes, edges, 1),
                     plane_band_weights(points, 0.5, edges, [0.2, 0.8])):
        assert np.allclose(apply_weights(operator, np.ones(6) * 0.73, 8), 0.73, atol=1e-12)
    # A continuous linear concentration is not represented by DG0; this check
    # instead validates the geometric first moment of a clipped rectangle.
    rectangle = np.zeros((1, 12, 2))
    rectangle[0, :4] = [(0, 0), (1, 0), (1, 1), (0, 1)]
    clipped, count = clip_polygons(rectangle, np.array([4]), 1, np.array([0.25]), True)
    clipped, count = clip_polygons(clipped, count, 1, np.array([0.75]), False)
    assert np.isclose(polygon_areas(clipped, count)[0], 0.5)
    p = clipped[0, :count[0]]
    next_p = np.roll(p, -1, axis=0)
    cross = p[:, 0] * next_p[:, 1] - next_p[:, 0] * p[:, 1]
    moment_z = np.sum((p[:, 1] + next_p[:, 1]) * cross) / 6
    assert np.isclose(abs(moment_z), 0.25)  # integral z dx dz on [0,1]x[.25,.75]
    z = np.array([0.05, 0.15, 0.25, 0.35])
    _, amp, heights = normalized_height([1, 1, 0.9, 0], z, 0.4, 1e-6)
    assert amp == 1 and np.isclose(heights[0.08][0], 0.15) and heights[0.08][1] == "ok"
    _, _, heights = normalized_height([1, 0.8, 1, 0], z, 0.4, 1e-6)
    assert np.isclose(heights[0.08][0], 0.25) and "disconnected_features" in heights[0.08][1]
    _, _, heights = normalized_height([0.8, 1, 1, 0], z, 0.4, 1e-6)
    assert "lower_roi_hit" in heights[0.08][1]
    _, _, heights = normalized_height([1, 1, 1, 1], z, 0.4, 1e-6)
    assert heights[0.08][0] is None and heights[0.08][1] == "tiny_amplitude"
    background = np.array([0.8, 1, 0.95, 0.9])
    later = background + np.array([0, 0, -0.001, -0.005])
    _, _, heights = normalized_height(later - background, z, 0.4, 1e-6)
    assert np.isclose(heights[0.08][0], 0.15) and heights[0.08][1] == "ok"
    _, _, heights = normalized_height(background - background, z, 0.4, 1e-6)
    assert heights[0.08][0] is None
    print("Self-test passed: exact cube clipping, plane/slab constant weights, linear polygon moment, first-crossing rule, background subtraction and flags.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, action="append", help="Repeat for comparison cases.")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--field", choices=("dye", "density"), default="dye",
                        help="Default dye for PLIF; density selects the buoyancy scalar.")
    parser.add_argument("--pattern", help="Override XDMF glob, relative to results. Default follows --field.")
    parser.add_argument("--roi-x", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--roi-z", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--sheet-y", type=float, help="Laser-sheet coordinate in metres; default tank midplane.")
    parser.add_argument("--band-mm", type=float, default=1.0, help="Requested maximum vertical band thickness.")
    parser.add_argument("--amplitude-floor", type=float, default=1e-6, help="Minimum profile concentration range for normalization.")
    parser.add_argument("--times", type=float, nargs="+", default=[20, 40, 60, 80, 100, 120], help="Profile-panel times, used only when saved in every case.")
    parser.add_argument("--time-origin", type=float, nargs="+", help="One explicit time-zero offset (seconds after injection) per case; required for experiment overlays, including offsets of 0.")
    parser.add_argument("--background-time", type=float, nargs="+", help="One physical reference time per case. Default: explicit --time-origin if supplied, otherwise earliest available. Must match a saved frame within half cadence.")
    parser.add_argument("--comparison-out", type=Path)
    parser.add_argument("--experiment", type=Path, help="CSV with time_s,h_m; epsilon=0.08, matching ROI and explicit --time-origin.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if not args.results:
            return
    if not args.results:
        parser.error("Supply at least one --results directory, or --self-test.")
    if args.labels and len(args.labels) != len(args.results):
        parser.error("Provide one --labels entry per --results directory.")
    if args.time_origin is not None and (len(args.time_origin) != len(args.results) or not all(math.isfinite(t) and t >= 0 for t in args.time_origin)):
        parser.error("Provide one finite nonnegative --time-origin offset per results directory.")
    if args.background_time is not None and (len(args.background_time) != len(args.results) or not all(math.isfinite(t) and t >= 0 for t in args.background_time)):
        parser.error("Provide one finite nonnegative --background-time per results directory.")
    if args.experiment is not None and args.time_origin is None:
        parser.error("Experiment overlays require explicit --time-origin for each case (0 for injection-based times, measured ceiling-contact time for post-impingement times).")
    if not math.isfinite(args.band_mm) or args.band_mm < 0.1 or not math.isfinite(args.amplitude_floor) or args.amplitude_floor <= 0:
        parser.error("--band-mm must be >=0.1; --amplitude-floor must be positive.")
    for roi in (args.roi_x, args.roi_z):
        if roi is not None and (not np.all(np.isfinite(roi)) or roi[0] >= roi[1]):
            parser.error("ROI bounds must be finite and increasing.")
    if args.sheet_y is not None and not math.isfinite(args.sheet_y):
        parser.error("--sheet-y must be finite.")
    if len(args.times) > 12 or not all(math.isfinite(t) and t >= 0 for t in args.times):
        parser.error("Supply at most 12 finite nonnegative --times.")
    labels = args.labels or [p.resolve().name for p in args.results]
    args.pattern = args.pattern or ("segments/*/dye_complement_dg0.xdmf" if args.field == "dye" else "segments/*/c_dg0.xdmf")
    origins = args.time_origin or [0.0] * len(args.results)
    backgrounds = args.background_time or [None] * len(args.results)
    if len(set(labels)) != len(labels):
        parser.error("Case labels must be unique; supply --labels if directory basenames repeat.")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    cases = [analyse_case(root, label, origin, background, args) for root, label, origin, background in zip(args.results, labels, origins, backgrounds)]
    destination = args.comparison_out or (cases[0]["root"] / "quantitative_analysis" / args.field / "comparison")
    compare_cases(cases, destination.expanduser().resolve(), args)
    print("Saved all-time metrics/profiles per case and comparison outputs in", destination)


if __name__ == "__main__":
    main()
