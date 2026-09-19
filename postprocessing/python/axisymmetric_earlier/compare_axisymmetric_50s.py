#!/usr/bin/env python3
"""Compare 50 s verification fields with the archived 70 cc/min calculation.

Requires NumPy only. Run after collecting the output folders:
  python3 compare_axisymmetric_50s.py /path/timestep /path/mesh --output comparison

Each positional path can be a run folder or a verification_final.npz file.
The supplied axisymmetric_70_baseline_50s.npz must remain beside this script,
unless --baseline gives another path. All comparisons use the fresh fraction,
f=1-c, without min-max normalisation. They describe sensitivity at 50 s, not
an estimate of exact error or proof of convergence at 1000 s.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read_field(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'verification_final.npz'
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in data.files}
    for key in ('geometry_s_z', 'topology', 'fresh', 'cell_volumes_m3', 'time_s'):
        if key not in result:
            raise ValueError('Missing {} in {}'.format(key, path))
    xy, cells = result['geometry_s_z'], result['topology']
    fresh, supplied_volumes = result['fresh'], result['cell_volumes_m3']
    if xy.ndim != 2 or xy.shape[1] != 2 or cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError('Expected triangular mesh in (s,z) coordinates')
    if fresh.shape != (len(cells),) or supplied_volumes.shape != fresh.shape:
        raise ValueError('Scalar values and volumes must be in mesh-cell order')
    if not all(np.all(np.isfinite(a)) for a in (xy, fresh, supplied_volumes)):
        raise ValueError('Non-finite field or mesh')
    pts = xy[cells]
    a, b = pts[:, 1]-pts[:, 0], pts[:, 2]-pts[:, 0]
    volumes = np.pi * np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    if np.any(volumes <= 0):
        raise ValueError('Degenerate triangle')
    np.testing.assert_allclose(supplied_volumes, volumes, rtol=2e-10, atol=1e-20)
    if abs(float(result['time_s'])-50.) > 1e-6:
        raise ValueError('Comparison requires a completed field at 50 s: '+str(path))
    result['path'] = str(path)
    return result


def coarse_averages(field, s_edges, z_edges):
    """Exact volume averages onto the original pre-barycentric rectangles.

    The verification mesh bisects every original physical radial/vertical
    interval. Its triangles remain within these coarse rectangles. Verify
    this before using cell centroids to assign an entire triangle. No
    interpolation, extrapolation, clipping or amplitude rescaling is used.
    """
    points = field['geometry_s_z'][field['topology']]
    centres = points.mean(axis=1)
    ns, nz = len(s_edges)-1, len(z_edges)-1
    i = np.searchsorted(s_edges, centres[:, 0], side='right')-1
    j = np.searchsorted(z_edges, centres[:, 1], side='right')-1
    if np.any((i < 0) | (i >= ns) | (j < 0) | (j >= nz)):
        raise ValueError('Field extends outside baseline enclosure')
    tol = 1e-12
    if (np.any(points[:, :, 0] < s_edges[i, None]-tol)
            or np.any(points[:, :, 0] > s_edges[i+1, None]+tol)
            or np.any(points[:, :, 1] < z_edges[j, None]-tol)
            or np.any(points[:, :, 1] > z_edges[j+1, None]+tol)):
        raise ValueError('Triangles cross comparison cells; exact intersection needed')
    ids = j*ns+i
    volumes = field['cell_volumes_m3']
    expected = 2*np.pi*np.diff(z_edges)[:, None]*np.diff(s_edges)[None, :]
    coverage = np.bincount(ids, weights=volumes, minlength=ns*nz).reshape(nz, ns)
    np.testing.assert_allclose(coverage, expected, rtol=2e-10, atol=1e-18)
    mass = np.bincount(ids, weights=volumes*field['fresh'], minlength=ns*nz).reshape(nz, ns)
    averages = mass/expected
    np.testing.assert_allclose(np.sum(averages*expected), volumes@field['fresh'], rtol=1e-12, atol=1e-18)
    return averages, expected


def relative_l2(reference, candidate, weights):
    denominator = np.sum(weights*reference**2)
    if denominator <= 0:
        raise ValueError('Reference norm is zero')
    return float(np.sqrt(np.sum(weights*(candidate-reference)**2)/denominator))


def self_test():
    baseline = read_field(Path(__file__).with_name('axisymmetric_70_baseline_50s.npz'))
    se, ze = baseline['coarse_s_edges'], baseline['coarse_z_edges']
    means, volume = coarse_averages(baseline, se, ze)
    # Subdivide each triangle into four; retain its piecewise-constant scalar.
    p = baseline['geometry_s_z'][baseline['topology']]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    triangles = np.stack((np.stack((a, ab, ca), axis=1),
                          np.stack((ab, b, bc), axis=1),
                          np.stack((ca, bc, c), axis=1),
                          np.stack((ab, bc, ca), axis=1)), axis=1).reshape(-1, 3, 2)
    refined = {'geometry_s_z': triangles.reshape(-1, 2),
               'topology': np.arange(triangles.shape[0]*3).reshape(-1, 3),
               'fresh': np.repeat(baseline['fresh'], 4),
               'cell_volumes_m3': np.repeat(baseline['cell_volumes_m3']/4, 4)}
    refined_means, unused = coarse_averages(refined, se, ze)
    np.testing.assert_allclose(refined_means, means, rtol=1e-13, atol=1e-15)
    assert relative_l2(means, refined_means, volume) < 1e-13
    scaled = means*1.1
    assert abs(relative_l2(means, scaled, volume)-.1) < 1e-12
    print('PASS: exact physical volumes, complete cell coverage, conservative subdivision and weighted comparison.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='*')
    parser.add_argument('--baseline', type=Path, default=Path(__file__).with_name('axisymmetric_70_baseline_50s.npz'))
    parser.add_argument('--output', type=Path, default=Path('axisymmetric_50s_comparison'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.runs:
        parser.error('Supply one or more verification output folders')
    baseline = read_field(args.baseline)
    se, ze = baseline['coarse_s_edges'], baseline['coarse_z_edges']
    original, volumes = coarse_averages(baseline, se, ze)
    dz = np.diff(ze)
    slab_volumes = volumes.sum(axis=1)
    original_profile = (original*volumes).sum(axis=1)/slab_volumes
    original_inventory = float(np.sum(original*volumes))
    summaries = []
    profiles = {'baseline': original_profile}
    for index, name in enumerate(args.runs):
        field = read_field(name)
        # The archived scalar belongs to the 70 cc/min Oseen calculation.
        # Reject another case rather than silently comparing different sources.
        if 'Q_cc_min' not in field or abs(float(field['Q_cc_min'])-70.) > 1e-12:
            raise ValueError('Verification field must identify Q_cc_min=70')
        if 'source_base_sha256' not in field or str(field['source_base_sha256']) != str(baseline['source_sha256']):
            raise ValueError('Verification source does not match the archived baseline')
        accepted, unused = coarse_averages(field, se, ze)
        profile = (accepted*volumes).sum(axis=1)/slab_volumes
        inventory = float(np.sum(accepted*volumes))
        label = str(field.get('case', 'run_{}'.format(index+1)))
        if label in profiles:
            label = label+'_{}'.format(index+1)
        profiles[label] = profile
        summaries.append({
            'label': label, 'file': field['path'], 'time_s': float(field['time_s']),
            'dt_s': float(field['dt_s']),
            'triangles': int(len(field['topology'])),
            'fresh_inventory_m3': inventory,
            'inventory_change_percent': 100*(inventory/original_inventory-1),
            'field_relative_L2_difference_percent': 100*relative_l2(original, accepted, volumes),
            'mean_profile_relative_L2_difference_percent': 100*relative_l2(original_profile, profile, dz),
            'minimum_fresh': float(field['fresh'].min()),
            'maximum_fresh': float(field['fresh'].max()),
        })
    # Avoid overwriting any earlier comparison.
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output/'comparison_summary.json'
    profile_file = args.output/'comparison_profiles.csv'
    if report.exists() or profile_file.exists():
        raise FileExistsError('Use a new comparison output directory')
    result = {'reference': str(args.baseline), 'baseline_inventory_m3': original_inventory,
              'interpretation': 'Sensitivity at 50 s; relative L2 uses physical volume/height weights. No min-max normalisation; no automatic convergence verdict.',
              'cases': summaries}
    report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with profile_file.open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['z_lower_m', 'z_upper_m', 'z_mid_m']+list(profiles))
        for j in range(len(dz)):
            writer.writerow([ze[j], ze[j+1], (ze[j]+ze[j+1])/2]+[p[j] for p in profiles.values()])
    print(json.dumps(result, indent=2, allow_nan=False))
    print('Saved', report, 'and', profile_file)


if __name__ == '__main__':
    main()