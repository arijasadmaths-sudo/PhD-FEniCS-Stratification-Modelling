#!/usr/bin/env python3
"""Audit the two compact Cartesian half-step RETURN folders and compare at 5 s.

Usage: python analyze_half_step.py COARSER_RETURN FINER_RETURN --out review
NumPy is required. The dt=0.001 reference states are inside each RETURN folder.
This reads saved results; it does not execute the coupled FEniCS solver.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def overlap(target, source):
    return np.maximum(0, np.minimum(target[1:, None], source[None, 1:])
                      - np.maximum(target[:-1, None], source[None, :-1]))


def same_coordinates(a, b):
    return a.shape == b.shape and np.allclose(a, b, atol=5e-16, rtol=0)


def field(path, label, dt):
    a = read_npz(path)
    for k, v in a.items():
        assert np.isfinite(v).all(), k
    x, z = a['x_edges_m'], a['y_edges_m']
    dx, dz = np.diff(x), np.diff(z)
    assert np.all(dx > 0) and np.all(dz > 0)
    np.testing.assert_allclose([x[0], x[-1], z[0], z[-1]], [0, .6, 0, .3], atol=1e-14, rtol=0)
    area = dz[:, None] * dx[None, :]
    np.testing.assert_allclose(area, a['cell_area_m2'], atol=1e-18, rtol=1e-13)
    np.testing.assert_allclose(a['x_m'], (x[1:] + x[:-1]) / 2, atol=1e-14, rtol=0)
    np.testing.assert_allclose(a['y_m'], (z[1:] + z[:-1]) / 2, atol=1e-14, rtol=0)
    assert a['c'].shape == area.shape
    assert abs(float(a['time_s']) - int(a['step']) * dt) < 1e-12
    f = 1 - a['c']  # Raw freshwater fraction: no clipping or normalisation.
    inventory = float(np.sum(f * area))
    assert inventory > 0
    fx, fy = a['transport_fx_m2_s'], a['transport_fy_m2_s']
    assert fx.shape == (len(dz), len(dx) + 1)
    assert fy.shape == (len(dz) + 1, len(dx))
    for corrected, raw, which in [(fx, a['raw_fx_m2_s'], 'x'), (fy, a['raw_fy_m2_s'], 'y')]:
        boundary = (slice(None), [0, -1]) if which == 'x' else ([0, -1], slice(None))
        assert np.array_equal(corrected[boundary], raw[boundary])
    divergence = np.diff(fx, axis=1) + np.diff(fy, axis=0)
    net = float(fx[:, -1].sum() - fx[:, 0].sum() + fy[-1].sum() - fy[0].sum())
    return dict(label=label, dt_s=dt, a=a, x=x, z=z, area=area, f=f,
                profile=f @ dx / (x[-1] - x[0]), inventory_m2=inventory,
                mean_height_mm=float(np.sum(f * area * a['y_m'][:, None]) / inventory * 1000),
                max_cell_flux_residual_m2_s=float(np.max(abs(divergence))),
                boundary_net_flux_m2_s=net)


def compare(ref, new, label):
    assert float(ref['a']['time_s']) == float(new['a']['time_s'])
    same = same_coordinates(ref['x'], new['x']) and same_coordinates(ref['z'], new['z'])
    ox, oz = overlap(ref['x'], new['x']), overlap(ref['z'], new['z'])
    np.testing.assert_allclose(ox.sum(axis=1), np.diff(ref['x']), atol=1e-14, rtol=0)
    np.testing.assert_allclose(oz.sum(axis=1), np.diff(ref['z']), atol=1e-14, rtol=0)
    mapped = new['f'] if same else (oz @ new['f'] @ ox.T) / ref['area']
    assert abs(np.sum(mapped * ref['area']) - new['inventory_m2']) < 1e-16
    norm2 = np.sum(ref['area'] * ref['f']**2)
    projected2 = np.sum(ref['area'] * (mapped - ref['f'])**2)
    # The exact piecewise-constant comparison retains subcell variation.
    exact2 = projected2 if same else norm2 + np.sum(new['area'] * new['f']**2) - 2 * np.sum(ref['f'] * (oz @ new['f'] @ ox.T))
    assert exact2 >= projected2 - 1e-17
    mapped_profile = mapped @ np.diff(ref['x']) / .6
    dz = np.diff(ref['z'])
    pn2 = np.sum(dz * ref['profile']**2)
    profile2 = np.sum(dz * (mapped_profile - ref['profile'])**2)
    exact_profile2 = profile2 if same else pn2 + np.sum(np.diff(new['z']) * new['profile']**2) - 2 * ref['profile'] @ oz @ new['profile']
    return dict(comparison=label, reference=ref['label'], new=new['label'],
                time_s=float(ref['a']['time_s']), same_mesh=same,
                field_l2_percent=float(100 * np.sqrt(projected2 / norm2)),
                exact_field_l2_percent=float(100 * np.sqrt(exact2 / norm2)),
                profile_l2_percent=float(100 * np.sqrt(profile2 / pn2)),
                exact_profile_l2_percent=float(100 * np.sqrt(exact_profile2 / pn2)),
                inventory_change_percent=100 * (new['inventory_m2'] / ref['inventory_m2'] - 1),
                mean_height_change_mm=new['mean_height_mm'] - ref['mean_height_mm'])


def audit_half(folder):
    cfg = read_json(folder / 'budget_configuration.json')
    launch = read_json(folder / 'launcher_status.json')
    status = read_json(folder / 'run_status.json')
    cp = read_npz(folder / 'checkpoint_latest.npz')
    meta = json.loads(str(cp['metadata_json']))
    b = np.genfromtxt(folder / 'scalar_budget.csv', names=True, delimiter=',')
    name = cfg['verification_case']
    dt = cfg['dt_s']
    assert dt == .0005 and cfg['requested_steps'] == 10000 and cfg['requested_duration_s'] == 5
    assert cfg['fresh_start'] and cfg['resume_from'] is None and cfg['initial_step'] == 0
    assert cfg['initial_time_s'] == 0 and cfg['initial_vf_m2'] == 0
    assert cfg['initial_cumulative_budget_m2'] == [0, 0, 0]
    assert status['status'] == 'completed' and status['last_completed_step'] == 10000
    assert status['last_completed_time_s'] == 5 and status == launch['simulation_status']
    assert launch['workload_exit_code'] == launch['log_capture_exit_code'] == 0
    for k in ['missing_files', 'errors', 'accepted_csv_missing_step_ranges', 'missing_elapsed_snapshots_s']:
        assert not launch[k], k
    assert launch['accepted_csv_rows'] == 10000 and launch['history_segment_count'] == 1
    assert sha(folder / 'checkpoint_latest.npz') == launch['checkpoint_sha256']
    source_hashes = {k: sha(folder / k) for k in launch['source_sha256']}
    assert source_hashes == launch['source_sha256']
    assert cfg['script_sha256'] == source_hashes['cartesian_half_step_5s.py'] == meta['source_sha256']
    assert meta['fingerprint'] == cfg['checkpoint_fingerprint']
    assert meta['case'] == name and meta['dt_s'] == dt
    final = field(folder / 'fv_final.npz', name + '_half', dt)
    assert int(cp['step']) == int(final['a']['step']) == 10000
    assert float(cp['time_s']) == float(final['a']['time_s']) == 5
    for k in ['c', 'x_edges_m', 'y_edges_m']:
        assert np.array_equal(cp[k], final['a'][k]), k
    for k in ['u_dofs', 'p_dofs', 'velocity_dof_xy_m', 'pressure_dof_xy_m', 'mesh_coordinates_m']:
        assert np.isfinite(cp[k]).all(), k
    assert cp['u_dofs'].size == cfg['velocity_dofs']
    assert cp['p_dofs'].size == cfg['pressure_dofs']
    assert cp['mesh_cells'].shape == (cfg['triangular_cells'], 3)
    assert list(final['a']['c'].shape[::-1]) == cfg['scalar_grid_cells']
    assert cfg['triangular_cells'] == 2 * final['a']['c'].size
    np.testing.assert_array_equal(b['step'], np.arange(1, 10001))
    np.testing.assert_allclose(b['time_s'], b['step'] * dt, atol=1e-12, rtol=0)
    assert (b['accepted'] == 1).all()
    assert not cfg['scalar_clipping_enabled'] and not cfg['global_scalar_mass_rescaling']
    assert (b['clipping_change_m2'] == 0).all() and (b['cum_clipping_change_m2'] == 0).all()
    assert b['c_min'].min() >= -cfg['bounds_stop_tolerance']
    assert b['c_max'].max() <= 1 + cfg['bounds_stop_tolerance']
    np.testing.assert_allclose(b['vf_previous_m2'][1:], b['vf_m2'][:-1], atol=1e-17, rtol=0)
    assert b['vf_previous_m2'][0] == 0
    rates = ['fresh_adv_in_m2_s', 'fresh_adv_out_m2_s', 'fresh_diff_in_m2_s']
    accumulated = ['cum_fresh_adv_in_m2', 'cum_fresh_adv_out_m2', 'cum_fresh_diff_in_m2']
    for i, (rate, total) in enumerate(zip(rates, accumulated)):
        np.testing.assert_allclose(np.cumsum(b[rate] * dt), b[total], atol=1e-17, rtol=0)
        assert abs(cp['cumulative_budget_m2'][i] - b[total][-1]) < 1e-17
    recomputed = b['vf_m2'] - (b[accumulated[0]] - b[accumulated[1]] + b[accumulated[2]])
    np.testing.assert_allclose(recomputed, b['cumulative_budget_residual_m2'], atol=1e-18, rtol=0)
    step_residual = b['vf_m2'] - b['vf_previous_m2'] - dt * (b[rates[0]] - b[rates[1]] + b[rates[2]])
    np.testing.assert_allclose(step_residual, b['step_budget_residual_m2'], atol=1e-18, rtol=0)
    for value in [float(cp['vf_m2']), b['vf_m2'][-1], status['final_vf_m2']]:
        assert abs(value - final['inventory_m2']) < 1e-17
    history = list((folder / 'history').iterdir())
    assert len(history) == 1
    for fn in ['scalar_budget.csv', 'budget_configuration.json', 'run_status.json']:
        assert (history[0] / fn).read_bytes() == (folder / fn).read_bytes(), fn
    snapshots = {}
    for t, filename in launch['physical_time_snapshots_s'].items():
        state = field(folder / filename, name + '_half', dt)
        assert float(state['a']['time_s']) == float(t)
        row = b[int(state['a']['step']) - 1]
        assert abs(state['inventory_m2'] - row['vf_m2']) < 1e-17
        assert abs(float(state['a']['c'].min()) - row['c_min']) < 1e-15
        assert abs(float(state['a']['c'].max()) - row['c_max']) < 1e-15
        snapshots[float(t)] = state
    assert sorted(snapshots) == [.5, 1, 2, 3, 4, 5]
    for k in final['a']:
        assert np.array_equal(final['a'][k], snapshots[5]['a'][k]), k
    totals = cp['cumulative_budget_m2']
    final_residual = final['inventory_m2'] - (totals[0] - totals[1] + totals[2])
    metrics = dict(case=name, dt_s=dt, final_time_s=5., accepted_steps=10000,
                   fv_grid=cfg['scalar_grid_cells'], triangles=cfg['triangular_cells'],
                   fresh_start=True, runtime_hours=status['elapsed_wall_s'] / 3600,
                   inventory_m2=final['inventory_m2'], mean_height_mm=final['mean_height_mm'],
                   final_budget_residual_m2=float(final_residual),
                   final_relative_budget_residual=float(abs(final_residual) / final['inventory_m2']),
                   max_abs_relative_budget_residual=float(np.max(abs(recomputed) / b['vf_m2'])),
                   max_abs_step_budget_residual_m2=float(np.max(abs(step_residual))),
                   min_c_recorded=float(b['c_min'].min()), max_c_recorded=float(b['c_max'].max()),
                   final_c_min=float(b['c_min'][-1]), final_c_max=float(b['c_max'][-1]),
                   cumulative_advective_input_m2=float(totals[0]),
                   cumulative_advective_output_m2=float(totals[1]),
                   cumulative_diffusive_input_m2=float(totals[2]),
                   diffusive_input_fraction_percent=float(100 * totals[2] / final['inventory_m2']),
                   max_abs_boundary_net_flux_m2_s=float(np.max(abs(b['boundary_net_volume_flux_m2_s']))),
                   max_cell_flux_residual_m2_s=float(np.max(abs(b['cell_flux_residual_after_max_m2_s']))),
                   max_scalar_row_residual_m2_s=float(np.max(abs(b['scalar_row_residual_max_m2_s']))),
                   max_flux_correction_relative=float(np.max(b['face_flux_correction_relative_l2'])),
                   final_flux_correction_relative=float(b['face_flux_correction_relative_l2'][-1]))
    integrity = dict(return_folder=folder.name, checkpoint_sha256=launch['checkpoint_sha256'],
                     source_hashes=source_hashes, checkpoint_final_and_snapshot_match=True,
                     csv_complete=True, history_matches=True, bounds_pass=True,
                     zero_clipping=True, fresh_start_identity_verified=True,
                     snapshots_verified=list(snapshots))
    return cfg, final, snapshots, metrics, integrity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('coarser_return', type=Path)
    p.add_argument('finer_return', type=Path)
    p.add_argument('--out', type=Path, default=Path('review'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folders = [args.coarser_return, args.finer_return]
    cases = ['compact312p5', 'compact156p25']
    results = [audit_half(folder) for folder in folders]
    assert [r[0]['verification_case'] for r in results] == cases
    cfgs, half, snapshots, audits, integrity = zip(*results)
    baseline = []
    physical_keys = ['domain_m', 'maximum_inlet_velocity_m_s', 'maximum_return_velocity_m_s',
                     'nozzle_width_m', 'nu_m2_s', 'D_m2_s', 'g_m_s2', 'beta', 'ambient_c',
                     'inlet_c', 'ramp_s', 'scalar_time_scheme', 'scalar_advection', 'scalar_diffusion',
                     'transport_velocity', 'core_x_m', 'core_z_m']
    for i, case in enumerate(cases):
        root = folders[0] / 'reference_dt001'
        cfg = read_json(root / case / 'budget_configuration.json')
        assert cfg['dt_s'] == .001 and cfg['requested_steps'] == 5000
        for key in physical_keys + ['scalar_grid_cells', 'triangular_cells', 'core_spacing_m']:
            assert cfg[key] == cfgs[i][key], key
        state = field(root / case / 'fv_final.npz', case + '_full', .001)
        assert float(state['a']['time_s']) == 5
        assert same_coordinates(state['x'], half[i]['x']) and same_coordinates(state['z'], half[i]['z'])
        integrity[i]['max_baseline_grid_coordinate_difference_m'] = float(max(
            np.max(abs(state['x'] - half[i]['x'])), np.max(abs(state['z'] - half[i]['z']))))
        baseline.append(state)
    for key in physical_keys:
        assert cfgs[0][key] == cfgs[1][key], key
    for i, folder in enumerate(folders):
        root = folder / 'reference_dt001'
        provenance = read_json(root / 'provenance.json')
        for case in cases:
            for fn, expected in provenance['references'][case]['files'].items():
                assert sha(root / case / fn) == expected
                assert (root / case / fn).read_bytes() == (folders[0] / 'reference_dt001' / case / fn).read_bytes()
        assert sha(root / 'original_dt001_solver.py') == provenance['original_solver_sha256']
        integrity[i]['reference_hashes_verified'] = 9
    comparisons = [compare(baseline[i], half[i], 'time_step_' + cases[i]) for i in range(2)]
    comparisons += [compare(*baseline, 'mesh_dt_0.001'), compare(*half, 'mesh_dt_0.0005')]
    startup = [compare(snapshots[0][t], snapshots[1][t], 'mesh_dt_0.0005') for t in sorted(snapshots[0])]
    baseline_metrics = [dict(case=c['label'], dt_s=c['dt_s'], inventory_m2=c['inventory_m2'],
                             mean_height_mm=c['mean_height_mm']) for c in baseline]
    metrics = dict(half_step_audits=audits, baseline_integrals=baseline_metrics,
                   comparisons_at_5s=comparisons, spatial_half_step_history=startup,
                   method='Raw f=1-c. Same-mesh temporal differences are direct area-weighted L2. '
                   'Spatial differences use exact rectangular-overlap conservative averaging onto '
                   'the coarser mesh; raw full-width mean profiles are compared on the same coarse '
                   'height slabs. Denominators use the first solution in each pair. Additional exact '
                   'piecewise-constant norms retain subcell variation. No scalar clipping, rescaling '
                   'or min-max profile normalisation is applied.',
                   limitations=['One time-step halving on each compact mesh does not establish a convergence order.',
                                'Matched dt=0.001 compact fields before 5 s are absent from these references; '
                                'no early-time temporal field accuracy claim is made.',
                                'This 5 s check does not validate the original 60 s trajectories or the '
                                'continued adequacy of the compact refined region.'],
                   coordinate_note='The coarser repeat grid has coordinate round-off differences no larger '
                   'than 1.12e-16 m; temporal norms use direct corresponding cells and reference areas. '
                   'Conservative remapping gives the same reported percentages. The finer grid is bitwise identical.')
    (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (args.out / 'integrity.json').write_text(json.dumps(integrity, indent=2) + '\n')
    for filename, rows in [('comparisons_at_5s.csv', comparisons), ('spatial_half_step_history.csv', startup),
                           ('half_step_integrals.csv', audits)]:
        with (args.out / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    profiles = {}
    for c in list(baseline) + list(half):
        profiles[c['label'] + '_z_edges_m'] = c['z']
        profiles[c['label'] + '_profile'] = c['profile']
    np.savez_compressed(args.out / 'profiles.npz', **profiles)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
