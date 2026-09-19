#!/usr/bin/env python3
"""Audit and compare the recovered extra-fine quarter-step return.

Usage: python analyze_quarter_check.py --return-dir EXTRACTED_RETURN --out OUTPUT
The recovered return includes all six earlier reference cases. No FEniCS run
is performed. Requires NumPy, SciPy, Shapely >= 2; plotting is a separate script.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
import shapely
import comparison_geometry as g


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readj(path):
    return json.loads(path.read_text())


def fingerprint(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def audit(root):
    manifest = readj(root / 'content_manifest.json')
    for name, item in manifest.items():
        path = root / name
        assert path.is_file() and path.stat().st_size == item['size_bytes'], name
        assert sha(path) == item['sha256'], name
    references = readj(root / 'reference_fields/REFERENCE_SHA256.json')
    for name, digest in references.items():
        assert sha(root / 'reference_fields' / name) == digest, name
    cfg = readj(root / 'verification_configuration.json')
    status = readj(root / 'verification_status.json')
    launcher = readj(root / 'launcher_status.json')
    assert cfg['mesh_cells'] == 200736 and cfg['dt_s'] == .00025
    assert cfg['fresh_start'] and cfg['initial_step'] == 0
    assert status['status'] == 'paused' and status['last_recorded_step'] == 20000
    assert status['last_recorded_time_s'] == 5.0 and status['stop_signal'] is None
    assert launcher['ready_for_5s_review'] and not launcher['missing_files']
    assert not launcher['metadata_errors']
    for filename, key in [('axisymmetric_70_study_50s.py', 'source_sha256'),
                          ('study_checkpoint.py', 'checkpoint_helper_sha256')]:
        assert sha(root / filename) == cfg[key]
    for folder in (root / 'reference_fields').iterdir():
        if folder.is_dir():
            rcfg = readj(folder / 'verification_configuration.json')
            assert sha(folder / 'axisymmetric_70_study_50s.py') == rcfg['source_sha256']
            assert sha(folder / 'study_checkpoint.py') == rcfg['checkpoint_helper_sha256']
    with (root / 'verification_budget.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    col = lambda key: np.array([float(row[key]) for row in rows])
    steps, times, dt = col('step'), col('time_s'), col('dt_s')
    assert np.array_equal(steps, np.arange(1, 20001))
    assert np.all(dt == cfg['dt_s']) and np.max(abs(times - steps * dt)) < 1e-12
    for key in ['fresh_stored_m3', 'fresh_in_cumulative_m3', 'fresh_out_cumulative_m3',
                'fresh_min', 'fresh_max', 'budget_residual_relative', 'boundary_net_m3_s']:
        assert np.all(np.isfinite(col(key))), key
    cin, cout = col('fresh_in_cumulative_m3'), col('fresh_out_cumulative_m3')
    assert np.allclose(np.cumsum(dt * col('fresh_in_rate_m3_s')), cin, rtol=1e-13, atol=1e-28)
    assert np.allclose(np.cumsum(dt * col('fresh_out_rate_m3_s')), cout, rtol=1e-13, atol=1e-28)
    residual = col('fresh_stored_m3') - (cin - cout)
    assert np.array_equal(residual, col('budget_residual_m3'))
    assert np.allclose(residual / np.maximum(cin, 1e-30), col('budget_residual_relative'), rtol=1e-14, atol=1e-28)
    with np.load(root / 'checkpoint_latest.npz', allow_pickle=False) as cp, \
         np.load(root / 'verification_paused.npz', allow_pickle=False) as paused, \
         np.load(root / 'field_step_000020000.npz', allow_pickle=False) as field:
        meta = json.loads(str(cp['metadata_json']))
        assert meta['identity'] == cfg['checkpoint_identity']
        assert meta['step'] == 20000 and meta['time_s'] == 5.0 and meta['dt_s'] == .00025
        for key in ('velocity_dofs', 'pressure_dofs', 'fresh_dofs'):
            assert np.all(np.isfinite(cp[key]))
        assert np.array_equal(cp['fresh_dofs'][paused['cell_dofs']], paused['fresh'])
        assert np.array_equal(cp['velocity_dofs'], paused['velocity_new_dofs'])
        assert np.array_equal(cp['pressure_dofs'], paused['pressure_dofs'])
        assert np.array_equal(field['fresh'], paused['fresh'])
        layout = dict(geometry_s_z=paused['geometry_s_z'], topology=paused['topology'],
                      cell_dofs=paused['cell_dofs'], velocity_coordinates=paused['velocity_dof_coordinates_s_z'],
                      velocity_ids=paused['velocity_component_dofs'], pressure_coordinates=paused['pressure_dof_coordinates_s_z'])
        layout_hash = fingerprint(layout)
        assert layout_hash == meta['identity']['mesh_and_dof_sha256']
        vol, fresh = field['cell_volumes_m3'], field['fresh']
        inventory = float(vol @ fresh)
        assert abs(inventory - float(rows[-1]['fresh_stored_m3'])) < 1e-19
        for key in ('fresh_in_cumulative_m3', 'fresh_out_cumulative_m3', 'fresh_stored_m3'):
            assert float(rows[-1][key]) == meta['last_row'][key] == status['last_budget_row'][key]
        raw, balanced = paused['facet_flux_raw_m3_s'], paused['facet_flux_balanced_m3_s']
        owner, neighbor = paused['facet_owner'], paused['facet_neighbor']
        interior, boundary = neighbor >= 0, neighbor < 0
        assert np.array_equal(raw[boundary], balanced[boundary])
        net = np.bincount(owner, weights=balanced, minlength=len(vol))
        net -= np.bincount(neighbor[interior], weights=balanced[interior], minlength=len(vol))
        assert np.allclose(net, paused['cell_net_balanced_m3_s'], rtol=0, atol=1e-20)
        boundary_sum = float(np.sum(balanced[boundary]))
        assert abs(boundary_sum) < 1e-18
    assert readj(root / 'restart_check.json')['status'] == 'passed'
    assert readj(root / 'underflow_check.json')['status'] == 'passed'
    return dict(manifest_files_verified=len(manifest), reference_hashes_verified=len(references),
                case=cfg['case'], cells=cfg['mesh_cells'], dt_s=cfg['dt_s'], fresh_start=True,
                accepted_step=20000, accepted_time_s=5., configured_target_time_s=cfg['target_time_s'],
                status='intentional 5 s review pause', budget_rows=len(rows), contiguous_history=True,
                checkpoint_matches_paused_fields_bitwise=True, mesh_and_dof_sha256=layout_hash,
                source_sha256=cfg['source_sha256'], checkpoint_helper_sha256=cfg['checkpoint_helper_sha256'],
                wall_time_hours=status['wall_elapsed_s']/3600,
                maximum_absolute_cumulative_budget_relative=float(np.max(abs(col('budget_residual_relative')))),
                final_cumulative_budget_residual_m3=float(residual[-1]),
                minimum_freshwater_fraction=float(np.min(col('fresh_min'))),
                maximum_freshwater_fraction=float(np.max(col('fresh_max'))),
                final_freshwater_inventory_m3=inventory, final_input_m3=float(cin[-1]), final_output_m3=float(cout[-1]),
                maximum_absolute_boundary_net_m3_s=float(np.max(abs(col('boundary_net_m3_s')))),
                final_boundary_flux_unchanged=True, final_recomputed_boundary_net_m3_s=boundary_sum,
                final_recomputed_maximum_cell_net_m3_s=float(np.max(abs(net))),
                maximum_constant_state_relative_residual=float(np.max(abs(col('constant_state_relative_residual')))),
                final_flux_correction_relative_L2=float(col('flux_correction_relative_L2')[-1]),
                maximum_flux_correction_relative_L2=float(np.max(col('flux_correction_relative_L2'))),
                checkpoint_smoke_check='passed: four coarse steps, full versus 2+2 restart',
                underflow_guard_check='passed',
                completeness_note='The 50 s target is not complete. All required files for the 5 s review are present.')


def same_mesh(mesh, folder):
    cfg = readj(folder / 'verification_configuration.json')
    with np.load(folder / 'field_step_000000000.npz') as data:
        for key, ref in [('geometry_s_z', 'xy'), ('topology', 'cells'), ('cell_volumes_m3', 'vol')]:
            assert np.array_equal(data[key], mesh[ref])
        assert np.all(data['fresh'] == 0)
    return dict(mesh, folder=folder, config=cfg)


def direct(mesh, first, second):
    v = mesh['vol']; delta = second-first
    error = float(np.sqrt(v @ (delta*delta)))
    norm = float(np.sqrt(v @ (first*first)))
    independent = math.sqrt(math.fsum(float(w)*float(d)*float(d) for w, d in zip(v, delta)))
    assert math.isclose(error, independent, rel_tol=2e-14)
    return dict(field_relative_L2_percent=100*error/norm, field_absolute_L2_m1p5=error,
                reference_field_norm_m1p5=norm, independent_fsum_L2_m1p5=independent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--return-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent)
    args = parser.parse_args(); root = args.return_dir.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    integrity = audit(root)
    (args.out / 'integrity.json').write_text(json.dumps(integrity, indent=2)+'\n')
    print('Integrity passed:', integrity['accepted_time_s'], 'seconds;', integrity['budget_rows'], 'rows', flush=True)
    names = dict(coarse_full='axisymmetric50_coarse_RETURN_18900751_0', fine_full='axisymmetric50_fine_RETURN_18900751_1',
                 fine_half='axisymmetric50_fine_half_RETURN_18900751_2', fine_quarter='axisymmetric5_fine_quarter_RETURN_18946228',
                 extra_full='axisymmetric50_extra_fine_RETURN_18946227', extra_half='axisymmetric50_extra_fine_half_RETURN_18954282')
    folders = {key: root/'reference_fields'/name for key, name in names.items()}; folders['extra_quarter'] = root
    levels = {key: ('coarse_full' if key.startswith('coarse') else 'fine_full' if key.startswith('fine') else 'extra_full') for key in folders}
    meshes = {key: g.build_mesh(folders[key]) for key in ('coarse_full','fine_full','extra_full')}
    for key in folders:
        if key not in meshes: meshes[key] = same_mesh(meshes[levels[key]], folders[key])
    physical_keys = ['H_m','R_m','Q_cc_min','Q_m3_s','nozzle_diameter_m','rho_fresh_kg_m3','rho_salt_kg_m3',
                     'nu_m2_s','g_m_s2','ramp_time_s','scalar','scalar_correction','scalar_diffusion','flux_projection']
    for key in physical_keys:
        assert all(m['config'][key] == meshes['extra_quarter']['config'][key] for m in meshes.values()), key
    overlaps = {}; checks = {}
    for a, b in [('coarse_full','fine_full'),('fine_full','extra_full')]:
        print('Computing exact cell overlaps:', a, b, flush=True)
        mat, pairs, check = g.overlap(meshes[a], meshes[b]); overlaps[(a,b)] = (mat,pairs); checks[a+'__'+b] = check
    grids = {'coarse_slabs': np.array(meshes['coarse_full']['config']['vertical_nodes_m']),
             'fine_slabs': np.array(meshes['fine_full']['config']['vertical_nodes_m'])}
    weights = {}; weight_checks = {}; upper_weights = {}; saved = {'times_s': np.array([2.,5.])}
    for tag, edges in grids.items():
        print('Computing full-radius profile weights:', tag, flush=True)
        saved[tag+'_edges_m'] = edges
        for key in ('coarse_full','fine_full','extra_full'):
            mat, vol, check = g.slab_weights(meshes[key], edges)
            weights[(tag,key)] = (mat,vol); weight_checks[tag+'__'+key] = check
        for key in meshes: saved[tag+'__'+key] = []
    for key in ('coarse_full','fine_full','extra_full'):
        mat, _, check = g.slab_weights(meshes[key], np.array([0.,.28,.3]))
        upper_weights[key] = mat.getrow(1); weight_checks['upper20mm__'+key] = check
    report = dict(method='Raw freshwater DG0 cell averages. Physical volume dV=2*pi*ds*dz with s=r^2/2. Same-mesh temporal L2 is direct; spatial L2 is reported both on exact cell overlaps and after conservative coarse-cell projection. Relative norms use the first/coarser result. Profile L2 is height weighted on identical full-radius horizontal slabs. No clipping or amplitude rescaling.',
                  upper_region_definition='Supplementary diagnostic: full-radius freshwater volume in 0.28 <= z <= 0.30 m, divided by total freshwater volume for the fraction. Exact cell intersections. Not an interface-height measurement.',
                  software=dict(numpy=np.__version__,scipy=scipy.__version__,shapely=shapely.__version__),
                  physical_parameters={key: meshes['extra_quarter']['config'][key] for key in physical_keys},
                  integrity=integrity, overlap_checks=checks, profile_weight_checks=weight_checks,
                  cases={key:dict(folder=folders[key].name,cells=len(m['vol']),dt_s=m['config']['dt_s'],source_sha256=m['config']['source_sha256']) for key,m in meshes.items()}, times={})
    temporal_pairs = [('fine_full','fine_half'),('fine_half','fine_quarter'),('extra_full','extra_half'),('extra_half','extra_quarter')]
    spatial_pairs = [('coarse_full','fine_full'),('fine_full','extra_full'),('fine_half','extra_half'),('fine_quarter','extra_quarter')]
    csv_rows = []; integral_rows = []
    for time in [2.,5.]:
        fields = {}; sources = {}; summaries = {}; profiles = {}
        for key, mesh in meshes.items():
            fields[key], source = g.get_field(mesh,time)
            source['path'] = str((folders[key]/Path(source['path']).name).relative_to(root)); sources[key] = source
            summaries[key] = g.field_summary(mesh,fields[key])
            upper = float(np.asarray(upper_weights[levels[key]] @ fields[key]).item())
            summaries[key]['upper20mm_inventory_m3'] = upper
            summaries[key]['upper20mm_inventory_fraction'] = upper/summaries[key]['inventory_m3']
            integral_rows.append(dict(time_s=time,case=key,**summaries[key]))
        for tag, edges in grids.items():
            profiles[tag] = {}
            for key in meshes:
                mat,vol = weights[(tag,levels[key])]
                profiles[tag][key] = np.asarray(mat @ fields[key]).ravel()/vol
                saved[tag+'__'+key].append(profiles[tag][key])
        out = dict(fields=summaries,sources=sources,temporal={},spatial={})
        for scope, pairs in [('temporal',temporal_pairs),('spatial',spatial_pairs)]:
            for a,b in pairs:
                if scope == 'temporal':
                    result = direct(meshes[a],fields[a],fields[b])
                else:
                    mat,triples = overlaps[(levels[a],levels[b])]
                    result = g.compare_fields(meshes[a],meshes[b],fields[a],fields[b],mat,triples,
                                              summaries['extra_quarter']['volume_weighted_L2_norm_m1p5'])
                result['profiles'] = {tag:g.compare_profiles(profiles[tag][a],profiles[tag][b],profiles[tag]['extra_quarter'],np.diff(edges)) for tag,edges in grids.items()}
                result['centroid_change_mm'] = 1000*(summaries[b]['fresh_z_centroid_m']-summaries[a]['fresh_z_centroid_m'])
                result['inventory_change_percent'] = 100*(summaries[b]['inventory_m3']/summaries[a]['inventory_m3']-1)
                result['upper20mm_fraction_change_percentage_points'] = 100*(summaries[b]['upper20mm_inventory_fraction']-summaries[a]['upper20mm_inventory_fraction'])
                out[scope][a+'__'+b] = result
                csv_rows.append(dict(time_s=time,comparison_type=scope,first_case=a,second_case=b,
                    field_direct_or_exact_L2_percent=result.get('field_relative_L2_percent',result.get('exact_common_refinement_L2_relative_to_low_pct')),
                    field_projected_L2_percent=result.get('projected_L2_relative_to_low_pct',''),
                    profile_51_slabs_L2_percent=result['profiles']['coarse_slabs']['L2_relative_to_low_pct'],
                    profile_102_slabs_L2_percent=result['profiles']['fine_slabs']['L2_relative_to_low_pct'],
                    inventory_change_percent=result['inventory_change_percent'],centroid_change_mm=result['centroid_change_mm'],
                    upper20mm_fraction_change_percentage_points=result['upper20mm_fraction_change_percentage_points']))
        out['temporal_successive_absolute_difference_ratios'] = {}
        for level in ['fine','extra']:
            first = out['temporal'][level+'_full__'+level+'_half']; second = out['temporal'][level+'_half__'+level+'_quarter']
            out['temporal_successive_absolute_difference_ratios'][level] = dict(
                field_L2=second['field_absolute_L2_m1p5']/first['field_absolute_L2_m1p5'],
                profile_51_slabs_L2=second['profiles']['coarse_slabs']['L2_absolute_m0p5']/first['profiles']['coarse_slabs']['L2_absolute_m0p5'],
                profile_102_slabs_L2=second['profiles']['fine_slabs']['L2_absolute_m0p5']/first['profiles']['fine_slabs']['L2_absolute_m0p5'])
        report['times'][str(time)] = out
        print('Time',time,'extra-fine temporal',out['temporal']['extra_half__extra_quarter']['field_relative_L2_percent'],
              'quarter-step mesh exact',out['spatial']['fine_quarter__extra_quarter']['exact_common_refinement_L2_relative_to_low_pct'],flush=True)
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.out/'profiles.npz',**{key:np.asarray(value) for key,value in saved.items()})
    for filename, rows in [('comparisons.csv',csv_rows),('integrals.csv',integral_rows)]:
        with (args.out/filename).open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Saved audited comparison:',args.out.resolve(),flush=True)


if __name__ == '__main__':
    main()
