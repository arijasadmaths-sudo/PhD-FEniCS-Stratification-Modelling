#!/usr/bin/env python3
"""Compare raw axisymmetric DG0 fields by exact polygon intersections.

Dependencies: Python 3, NumPy, SciPy, Shapely >=2.
Run: python compare_spatial.py --data ../data --out .
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import scipy.sparse as sp
import shapely
from shapely import STRtree


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_float(x):
    return float(np.asarray(x))


def build_mesh(folder):
    conf = json.loads((folder / 'verification_configuration.json').read_text())
    with np.load(folder / 'field_step_000000000.npz') as d:
        xy = d['geometry_s_z'].copy()
        cells = d['topology'].copy()
        vol = d['cell_volumes_m3'].copy()
    tri = xy[cells]
    poly = shapely.polygons(tri)
    geom_vol = 2*np.pi*shapely.area(poly)
    vol_error = np.max(np.abs(geom_vol-vol)/vol)
    assert vol_error < 1e-10, vol_error
    assert np.all(shapely.is_valid(poly))
    return dict(folder=folder, config=conf, xy=xy, cells=cells, tri=tri,
                poly=poly, vol=vol, zcent=tri[:,:,1].mean(axis=1),
                geometry_volume_max_relative_error=float(vol_error))


def overlap(low, high):
    """M[i,j] is the physical volume shared by low cell i and high cell j."""
    tree = STRtree(low['poly'])
    # Query returns input (high) indices first, tree (low) indices second.
    jj, ii = tree.query(high['poly'], predicate='intersects')
    vals = 2*np.pi*shapely.area(shapely.intersection(low['poly'][ii], high['poly'][jj]))
    keep = vals > 0
    ii, jj, vals = ii[keep], jj[keep], vals[keep]
    matrix = sp.csr_matrix((vals, (ii,jj)), shape=(len(low['vol']),len(high['vol'])))
    row = np.asarray(matrix.sum(axis=1)).ravel()
    col = np.asarray(matrix.sum(axis=0)).ravel()
    error_low = float(np.max(np.abs(row-low['vol'])/low['vol']))
    error_high = float(np.max(np.abs(col-high['vol'])/high['vol']))
    assert max(error_low,error_high) < 1e-10, (error_low,error_high)
    diag = dict(nonzero_overlaps=matrix.nnz,
                low_cell_volume_max_relative_closure_error=error_low,
                high_cell_volume_max_relative_closure_error=error_high,
                total_overlap_volume_m3=float(matrix.sum()))
    return matrix, (ii,jj,vals), diag


def slab_weights(mesh, edges):
    smax = float(mesh['xy'][:,0].max())
    slabs = shapely.box(np.zeros(len(edges)-1), edges[:-1],
                        np.full(len(edges)-1,smax), edges[1:])
    tree = STRtree(slabs)
    jj, ii = tree.query(mesh['poly'], predicate='intersects')
    vals = 2*np.pi*shapely.area(shapely.intersection(slabs[ii],mesh['poly'][jj]))
    keep = vals > 0
    mat = sp.csr_matrix((vals[keep],(ii[keep],jj[keep])),
                        shape=(len(edges)-1,len(mesh['vol'])))
    expected = 2*np.pi*smax*np.diff(edges)
    row = np.asarray(mat.sum(axis=1)).ravel()
    col = np.asarray(mat.sum(axis=0)).ravel()
    err_slab = float(np.max(np.abs(row-expected)/expected))
    err_cell = float(np.max(np.abs(col-mesh['vol'])/mesh['vol']))
    assert max(err_slab,err_cell) < 1e-10, (err_slab,err_cell)
    return mat, expected, dict(slab_max_relative_closure_error=err_slab,
                               cell_max_relative_closure_error=err_cell)


def get_field(mesh, time):
    step = round(time/mesh['config']['dt_s'])
    p = mesh['folder']/f'field_step_{step:09d}.npz'
    with np.load(p) as d:
        assert float(d['time_s']) == time
        assert np.array_equal(d['geometry_s_z'],mesh['xy'])
        assert np.array_equal(d['topology'],mesh['cells'])
        assert np.array_equal(d['cell_volumes_m3'],mesh['vol'])
        fresh = d['fresh'].copy()
    assert np.all(np.isfinite(fresh))
    return fresh, dict(path=str(p.resolve()), sha256=sha256(p),
                      step=step, time_s=time)


def field_summary(mesh, fresh):
    v = mesh['vol']
    inventory = float(v@fresh)
    return dict(inventory_m3=inventory,
                fresh_z_centroid_m=float((v*fresh)@mesh['zcent']/inventory),
                fresh_min=float(fresh.min()),fresh_max=float(fresh.max()),
                volume_weighted_L2_norm_m1p5=float(np.sqrt(v@(fresh*fresh))))


def compare_fields(low, high, fl, fh, weights, pairs, ref_norm):
    ii,jj,vol = pairs
    projected = np.asarray(weights@fh).ravel()/low['vol']
    d_projected = projected-fl
    d_exact = fh[jj]-fl[ii]
    norm_low = np.sqrt(low['vol']@(fl*fl))
    l2p = float(np.sqrt(low['vol']@(d_projected*d_projected)))
    l2e = float(np.sqrt(vol@(d_exact*d_exact)))
    ip = float(low['vol']@projected)
    ih = float(high['vol']@fh)
    il = float(low['vol']@fl)
    return dict(
        low_reference_L2_norm_m1p5=float(norm_low),
        common_extra_fine_reference_L2_norm_m1p5=ref_norm,
        projected_L2_absolute_m1p5=l2p,
        projected_L2_relative_to_low_pct=100*l2p/norm_low,
        projected_L2_relative_to_common_extra_fine_pct=100*l2p/ref_norm,
        exact_common_refinement_L2_absolute_m1p5=l2e,
        exact_common_refinement_L2_relative_to_low_pct=100*l2e/norm_low,
        exact_common_refinement_L2_relative_to_common_extra_fine_pct=100*l2e/ref_norm,
        projected_L1_absolute_m3=float(low['vol']@np.abs(d_projected)),
        exact_common_refinement_L1_absolute_m3=float(vol@np.abs(d_exact)),
        projection_inventory_relative_error=float((ip-ih)/ih),
        high_minus_low_inventory_m3=ih-il,
        high_minus_low_inventory_relative_to_low_pct=100*(ih-il)/il)


def compare_profiles(pl,ph,ref,dz):
    l2 = float(np.sqrt(dz@((ph-pl)**2)))
    norm = float(np.sqrt(dz@(pl**2)))
    ref_norm = float(np.sqrt(dz@(ref**2)))
    return dict(L2_absolute_m0p5=l2,
                L2_relative_to_low_pct=100*l2/norm,
                L2_relative_to_common_extra_fine_pct=100*l2/ref_norm,
                low_reference_L2_norm_m0p5=norm,
                common_extra_fine_reference_L2_norm_m0p5=ref_norm,
                L1_absolute_m=float(dz@np.abs(ph-pl)),
                max_absolute_difference=float(np.max(np.abs(ph-pl))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    names=dict(coarse='axisymmetric50_coarse_RETURN_18900751_0',
               fine='axisymmetric50_fine_RETURN_18900751_1',
               extra_fine='axisymmetric50_extra_fine_RETURN_18946227')
    meshes={k:build_mesh(args.data/v) for k,v in names.items()}
    for mesh in meshes.values():
        assert mesh['config']['dt_s']==0.001
    pair_names=[('coarse','fine'),('fine','extra_fine'),('coarse','extra_fine')]
    overlap_data={}
    metrics=dict(method='Exact intersections of piecewise-constant triangle fields in s=r^2/2,z; dV=2*pi*ds*dz. No clipping or rescaling.',
                 software=dict(numpy=np.__version__,shapely=shapely.__version__),
                 meshes={},overlap_checks={},profile_weights={},times={})
    for k,m in meshes.items():
        metrics['meshes'][k]=dict(cells=len(m['vol']),vertices=len(m['xy']),
              dt_s=m['config']['dt_s'],domain_volume_m3=float(m['vol'].sum()),
              geometry_volume_max_relative_error=m['geometry_volume_max_relative_error'],
              source_sha256=m['config']['source_sha256'])
    for a,b in pair_names:
        print('Computing overlap',a,b,flush=True)
        mat,triples,checks=overlap(meshes[a],meshes[b])
        overlap_data[(a,b)]=(mat,triples)
        metrics['overlap_checks'][a+'_to_'+b]=checks
    # Primary profile grid is the original coarse BASE vertical partition,
    # excluding barycentric vertex heights. All comparisons use identical slabs.
    primary_edges=np.array(meshes['coarse']['config']['vertical_nodes_m'])
    secondary_edges=np.array(meshes['fine']['config']['vertical_nodes_m'])
    profile_grids=dict(original_coarse_slabs=primary_edges, fine_slabs=secondary_edges)
    saved_profiles=dict(times_s=np.array([2.,5.]))
    profile_maps={}
    for grid,edges in profile_grids.items():
        saved_profiles[grid+'_edges_m']=edges
        metrics['profile_weights'][grid]={}
        for k,m in meshes.items():
            mat,vol,checks=slab_weights(m,edges)
            profile_maps[(grid,k)]=(mat,vol)
            metrics['profile_weights'][grid][k]=checks
            saved_profiles[grid+'_'+k+'_fresh_mean']=[]
    for t in (2.,5.):
        fresh,src={},{}
        summaries={}
        for k,m in meshes.items():
            fresh[k],src[k]=get_field(m,t)
            summaries[k]=field_summary(m,fresh[k])
        profiles={}
        for grid in profile_grids:
            profiles[grid]={}
            for k,m in meshes.items():
                mat,vol=profile_maps[(grid,k)]
                profile=np.asarray(mat@fresh[k]).ravel()/vol
                profiles[grid][k]=profile
                saved_profiles[grid+'_'+k+'_fresh_mean'].append(profile)
        ref_norm=summaries['extra_fine']['volume_weighted_L2_norm_m1p5']
        time_metrics=dict(sources=src,fields=summaries,comparisons={})
        for a,b in pair_names:
            mat,triples=overlap_data[(a,b)]
            result=compare_fields(meshes[a],meshes[b],fresh[a],fresh[b],mat,triples,ref_norm)
            result['high_minus_low_centroid_z_m']=summaries[b]['fresh_z_centroid_m']-summaries[a]['fresh_z_centroid_m']
            result['profiles']={}
            for grid,edges in profile_grids.items():
                result['profiles'][grid]=compare_profiles(profiles[grid][a],profiles[grid][b],profiles[grid]['extra_fine'],np.diff(edges))
            time_metrics['comparisons'][a+'_to_'+b]=result
        c=time_metrics['comparisons']
        # Absolute differences and common denominators establish shrinkage
        # without changing each pair's normalization.
        time_metrics['successive_difference_ratio_fine_extra_over_coarse_fine']=dict(
           exact_common_refinement_L2=c['fine_to_extra_fine']['exact_common_refinement_L2_absolute_m1p5']/c['coarse_to_fine']['exact_common_refinement_L2_absolute_m1p5'],
           projected_L2=c['fine_to_extra_fine']['projected_L2_absolute_m1p5']/c['coarse_to_fine']['projected_L2_absolute_m1p5'],
           original_coarse_slab_profile_L2=c['fine_to_extra_fine']['profiles']['original_coarse_slabs']['L2_absolute_m0p5']/c['coarse_to_fine']['profiles']['original_coarse_slabs']['L2_absolute_m0p5'])
        metrics['times'][str(t)]=time_metrics
        print('Time',t,json.dumps(time_metrics['successive_difference_ratio_fine_extra_over_coarse_fine']),flush=True)
    for k,v in saved_profiles.items():
        saved_profiles[k]=np.asarray(v)
    (args.out/'spatial_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    np.savez_compressed(args.out/'spatial_profiles.npz',**saved_profiles)
    print('Wrote',args.out/'spatial_metrics.json',flush=True)


if __name__=='__main__':
    main()
