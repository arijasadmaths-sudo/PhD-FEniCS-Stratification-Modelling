#!/usr/bin/env python3
"""Compare the axisymmetric mesh/time-step matrix using exact volume overlap.
Run with --data holding five previous RETURN dirs and --current holding new RETURN.
Dependencies: NumPy, SciPy, Shapely>=2, Matplotlib (plot script).
"""
import argparse,json
from pathlib import Path
import numpy as np
import comparison_geometry as g

def same_mesh(mesh,folder):
    cfg=json.loads((folder/'verification_configuration.json').read_text())
    with np.load(folder/'field_step_000000000.npz') as d:
        for key,ref in [('geometry_s_z','xy'),('topology','cells'),('cell_volumes_m3','vol')]:assert np.array_equal(d[key],mesh[ref]),key
    return dict(mesh,folder=folder,config=cfg)

def direct(a,b,fa,fb):
    v=a['vol'];d=fb-fa;norm=float(np.sqrt(v@(fa*fa)));e=float(np.sqrt(v@(d*d)))
    sa,sb=g.field_summary(a,fa),g.field_summary(b,fb)
    return {'field_relative_L2_percent':100*e/norm,'field_absolute_L2_m1p5':e,
      'reference_field_norm_m1p5':norm,'centroid_change_mm':1000*(sb['fresh_z_centroid_m']-sa['fresh_z_centroid_m']),
      'inventory_change_percent':100*(sb['inventory_m3']/sa['inventory_m3']-1)}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--current',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path(__file__).parent);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    names={'coarse_full':'axisymmetric50_coarse_RETURN_18900751_0','fine_full':'axisymmetric50_fine_RETURN_18900751_1','fine_half':'axisymmetric50_fine_half_RETURN_18900751_2','fine_quarter':'axisymmetric5_fine_quarter_RETURN_18946228','extra_full':'axisymmetric50_extra_fine_RETURN_18946227'}
    meshes={k:g.build_mesh(args.data/names[k]) for k in ['coarse_full','fine_full','extra_full']}
    for k in ['fine_half','fine_quarter']:meshes[k]=same_mesh(meshes['fine_full'],args.data/names[k])
    meshes['extra_half']=same_mesh(meshes['extra_full'],args.current)
    levels={'coarse_full':'coarse_full','fine_full':'fine_full','fine_half':'fine_full','fine_quarter':'fine_full','extra_full':'extra_full','extra_half':'extra_full'}
    keys=['H_m','R_m','Q_cc_min','Q_m3_s','nozzle_diameter_m','rho_fresh_kg_m3','rho_salt_kg_m3','nu_m2_s','g_m_s2','ramp_time_s']
    for key in keys:assert all(m['config'][key]==meshes['fine_full']['config'][key] for m in meshes.values()),key
    checks={};overlaps={}
    for lo,hi in [('coarse_full','fine_full'),('fine_full','extra_full')]:
        print('Exact overlap:',lo,hi,flush=True)
        mat,pairs,checks[lo+'__'+hi]=g.overlap(meshes[lo],meshes[hi]);overlaps[(lo,hi)]=(mat,pairs)
    grids={'coarse_slabs':np.asarray(meshes['coarse_full']['config']['vertical_nodes_m']),'fine_slabs':np.asarray(meshes['fine_full']['config']['vertical_nodes_m'])}
    weights={};weight_checks={};saved={'times_s':np.array([2.,5.])}
    for tag,edges in grids.items():
        saved[tag+'_edges_m']=edges
        for k in ['coarse_full','fine_full','extra_full']:
            mat,vol,check=g.slab_weights(meshes[k],edges);weights[(tag,k)]=(mat,vol);weight_checks[tag+'__'+k]=check
        for k in meshes:saved[tag+'__'+k]=[]
    report={'method':'Exact nonnested triangle overlaps in (s=r²/2,z), physical dV=2π ds dz. Raw freshwater fractions; no clipping/rescaling. Same-mesh temporal L2 direct, spatial L2 both projected to coarser cells and exact common refinement. Relative denominators are first/low solution norms. Profiles exact full-radius slab means with height weights; both original coarse and finer slabs retained.',
        'overlap_checks':checks,'profile_weight_checks':weight_checks,'times':{},'meshes':{k:{'cells':len(m['vol']),'dt_s':m['config']['dt_s'],'source_sha256':m['config']['source_sha256'],'folder':str(m['folder'])} for k,m in meshes.items()}}
    for t in [2.,5.]:
        fields={};source={};summaries={};profiles={}
        for k,m in meshes.items():
            fields[k],source[k]=g.get_field(m,t);summaries[k]=g.field_summary(m,fields[k])
        for tag,edges in grids.items():
            profiles[tag]={}
            for k in meshes:
                mat,vol=weights[(tag,levels[k])];profiles[tag][k]=np.asarray(mat@fields[k]).ravel()/vol;saved[tag+'__'+k].append(profiles[tag][k])
        out={'fields':summaries,'sources':source,'temporal':{},'spatial':{}}
        for a,b in [('fine_full','fine_half'),('fine_half','fine_quarter'),('extra_full','extra_half')]:
            result=direct(meshes[a],meshes[b],fields[a],fields[b]);result['profiles']={tag:g.compare_profiles(profiles[tag][a],profiles[tag][b],profiles[tag][b],np.diff(edges)) for tag,edges in grids.items()};out['temporal'][a+'__'+b]=result
        for a,b in [('coarse_full','fine_full'),('fine_full','extra_full'),('fine_half','extra_half')]:
            mat,pairs=overlaps[(levels[a],levels[b])]
            result=g.compare_fields(meshes[a],meshes[b],fields[a],fields[b],mat,pairs,summaries['extra_half']['volume_weighted_L2_norm_m1p5'])
            result['profiles']={tag:g.compare_profiles(profiles[tag][a],profiles[tag][b],profiles[tag]['extra_half'],np.diff(edges)) for tag,edges in grids.items()}
            result['centroid_change_mm']=1000*(summaries[b]['fresh_z_centroid_m']-summaries[a]['fresh_z_centroid_m']);out['spatial'][a+'__'+b]=result
        for scope in ['temporal','spatial']:
            print('Time',t,scope,{k:{'field%':v.get('field_relative_L2_percent',v.get('projected_L2_relative_to_low_pct')),'profile%':v['profiles']['coarse_slabs']['L2_relative_to_low_pct'],'fine_profile%':v['profiles']['fine_slabs']['L2_relative_to_low_pct']} for k,v in out[scope].items()},flush=True)
        report['times'][str(t)]=out
    (args.out/'axisymmetric_half_matrix_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.out/'axisymmetric_half_matrix_profiles.npz',**{k:np.asarray(v) for k,v in saved.items()})
if __name__=='__main__':main()
