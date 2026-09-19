#!/usr/bin/env python3
"""Plot the saved audited metrics; no simulation or data normalization."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent,
                    help="Folder containing the saved analysis metrics and profiles")
root = parser.parse_args().directory.resolve()
report = json.loads((root/'metrics.json').read_text())
profiles = np.load(root/'profiles.npz')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':200,'svg.fonttype':'none'})
blue, orange, green = '#24648c','#ba5b27','#2a7c66'
fig, axes = plt.subplots(1,2,figsize=(12.0,4.7),layout='constrained')
x=np.arange(2);width=.34
for i,t in enumerate(['2.0','5.0']):
    keys=['extra_full__extra_half','extra_half__extra_quarter']
    values=[report['times'][t]['temporal'][key]['field_relative_L2_percent'] for key in keys]
    bars=axes[0].bar(x+(i-.5)*width,values,width,color=[blue,orange][i],label=f't = {float(t):g} s')
    axes[0].bar_label(bars,fmt='%.2f',padding=4)
axes[0].set_xticks(x,['0.001 → 0.0005 s','0.0005 → 0.00025 s'])
axes[0].set_xlabel('Successive time-step changes')
axes[0].set_ylabel('Direct volume-weighted L₂ difference (%)')
axes[0].set_ylim(0,23)
axes[0].set_title('Extra-fine mesh: temporal sensitivity',loc='left',pad=12)
axes[0].legend(frameon=False)
x=np.arange(3)
keys=['fine_full__extra_full','fine_half__extra_half','fine_quarter__extra_quarter']
for i,(metric,label,color) in enumerate([
    ('exact_common_refinement_L2_relative_to_low_pct','Exact cell intersections',green),
    ('projected_L2_relative_to_low_pct','Conservative coarse-cell projection','#9bc7b6')]):
    values=[report['times']['5.0']['spatial'][key][metric] for key in keys]
    bars=axes[1].bar(x+(i-.5)*width,values,width,color=color,label=label)
    axes[1].bar_label(bars,fmt='%.2f',padding=4,fontsize=9)
axes[1].set_xticks(x,['0.001 s','0.0005 s','0.00025 s'])
axes[1].set_xlabel('Time step shared by both meshes')
axes[1].set_ylabel('Volume-weighted L₂ difference (%)')
axes[1].set_ylim(0,45)
axes[1].set_title('Fine → extra-fine mesh at 5 s',loc='left',pad=12)
axes[1].legend(frameon=False,fontsize=9,loc='upper right')
for ax in axes: ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
fig.suptitle('Axisymmetric verification: time-step changes shrink; mesh sensitivity remains',fontsize=13)
fig.savefig(root/'field_sensitivity.png');fig.savefig(root/'field_sensitivity.svg');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(10.8,8.4),layout='constrained')
edges=profiles['fine_slabs_edges_m']*1000
for i,time in enumerate([2.,5.]):
    for j,(keys,labels,title) in enumerate([
        (['extra_full','extra_half','extra_quarter'],['Δt = 0.001 s','Δt = 0.0005 s','Δt = 0.00025 s'],'Extra-fine time steps'),
        (['fine_quarter','extra_quarter'],['Fine: 50,184 cells','Extra-fine: 200,736 cells'],'Meshes at Δt = 0.00025 s')]):
        ax=axes[i,j]
        for key,label,color in zip(keys,labels,[blue,orange,green]):
            ax.stairs(profiles['fine_slabs__'+key][i]*1e4,edges,orientation='horizontal',
                      baseline=None,label=label,color=color,linewidth=1.4)
        ax.set_ylim(0,300);ax.set_xlim(left=0)
        ax.set_xlabel('Full-radius mean freshwater fraction (×10⁻⁴)')
        ax.set_ylabel('Height (mm)');ax.grid(alpha=.18)
        ax.set_title(f'{title}, t = {time:g} s',loc='left')
        ax.legend(frameon=False,fontsize=9,loc='lower right')
fig.suptitle('Raw freshwater profiles on 102 identical horizontal slabs',fontsize=13)
fig.savefig(root/'mean_profiles.png');fig.savefig(root/'mean_profiles.svg');plt.close(fig)
print('Saved field_sensitivity and mean_profiles as PNG and SVG')
