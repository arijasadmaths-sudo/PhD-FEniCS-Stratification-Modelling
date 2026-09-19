#!/usr/bin/env python3
"""Plot saved matched-time metrics and exact horizontal-slab averages."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
metrics = json.loads((ROOT/'axisymmetric_half_matrix_metrics.json').read_text())
profiles = np.load(ROOT/'axisymmetric_half_matrix_profiles.npz')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 180})
colors = ['#27679b', '#d46a32', '#31816c']
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
x = np.arange(3); width = .35
labels = ['Fine\n0.001 → 0.0005', 'Fine\n0.0005 → 0.00025', 'Extra-fine\n0.001 → 0.0005']
keys = ['fine_full__fine_half','fine_half__fine_quarter','extra_full__extra_half']
for idx, time in enumerate(['2.0', '5.0']):
    values = [metrics['times'][time]['temporal'][k]['field_relative_L2_percent'] for k in keys]
    bars = axes[0].bar(x+(idx-.5)*width, values, width, color=colors[idx], label=f'{float(time):g} s')
    axes[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
axes[0].set_xticks(x, labels); axes[0].set_ylim(0, 23)
axes[0].set_ylabel('Volume-weighted field L₂ difference (%)')
axes[0].set_title('Time-step changes on each mesh', loc='left', weight='bold')
axes[0].legend(frameon=False)
keys = ['coarse_full__fine_full','fine_full__extra_full','fine_half__extra_half']
labels = ['Coarse → fine\nΔt = 0.001 s', 'Fine → extra-fine\nΔt = 0.001 s', 'Fine → extra-fine\nΔt = 0.0005 s']
for idx, (key, title) in enumerate([('exact_common_refinement_L2_relative_to_low_pct','Exact cell overlaps'),('projected_L2_relative_to_low_pct','After coarse-cell projection')]):
    values = [metrics['times']['5.0']['spatial'][k][key] for k in keys]
    bars = axes[1].bar(x+(idx-.5)*width, values, width, color=colors[idx], label=title)
    axes[1].bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
axes[1].set_xticks(x, labels); axes[1].set_ylim(0, 43)
axes[1].set_ylabel('Volume-weighted field L₂ difference (%)')
axes[1].set_title('Mesh changes at 5 seconds', loc='left', weight='bold')
axes[1].legend(frameon=False, fontsize=9)
for ax in axes: ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
fig.suptitle('Axisymmetric verification: sensitivity remains on the extra-fine mesh', fontsize=14, weight='bold')
fig.savefig(ROOT/'field_sensitivity.png'); fig.savefig(ROOT/'field_sensitivity.pdf'); plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
edges = profiles['fine_slabs_edges_m']*1000
for i, time in enumerate([2., 5.]):
    for j, (keys, labels, title) in enumerate([
        (['extra_full','extra_half'], ['Δt = 0.001 s','Δt = 0.0005 s'], 'Extra-fine time steps'),
        (['fine_half','extra_half'], ['Fine: 50,184 cells','Extra-fine: 200,736 cells'], 'Meshes at Δt = 0.0005 s')]):
        ax=axes[i,j]
        for k,label,color in zip(keys,labels,colors):
            values=profiles['fine_slabs__'+k][i]*1e4
            ax.stairs(values, edges, orientation='horizontal', baseline=None, label=label, color=color, linewidth=1.5)
        ax.set_ylim(0, 180 if time==2 else 300)
        ax.set_xlim(left=0)
        ax.set_xlabel('Full-radius mean freshwater fraction (×10⁻⁴)')
        ax.set_ylabel('Height (mm)'); ax.grid(alpha=.18)
        ax.set_title(f'{title}, t = {time:g} s', loc='left', weight='bold')
        ax.legend(frameon=False, fontsize=9, loc='lower right')
fig.suptitle('Mean profiles on 102 identical horizontal slabs', fontsize=14, weight='bold')
fig.savefig(ROOT/'mean_profiles.png'); fig.savefig(ROOT/'mean_profiles.pdf'); plt.close(fig)
