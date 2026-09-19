#!/usr/bin/env python3
"""Create the review figure from metrics.json and profiles.npz in this folder."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent,
                    help="Folder containing the saved analysis metrics and profiles")
root = parser.parse_args().directory.resolve()
metrics = json.loads((root / 'metrics.json').read_text())
rows = metrics['comparisons_at_5s']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, (ax, profile) = plt.subplots(1, 2, figsize=(11.6, 4.9),
                                gridspec_kw={'width_ratios': [1.25, 1]}, layout='constrained')
field_color, profile_color = '#146A86', '#C45C30'
y = np.arange(4)
for offset, key, color, label in [(-.14, 'field_l2_percent', field_color, 'Field'),
                                  (.14, 'profile_l2_percent', profile_color, 'Mean profile')]:
    values = [r[key] for r in rows]
    ax.scatter(values, y + offset, color=color, s=40, label=label, zorder=3)
    for i, value in enumerate(values):
        ax.annotate(f'{value:.3f}%' if i < 2 else f'{value:.2f}%',
                    (value, y[i] + offset), xytext=(7, 0), textcoords='offset points',
                    va='center', fontsize=9, color=color)
ax.set_yticks(y, ['Halve time step\n0.3125 mm mesh', 'Halve time step\n0.15625 mm mesh',
                  'Refine mesh\nΔt = 0.001 s', 'Refine mesh\nΔt = 0.0005 s'])
ax.set_xscale('log')
ax.set_xlim(.05, 60)
ax.set_xticks([.1, 1, 10], ['0.1', '1', '10'])
ax.set_ylim(3.55, -.65)
ax.set_xlabel('Relative difference (%) · logarithmic scale')
ax.set_title('(a) Time-step and mesh sensitivity', loc='left', pad=12)
ax.grid(axis='x', which='major', color='#D7DDE2', linewidth=.7)
ax.axhline(1.5, color='#D7DDE2', linewidth=.7)
ax.legend(loc='lower left', frameon=False, ncols=2, fontsize=9)

colors = {'compact312p5': '#C45C30', 'compact156p25': '#146A86'}
with np.load(root / 'profiles.npz') as data:
    for case, color in colors.items():
        for suffix, style in [('full', '-'), ('half', '--')]:
            label = case + '_' + suffix
            values, edges = data[label + '_profile'], data[label + '_z_edges_m'] * 1000
            profile.stairs(values, edges, orientation='horizontal', baseline=None,
                           color=color, linestyle=style, linewidth=1.45)
profile.set(xlim=(-.00035, .0175), ylim=(0, 40),
            xlabel='Full-width mean freshwater fraction', ylabel='Height (mm)')
profile.set_xticks([0, .005, .010, .015], ['0', '0.005', '0.010', '0.015'])
profile.set_title('(b) Raw mean profiles', loc='left', pad=12)
profile.grid(alpha=.18)
profile.legend(handles=[Line2D([0], [0], color=colors['compact312p5'], label='0.3125 mm'),
                        Line2D([0], [0], color=colors['compact156p25'], label='0.15625 mm'),
                        Line2D([0], [0], color='#555555', linestyle='-', label='Δt = 0.001 s'),
                        Line2D([0], [0], color='#555555', linestyle='--', label='Δt = 0.0005 s')],
               frameon=False, fontsize=9, loc='upper right')
fig.suptitle('Cartesian compact-grid verification at 5 seconds', fontsize=14)
fig.savefig(root / 'Cartesian_half_step_comparison.pdf')
fig.savefig(root / 'Cartesian_half_step_comparison.png', dpi=190)
plt.close(fig)
