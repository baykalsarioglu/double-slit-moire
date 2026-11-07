"""
Figure 1 — Three panels: Slit 1, Slit 2, Sum vs QM.

Uses (½(1+cos kx))² probability function:
  - Sharper teeth than ½(1+cos) → better visual alignment with QM
  - Same fringe count as ½(1+cos) → correct physics
  - Mathematically: cos⁴(kx/2), normalized to [0,1]

Parameters:
  k₁ = 240π, k₂ = 250π, Δk = 10π
  env = sinc²(1.8x)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

# ──── PARAMETERS ────
k1 = 240 * np.pi
k2 = 250 * np.pi
dk = k2 - k1        # 10π
k_avg = (k1+k2)/2   # 245π → k_avg/dk = 24.5 (half-integer)

x = np.linspace(-1, 1, 300000)
env = np.sinc(1.8 * x)**2

# ──── PROBABILITY: (½(1+cos))² = cos⁴(kx/2) ────
P1 = env * (0.5 * (1 + np.cos(k1 * x)))**2
P2 = env * (0.5 * (1 + np.cos(k2 * x)))**2
P_sum = P1 + P2
QM = 2 * env * np.cos(dk * x / 2)**2

print(f"k₁={k1/np.pi:.0f}π  k₂={k2/np.pi:.0f}π  Δk={dk/np.pi:.0f}π  k_avg/Δk={k_avg/dk:.1f}")
print(f"P_sum(0) = {P_sum[len(x)//2]:.4f}  QM(0) = {QM[len(x)//2]:.4f}")
for n in range(-3, 4):
    x0 = 2 * n * np.pi / dk
    mask = (x > x0-0.06) & (x < x0+0.06)
    if np.any(mask) and np.max(QM[mask]) > 0.1:
        qm_x = x[mask][np.argmax(QM[mask])]
        ps_x = x[mask][np.argmax(P_sum[mask])]
        print(f"  n={n:+d}: δ={abs(qm_x-ps_x):.5f}")

# ──── FIGURE ────
fig, axes = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True,
                          gridspec_kw={'hspace': 0.08})

c1 = '#1d4ed8'
c2 = '#b91c1c'
c_sum = '#4338ca'
c_qm = '#f59e0b'

# (a) Slit 1
ax = axes[0]
ax.fill_between(x, 0, P1, color=c1, alpha=0.9, linewidth=0, rasterized=True)
ax.plot(x, env, '#64748b', lw=0.7, ls='--', alpha=0.4)
ax.set_ylabel(r'$P_1(y)$')
ax.set_ylim(-0.03, 1.15)
ax.text(0.015, 0.85, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

# (b) Slit 2
ax = axes[1]
ax.fill_between(x, 0, P2, color=c2, alpha=0.9, linewidth=0, rasterized=True)
ax.plot(x, env, '#64748b', lw=0.7, ls='--', alpha=0.4)
ax.set_ylabel(r'$P_2(y)$')
ax.set_ylim(-0.03, 1.15)
ax.text(0.015, 0.85, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

# (c) Sum vs QM
ax = axes[2]
ax.fill_between(x, 0, P_sum, color=c_sum, alpha=0.85, linewidth=0,
                rasterized=True, label=r'$P_1 + P_2$')
ax.plot(x, QM, color=c_qm, lw=2.0, label='QM', zorder=5)
ax.set_xlabel('Screen position $y$')
ax.set_ylabel(r'$P_1 + P_2$')
ax.set_ylim(-0.1, 2.2)
ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
ax.text(0.015, 0.88, '(c)', transform=ax.transAxes, fontsize=12, fontweight='bold')

for a in axes:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

# ──── ZOOM INSETS with dashed lines ────
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
carrier_period = 2 * np.pi / k_avg
zoom_lo = -0.005
zoom_hi =  0.105

zoom_configs = [
    (axes[0], P1, c1),
    (axes[1], P2, c2),
]

for ax_main, P_data, color in zoom_configs:
    # Dashed lines marking zoom region
    ax_main.axvline(zoom_lo, color='#888', ls='--', lw=0.8, alpha=0.5)
    ax_main.axvline(zoom_hi, color='#888', ls='--', lw=0.8, alpha=0.5)

    ax_in = inset_axes(ax_main, width="25%", height="50%", loc='upper right',
                       borderpad=0.6)
    mask = (x > zoom_lo) & (x < zoom_hi)
    ax_in.fill_between(x[mask], 0, P_data[mask], color=color, alpha=0.9, linewidth=0)
    ax_in.set_xlim(zoom_lo, zoom_hi)
    ax_in.set_ylim(-0.02, 1.15)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.patch.set_alpha(0.95)
    for spine in ax_in.spines.values():
        spine.set_edgecolor('#888')
        spine.set_linewidth(1.0)
        spine.set_linestyle('--')

# Panel (c) zoom
axes[2].axvline(zoom_lo, color='#888', ls='--', lw=0.8, alpha=0.5)
axes[2].axvline(zoom_hi, color='#888', ls='--', lw=0.8, alpha=0.5)

ax_in_c = inset_axes(axes[2], width="25%", height="50%", loc='center left',
                     borderpad=1.2)
mask = (x > zoom_lo) & (x < zoom_hi)
ax_in_c.fill_between(x[mask], 0, P1[mask], color=c1, alpha=0.5, linewidth=0)
ax_in_c.fill_between(x[mask], 0, P2[mask], color=c2, alpha=0.5, linewidth=0)
ax_in_c.plot(x[mask], P_sum[mask], color='black', lw=0.7, alpha=0.8)
ax_in_c.set_xlim(zoom_lo, zoom_hi)
ax_in_c.set_ylim(-0.02, 2.2)
ax_in_c.set_xticks([])
ax_in_c.set_yticks([])
ax_in_c.patch.set_alpha(0.95)
for spine in ax_in_c.spines.values():
    spine.set_edgecolor('#888')
    spine.set_linewidth(1.0)
    spine.set_linestyle('--')
ax_in_c.annotate('overlap', xy=(0.015, 1.95), fontsize=6, color='#333',
                 ha='center', fontweight='bold')
ax_in_c.annotate('interlace', xy=(0.085, 1.05), fontsize=6, color='#333',
                 ha='center', fontweight='bold')

import os
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, 'fig1_microstripes.png'))
plt.savefig(os.path.join(out_dir, 'fig1_microstripes.pdf'))
print("Saved fig1")
