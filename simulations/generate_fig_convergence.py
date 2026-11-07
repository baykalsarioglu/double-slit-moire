"""
Figure: Convergence of detector memory to QM with increasing microstripe density.

Left column: linear readout (flat at all densities).
Right column: detector memory (fringes emerge as density increases).

Parameters:
  f(θ) = [½(1+cos θ)]²
  k values: 300π, 1000π, 3000π, 10000π, 100000π  (peaks per slit)
  80 bins, kill radius = 10 carrier periods
  5×10⁵ particles per run

Output:
  paper/fig_convergence.png
  paper/fig_convergence.pdf
"""
import numpy as np
from scipy.signal import argrelmax
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'legend.fontsize': 7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})


def run_simulation(k_base, n_bins=80, kill_r_periods=10, N_total=500_000):
    """Run screen memory + linear readout for given k_base (in units of π)."""
    dk = 10 * np.pi
    k1 = k_base * np.pi
    k2 = k1 + dk
    k_avg = (k1 + k2) / 2
    carrier_period = 2 * np.pi / k_avg

    n_per_period = 20
    n_grid = int(1.0 / carrier_period * n_per_period)
    n_grid = min(n_grid, 5_000_000)
    x = np.linspace(-0.5, 0.5, n_grid)
    dx = x[1] - x[0]

    env = np.sinc(1.8 * x)**2
    P1 = env * (0.5 * (1 + np.cos(k1 * x)))**2
    P2 = env * (0.5 * (1 + np.cos(k2 * x)))**2
    P_sum = P1 + P2

    # Find peaks
    min_dist = max(1, int(0.4 * carrier_period / dx))
    peaks_idx = argrelmax(P_sum, order=min_dist)[0]
    good = P_sum[peaks_idx] > 0.001
    peaks_idx = peaks_idx[good]

    peak_x = x[peaks_idx]
    peak_h = P_sum[peaks_idx]
    n_peaks = len(peak_x)
    if n_peaks < 10:
        return None

    kill_radius = kill_r_periods * carrier_period
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    peak_bin = np.digitize(peak_x, bin_edges) - 1

    # Precompute neighbors
    neighbors = []
    for i in range(n_peaks):
        lo = np.searchsorted(peak_x, peak_x[i] - kill_radius)
        hi = np.searchsorted(peak_x, peak_x[i] + kill_radius)
        neighbors.append([n for n in range(lo, hi) if n != i])

    # QM reference
    x_fine = np.linspace(-0.5, 0.5, 100000)
    QM_fine = 2 * np.sinc(1.8 * x_fine)**2 * np.cos(dk * x_fine / 2)**2
    qm_hist = np.zeros(n_bins)
    bi = np.digitize(x_fine, bin_edges) - 1
    for b in range(n_bins):
        m = bi == b
        if m.any():
            qm_hist[b] = QM_fine[m].mean()

    # Linear readout
    P_full_norm = P_sum / P_sum.sum()
    hits_x = np.random.choice(len(x), size=N_total, p=P_full_norm)
    mean_hist = np.histogram(x[hits_x], bins=bin_edges)[0].astype(float)

    # Screen memory
    alive = np.ones(n_peaks, dtype=bool)
    hit_count = np.zeros(n_peaks)
    prev_alive = alive.sum()
    converged_at = 0
    for i in range(n_peaks * 3):
        alive_h = peak_h * alive
        total = alive_h.sum()
        if total == 0:
            break
        probs = alive_h / total
        h = np.random.choice(n_peaks, p=probs)
        if alive[h]:
            hit_count[h] += 1
            for nb in neighbors[h]:
                alive[nb] = False
        if (i + 1) % 200 == 0:
            n_alive = alive.sum()
            if n_alive == prev_alive:
                converged_at = i + 1
                break
            prev_alive = n_alive

    if converged_at == 0:
        converged_at = min(i + 1, n_peaks * 3)

    remaining = N_total - converged_at
    alive_h = peak_h * alive
    total = alive_h.sum()
    if total > 0:
        probs = alive_h / total
        batch_hits = np.random.choice(n_peaks, size=remaining, p=probs)
        for h in batch_hits:
            hit_count[h] += 1

    memory_hist = np.zeros(n_bins)
    for i, b in enumerate(peak_bin):
        if 0 <= b < n_bins:
            memory_hist[b] += hit_count[i]

    def norm01(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn) if mx > mn else a * 0

    r_mean = np.corrcoef(norm01(mean_hist), norm01(qm_hist))[0, 1]
    r_memory = np.corrcoef(norm01(memory_hist), norm01(qm_hist))[0, 1]
    ppb = n_peaks // n_bins

    return {
        'r_mean': r_mean, 'r_memory': r_memory, 'ppb': ppb,
        'n_peaks': n_peaks,
        'memory_hist': memory_hist, 'mean_hist': mean_hist,
        'qm_hist': qm_hist, 'bin_centers': bin_centers,
    }


# ──── RUN FOR MULTIPLE k VALUES ────
k_values = [300, 1000, 3000, 10000, 100000]  # in units of π

results = []
for kv in k_values:
    np.random.seed(42)
    print(f"Running k = {kv}π ...")
    res = run_simulation(kv, n_bins=80, kill_r_periods=10, N_total=500_000)
    if res:
        res['k_val'] = kv
        results.append(res)
        print(f"  peaks/bin={res['ppb']}, r(linear)={res['r_mean']:.3f}, "
              f"r(memory)={res['r_memory']:.3f}")
    else:
        print(f"  SKIPPED (too few peaks)")

# ──── FIGURE ────
n_rows = len(results)
fig, axes = plt.subplots(n_rows, 2, figsize=(7, 2.2 * n_rows), sharex=True)

if n_rows == 1:
    axes = axes.reshape(1, -1)

c_bar_mean = '#94a3b8'
c_bar_mem = '#6366f1'
c_qm = '#dc2626'


def norm01(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn) if mx > mn else a * 0


for i, res in enumerate(results):
    bc = res['bin_centers']
    bw = (bc[1] - bc[0]) * 0.88

    mean_n = norm01(res['mean_hist'])
    mem_n = norm01(res['memory_hist'])
    qm_n = norm01(res['qm_hist'])

    # Left: linear readout
    ax = axes[i, 0]
    ax.bar(bc, mean_n, width=bw, color=c_bar_mean, alpha=0.8,
           edgecolor='#475569', linewidth=0.3)
    ax.plot(bc, qm_n, color=c_qm, lw=1.8, zorder=5)
    ax.set_ylim(-0.05, 1.25)
    ax.set_ylabel('Normalized')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    bbox = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85)
    ax.text(0.02, 0.92, f'({chr(97 + 2*i)})',
            transform=ax.transAxes, fontsize=10, fontweight='bold')
    ax.text(0.02, 0.78, 'Linear readout',
            transform=ax.transAxes, fontsize=8, color='#475569')
    ax.text(0.55, 0.92,
            f'{res["ppb"]} microstripes/bin, $r = {res["r_mean"]:.2f}$',
            transform=ax.transAxes, fontsize=8, ha='center', color='#64748b',
            bbox=bbox)

    # Right: detector memory
    ax = axes[i, 1]
    ax.bar(bc, mem_n, width=bw, color=c_bar_mem, alpha=0.7,
           edgecolor='#4338ca', linewidth=0.3)
    ax.plot(bc, qm_n, color=c_qm, lw=1.8, zorder=5)
    ax.set_ylim(-0.05, 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.02, 0.92, f'({chr(97 + 2*i + 1)})',
            transform=ax.transAxes, fontsize=10, fontweight='bold')
    ax.text(0.02, 0.78, 'Detector memory',
            transform=ax.transAxes, fontsize=8, color='#4338ca')
    ax.text(0.55, 0.92,
            f'{res["ppb"]} microstripes/bin, $r = {res["r_memory"]:.2f}$',
            transform=ax.transAxes, fontsize=8, ha='center', color='#4338ca',
            bbox=bbox)

axes[-1, 0].set_xlabel('Screen position $y$')
axes[-1, 1].set_xlabel('Screen position $y$')

plt.tight_layout()

import os
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, 'fig_convergence.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'fig_convergence.pdf'),
            bbox_inches='tight')
print(f"\nSaved fig_convergence.png and fig_convergence.pdf")

# Print values for paper
print("\n\nVALUES FOR PAPER (Table 1):")
print(f"{'Peaks/slit':>12} {'Peaks/bin':>10} {'r(memory)':>10}")
print("-" * 35)
for res in results:
    print(f"{res['n_peaks']:>12,} {res['ppb']:>10} {res['r_memory']:>10.2f}")
