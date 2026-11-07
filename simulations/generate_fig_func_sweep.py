"""
Paper figure: Probability function sweep for screen memory.
Shows that different f(θ) give different screen memory performance.
Paper-quality formatting.
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

np.random.seed(42)

def run_memory(k_base, kill_r_periods, prob_func_fn, n_bins=80, N_total=500_000):
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
    P1 = env * prob_func_fn(k1 * x)
    P2 = env * prob_func_fn(k2 * x)
    P_sum = P1 + P2

    min_dist = max(1, int(0.4 * carrier_period / dx))

    # Find peaks per slit separately, then merge
    # (fixes flat-top bias for rect-like functions)
    peaks_p1 = argrelmax(P1, order=min_dist)[0]
    peaks_p1 = peaks_p1[P1[peaks_p1] > 0.001]
    peaks_p2 = argrelmax(P2, order=min_dist)[0]
    peaks_p2 = peaks_p2[P2[peaks_p2] > 0.001]
    peaks_idx = np.unique(np.concatenate([peaks_p1, peaks_p2]))
    peaks_idx = peaks_idx[P_sum[peaks_idx] > 0.001]

    peak_x = x[peaks_idx]
    peak_h = P_sum[peaks_idx]
    n_peaks = len(peak_x)
    if n_peaks < 10:
        return None

    kill_radius = kill_r_periods * carrier_period
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    peak_bin = np.digitize(peak_x, bin_edges) - 1

    neighbors = []
    for i in range(n_peaks):
        lo = np.searchsorted(peak_x, peak_x[i] - kill_radius)
        hi = np.searchsorted(peak_x, peak_x[i] + kill_radius)
        neighbors.append([n for n in range(lo, hi) if n != i])

    x_fine = np.linspace(-0.5, 0.5, 100000)
    QM_fine = 2 * np.sinc(1.8 * x_fine)**2 * np.cos(dk * x_fine / 2)**2
    qm_hist = np.zeros(n_bins)
    bi = np.digitize(x_fine, bin_edges) - 1
    for b in range(n_bins):
        m = bi == b
        if m.any():
            qm_hist[b] = QM_fine[m].mean()

    # Linear
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
        if total == 0: break
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

    n_alive = alive.sum()
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
        'n_peaks': n_peaks, 'n_alive': n_alive,
        'memory_hist': memory_hist, 'mean_hist': mean_hist,
        'qm_hist': qm_hist, 'bin_centers': bin_centers
    }


# ══════════════════════════════════════════════════════
# Select functions for the paper figure
# ══════════════════════════════════════════════════════
paper_funcs = [
    ('cos1',   r'$\frac{1}{2}(1+\cos\theta)$',
     lambda t: 0.5 * (1 + np.cos(t))),
    ('cos2',   r'$[\frac{1}{2}(1+\cos\theta)]^2$',
     lambda t: (0.5 * (1 + np.cos(t)))**2),
    ('arcsin', r'$1/|\sin(\theta/2)|$ (dwell time)',
     lambda t: np.minimum(1.0 / np.maximum(np.abs(np.sin(t * 0.5)), 0.2) / 5.0, 1.0)),
    ('gauss5', r'$e^{-5(1-\cos\theta)^2}$',
     lambda t: np.exp(-5.0 * (1 - np.cos(t))**2)),
    ('rect30', r'rect$(30\%)$',
     lambda t: (np.cos(t) > np.cos(0.3 * np.pi)).astype(float)),
    ('cos4',   r'$[\frac{1}{2}(1+\cos\theta)]^4$',
     lambda t: (0.5 * (1 + np.cos(t)))**4),
]

# Run all
results = []
for name, label, fn in paper_funcs:
    np.random.seed(42)
    res = run_memory(50000, kill_r_periods=10, prob_func_fn=fn, n_bins=80, N_total=500_000)
    if res:
        res['name'] = name
        res['label'] = label
        res['fn'] = fn  # store for silhouette
        results.append(res)
        print(f"{name:>8}: r_mem={res['r_memory']:.4f}, r_mean={res['r_mean']:.4f}, "
              f"ppb={res['ppb']}, alive={res['n_alive']}")

# Sort by r_memory
results.sort(key=lambda x: x['r_memory'], reverse=True)

# ══════════════════════════════════════════════════════
# Precompute low-k microstripe silhouettes (same as Fig 3)
# ══════════════════════════════════════════════════════
k1_low = 240 * np.pi
k2_low = 250 * np.pi
x_sil = np.linspace(-0.5, 0.5, 300000)
env_sil = np.sinc(1.8 * x_sil)**2

# ══════════════════════════════════════════════════════
# PAPER FIGURE
# ══════════════════════════════════════════════════════
n_show = len(results)
fig, axes = plt.subplots(n_show, 1, figsize=(7, 2.2 * n_show), sharex=True)

def norm01(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn) if mx > mn else a * 0

colors = ['#2563eb', '#7c3aed', '#dc2626', '#059669', '#d97706', '#64748b']

for i, res in enumerate(results):
    ax = axes[i]
    bc = res['bin_centers']
    
    mem_n = norm01(res['memory_hist'])
    qm_n = norm01(res['qm_hist'])

    # Faint microstripe silhouette (low-k analytical P1+P2)
    fn = res['fn']
    P1_sil = env_sil * fn(k1_low * x_sil)
    P2_sil = env_sil * fn(k2_low * x_sil)
    P_sum_sil = P1_sil + P2_sil
    sil_n = norm01(P_sum_sil)
    ax.fill_between(x_sil, 0, sil_n, color='#94a3b8', alpha=0.18,
                    linewidth=0, rasterized=True, zorder=1)

    ax.plot(bc, mem_n, color=colors[i], lw=1.8, zorder=3,
            label=f'Detector memory ($r = {res["r_memory"]:.3f}$)')
    ax.plot(bc, qm_n, color='#dc2626', lw=1.5, ls='--', alpha=0.8, zorder=4,
            label='QM prediction')
    ax.fill_between(bc, 0, mem_n, color=colors[i], alpha=0.12, zorder=2)
    
    ax.text(0.02, 0.82, f'({chr(97+i)})', transform=ax.transAxes,
            fontsize=11, fontweight='bold')
    ax.text(0.08, 0.82, f'$f(\\theta) = $ {res["label"]}',
            transform=ax.transAxes, fontsize=9)
    
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel('Normalized')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.92)

axes[-1].set_xlabel('Screen position $y$')

import os
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, 'fig_func_sweep.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'fig_func_sweep.pdf'),
            bbox_inches='tight')
print("\nSaved fig_func_sweep.png and fig_func_sweep.pdf")

# Also print the values we need for the paper
print("\n\nVALUES FOR PAPER:")
print(f"{'Function':>12} {'r(memory)':>10} {'r(linear)':>10}")
print("-" * 35)
for res in results:
    print(f"{res['name']:>12} {res['r_memory']:>10.3f} {res['r_mean']:>10.3f}")
