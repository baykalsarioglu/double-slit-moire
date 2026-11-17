"""
Figure: Screen Memory — HONEST simulation.

Key insight (from user): peaks are PROBABILITY peaks. The tallest
peak gets hit FIRST (highest probability), kills its neighbors,
then the next tallest surviving peak gets hit, etc.

Process:
Phase 1: Fire particles ONE AT A TIME until alive mask converges.
          Each hit kills neighbors. Taller peaks dominate.
Phase 2: Alive mask is stable. Batch-fire remaining particles.
          Each particle lands on a surviving peak ∝ its height.

Output:
  paper/fig_screen_memory.png
  paper/fig_screen_memory.pdf
"""
import numpy as np
from scipy.signal import argrelmax
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

np.random.seed(42)

# ──── PARAMETERS ────
k1 = 50000 * np.pi
k2 = 50010 * np.pi
dk = k2 - k1

k_avg = (k1 + k2) / 2
carrier_period = 2 * np.pi / k_avg

print(f"k1={k1/np.pi:.0f}π  k2={k2/np.pi:.0f}π  dk={dk/np.pi:.0f}π")

# Fine grid to find TRUE peaks
n_per_period = 20
n_grid = int(1.0 / carrier_period * n_per_period)
x = np.linspace(-0.5, 0.5, n_grid)
dx = x[1] - x[0]

env = np.sinc(1.8 * x)**2
P1 = env * (0.5 * (1 + np.cos(k1 * x)))**2
P2 = env * (0.5 * (1 + np.cos(k2 * x)))**2
P_sum = P1 + P2

# Find true peaks
min_dist = max(1, int(0.4 * carrier_period / dx))
peaks_idx = argrelmax(P_sum, order=min_dist)[0]
good = P_sum[peaks_idx] > 0.001
peaks_idx = peaks_idx[good]

peak_x = x[peaks_idx]
peak_h = P_sum[peaks_idx]
n_peaks = len(peak_x)

kill_radius = 10 * carrier_period

print(f"Peaks: {n_peaks}")
print(f"Kill radius: {kill_radius/carrier_period:.0f} carrier periods")

# ──── BINNING ────
n_bins = 80
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

# ──── METHOD 1: Linear readout (no memory) ────
# Sample from the FULL continuous P_sum (carrier averages out in bins)
N_total = 500_000
P_full_norm = P_sum / P_sum.sum()
hits_x = np.random.choice(len(x), size=N_total, p=P_full_norm)
mean_hist = np.histogram(x[hits_x], bins=bin_edges)[0].astype(float)

# ──── METHOD 2: Screen Memory (one-at-a-time then batch) ────
alive = np.ones(n_peaks, dtype=bool)
hit_count = np.zeros(n_peaks)

# PHASE 1: Fire one particle at a time until alive converges
print("\n── Phase 1: sequential firing (kills happening) ──")
prev_alive = alive.sum()
converged_at = 0
for i in range(n_peaks * 3):  # at most 3x peaks
    alive_h = peak_h * alive
    total = alive_h.sum()
    if total == 0:
        break
    probs = alive_h / total

    # Sample ONE particle
    h = np.random.choice(n_peaks, p=probs)

    # Only count if peak is still alive
    if alive[h]:
        hit_count[h] += 1
        # Kill neighbors
        for nb in neighbors[h]:
            alive[nb] = False

    if (i + 1) % 500 == 0:
        n_alive = alive.sum()
        if n_alive == prev_alive:
            converged_at = i + 1
            break
        prev_alive = n_alive
        print(f"  Particle {i+1:5d}: {n_alive:5d}/{n_peaks} alive "
              f"({100*n_alive/n_peaks:.1f}%)")

n_alive = alive.sum()
print(f"  Converged at particle {converged_at}: "
      f"{n_alive}/{n_peaks} alive ({100*n_alive/n_peaks:.1f}%)")

# PHASE 2: Alive mask is stable. Batch-fire remaining particles.
print("\n── Phase 2: batch firing (stable mask) ──")
remaining = N_total - converged_at
alive_h = peak_h * alive
total = alive_h.sum()
probs = alive_h / total

batch_hits = np.random.choice(n_peaks, size=remaining, p=probs)
for h in batch_hits:
    hit_count[h] += 1

memory_hist = np.zeros(n_bins)
for i, b in enumerate(peak_bin):
    if 0 <= b < n_bins:
        memory_hist[b] += hit_count[i]

# ── Correlations ──
def norm01(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn) if mx > mn else a * 0

r_mean = np.corrcoef(norm01(mean_hist), norm01(qm_hist))[0, 1]
r_memory = np.corrcoef(norm01(memory_hist), norm01(qm_hist))[0, 1]

print(f"\nStandard counting:  r = {r_mean:.4f}")
print(f"Screen memory:      r = {r_memory:.4f}")

# ──── FIGURE ────
fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True,
                          gridspec_kw={'hspace': 0.15})

bw = (bin_centers[1] - bin_centers[0]) * 0.88
c_bar_mean = '#94a3b8'
c_bar_mem = '#6366f1'
c_qm = '#dc2626'

scale_mean = np.max(mean_hist) / np.max(qm_hist)
scale_mem = np.max(memory_hist) / np.max(qm_hist) if np.max(memory_hist) > 0 else 1

ax = axes[0]
ax.bar(bin_centers, mean_hist, width=bw, color=c_bar_mean, alpha=0.8,
       edgecolor='#475569', linewidth=0.3, label='Linear readout')
ax.plot(bin_centers, qm_hist * scale_mean, color=c_qm, lw=2.0,
        zorder=5, label='QM (scaled)')
ax.set_ylabel('Hit count')
ax.set_ylim(0, np.max(mean_hist) * 1.15)
ax.legend(loc='upper right', framealpha=0.9)
ax.text(0.02, 0.88, '(a)', transform=ax.transAxes,
        fontsize=12, fontweight='bold')
ax.text(0.50, 0.88,
        f'{N_total//1000}k particles, $r = {r_mean:.3f}$',
        transform=ax.transAxes, fontsize=9, color='#dc2626', ha='center')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[1]
ax.bar(bin_centers, memory_hist, width=bw, color=c_bar_mem, alpha=0.7,
       edgecolor='#4338ca', linewidth=0.3, label='Detector memory')
ax.plot(bin_centers, qm_hist * scale_mem, color=c_qm, lw=2.0,
        zorder=5, label='QM (scaled)')
ax.set_xlabel('Screen position $y$')
ax.set_ylabel('Hit count')
ax.set_ylim(0, np.max(memory_hist) * 1.15 if np.max(memory_hist) > 0 else 1)
ax.legend(loc='upper right', framealpha=0.9)
ax.text(0.02, 0.88, '(b)', transform=ax.transAxes,
        fontsize=12, fontweight='bold')
ax.text(0.50, 0.88,
        f'{N_total//1000}k particles, $r = {r_memory:.3f}$',
        transform=ax.transAxes, fontsize=9, color='#16a34a', ha='center')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

import os
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, 'fig_screen_memory.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'fig_screen_memory.pdf'),
            bbox_inches='tight')
print("\nSaved fig_screen_memory")

# Print Table 2: Local baseline thresholding sweep
print("\nTABLE 2 — Local Baseline Thresholding Sweep:")
surv_x = peak_x[alive]
surv_h = peak_h[alive]
surv_env = np.sinc(1.8 * surv_x)**2
for tau in [0.0, 0.5, 0.75, 0.9, 1.0]:
    w_th = np.maximum(0.0, surv_h - tau * surv_env)
    w_th /= np.sum(w_th)
    hits_th = rng.choice(surv_x, size=N_total, p=w_th)
    hist_th, _ = np.histogram(hits_th, bins=bin_edges)
    r_th = np.corrcoef(hist_th, qm_hist)[0, 1]
    print(f"  tau = {tau:.2f} -> r(MC) = {r_th:.3f}")
