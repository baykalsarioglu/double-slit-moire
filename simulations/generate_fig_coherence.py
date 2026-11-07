"""
Figure: Coherence requirement — coherent vs incoherent source.

Physical picture:
  The SCREEN oscillates at frequency ω (synchronized via mechanical
  coupling). A particle emitted at time t_emit arrives at position x
  after traveling distance d_j(x), encountering the screen at phase:
    Φ = ω·t_emit + k·d_j(x) = φ₀ + k·d_j(x)
  where φ₀ = ω·t_emit is the SHARED emission-time phase.

  - Coherent source: particles emitted at synchronized times → same φ₀
    → microstripe peaks land at same positions → detector memory works.
  - Incoherent source: random emission times → random φ₀ per particle
    → carrier peaks shift by φ₀/k → detector sites sample random phases
    → averaged over many φ₀, every site sees ⟨f(random)⟩ = f̄ → flat.

  Note: φ₀ is SHARED between both slits (one particle, one emission
  time). The moiré envelope position (which depends on k·Δd) does NOT
  shift with φ₀ — but the carrier peaks DO, and the detector operates
  at the carrier peak level.

Implementation:
  1. Converge alive mask using canonical screen memory (φ₀=0)
  2. Coherent: batch-fire N particles with φ₀=0, sample ∝ P_sum.
  3. Incoherent: for each particle, draw random φ₀, evaluate
     P_sum(x_det; φ₀) at the SAME fixed detector sites.
     Accumulate absolute (unnormalized) probabilities across all
     particles, then normalize once and sample.

Parameters:
  f(θ) = [½(1+cos θ)]²
  k₁ = 50000π, k₂ = 50010π, Δk = 10π
  80 bins, kill radius = 10 carrier periods
  5×10⁵ particles per case

Output:
  paper/fig_coherence.png
  paper/fig_coherence.pdf
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

n_per_period = 20
n_grid = int(1.0 / carrier_period * n_per_period)
x = np.linspace(-0.5, 0.5, n_grid)
dx = x[1] - x[0]

env = np.sinc(1.8 * x)**2

n_bins = 80
bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

N_total = 500_000
kill_r_periods = 10
kill_radius = kill_r_periods * carrier_period
min_dist = max(1, int(0.4 * carrier_period / dx))

print(f"k1={k1/np.pi:.0f}π  k2={k2/np.pi:.0f}π  dk={dk/np.pi:.0f}π")
print(f"Bins: {n_bins}, Kill radius: {kill_r_periods} carrier periods")
print(f"Particles: {N_total:,}")

# ──── QM reference ────
x_fine = np.linspace(-0.5, 0.5, 100000)
QM_fine = 2 * np.sinc(1.8 * x_fine)**2 * np.cos(dk * x_fine / 2)**2
qm_hist = np.zeros(n_bins)
bi = np.digitize(x_fine, bin_edges) - 1
for b in range(n_bins):
    m = bi == b
    if m.any():
        qm_hist[b] = QM_fine[m].mean()


def norm01(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn) if mx > mn else a * 0


# ══════════════════════════════════════════════════════════
# STEP 1: Converge alive mask (shared by both cases)
# Same as generate_fig_screen_memory.py
# ══════════════════════════════════════════════════════════
print("\n── Converging alive mask ──")

P1_ref = env * (0.5 * (1 + np.cos(k1 * x)))**2
P2_ref = env * (0.5 * (1 + np.cos(k2 * x)))**2
P_sum_ref = P1_ref + P2_ref

peaks_idx = argrelmax(P_sum_ref, order=min_dist)[0]
good = P_sum_ref[peaks_idx] > 0.001
peaks_idx = peaks_idx[good]

peak_x = x[peaks_idx]
peak_h = P_sum_ref[peaks_idx]
n_peaks = len(peak_x)
peak_bin = np.digitize(peak_x, bin_edges) - 1
print(f"Reference peaks: {n_peaks}")

# Precompute neighbors
neighbors = []
for i in range(n_peaks):
    lo = np.searchsorted(peak_x, peak_x[i] - kill_radius)
    hi = np.searchsorted(peak_x, peak_x[i] + kill_radius)
    neighbors.append([n for n in range(lo, hi) if n != i])

# Fire until stable
alive = np.ones(n_peaks, dtype=bool)
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
        for nb in neighbors[h]:
            alive[nb] = False
    if (i + 1) % 500 == 0:
        n_alive = alive.sum()
        if n_alive == prev_alive:
            converged_at = i + 1
            break
        prev_alive = n_alive

if converged_at == 0:
    converged_at = min(i + 1, n_peaks * 3)

surviving = np.where(alive)[0]
det_x = peak_x[surviving]
det_bin = peak_bin[surviving]
n_det = len(surviving)
print(f"Converged at {converged_at}: {n_det}/{n_peaks} sites alive "
      f"({100*n_det/n_peaks:.1f}%)")

# Envelope at detector positions (for evaluating P with different φ₀)
env_at_det = np.sinc(1.8 * det_x)**2


# ══════════════════════════════════════════════════════════
# CASE A: COHERENT (φ₀ = 0)
# Every particle's peaks align with detector sites.
# ══════════════════════════════════════════════════════════
print(f"\n══ Case A: Coherent source (φ₀ = 0) ══")
np.random.seed(42)

# P_sum at fixed detector sites with φ₀=0
P1_det = env_at_det * (0.5 * (1 + np.cos(k1 * det_x)))**2
P2_det = env_at_det * (0.5 * (1 + np.cos(k2 * det_x)))**2
P_sum_det = P1_det + P2_det
probs_a = P_sum_det / P_sum_det.sum()

# Batch-fire all particles (alive mask is stable)
hits_a = np.random.choice(n_det, size=N_total, p=probs_a)
hist_coherent = np.zeros(n_bins)
for h in hits_a:
    b = det_bin[h]
    if 0 <= b < n_bins:
        hist_coherent[b] += 1

r_coherent = np.corrcoef(norm01(hist_coherent), norm01(qm_hist))[0, 1]
print(f"  r = {r_coherent:.3f}")


# ══════════════════════════════════════════════════════════
# CASE B: INCOHERENT (random shared φ₀ per particle)
# Same detector sites as Case A — the detector is one
# physical object. Each particle has a random emission time,
# so φ₀ = ω·t_emit is random. But φ₀ is SHARED between
# both slits (one particle, one emission event).
# The carrier peaks shift by φ₀/k; the moiré envelope
# stays put — but the detector sites no longer sit on peaks.
# ══════════════════════════════════════════════════════════
print(f"\n══ Case B: Incoherent source (random shared φ₀) ══")
np.random.seed(42)

hist_incoherent = np.zeros(n_bins)
P_acc_b = np.zeros(n_det)

batch_size = 10000
for i in range(N_total // batch_size):
    phi0 = np.random.uniform(0, 2 * np.pi, (batch_size, 1))

    # Evaluate P_sum at SAME detector sites with this particle's φ₀
    # φ₀ is shared — same phase offset for both slit contributions
    P1_batch = env_at_det * (0.5 * (1 + np.cos(k1 * det_x + phi0)))**2
    P2_batch = env_at_det * (0.5 * (1 + np.cos(k2 * det_x + phi0)))**2
    P_sum_batch = P1_batch + P2_batch
    
    P_acc_b += P_sum_batch.sum(axis=0)

probs_b = P_acc_b / P_acc_b.sum()
hits_b = np.random.choice(n_det, size=N_total, p=probs_b)
for h in hits_b:
    b = det_bin[h]
    if 0 <= b < n_bins:
        hist_incoherent[b] += 1

r_incoherent = np.corrcoef(norm01(hist_incoherent), norm01(qm_hist))[0, 1]
print(f"  FINAL r = {r_incoherent:.3f}")


# ──── FIGURE ────
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

bw = (bin_centers[1] - bin_centers[0]) * 0.88
c_qm = '#dc2626'

# Independent scaling matching the screen memory example
scale_coh = np.max(hist_coherent) / np.max(qm_hist)
scale_inc = np.max(hist_incoherent) / np.max(qm_hist) if np.max(hist_incoherent) > 0 else 1

# (a) Coherent
ax = axes[0]
ax.bar(bin_centers, hist_coherent, width=bw, color='#6366f1', alpha=0.7,
       edgecolor='#4338ca', linewidth=0.3, label='Detector memory')
ax.plot(bin_centers, qm_hist * scale_coh, color=c_qm, lw=2.0, zorder=5, label='QM (scaled)')
ax.set_xlabel('Screen position $y$')
ax.set_ylabel('Hit count')
ax.set_ylim(0, np.max(hist_coherent) * 1.15)
ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
ax.text(0.02, 0.88, f'(a)  $r = {r_coherent:.2f}$', transform=ax.transAxes,
        fontsize=11, fontweight='bold', color='#16a34a')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (b) Incoherent
ax = axes[1]
ax.bar(bin_centers, hist_incoherent, width=bw, color='#f59e0b', alpha=0.7,
       edgecolor='#d97706', linewidth=0.3, label='Detector memory')
ax.plot(bin_centers, qm_hist * scale_inc, color=c_qm, lw=2.0, zorder=5, label='QM (scaled)')
ax.set_xlabel('Screen position $y$')
ax.set_ylim(0, np.max(hist_incoherent) * 1.15 if np.max(hist_incoherent) > 0 else 1)
ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
ax.text(0.02, 0.88, f'(b)  $r = {r_incoherent:.2f}$', transform=ax.transAxes,
        fontsize=11, fontweight='bold', color='#dc2626')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

import os
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, 'fig_coherence.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'fig_coherence.pdf'),
            bbox_inches='tight')
print(f"\nSaved fig_coherence.png and fig_coherence.pdf")
print(f"\nFINAL:")
print(f"  Coherent:   r = {r_coherent:.3f}")
print(f"  Incoherent: r = {r_incoherent:.3f}")
