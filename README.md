# Extracting Double-Slit Interference Envelopes from Classical Probability Combs via Moiré Sampling and Nonlinear Detector Memory

Companion simulation suite and interactive web application for:

**B. Sarioglu**, *"Extracting double-slit interference envelopes from classical probability combs via moiré sampling and nonlinear detector memory"* (2026).

## [▶ Launch Interactive Web Explorer](https://baykalsarioglu.github.io/double-slit-moire/moire_explorer.html)

---

## Overview

Textbook impossibility arguments for the double-slit experiment conclude that adding single-slit classical probabilities ($P_{12} = P_1 + P_2$) cannot produce interference fringes. This repository demonstrates the unstated premise behind that conclusion—**linear, passive detector readout**—and provides a complete analytical and numerical toolkit for extracting interference envelopes from summed positive probability combs:

1. **Multi-Harmonic Moiré Carrier Beat:** Summing two non-negative periodic acceptance combs $P_j(y) = S(y)\,f(k\cdot d_j(y))$ generates a high-frequency spatial carrier whose upper envelope modulates at the exact double-slit interference phase $\delta(y) = k\,\Delta d(y)$.
2. **Exact Pitchfork Bifurcation & Analytical Envelope:** For $f(\theta) = [\tfrac{1}{2}(1+\cos\theta)]^2$, the carrier peaks undergo a supercritical pitchfork bifurcation at $|\cos(\delta/2)| = 1/2$ ($\delta \equiv \pm 2\pi/3 \pmod{2\pi}$), yielding an exact piecewise closed-form peak envelope $E_{\text{peak}}(y)$ with Pearson correlation $r(E_{\text{peak}}, I_{\text{QM}}) = 0.892$ and fringe visibility $V \approx 0.646$.
3. **Asymptotic Low-Pass Erasure:** Linear pixel integration over $W \gg \lambda_{\text{dB}}$ cancels all oscillatory harmonics as $\mathcal{O}(\lambda_{\text{dB}}/W)$, recovering the smooth classical sum $2\bar{f}\,S(y)$ ($r = 0.625$).
4. **Nonlinear Detector Memory (Lateral Depletion) & Local Thresholding:** A height-biased spatial dead-time / lateral-depletion Markov process ($R_{\text{kill}} \gg \lambda_{\text{dB}}$) acts as a spatial envelope detector, converging in finite stopping time to a jammed survivor set that tracks $E_{\text{peak}}(y)$ ($r = 0.891\text{–}0.895$, $V \approx 0.65$ under unthresholded readout). Incorporating a local registration threshold calibrated to the local mean arrival rate ($\Theta = 2\bar{f}\,S(y)$, available strictly locally) suppresses the interlaced destructive pedestal, raising agreement with the quantum intensity profile to $r = 0.988\text{–}0.997$ and restoring unit fringe visibility ($V \to 1.00$).

---

## Repository Structure

```text
├── moire_explorer.html                   # Interactive browser-based simulation (HTML5 Canvas)
├── simulations/
│   ├── generate_fig_setup.py             # Fig. 1: Geometric setup & active detector schematic
│   ├── generate_fig1.py                  # Fig. 2: Slit 1, Slit 2, and moiré sum P1 + P2 vs QM
│   ├── generate_fig_screen_memory.py     # Fig. 3: Linear readout (r=0.625) vs lateral depletion (r=0.891)
│   ├── generate_fig_func_sweep.py        # Fig. 4 & Table 1: Six periodic acceptance profiles f(θ)
│   ├── generate_fig_convergence.py       # Table 2: Convergence across microstripe density k
│   └── generate_fig_coherence.py         # Fig. 5: Coherent vs incoherent source-detector phase
└── LICENSE                               # MIT License
```

## Reproducing the Figures and Tables

Requires Python 3.9+ with `numpy`, `scipy`, and `matplotlib`:

```bash
pip install numpy scipy matplotlib
python3 simulations/generate_fig_setup.py
python3 simulations/generate_fig1.py
python3 simulations/generate_fig_screen_memory.py
python3 simulations/generate_fig_func_sweep.py
python3 simulations/generate_fig_convergence.py
python3 simulations/generate_fig_coherence.py
```

## License

MIT License — Copyright (c) 2026 Baykal Sarioglu
