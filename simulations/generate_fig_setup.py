import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set up matplotlib for paper-quality output
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 18,
})

fig, ax = plt.subplots(figsize=(12, 7))

# Coordinates
slit_x = 0
screen_x = 7
center_y = 0
s = 2.6     # slit separation
y_pt = 2.5  # point on the screen

# Draw center line (optical axis)
ax.plot([-4.5, screen_x + 2], [center_y, center_y], 'k-.', lw=1, alpha=0.5)

# Draw the barrier with slits
barrier_width = 0.2
slit_width = 0.3
color_barrier = '#555555'

ax.add_patch(patches.Rectangle((slit_x - barrier_width/2, s/2 + slit_width/2), 
                               barrier_width, 4, facecolor=color_barrier, edgecolor='black', zorder=3))
ax.add_patch(patches.Rectangle((slit_x - barrier_width/2, -s/2 + slit_width/2), 
                               barrier_width, s - slit_width, facecolor=color_barrier, edgecolor='black', zorder=3))
ax.add_patch(patches.Rectangle((slit_x - barrier_width/2, -4), 
                               barrier_width, 4 - s/2 - slit_width/2, facecolor=color_barrier, edgecolor='black', zorder=3))

# Draw the detector screen with oscillating nodes
ax.plot([screen_x, screen_x], [-3.8, 3.8], 'k-', lw=2, zorder=3)
ax.text(screen_x + 0.3, 3.6, 'Phase-locked\noscillating nodes\n(frequency $\\omega$)', fontsize=16, va='bottom', ha='left')

# Draw nodes on the screen — each with a small sine wave to show oscillation
node_ys = np.linspace(-3.5, 3.5, 15)
sq_half = 0.18  # half-size of square in data coords
t_wave = np.linspace(0, 2*np.pi, 30)
for i, ny in enumerate(node_ys):
    # Draw the square
    ax.add_patch(patches.FancyBboxPatch(
        (screen_x - sq_half, ny - sq_half), 2*sq_half, 2*sq_half,
        boxstyle="square,pad=0", facecolor='white', edgecolor='black', lw=1.2, zorder=4))
    # Inner circle centered
    ax.plot(screen_x, ny, marker='o', color='black', markersize=6, mfc='gray', zorder=5)
    # Small sine wave next to each node
    wave_x = screen_x + 0.28 + 0.18 * t_wave / (2*np.pi)
    wave_y = ny + 0.10 * np.sin(t_wave)
    ax.plot(wave_x, wave_y, 'k-', lw=1.0, zorder=5)

# Emphasize the node at y_pt
ax.add_patch(patches.FancyBboxPatch(
    (screen_x - 0.25, y_pt - 0.25), 0.5, 0.5,
    boxstyle="square,pad=0", facecolor='white', edgecolor='red', lw=2.5, zorder=5))
ax.plot(screen_x, y_pt, marker='o', color='red', markersize=10, mfc='red', zorder=6)
ax.text(screen_x + 0.6, y_pt, 'Detector node\nat position $y$', fontsize=18, va='center')


# Draw rays from slits to screen
ax.plot([slit_x, screen_x], [s/2, y_pt], color='#1f77b4', lw=2, zorder=2)
ax.plot([slit_x, screen_x], [-s/2, y_pt], color='#d62728', lw=2, zorder=2)

# Labels for slits
ax.text(slit_x - 0.3, s/2, 'Slit 1 ', va='center', ha='right', fontsize=18)
ax.text(slit_x - 0.3, -s/2, 'Slit 2 ', va='center', ha='right', fontsize=18)

# Labels for path lengths
ax.text(screen_x/2 - 0.8, (y_pt + s/2)/2 + 0.3, '$d_1(y)$', fontsize=20, color='#1f77b4')
ax.text(screen_x/2 + 0.5, (y_pt - s/2)/2 - 0.5, '$d_2(y)$', fontsize=20, color='#d62728')

# Annotate L (Screen Distance)
ax.annotate('', xy=(slit_x, -3.3), xytext=(screen_x, -3.3), 
            arrowprops=dict(arrowstyle='<|-|>', lw=2.0, color='black'))
ax.text(screen_x/2, -3.6, 'Distance $L$', ha='center', fontsize=18)

# Annotate s (Slit Separation)
ax.annotate('', xy=(slit_x - 1.8, s/2), xytext=(slit_x - 1.8, -s/2), 
            arrowprops=dict(arrowstyle='<|-|>', lw=2.0, color='black'))
ax.text(slit_x - 2.1, 0, 'Slit separation $s$', va='center', ha='center', rotation=90, fontsize=18)

# Annotate y (Position on screen)
ax.annotate('', xy=(screen_x - 0.6, center_y), xytext=(screen_x - 0.6, y_pt), 
            arrowprops=dict(arrowstyle='<|-|>', lw=2.0, color='black'))
ax.text(screen_x - 0.8, y_pt/2, '$y$', va='center', ha='right', fontsize=18)

# Source
source_x = -4.5
ax.plot(source_x, 0, 'ko', markersize=10, zorder=4)
ax.text(source_x - 0.3, 0, 'Particle\nSource', va='center', ha='right', fontsize=18)



# Set axis limits and remove box
ax.set_xlim(-6.5, 10.5)
ax.set_ylim(-4.2, 4.2)
ax.axis('off')

# Save figures
out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figures', 'fig_setup.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'figures', 'fig_setup.png'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(out_dir, 'fig_setup.pdf'), bbox_inches='tight')
print("Generated figures/fig_setup.pdf and fig_setup.png")
