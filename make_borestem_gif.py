import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

NFRAMES = 120
FPS = 24

fig, ax = plt.subplots(figsize=(9, 11), facecolor='#060614')
ax.set_facecolor('#060614')
ax.axis('off')
ax.set_xlim(-1, 1)
ax.set_ylim(-3.2, 1.5)

rng = np.random.default_rng(7)

# ── Stars
sx = rng.uniform(-1, 1, 200)
sy = rng.uniform(0.15, 1.5, 200)
ss = rng.uniform(0.3, 4, 200)
ax.scatter(sx, sy, s=ss, c='white', alpha=0.8, zorder=1)

# ── Sun
sun = Circle((0.78, 1.22), 0.16, color='#FFE44D', zorder=6, lw=0)
ax.add_patch(sun)
for ang in np.linspace(0, 360, 14, endpoint=False):
    r = np.radians(ang)
    ax.plot([0.78 + 0.19*np.cos(r), 0.78 + 0.30*np.cos(r)],
            [1.22 + 0.19*np.sin(r), 1.22 + 0.30*np.sin(r)],
            color='#FFE44D', lw=2.2, zorder=6, alpha=0.9)

# ── Regolith gradient
cmap_reg = LinearSegmentedColormap.from_list('reg', [
    '#1a0a3a', '#1e1060', '#243070', '#2a5070',
    '#5a7a90', '#8a9090', '#aaaaaa', '#c8c8c8'], N=256)
z_bg = np.linspace(0, 1, 300).reshape(-1, 1) * np.ones((300, 1))
ax.imshow(z_bg, extent=[-1, 1, -3.2, 0], aspect='auto',
          cmap=cmap_reg, origin='upper', zorder=2, alpha=0.92)

# ── Surface line
ax.plot([-1, 1], [0, 0], color='#d8d8d8', lw=2.5, zorder=7)
ax.text(-0.95, 0.07, 'Lunar Surface', color='#e8e8e8',
        fontsize=11, fontweight='bold', zorder=10,
        path_effects=[pe.withStroke(linewidth=3, foreground='#060614')])

# ── Borestem tube
TW = 0.055
TDEPTH = -2.4
bore_bg = Rectangle((-TW, TDEPTH), TW*2, -TDEPTH, color='#8B5E3C', zorder=8, lw=0)
ax.add_patch(bore_bg)
bore_in = Rectangle((-TW*0.55, TDEPTH), TW*1.1, -TDEPTH, color='#1a1a2e', zorder=9, lw=0)
ax.add_patch(bore_in)
for sgn in [-1, 1]:
    ax.plot([sgn*TW, sgn*TW], [TDEPTH, 0], color='#C4894E', lw=1.8, zorder=10)

# Tube label
ax.text(TW + 0.04, -0.5, 'Fiberglass\nborestem', color='#C4894E',
        fontsize=8.5, va='center', zorder=10,
        path_effects=[pe.withStroke(linewidth=2, foreground='#060614')])

# ── Depth labels
for depth_m, label in [(0.8, '80 cm'), (1.0, '1.0 m'), (1.5, '1.5 m'), (2.0, '2.0 m')]:
    y = -depth_m
    ax.plot([-0.95, -0.82], [y, y], color='#aaaaaa', lw=1, zorder=7, alpha=0.6)
    ax.text(-0.98, y, label, color='#cccccc', fontsize=8.5,
            va='center', ha='right', zorder=10, alpha=0.8)

# ── Probe tip
tip = plt.Polygon([[-TW, TDEPTH], [TW, TDEPTH], [0, TDEPTH - 0.08]],
                   color='#C4894E', zorder=10)
ax.add_patch(tip)

# ── Sensor dots
sensor_depths = [-0.84, -0.91, -1.01, -1.29, -1.39]
for sd in sensor_depths:
    ax.plot(0, sd, 'o', ms=7, color='#FFD700', zorder=12, mec='white', mew=0.8)

# ── k=0.04 label
ax.text(0, -2.55, 'k = 0.04 W/m/K\n(40x regolith)', color='#C4894E',
        fontsize=8, ha='center', va='center', zorder=12,
        path_effects=[pe.withStroke(linewidth=2, foreground='#060614')])

# ── Dynamic elements
part_dots, = ax.plot([], [], 'o', ms=5, color='#FF6A00', zorder=13, alpha=0.9, mec='none')
bore_glow = Rectangle((-TW, TDEPTH), TW*2, -TDEPTH, color='#FF6A00', zorder=8, lw=0, alpha=0)
ax.add_patch(bore_glow)
surf_glow = Rectangle((-1, -0.2), 2, 0.2, color='#FF4400', zorder=3, alpha=0)
ax.add_patch(surf_glow)

# Heat flow arrow annotation
heat_arrow = ax.annotate('', xy=(TW + 0.12, -1.6), xytext=(TW + 0.12, -0.3),
    arrowprops=dict(arrowstyle='->', color='#FF6A00', lw=2.5), zorder=15, alpha=0)
heat_label = ax.text(TW + 0.15, -1.0, 'Heat\nleak\ndown', color='#FF6A00',
    fontsize=9, fontweight='bold', va='center', zorder=15, alpha=0,
    path_effects=[pe.withStroke(linewidth=2, foreground='#060614')])

# Correction elements
corr_line = ax.axvline(x=0, color='#00FFAA', lw=0, zorder=14, alpha=0)
corr_text = ax.text(TW + 0.04, -1.85, 'CORRECTION\nAPPLIED', color='#00FFAA',
    fontsize=10, fontweight='bold', va='center', zorder=15, alpha=0,
    path_effects=[pe.withStroke(linewidth=3, foreground='#060614')])
checkmark = ax.text(0, -0.8, '', color='#00FFAA', fontsize=40,
    ha='center', va='center', zorder=15, alpha=0)

# Phase label
phase_txt = ax.text(0, -0.25, '', color='#FF9944', fontsize=10,
    fontweight='bold', ha='center', va='center', zorder=20,
    path_effects=[pe.withStroke(linewidth=3, foreground='#060614')])

# Temp readout boxes
temp_raw = ax.text(-0.95, -1.6, '', color='white', fontsize=9,
    ha='left', va='center', zorder=20, alpha=0,
    bbox=dict(boxstyle='round,pad=0.5', fc='#1a0505', ec='#FF6A00', lw=2))
temp_corr = ax.text(-0.95, -2.1, '', color='white', fontsize=9,
    ha='left', va='center', zorder=20, alpha=0,
    bbox=dict(boxstyle='round,pad=0.5', fc='#051a0a', ec='#00FFAA', lw=2))

# Title
ax.text(0, 1.38, 'Apollo HFE — Borestem Thermal Short-Circuit',
    color='white', fontsize=12, fontweight='bold', ha='center', va='center', zorder=20,
    path_effects=[pe.withStroke(linewidth=4, foreground='#060614')])
ax.text(0, 1.15, 'How fiberglass casing contaminates temperature readings',
    color='#aaaaaa', fontsize=9, ha='center', va='center', zorder=20)

# Geothermal arrow (always present, subtle)
ax.annotate('', xy=(- 0.85, -2.8), xytext=(-0.85, -3.1),
    arrowprops=dict(arrowstyle='->', color='#4466FF', lw=1.8), zorder=7)
ax.text(-0.82, -2.95, 'Geothermal\nheat  18 mW/m²', color='#4466FF',
    fontsize=7.5, va='center', zorder=7,
    path_effects=[pe.withStroke(linewidth=2, foreground='#060614')])

part_x = rng.uniform(-TW*0.4, TW*0.4, 20)


def animate(i):
    t = i / NFRAMES

    phase2 = max(0.0, min(1.0, (t - 0.30) / 0.40))   # borestem heating
    phase3 = max(0.0, min(1.0, (t - 0.72) / 0.28))   # correction

    # Particles falling down tube
    if phase2 > 0 and phase3 < 0.95:
        speed = 1.6
        n_show = int(20 * min(1, phase2 * 2) * (1 - phase3))
        py = ((part_x * 0 + np.linspace(0, 1, 20) - t * speed) % 1.0) * TDEPTH
        px = part_x.copy()
        part_dots.set_data(px[:n_show], py[:n_show])
        part_dots.set_alpha(min(1, phase2 * 3) * (1 - phase3))
    else:
        part_dots.set_data([], [])

    # Glow
    bore_glow.set_alpha(phase2 * (1 - phase3) * 0.4)
    surf_glow.set_alpha(phase2 * (1 - phase3) * 0.5)

    # Heat arrow
    arrow_alpha = phase2 * (1 - phase3)
    heat_arrow.set_alpha(arrow_alpha)
    heat_label.set_alpha(arrow_alpha)

    # Phase label
    if t < 0.30:
        phase_txt.set_text('Geothermal heat slowly rises from interior...')
        phase_txt.set_color('#6688FF')
    elif t < 0.72:
        phase_txt.set_text('Fiberglass conducts surface heat DOWNWARD')
        phase_txt.set_color('#FF6A00')
    else:
        phase_txt.set_text('Borestem correction removes fake warmth')
        phase_txt.set_color('#00FFAA')

    # Temp boxes
    box_alpha = min(1, phase2 * 5)
    temp_raw.set_alpha(box_alpha * (1 - phase3 * 0.5))
    temp_raw.set_text('Probe records:  253.5 K\n(contaminated by borestem)')

    temp_corr.set_alpha(phase3)
    temp_corr.set_text('True regolith:  252.0 K\n(after correction  -1.5 K)')

    # Correction text
    corr_text.set_alpha(phase3)
    checkmark.set_text('\u2713' if phase3 > 0.5 else '')
    checkmark.set_alpha(phase3)

    return (part_dots, bore_glow, surf_glow, heat_arrow, heat_label,
            phase_txt, temp_raw, temp_corr, corr_text, checkmark)


ani = animation.FuncAnimation(fig, animate, frames=NFRAMES,
                               interval=1000 // FPS, blit=True)
ani.save('figures/borestem_animation.gif', writer='pillow', fps=FPS, dpi=130)
plt.close()
print('Saved figures/borestem_animation.gif')
