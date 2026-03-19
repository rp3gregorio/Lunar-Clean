"""
plots.py — All plotting functions for the lunar thermal model.

Every function takes pre-computed data (stats dicts, cycle dicts, etc.)
and returns a matplotlib Figure.  This keeps the notebook cells short:
  fig = plots.diurnal_cycles(cycles, lat, lon, sunscale)
  fig.show()

The same functions are called by the single-point analysis, the model
comparison, the sensitivity study, and the batch summary — so every
section of the notebook uses a consistent visual style.

Public functions
----------------
diurnal_cycles()         — Temperature vs time at multiple depths.
heatmap()                — 2-D temperature field (depth × time).
apollo_comparison()      — Model profile vs single Apollo HFE site.
dual_apollo_comparison() — Both Apollo 15 & 17 side-by-side.
model_comparison()       — Two or more models side-by-side.
sensitivity_sweep()   — Parameter sensitivity: 6-panel summary.
batch_summary()       — Grid of bar/line plots for batch results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ── Shared style — publication quality ───────────────────────────────────────
import matplotlib as _mpl
plt.style.use('seaborn-v0_8-whitegrid')
_mpl.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'legend.framealpha':  0.9,
    'figure.dpi':         130,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.edgecolor':     '#444444',
    'axes.linewidth':     0.8,
    'grid.color':         '#dddddd',
    'grid.linewidth':     0.6,
})

_MODEL_COLORS = {
    'discrete':           '#C0392B',   # deep red
    'hayne_exponential':  '#2471A3',   # deep blue
    'custom':             '#1E8449',   # deep green
}
_MODEL_LABELS = {
    'discrete':          'Discrete Layers',
    'hayne_exponential': 'Hayne 2017',
    'custom':            'Custom',
}
_MODEL_LINES = {
    'discrete':          '-',
    'hayne_exponential': '--',
    'custom':            '-.',
}


def _model_style(model_name):
    """Return (color, linestyle, label) for a model name."""
    return (
        _MODEL_COLORS.get(model_name, 'black'),
        _MODEL_LINES.get(model_name, '-'),
        _MODEL_LABELS.get(model_name, model_name),
    )


def _subtitle(lat, lon, model_name=None, extra=''):
    parts = [f'{lat:.3f}°N, {lon:.3f}°E']
    if model_name:
        parts.append(_MODEL_LABELS.get(model_name, model_name))
    if extra:
        parts.append(extra)
    return ' | '.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DIURNAL TEMPERATURE CYCLES
# ─────────────────────────────────────────────────────────────────────────────

def diurnal_cycles(cycles, lat, lon, model_name=None, sunscale=None,
                   figsize=(13, 7)):
    """
    Plot temperature vs time (hours) at several depths for the final lunar day.

    Parameters
    ----------
    cycles     : dict from analysis.get_diurnal_cycles()
    lat, lon   : location (degrees)
    model_name : model key (used for subtitle)
    sunscale   : SUNSCALE value (used for subtitle)
    figsize    : figure size tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    depth_colors = ['#E74C3C', '#E67E22', '#2980B9', '#8E44AD',
                    '#27AE60', '#16A085', '#D35400', '#C0392B']

    fig, ax = plt.subplots(figsize=figsize)

    for i, (depth, data) in enumerate(sorted(cycles.items())):
        color = depth_colors[i % len(depth_colors)]
        ax.plot(data['time_h'], data['temperature'],
                color=color, linewidth=2.5,
                label=f"{depth * 100:.0f} cm depth")

    ax.set_xlabel('Time in lunar day (hours)', fontsize=13, weight='bold')
    ax.set_ylabel('Temperature (K)', fontsize=13, weight='bold')

    extra = f'SUNSCALE={sunscale:.2f}' if sunscale is not None else ''
    ax.set_title('Diurnal Temperature Cycles\n' + _subtitle(lat, lon, model_name, extra),
                 fontsize=14, weight='bold', pad=12)

    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEMPERATURE HEATMAP (depth × time)
# ─────────────────────────────────────────────────────────────────────────────

def heatmap(T_profile, t_arr, z_grid, lat, lon, model_name=None,
            depth_limit=1.5, colormap='hot', show_contours=True,
            figsize=(13, 7)):
    """
    2-D filled-contour plot of temperature as a function of depth and time.

    Parameters
    ----------
    depth_limit  : only show the top *depth_limit* metres
    colormap     : matplotlib colormap name
    show_contours: overlay contour lines
    """
    from lunar.constants import LUNAR_DAY

    t_start  = t_arr[-1] - LUNAR_DAY
    idx_t    = np.where(t_arr >= t_start)[0]
    idx_z    = np.where(z_grid <= depth_limit)[0]

    t_hours = (t_arr[idx_t] - t_start) / 3600.0
    z_cm    = z_grid[idx_z] * 100.0
    T_sub   = T_profile[np.ix_(idx_t, idx_z)]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.contourf(t_hours, z_cm, T_sub.T,
                     levels=50, cmap=colormap,
                     vmin=np.min(T_sub), vmax=np.max(T_sub))

    if show_contours:
        cs = ax.contour(t_hours, z_cm, T_sub.T,
                        levels=10, colors='black',
                        linewidths=0.5, alpha=0.3)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f K')

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Temperature (K)', fontsize=12, weight='bold')

    ax.set_xlabel('Time in lunar day (hours)', fontsize=13, weight='bold')
    ax.set_ylabel('Depth (cm)', fontsize=13, weight='bold')
    ax.set_title('Temperature Evolution — Depth × Time\n' +
                 _subtitle(lat, lon, model_name),
                 fontsize=14, weight='bold', pad=12)
    ax.invert_yaxis()
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. APOLLO SITE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def apollo_comparison(stats, errors, site_name, model_name,
                      sunscale, chi, albedo, figsize=(12, 9)):
    """
    Compare model predictions with Apollo HFE measurements.

    Parameters
    ----------
    stats      : dict from analysis.extract_stats()
    errors     : dict from analysis.compute_apollo_errors()
    site_name  : 'Apollo 15' or 'Apollo 17'
    model_name : used in title / legend
    figsize    : figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    z_grid       = stats['depth']
    a_depths     = errors['apollo_depths']
    a_temps      = errors['apollo_temps']
    residuals    = errors['residuals']
    color, ls, label = _model_style(model_name)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.35, wspace=0.3)

    # ── Panel 1: Temperature profiles ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    ax1.plot(stats['T_mean'], z_grid * 100, color=color, ls=ls,
             linewidth=3, label=f'{label} (mean)')
    ax1.fill_betweenx(z_grid * 100, stats['T_min'], stats['T_max'],
                      color=color, alpha=0.15, label='Diurnal range')
    ax1.plot(a_temps, a_depths * 100, 'o', color='#1A5276',
             markersize=9, markeredgewidth=1.5, markeredgecolor='white',
             label=f'{site_name} measured', zorder=5)

    max_meas_cm = float(np.max(a_depths * 100))
    y_max_cm    = max(30.0, max_meas_cm * 2.2)
    mask = z_grid * 100 <= y_max_cm * 1.05

    ax1.set_xlabel('Temperature (K)', fontsize=12, weight='bold')
    ax1.set_ylabel('Depth (cm)', fontsize=12, weight='bold')
    ax1.set_title(f'{site_name} Validation — RMSE: {errors["rmse"]:.2f} K  '
                  f'Bias: {errors["bias"]:+.2f} K  MAE: {errors["mae"]:.2f} K',
                  fontsize=13, weight='bold')
    ax1.legend(fontsize=10, framealpha=0.95)
    ax1.set_ylim(y_max_cm, 0)

    # ── Panel 2: Residuals lollipop chart ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    d_cm = a_depths * 100
    ax2.hlines(d_cm, 0, residuals, color='#2471A3', linewidth=2.0, alpha=0.85)
    ax2.scatter(residuals, d_cm,
                s=50, color='#2471A3', edgecolors='white',
                linewidths=0.8, zorder=4)
    ax2.axvline(0, color='#333333', linewidth=1.2, ls='--')
    ax2.set_xlabel('Residual (K)',  fontsize=11, weight='bold')
    ax2.set_ylabel('Depth (cm)',   fontsize=11, weight='bold')
    ax2.set_title('Model − Measured', fontsize=12, weight='bold')
    ax2.set_ylim(y_max_cm * 0.55, max(0, float(np.min(d_cm)) - 2))
    ax2.invert_yaxis()

    # ── Panel 3: Statistics text ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    r2 = 1.0 - np.sum(residuals**2) / np.sum((a_temps - np.mean(a_temps))**2)
    txt = (
        f"Validation Statistics\n"
        f"{'─'*28}\n"
        f"RMSE : {errors['rmse']:.3f} K\n"
        f"Bias : {errors['bias']:+.3f} K\n"
        f"MAE  : {errors['mae']:.3f} K\n"
        f"R²   : {r2:.3f}\n\n"
        f"Configuration\n"
        f"{'─'*28}\n"
        f"SUNSCALE : {sunscale:.2f}\n"
        f"CHI      : {chi:.1f}\n"
        f"ALBEDO   : {albedo:.3f}\n"
        f"Model    : {label}"
    )
    ax3.text(0.05, 0.5, txt, fontsize=10, family='monospace',
             va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.35, pad=0.8))

    plt.suptitle(f'{site_name} Model Validation', fontsize=15, weight='bold', y=0.99)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3b. DUAL APOLLO COMPARISON  (Apollo 15 and Apollo 17 side-by-side)
# ─────────────────────────────────────────────────────────────────────────────

def dual_apollo_comparison(apollo_results, model_name, sunscale, chi, albedo,
                           figsize=(16, 10)):
    """
    Show both Apollo 15 and Apollo 17 validation side-by-side in one figure.

    Parameters
    ----------
    apollo_results : dict with keys 'Apollo 15' and 'Apollo 17', each mapping
                     to {'stats': extract_stats() output,
                         'errors': compute_apollo_errors() output}
    model_name     : used in title / legend
    sunscale, chi, albedo : for the config text box
    figsize        : figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    sites  = ['Apollo 15', 'Apollo 17']
    colors = ['#1A5276', '#7D3C98']   # navy blue for A15, purple for A17
    color, ls, label = _model_style(model_name)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 2, height_ratios=[2.8, 1.0],
                            hspace=0.38, wspace=0.38)

    for col, site_name in enumerate(sites):
        if site_name not in apollo_results:
            continue
        stats     = apollo_results[site_name]['stats']
        errors    = apollo_results[site_name]['errors']
        z_grid    = stats['depth']
        a_depths  = errors['apollo_depths']          # metres
        a_temps   = errors['apollo_temps']
        residuals = errors['residuals']
        dot_color = colors[col]

        # ── Depth axis: zoom to the measurement region with context ───────────
        # Apollo 15 sensors: 8–14 cm  |  Apollo 17 sensors: 13–23 cm
        # Show down to 2× the deepest sensor so the profile shape is visible.
        max_meas_cm = float(np.max(a_depths * 100))
        y_max_cm    = max(30.0, max_meas_cm * 2.2)   # at least 30 cm

        # ── Row 0: temperature profiles ───────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])

        # Clip model arrays to the displayed depth range (avoids confusing
        # flat tail that dominates a 0–300 cm axis)
        mask = z_grid * 100 <= y_max_cm * 1.05
        ax0.plot(stats['T_mean'][mask], z_grid[mask] * 100,
                 color=color, ls=ls, linewidth=2.5, label=f'{label} (mean)')
        ax0.fill_betweenx(z_grid[mask] * 100,
                          stats['T_min'][mask], stats['T_max'][mask],
                          color=color, alpha=0.18, label='Diurnal range')

        # Measurement dots — each sensor at its own depth
        ax0.plot(a_temps, a_depths * 100, 'o', color=dot_color,
                 markersize=8, markeredgewidth=1.2,
                 markeredgecolor='white', zorder=5,
                 label=f'{site_name} measured')

        # Annotate deepest measurement depth
        ax0.axhline(max_meas_cm, color=dot_color, ls='--',
                    lw=0.8, alpha=0.45)

        ax0.set_xlabel('Temperature (K)', fontsize=12, weight='bold')
        ax0.set_ylabel('Depth (cm)',       fontsize=12, weight='bold')
        ax0.set_ylim(y_max_cm, 0)          # zoomed — surface at top
        ax0.set_title(
            f'{site_name}\n'
            f'RMSE {errors["rmse"]:.2f} K  |  Bias {errors["bias"]:+.2f} K',
            fontsize=13, weight='bold', pad=8,
        )
        ax0.legend(fontsize=9, framealpha=0.9, loc='lower left')

        # ── Row 1: residual lollipop chart ────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])

        d_cm  = a_depths * 100        # depths in cm
        # Lollipop: horizontal lines + circles (clean for publication)
        ax1.hlines(d_cm, 0, residuals, color='#2471A3',
                   linewidth=1.8, alpha=0.85)
        ax1.scatter(residuals, d_cm,
                    s=45, color='#2471A3', edgecolors='white',
                    linewidths=0.8, zorder=4)
        ax1.axvline(0, color='#333333', linewidth=1.2, ls='--')

        ax1.set_xlabel('Residual  (Model − Measured, K)',
                       fontsize=11, weight='bold')
        ax1.set_ylabel('Depth (cm)', fontsize=11, weight='bold')
        ax1.set_title('Model − Measured', fontsize=12, weight='bold', pad=6)
        ax1.set_ylim(y_max_cm * 0.55, max(0, float(np.min(d_cm)) - 2))
        ax1.invert_yaxis()

        # Minimal stats annotation inside the residuals panel
        r2 = 1.0 - (np.sum(residuals**2) /
                    np.sum((a_temps - np.mean(a_temps))**2))
        ax1.text(0.97, 0.05,
                 f'RMSE {errors["rmse"]:.3f} K\n'
                 f'Bias  {errors["bias"]:+.3f} K\n'
                 f'MAE  {errors["mae"]:.3f} K\n'
                 f'R²    {r2:.3f}',
                 fontsize=8, family='monospace',
                 ha='right', va='bottom', transform=ax1.transAxes,
                 bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='#f7f7f7', edgecolor='#cccccc',
                           alpha=0.9))

    # ── Shared super-title ────────────────────────────────────────────────────
    cfg = (f'Model: {label}   |   SUNSCALE {sunscale:.2f}   '
           f'CHI {chi:.1f}   ALBEDO {albedo:.3f}')
    plt.suptitle(f'Apollo 15 & 17 Dual Validation — Discrete Layers\n'
                 f'{cfg}',
                 fontsize=13, weight='bold', y=1.01)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL COMPARISON  (Discrete vs Hayne — or any set of models)
# ─────────────────────────────────────────────────────────────────────────────

def model_comparison(results_dict, z_grid, lat, lon,
                     apollo_errors=None, figsize=(15, 9)):
    """
    Side-by-side comparison of two or more density models.

    Parameters
    ----------
    results_dict : {model_name: stats_dict, ...}
                   stats_dict from analysis.extract_stats()
    z_grid       : shared depth array (m)
    lat, lon     : location (for title)
    apollo_errors: {model_name: errors_dict} or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    model_names = list(results_dict.keys())

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 3, height_ratios=[1.5, 1],
                            hspace=0.35, wspace=0.30)

    # ── Panel 1: Mean temperature profiles ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for name in model_names:
        c, ls, lbl = _model_style(name)
        s = results_dict[name]
        ax1.plot(s['T_mean'], z_grid * 100, color=c, ls=ls, lw=2.5, label=lbl)

    if apollo_errors:
        # Use Apollo data from first model that has it
        for name in model_names:
            err = (apollo_errors or {}).get(name)
            if err is not None:
                ax1.plot(err['apollo_temps'], err['apollo_depths'] * 100,
                         'o', color='green', markersize=9,
                         markeredgewidth=2, markeredgecolor='darkgreen',
                         label='Apollo Measured', zorder=10)
                break

    ax1.set_xlabel('Temperature (K)', fontsize=11, weight='bold')
    ax1.set_ylabel('Depth (cm)', fontsize=11, weight='bold')
    ax1.set_title('Mean Temperature Profiles', fontsize=12, weight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # ── Panel 2: Diurnal amplitude ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for name in model_names:
        c, ls, lbl = _model_style(name)
        s = results_dict[name]
        ax2.plot(s['T_amplitude'], z_grid * 100,
                 color=c, ls=ls, lw=2.5, label=lbl)

    ax2.set_xlabel('Amplitude (K)', fontsize=11, weight='bold')
    ax2.set_ylabel('Depth (cm)', fontsize=11, weight='bold')
    ax2.set_title('Diurnal Temperature Amplitude', fontsize=12, weight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    # ── Panel 3: Difference between first two models ───────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if len(model_names) >= 2:
        m1, m2  = model_names[0], model_names[1]
        l1, l2  = _model_style(m1)[2], _model_style(m2)[2]
        T_diff  = results_dict[m1]['T_mean'] - results_dict[m2]['T_mean']
        ax3.plot(T_diff, z_grid * 100, 'k-', lw=2.5)
        ax3.axvline(0, color='gray', ls='--', lw=1)
        ax3.set_xlabel('ΔT (K)', fontsize=11, weight='bold')
        ax3.set_ylabel('Depth (cm)', fontsize=11, weight='bold')
        ax3.set_title(f'{l1} − {l2}', fontsize=12, weight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()

    # ── Panel 4: Statistics table ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    lines = ['Model Statistics', '─' * 55, '']
    if apollo_errors and any(apollo_errors.get(n) for n in model_names):
        lines.append(f"{'Model':<22} {'RMSE (K)':<12} {'Bias (K)':<12} {'MAE (K)':<12}")
        lines.append('─' * 55)
        for name in model_names:
            lbl = _model_labels_get(name)
            err = (apollo_errors or {}).get(name)
            if err:
                lines.append(f"{lbl:<22} {err['rmse']:<12.3f} {err['bias']:<+12.3f} {err['mae']:<12.3f}")
            else:
                lines.append(f"{lbl:<22} — no Apollo data —")
    else:
        lines.append('No Apollo data available at this location.')
        lines.append('Run at Apollo 15 (26.13°N, 3.63°E) or Apollo 17 (20.19°N, 30.76°E) to compare.')

    ax4.text(0.05, 0.5, '\n'.join(lines), fontsize=10, family='monospace',
             va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=0.8))

    plt.suptitle('Model Comparison — ' +
                 ' vs '.join(_model_style(n)[2] for n in model_names) +
                 '\n' + _subtitle(lat, lon),
                 fontsize=14, weight='bold', y=0.99)
    plt.tight_layout()
    return fig


def _model_labels_get(name):
    return _MODEL_LABELS.get(name, name)


# ─────────────────────────────────────────────────────────────────────────────
# 5. HFE TIME-SERIES  (temperature over probe lifetime)
# ─────────────────────────────────────────────────────────────────────────────

def hfe_timeseries(site_name, figsize=(16, 6)):
    """
    Plot the full temperature time-series for every sensor at *site_name*,
    one subplot per probe.

    Highlights
    ----------
    - Sensors coloured by depth (shallow = warm, deep = cool).
    - Shaded green band marks the stable equilibrium window (last 25 % of
      each probe's record) used for model validation.
    - Dashed vertical lines mark the start and end of the stable window.

    Parameters
    ----------
    site_name : 'Apollo 15' or 'Apollo 17'

    Returns
    -------
    matplotlib.figure.Figure
    """
    from lunar.hfe_loader import get_timeseries, _STABLE_FRACTION

    probes   = get_timeseries(site_name)
    n_probes = len(probes)

    fig, axes = plt.subplots(1, n_probes, figsize=figsize,
                             sharey=False, squeeze=False)
    axes = axes[0]

    # Diverging-friendly depth colormap: shallow=yellow, deep=indigo
    _depth_cmap = plt.get_cmap('viridis_r')

    # Collect overall depth range for shared colorbar
    all_depths_global = sorted({d['depth_mm']
                                 for probe in probes
                                 for d in probe.values()})
    g_dmin = all_depths_global[0]
    g_dmax = all_depths_global[-1]

    for ax, probe in zip(axes, probes):
        all_depths = sorted({d['depth_mm'] for d in probe.values()})
        d_min, d_max = all_depths[0], all_depths[-1]
        norm = Normalize(vmin=g_dmin, vmax=g_dmax)

        t_stable_final = None
        t_end_final    = None

        # --- plot each sensor ---
        for sensor, data in sorted(probe.items(),
                                   key=lambda kv: kv[1]['depth_mm']):
            times = data['times']
            temps = data['temps']
            d_mm  = data['depth_mm']
            color = _depth_cmap(norm(d_mm))

            t_num = np.array([t.timestamp() / 86400 for t in times])
            t0    = t_num[0]
            t_num = t_num - t0        # days since emplacement

            n_stable = max(1, int(len(temps) * _STABLE_FRACTION))
            t_stable = t_num[-n_stable]

            # Thicker line for deep (equilibrium) sensors
            lw = 1.5 if d_mm >= 80 else 1.0

            ax.plot(t_num, temps, lw=lw, color=color, alpha=0.92,
                    label=f'{sensor}  ({d_mm} mm)')

            # Track stable window bounds
            if t_stable_final is None or t_stable < t_stable_final:
                t_stable_final = t_stable
            if t_end_final is None or t_num[-1] > t_end_final:
                t_end_final = t_num[-1]

        # Stable-window shading and boundary lines (once per probe)
        ax.axvspan(t_stable_final, t_end_final,
                   alpha=0.13, color='#2ECC71',
                   label='Stable window (validation)')
        ax.axvline(t_stable_final, color='#1E8449', ls='--',
                   lw=1.4, alpha=0.75, zorder=3)
        ax.axvline(t_end_final,   color='#1E8449', ls=':',
                   lw=1.2, alpha=0.55, zorder=3)

        probe_label = next(iter(probe.values()))['probe_label']
        ax.set_title(probe_label, fontsize=13, weight='bold', pad=8)
        ax.set_xlabel('Days since emplacement', fontsize=12, weight='bold')
        ax.set_ylabel('Temperature (K)',        fontsize=12, weight='bold')

        # Legend: right column for deeper sensors (more relevant)
        leg = ax.legend(fontsize=7.5, ncol=1,
                        loc='upper right', framealpha=0.92,
                        edgecolor='#cccccc',
                        handlelength=1.6)
        for line in leg.get_lines():
            line.set_linewidth(2.0)

    # Shared colorbar across all probe panels
    sm = ScalarMappable(cmap=_depth_cmap,
                        norm=Normalize(vmin=g_dmin, vmax=g_dmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02, aspect=22)
    cbar.set_label('Sensor depth (mm)', fontsize=11, weight='bold')
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        f'{site_name} — HFE Probe Temperature History\n'
        'Green band = stable equilibrium window used for validation',
        fontsize=13, weight='bold', y=1.02,
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. HFE THERMAL SHUNTING EVIDENCE
# ─────────────────────────────────────────────────────────────────────────────

def hfe_shunting(site_name, n_snapshots=4, figsize=(15, 6)):
    """
    Evidence of the thermal disturbance (drilling residual heat + thermal
    shunting along the probe casing) in the HFE data.

    Left panel  — Temperature–depth profiles at several time snapshots.
                  Shortly after emplacement the profile is highly disturbed;
                  it converges toward the true geothermal gradient over months.

    Right panel — Temperature difference between gradient-bridge A and B
                  sensors (ΔT = T_A − T_B) over time for every TG bridge.
                  Shunting reduces the apparent gradient: the initial large ΔT
                  shrinks rapidly during the transient recovery, then settles to
                  a residual offset that reflects the true heat flow signal
                  modified by the probe's thermal conductance.

    Parameters
    ----------
    site_name   : 'Apollo 15' or 'Apollo 17'
    n_snapshots : number of time snapshots for the T-depth profile

    Returns
    -------
    matplotlib.figure.Figure
    """
    from lunar.hfe_loader import get_timeseries

    probes = get_timeseries(site_name)
    fig, (ax_prof, ax_grad) = plt.subplots(1, 2, figsize=figsize)

    # ── Left: T-depth profiles at successive snapshots ────────────────────────
    snap_cmap  = plt.get_cmap('viridis')
    all_depths_all = []  # collect across probes

    for probe in probes:
        for data in probe.values():
            all_depths_all.append(data['depth_mm'])

    # Build a time-indexed merged view: {timestamp: {depth_mm: T}}
    from collections import defaultdict
    ts_index = defaultdict(dict)
    for probe in probes:
        for sensor, data in probe.items():
            d_mm = data['depth_mm']
            for t, T in zip(data['times'], data['temps']):
                ts_index[t][d_mm] = T

    # Sort all unique timestamps
    all_times = sorted(ts_index.keys())
    n_t = len(all_times)
    snapshot_indices = [int(i * (n_t - 1) / (n_snapshots - 1))
                        for i in range(n_snapshots)]

    for snap_i, t_idx in enumerate(snapshot_indices):
        t_snap = all_times[t_idx]
        day    = (t_snap - all_times[0]).total_seconds() / 86400
        profile = ts_index[t_snap]
        depths  = sorted(profile.keys())
        temps   = [profile[d] for d in depths]
        color   = snap_cmap(snap_i / max(1, n_snapshots - 1))
        lbl     = (f'Day {day:.0f}' if day > 0 else 'Emplacement')
        ax_prof.plot(temps, [d / 10 for d in depths],  # depth in cm
                     'o-', color=color, lw=1.5, ms=5, label=lbl)

    ax_prof.invert_yaxis()
    ax_prof.set_xlabel('Temperature (K)', fontsize=11, weight='bold')
    ax_prof.set_ylabel('Depth (cm)', fontsize=11, weight='bold')
    ax_prof.set_title('T–Depth Profile at Time Snapshots\n'
                      '(disturbance convergence)', fontsize=11, weight='bold')
    ax_prof.legend(fontsize=9)
    ax_prof.grid(True, alpha=0.3)

    # ── Right: ΔT between bridge A and B sensors over time ───────────────────
    grad_cmap  = plt.get_cmap('tab10')
    bridge_idx = 0

    for probe in probes:
        # Identify TG bridges: pairs sharing the same numeric id (e.g. TG11 → A,B)
        tg_sensors = {s: d for s, d in probe.items() if s.startswith('TG')}
        # Group by bridge root (everything except trailing A/B)
        bridges = defaultdict(dict)
        for s, data in tg_sensors.items():
            root = s[:-1]   # e.g. 'TG12A' → 'TG12'
            ab   = s[-1]    # 'A' or 'B'
            bridges[root][ab] = data

        for bridge_name, ab_dict in sorted(bridges.items()):
            if 'A' not in ab_dict or 'B' not in ab_dict:
                continue
            dA, dB = ab_dict['A'], ab_dict['B']
            # Align on common timestamps (inner join via dict)
            tA_map = {t: T for t, T in zip(dA['times'], dA['temps'])}
            tB_map = {t: T for t, T in zip(dB['times'], dB['temps'])}
            common = sorted(set(tA_map) & set(tB_map))
            if not common:
                continue
            t0     = common[0]
            days   = np.array([(t - t0).total_seconds() / 86400
                                for t in common])
            delta  = np.array([tA_map[t] - tB_map[t] for t in common])
            color  = grad_cmap(bridge_idx % 10)
            depth_label = f'{dA["depth_mm"]}–{dB["depth_mm"]} mm'
            ax_grad.plot(days, delta, lw=0.9, color=color,
                         label=f'{bridge_name}  ({depth_label})')
            bridge_idx += 1

    ax_grad.axhline(0, color='k', ls='--', lw=0.8)
    ax_grad.set_xlabel('Days since emplacement', fontsize=11, weight='bold')
    ax_grad.set_ylabel('ΔT  (T_A − T_B)  [K]', fontsize=11, weight='bold')
    ax_grad.set_title('Gradient-Bridge ΔT Over Time\n'
                      '(thermal shunting signature)', fontsize=11, weight='bold')
    ax_grad.legend(fontsize=8)
    ax_grad.grid(True, alpha=0.3)

    fig.suptitle(f'{site_name} — HFE Thermal Disturbance & Shunting Evidence',
                 fontsize=13, weight='bold')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. LOCAL TERRAIN MAP  (DEM overview around target)
# ─────────────────────────────────────────────────────────────────────────────

def dem_overview(elev_m, map_res, target_lat, target_lon,
                 window_deg=5, figsize=(14, 6)):
    """
    Show the global Moon DEM and a zoomed local terrain map around the target.

    Left panel  : Full Moon elevation map with target location marked.
    Right panel : Local terrain (±window_deg) with contour lines.

    Parameters
    ----------
    elev_m      : (H, W) float32 — full DEM elevation grid in metres
    map_res     : pixels per degree
    target_lat  : target latitude (degrees)
    target_lon  : target longitude (degrees, 0–360)
    window_deg  : half-width of the local zoom box in degrees
    """
    H, W    = elev_m.shape
    pix_deg = 1.0 / map_res

    # Target pixel
    row_t = int(round((90.0 - target_lat) / pix_deg - 0.5))
    col_t = int(round(target_lon          / pix_deg - 0.5))
    row_t = max(0, min(H - 1, row_t))
    col_t = max(0, min(W - 1, col_t))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ── Left: Full Moon DEM ────────────────────────────────────────────────────
    step    = max(1, H // 360)
    elev_ds = elev_m[::step, ::step]
    lon_full = np.linspace(0, 360, elev_ds.shape[1])
    lat_full = np.linspace(90, -90, elev_ds.shape[0])

    im1 = ax1.pcolormesh(lon_full, lat_full, elev_ds / 1000.0,
                         cmap='gist_earth', shading='auto',
                         vmin=-9, vmax=10)
    plt.colorbar(im1, ax=ax1, label='Elevation (km)', shrink=0.75)
    ax1.plot(target_lon, target_lat, 'r*',
             markersize=14, markeredgewidth=1, markeredgecolor='white',
             label='Target', zorder=10)
    ax1.set_xlabel('Longitude (°E)', fontsize=11, weight='bold')
    ax1.set_ylabel('Latitude (°N)', fontsize=11, weight='bold')
    ax1.set_title('Global DEM — LOLA (LRO)', fontsize=12, weight='bold')
    ax1.legend(fontsize=9)
    ax1.set_aspect('equal')

    # ── Right: Zoomed local DEM ────────────────────────────────────────────────
    lat_min = max(-90, target_lat - window_deg)
    lat_max = min( 90, target_lat + window_deg)
    lon_min = max(  0, target_lon - window_deg)
    lon_max = min(360, target_lon + window_deg)

    r0 = max(0, int(round((90 - lat_max) / pix_deg - 0.5)))
    r1 = min(H, int(round((90 - lat_min) / pix_deg - 0.5)) + 1)
    c0 = max(0, int(round(lon_min / pix_deg - 0.5)))
    c1 = min(W, int(round(lon_max / pix_deg - 0.5)) + 1)

    elev_local = elev_m[r0:r1, c0:c1]
    lons_local = np.linspace(lon_min, lon_max, elev_local.shape[1])
    lats_local = np.linspace(lat_max, lat_min, elev_local.shape[0])

    im2 = ax2.pcolormesh(lons_local, lats_local, elev_local / 1000.0,
                         cmap='gist_earth', shading='auto')
    plt.colorbar(im2, ax=ax2, label='Elevation (km)', shrink=0.75)

    cs = ax2.contour(lons_local, lats_local, elev_local / 1000.0,
                     levels=8, colors='black', linewidths=0.4, alpha=0.4)
    ax2.clabel(cs, inline=True, fontsize=7, fmt='%.1f km')

    ax2.plot(target_lon, target_lat, 'r*',
             markersize=16, markeredgewidth=1, markeredgecolor='white',
             label=f'{target_lat:.3f}°N, {target_lon:.3f}°E', zorder=10)
    ax2.set_xlabel('Longitude (°E)', fontsize=11, weight='bold')
    ax2.set_ylabel('Latitude (°N)', fontsize=11, weight='bold')
    ax2.set_title(f'Local Terrain  (±{window_deg}°)', fontsize=12, weight='bold')
    ax2.legend(fontsize=9)
    ax2.set_aspect('equal')

    plt.suptitle('Digital Elevation Model', fontsize=14, weight='bold')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. HORIZON PROFILE  (polar plot)
# ─────────────────────────────────────────────────────────────────────────────

def horizon_polar(horizons, az_angles, svf, lat, lon, figsize=(8, 8)):
    """
    Polar plot of the terrain horizon profile at the target location.

    The radial axis shows how high the surrounding terrain appears in each
    compass direction (in degrees above the local horizontal).  A zero value
    means a flat, unobstructed view; a larger value means a hill or crater
    wall is blocking the sky in that direction.

    Parameters
    ----------
    horizons  : 1-D array from compute_horizon_profile() (radians)
    az_angles : same azimuth array used when computing horizons (radians)
    svf       : sky-view factor (0–1)
    lat, lon  : location (degrees)
    """
    az_plot    = np.append(az_angles, az_angles[0])
    horiz_deg  = np.degrees(np.append(horizons, horizons[0]))
    horiz_pos  = np.maximum(horiz_deg, 0.0)

    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': 'polar'})

    ax.fill(az_plot, horiz_pos, alpha=0.45, color='saddlebrown',
            label='Terrain horizon')
    ax.plot(az_plot, horiz_pos, color='saddlebrown', lw=2)

    # Sky dome (reference)
    ax.fill_between(az_plot, horiz_pos, np.full_like(az_plot, 90.0),
                    alpha=0.08, color='steelblue', label='Open sky')

    # Compass: North at top, clockwise
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                      ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                      fontsize=11, weight='bold')

    max_h = max(float(np.degrees(np.max(horizons))), 5.0)
    ax.set_ylim(0, max_h * 1.25)
    ax.set_ylabel('Horizon elevation (°)', labelpad=18, fontsize=10)
    ax.set_title(
        f'Terrain Horizon Profile\n'
        f'{lat:.3f}°N, {lon:.3f}°E\n'
        f'Sky-View Factor: {svf:.3f}  '
        f'(1.0 = completely open)',
        fontsize=12, weight='bold', pad=20)
    ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.25, 1.15))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. ILLUMINATION TIMELINE
# ─────────────────────────────────────────────────────────────────────────────

def illumination_timeline(lat_deg, lon_deg, slope, aspect,
                          horizons, az_angles,
                          sunscale=1.0, albedo=0.09,
                          n_pts=1000, figsize=(13, 9)):
    """
    Show solar illumination and absorbed flux over one complete lunar day.

    Computes solar position at n_pts evenly-spaced times and checks whether
    the surface is lit (accounting for topographic shadowing from the horizon
    profile).

    Panels
    ------
    1. Solar elevation angle vs time — gold = sunlit, grey = shadow, blue = below horizon
    2. Absorbed solar flux (W/m²) vs time
    3. Running illumination percentage throughout the day

    Parameters
    ----------
    lat_deg, lon_deg : location (degrees)
    slope, aspect    : terrain geometry (radians)
    horizons         : horizon-profile array from compute_horizon_profile()
    az_angles        : azimuth array (radians)
    sunscale         : solar flux multiplier
    albedo           : fraction of sunlight reflected
    n_pts            : number of time steps to evaluate
    """
    from lunar.solar   import solar_geometry, direct_solar_flux
    from lunar.horizon import check_illumination
    from lunar.constants import LUNAR_DAY, S0

    t_arr    = np.linspace(0, LUNAR_DAY, n_pts)
    sol_elev = np.zeros(n_pts)
    sol_flux = np.zeros(n_pts)
    is_lit   = np.zeros(n_pts, dtype=bool)

    for i in range(n_pts):
        zen, az, _ = solar_geometry(lat_deg, lon_deg, t_arr[i])
        lit         = bool(check_illumination(zen, az, horizons, az_angles))
        sol_elev[i] = 90.0 - np.degrees(zen)
        is_lit[i]   = lit
        if lit:
            sol_flux[i] = float(direct_solar_flux(zen, az, slope, aspect,
                                                   sunscale, albedo))

    illum_frac = float(np.mean(is_lit))
    t_hours    = t_arr / 3600.0
    day_h      = LUNAR_DAY / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                              gridspec_kw={'hspace': 0.12})

    # ── Panel 1: Solar elevation ───────────────────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(t_hours, sol_elev, 0,
                     where=(sol_elev > 0) & is_lit,
                     alpha=0.35, color='gold', label='Sunlit')
    shadow = (sol_elev > 0) & ~is_lit
    ax1.fill_between(t_hours, sol_elev, 0,
                     where=shadow,
                     alpha=0.6, color='dimgray', label='Topographic shadow')
    ax1.fill_between(t_hours, sol_elev.clip(max=0), 0,
                     alpha=0.20, color='steelblue', label='Below horizon (night)')
    ax1.plot(t_hours, sol_elev, color='#E67E22', lw=1.8)
    ax1.axhline(0, color='black', lw=1, ls='--', alpha=0.6)
    ax1.set_ylabel('Solar Elevation (°)', fontsize=11, weight='bold')
    ax1.set_title('Solar Illumination Analysis  —  one full lunar day',
                  fontsize=13, weight='bold')
    ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Absorbed flux ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(t_hours, sol_flux, alpha=0.4, color='#E74C3C')
    ax2.plot(t_hours, sol_flux, color='#C0392B', lw=1.5, label='Absorbed solar flux')
    ax2.set_ylabel('Absorbed Flux (W/m²)', fontsize=11, weight='bold')
    ax2.set_title('Absorbed Solar Energy', fontsize=12, weight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    peak_idx = int(np.argmax(sol_flux))
    if sol_flux[peak_idx] > 0:
        ax2.annotate(f'Peak: {sol_flux[peak_idx]:.0f} W/m²',
                     xy=(t_hours[peak_idx], sol_flux[peak_idx]),
                     xytext=(t_hours[peak_idx] + day_h * 0.04,
                             sol_flux[peak_idx] * 0.82),
                     fontsize=9, color='darkred', weight='bold',
                     arrowprops=dict(arrowstyle='->', color='darkred'))

    # ── Panel 3: Cumulative illumination ───────────────────────────────────────
    ax3 = axes[2]
    running = np.cumsum(is_lit) / np.arange(1, n_pts + 1)
    ax3.plot(t_hours, running * 100.0, color='#27AE60', lw=2.5)
    ax3.fill_between(t_hours, running * 100.0, alpha=0.25, color='#27AE60')
    ax3.axhline(illum_frac * 100.0, color='#1E8449', ls='--', lw=1.5, alpha=0.9,
                label=f'Total: {illum_frac*100:.1f}% of day is sunlit')
    ax3.set_xlabel('Time in lunar day (hours)', fontsize=11, weight='bold')
    ax3.set_ylabel('Illumination (%)', fontsize=11, weight='bold')
    ax3.set_title('Cumulative Illumination Fraction', fontsize=12, weight='bold')
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f'{lat_deg:.3f}°N, {lon_deg:.3f}°E  |  '
                 f'SVF check via horizon profile',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. DENSITY PROFILE VIEWER
# ─────────────────────────────────────────────────────────────────────────────

def density_profile(z_grid, model_name, rho_surface=1100.0, h_param=0.07,
                    figsize=(8, 7)):
    """
    Plot regolith density vs depth for the selected model.

    Helps users understand what "top-layer surface density" means physically.
    The density controls how quickly heat diffuses through the soil — lower
    density (fluffier, more porous regolith) means faster temperature changes.

    Parameters
    ----------
    z_grid      : depth array from create_depth_grid() (metres)
    model_name  : 'discrete' or 'hayne_exponential'
    rho_surface : surface (top-layer) density in kg/m³ (default 1100)
    h_param     : layer-1 thickness (discrete) or scale height (Hayne) in metres
    """
    from lunar.models import density_discrete_py, density_hayne_py

    if model_name == 'discrete':
        rho = np.array([density_discrete_py(float(z), H=h_param,
                                            rho_surface=rho_surface)
                        for z in z_grid])
    else:
        rho = np.array([density_hayne_py(float(z), H=h_param,
                                          rho_surface=rho_surface)
                        for z in z_grid])

    color, ls, label = _model_style(model_name)
    z_cm = z_grid * 100.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rho, z_cm, color=color, ls=ls, lw=3, label=label)
    ax.fill_betweenx(z_cm, 0, rho, alpha=0.15, color=color)

    # Layer annotations for the discrete model
    if model_name == 'discrete':
        h_cm = h_param * 100.0
        ax.axhline(h_cm, color='gray', ls='--', lw=1.5, alpha=0.7,
                   label=f'Layer 1/2 boundary ({h_cm:.0f} cm)')
        ax.axhline(20, color='gray', ls=':', lw=1.5, alpha=0.7,
                   label='Layer 2/3 boundary (20 cm)')
        ax.text(rho_surface + 25, h_cm * 0.35,
                f'Layer 1\nFluffy surface dust\nρ = {rho_surface:.0f} kg/m³',
                fontsize=8.5, color='gray', va='center')
        ax.text(rho_surface + 25, (h_cm + 20) / 2,
                'Layer 2\nTransitional',
                fontsize=8.5, color='gray', va='center')
        ax.text(rho_surface + 25, 30,
                'Layer 3\nConsolidated regolith',
                fontsize=8.5, color='gray', va='center')

    ax.set_xlabel('Density (kg/m³)', fontsize=12, weight='bold')
    ax.set_ylabel('Depth (cm)', fontsize=12, weight='bold')
    ax.set_title(
        f'Regolith Density Profile — {label}\n'
        f'Surface density: {rho_surface:.0f} kg/m³  |  '
        f'Layer thickness: {h_param * 100:.0f} cm',
        fontsize=12, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 11. SURFACE TEMPERATURE MAP  (analytical equilibrium estimate)
# ─────────────────────────────────────────────────────────────────────────────

def surface_temperature_map(elev_m, map_res, target_lat, target_lon,
                             albedo=0.09, emissivity=0.95,
                             window_deg=5, T_simulated_max=None,
                             figsize=(14, 6)):
    """
    Estimated peak and mean surface temperature map for the local terrain.

    Uses the analytical equilibrium formula for a flat, airless body:
        T_noon  = ( (1-A)·S₀·cos(lat) / (ε·σ) )^¼    [peak noon temperature]
        T_night ≈ 95 K  (approximate lunar nighttime minimum)

    This is a fast spatial estimate — the full thermal model gives a more
    accurate single-point value for the target location.

    Parameters
    ----------
    elev_m          : full DEM elevation grid
    map_res         : pixels per degree
    target_lat/lon  : target location (degrees)
    albedo          : Bond albedo
    emissivity      : IR emissivity
    window_deg      : half-width of displayed region (degrees)
    T_simulated_max : optional — simulated surface T_max from the full model,
                      shown as an annotation at the target point
    """
    from lunar.constants import S0, sigma as _sigma

    H, W    = elev_m.shape
    pix_deg = 1.0 / map_res

    row_t = int(round((90.0 - target_lat) / pix_deg - 0.5))
    col_t = int(round(target_lon          / pix_deg - 0.5))
    row_t = max(0, min(H - 1, row_t))
    col_t = max(0, min(W - 1, col_t))

    # Local window
    win   = int(window_deg * map_res)
    r0    = max(0, row_t - win);  r1 = min(H, row_t + win)
    c0    = max(0, col_t - win);  c1 = min(W, col_t + win)

    elev_local  = elev_m[r0:r1, c0:c1]
    lats_local  = 90.0 - (np.arange(r0, r1) + 0.5) * pix_deg
    lons_local  = (np.arange(c0, c1) + 0.5) * pix_deg

    # Analytical peak temperature (flat terrain, local noon)
    cos_lat = np.maximum(0.005, np.cos(np.radians(lats_local)))
    T_noon_1d = ((1.0 - albedo) * S0 * cos_lat /
                 (emissivity * _sigma)) ** 0.25
    # Tile across longitude columns so T_noon has shape (n_rows, n_cols)
    T_noon = np.tile(T_noon_1d[:, None], (1, len(lons_local)))

    # Night-side minimum (empirical lunar average)
    T_night = np.where(np.abs(lats_local)[:, None] < 60, 95.0, 70.0)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    extent = [lons_local[0], lons_local[-1], lats_local[-1], lats_local[0]]

    # ── Panel 1: Elevation ─────────────────────────────────────────────────────
    ax1 = axes[0]
    im1 = ax1.imshow(elev_local / 1000.0, cmap='gist_earth',
                     aspect='auto', extent=extent)
    plt.colorbar(im1, ax=ax1, label='Elevation (km)', shrink=0.8)
    ax1.plot(target_lon, target_lat, 'r*', markersize=14, markeredgewidth=1,
             markeredgecolor='white', zorder=10)
    ax1.set_xlabel('Longitude (°E)', fontsize=10, weight='bold')
    ax1.set_ylabel('Latitude (°N)', fontsize=10, weight='bold')
    ax1.set_title('Terrain Elevation', fontsize=11, weight='bold')

    # ── Panel 2: Peak (noon) temperature ──────────────────────────────────────
    ax2 = axes[1]
    im2 = ax2.imshow(T_noon, cmap='hot', aspect='auto',
                     extent=extent, vmin=200, vmax=420)
    plt.colorbar(im2, ax=ax2, label='Est. Peak Temp (K)', shrink=0.8)
    annot_txt = (f'Model: {T_simulated_max:.0f} K'
                 if T_simulated_max is not None
                 else f'Est.: {T_noon[row_t - r0, col_t - c0]:.0f} K')
    ax2.plot(target_lon, target_lat, 'c*', markersize=14, markeredgewidth=1,
             markeredgecolor='black', zorder=10, label=annot_txt)
    ax2.set_xlabel('Longitude (°E)', fontsize=10, weight='bold')
    ax2.set_ylabel('Latitude (°N)', fontsize=10, weight='bold')
    ax2.set_title('Estimated Peak Temperature\n(noon, flat terrain)',
                  fontsize=11, weight='bold')
    ax2.legend(fontsize=9)

    # ── Panel 3: Day-night temperature swing ───────────────────────────────────
    ax3 = axes[2]
    T_swing = T_noon - T_night
    im3 = ax3.imshow(T_swing, cmap='RdYlBu_r', aspect='auto',
                     extent=extent, vmin=0)
    plt.colorbar(im3, ax=ax3, label='Temp Swing (K)', shrink=0.8)
    ax3.plot(target_lon, target_lat, 'g*', markersize=14, markeredgewidth=1,
             markeredgecolor='black', zorder=10)
    ax3.set_xlabel('Longitude (°E)', fontsize=10, weight='bold')
    ax3.set_ylabel('Latitude (°N)', fontsize=10, weight='bold')
    ax3.set_title('Estimated Day–Night\nTemperature Swing (K)',
                  fontsize=11, weight='bold')

    plt.suptitle(
        f'Surface Temperature Estimates — {target_lat:.3f}°N, {target_lon:.3f}°E  '
        f'(±{window_deg}°)\n'
        f'Albedo = {albedo:.2f}  |  Emissivity = {emissivity:.2f}',
        fontsize=12, weight='bold')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. PARAMETER SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_sweep(sens_results, param_name, z_grid, lat, lon,
                      model_name=None, figsize=(15, 9)):
    """
    Six-panel summary of a parameter sensitivity sweep.

    Parameters
    ----------
    sens_results : list of dicts from analysis.run_sensitivity()
    param_name   : name of the varied parameter (for axis labels)
    z_grid       : depth array (m)
    lat, lon     : location

    Panels
    ------
    1. Mean temperature profiles for every tested value (coloured by value)
    2. Surface temperature (min / mean / max) vs parameter value
    3. RMSE & bias vs parameter (if Apollo data available)
    4. Mean temperature at 1 m depth vs parameter
    5. Surface diurnal amplitude vs parameter
    6. Text summary
    """
    values     = [r['value']  for r in sens_results]
    has_apollo = any(r['errors'] is not None for r in sens_results)

    cmap   = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(values)))

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.33)

    # ── Panel 1: T profiles ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for r, col in zip(sens_results, colors):
        ax1.plot(r['stats']['T_mean'], z_grid * 100,
                 color=col, lw=2, label=f'{r["value"]:.3g}')
    if has_apollo:
        for r in sens_results:
            if r['errors']:
                err = r['errors']
                ax1.plot(err['apollo_temps'], err['apollo_depths'] * 100,
                         'ro', markersize=8, markeredgewidth=2,
                         markeredgecolor='darkred', label='Apollo', zorder=10)
                break
    ax1.set_xlabel('Temperature (K)', fontsize=11, weight='bold')
    ax1.set_ylabel('Depth (cm)', fontsize=11, weight='bold')
    ax1.set_title(f'Profiles vs {param_name}', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    # Colourbar instead of legend (cleaner for many lines)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=min(values), vmax=max(values)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label=param_name, shrink=0.8)

    # ── Panel 2: Surface T vs parameter ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    T_min  = [r['stats']['T_min'][0]  for r in sens_results]
    T_max  = [r['stats']['T_max'][0]  for r in sens_results]
    T_mean = [r['stats']['T_mean'][0] for r in sens_results]
    ax2.fill_between(values, T_min, T_max, alpha=0.25, color='orange')
    ax2.plot(values, T_max,  'r-o', lw=2, ms=7, label='Maximum')
    ax2.plot(values, T_mean, 'k-s', lw=2, ms=7, label='Mean')
    ax2.plot(values, T_min,  'b-o', lw=2, ms=7, label='Minimum')
    ax2.set_xlabel(param_name, fontsize=11, weight='bold')
    ax2.set_ylabel('Surface Temperature (K)', fontsize=11, weight='bold')
    ax2.set_title('Surface Temperature Response', fontsize=12, weight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: RMSE / bias vs parameter (or placeholder) ────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if has_apollo:
        rmse_v = [r['errors']['rmse'] for r in sens_results if r['errors']]
        bias_v = [r['errors']['bias'] for r in sens_results if r['errors']]
        v_ap   = [r['value']          for r in sens_results if r['errors']]
        ax3b   = ax3.twinx()
        l1,    = ax3.plot(v_ap, rmse_v, 'ro-', lw=2, ms=7, label='RMSE')
        l2,    = ax3b.plot(v_ap, bias_v, 'bs-', lw=2, ms=7, label='Bias')
        # Optimal value
        best_i = int(np.argmin(rmse_v))
        ax3.axvline(v_ap[best_i], color='green', ls='--', lw=2, alpha=0.7)
        ax3.text(v_ap[best_i], max(rmse_v) * 0.97,
                 f'  Opt={v_ap[best_i]:.3g}',
                 fontsize=9, color='green', weight='bold', va='top')
        ax3.set_xlabel(param_name, fontsize=11, weight='bold')
        ax3.set_ylabel('RMSE (K)', fontsize=11, weight='bold', color='red')
        ax3b.set_ylabel('Bias (K)', fontsize=11, weight='bold', color='blue')
        ax3.tick_params(axis='y', labelcolor='red')
        ax3b.tick_params(axis='y', labelcolor='blue')
        ax3.set_title('Accuracy vs Parameter', fontsize=12, weight='bold')
        ax3.legend([l1, l2], ['RMSE', 'Bias'], fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Apollo data\nfor validation',
                 ha='center', va='center', fontsize=13, weight='bold',
                 transform=ax3.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
        ax3.set_axis_off()

    # ── Panel 4: Deep temperature (1 m) ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    T_1m = [float(np.interp(1.0, z_grid, r['stats']['T_mean']))
            for r in sens_results]
    ax4.plot(values, T_1m, 'go-', lw=2.5, ms=9, mew=2, mec='darkgreen')
    ax4.fill_between(values, T_1m, alpha=0.2, color='green')
    ax4.set_xlabel(param_name, fontsize=11, weight='bold')
    ax4.set_ylabel('Temperature at 1 m (K)', fontsize=11, weight='bold')
    ax4.set_title('Deep Temperature Sensitivity', fontsize=12, weight='bold')
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Surface amplitude ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    amp  = [r['stats']['T_amplitude'][0] for r in sens_results]
    ax5.plot(values, amp, 'mo-', lw=2.5, ms=9, mew=2, mec='darkmagenta')
    ax5.fill_between(values, amp, alpha=0.2, color='magenta')
    ax5.set_xlabel(param_name, fontsize=11, weight='bold')
    ax5.set_ylabel('Diurnal Amplitude (K)', fontsize=11, weight='bold')
    ax5.set_title('Surface Amplitude Sensitivity', fontsize=12, weight='bold')
    ax5.grid(True, alpha=0.3)

    # ── Panel 6: Text summary ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    lines = [
        f'Parameter: {param_name}',
        f'Range: {min(values):.4g} → {max(values):.4g}',
        f'Points tested: {len(values)}',
        '',
        'Surface T:',
        f'  Max range: {min(T_max):.1f} – {max(T_max):.1f} K',
        f'  Min range: {min(T_min):.1f} – {max(T_min):.1f} K',
        f'  Change:    {max(T_max) - min(T_max):.1f} K',
        '',
        'Temperature at 1 m:',
        f'  Range: {min(T_1m):.1f} – {max(T_1m):.1f} K',
        f'  Sensitivity: {max(T_1m) - min(T_1m):.2f} K',
    ]
    if has_apollo:
        lines += [
            '',
            f'Optimal value: {v_ap[best_i]:.4g}',
            f'Best RMSE: {min(rmse_v):.3f} K',
        ]
    ax6.text(0.05, 0.5, '\n'.join(lines), fontsize=9.5, family='monospace',
             va='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4, pad=0.8))

    plt.suptitle(f'Parameter Sensitivity: {param_name}\n' +
                 _subtitle(lat, lon, model_name),
                 fontsize=13, weight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. BATCH SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def batch_summary(batch_results, z_grid, figsize=(15, 10)):
    """
    Multi-panel summary of batch-processing results.

    Panels
    ------
    1. Surface T range (min/max bar chart per location)
    2. Mean temperature at 0 cm and 50 cm per location
    3. RMSE vs Apollo (for locations near Apollo sites)
    4. All mean T profiles overlaid

    Parameters
    ----------
    batch_results : list from analysis.run_batch()
    z_grid        : depth grid (m)
    """
    names = [r['name'] for r in batch_results]
    n     = len(names)
    xs    = np.arange(n)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30)

    # ── Panel 1: Surface temperature range ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    T_min_surf = [r['stats']['T_min'][0]  for r in batch_results]
    T_max_surf = [r['stats']['T_max'][0]  for r in batch_results]
    T_mean_surf= [r['stats']['T_mean'][0] for r in batch_results]
    ax1.bar(xs, T_max_surf, color='#E74C3C', alpha=0.7, label='Max')
    ax1.bar(xs, T_mean_surf, color='#5D6D7E', alpha=0.7, label='Mean')
    ax1.bar(xs, T_min_surf, color='#2471A3', alpha=0.7, label='Min')
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Surface Temperature (K)', fontsize=11, weight='bold')
    ax1.set_title('Surface Temperature Range', fontsize=12, weight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # ── Panel 2: Temperature at 0 and 50 cm ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    T_50cm = [float(np.interp(0.5, z_grid, r['stats']['T_mean']))
              for r in batch_results]
    ax2.plot(xs, T_mean_surf, 'ro-', lw=2, ms=7, label='Surface (0 cm)')
    ax2.plot(xs, T_50cm, 'bs-', lw=2, ms=7, label='Depth 50 cm')
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Mean Temperature (K)', fontsize=11, weight='bold')
    ax2.set_title('Mean Temperature at Key Depths', fontsize=12, weight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: RMSE where Apollo data exists ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    apollo_names = [r['name'] for r in batch_results if r['errors']]
    rmse_vals    = [r['errors']['rmse'] for r in batch_results if r['errors']]
    bias_vals    = [r['errors']['bias'] for r in batch_results if r['errors']]

    if apollo_names:
        xA = np.arange(len(apollo_names))
        ax3.bar(xA - 0.2, rmse_vals, width=0.35, color='#E74C3C',
                alpha=0.8, label='RMSE')
        ax3.bar(xA + 0.2, bias_vals, width=0.35, color='#2471A3',
                alpha=0.8, label='Bias')
        ax3.axhline(0, color='black', lw=1)
        ax3.set_xticks(xA)
        ax3.set_xticklabels(apollo_names, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Error (K)', fontsize=11, weight='bold')
        ax3.set_title('Apollo Validation Errors', fontsize=12, weight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No Apollo sites\nin batch', ha='center', va='center',
                 fontsize=13, weight='bold', transform=ax3.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
        ax3.axis('off')

    # ── Panel 4: All T profiles overlaid ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    cmap   = plt.cm.tab10
    for i, r in enumerate(batch_results):
        col = cmap(i % 10)
        ax4.plot(r['stats']['T_mean'], z_grid * 100, color=col, lw=2,
                 label=r['name'])
    ax4.set_xlabel('Mean Temperature (K)', fontsize=11, weight='bold')
    ax4.set_ylabel('Depth (cm)', fontsize=11, weight='bold')
    ax4.set_title('All Mean Temperature Profiles', fontsize=12, weight='bold')
    ax4.legend(fontsize=8, ncol=max(1, n // 5), loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()

    plt.suptitle('Batch Processing Summary', fontsize=14, weight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
