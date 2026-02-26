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
diurnal_cycles()      — Temperature vs time at multiple depths.
heatmap()             — 2-D temperature field (depth × time).
apollo_comparison()   — Model profile vs Apollo HFE measurements.
model_comparison()    — Two or more models side-by-side.
sensitivity_sweep()   — Parameter sensitivity: 6-panel summary.
batch_summary()       — Grid of bar/line plots for batch results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ── Shared style ──────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')

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
    ax1.plot(a_temps, a_depths * 100, 'o', color='green',
             markersize=11, markeredgewidth=2, markeredgecolor='darkgreen',
             label=f'{site_name} measured', zorder=5)

    ax1.set_xlabel('Temperature (K)', fontsize=12, weight='bold')
    ax1.set_ylabel('Depth (cm)', fontsize=12, weight='bold')
    ax1.set_title(f'{site_name} Validation — RMSE: {errors["rmse"]:.2f} K  '
                  f'Bias: {errors["bias"]:+.2f} K  MAE: {errors["mae"]:.2f} K',
                  fontsize=13, weight='bold')
    ax1.legend(fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # ── Panel 2: Residuals bar chart ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    bar_colors = ['#E74C3C' if r > 0 else '#2471A3' for r in residuals]
    ax2.barh(a_depths * 100, residuals, height=4,
             color=bar_colors, alpha=0.75, edgecolor='black')
    ax2.axvline(0, color='black', linewidth=1.5)
    ax2.set_xlabel('Residual (K)',  fontsize=11, weight='bold')
    ax2.set_ylabel('Depth (cm)',   fontsize=11, weight='bold')
    ax2.set_title('Model − Measured', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
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
