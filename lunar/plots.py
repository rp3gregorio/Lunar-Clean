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
diurnal_cycles()           — Temperature vs time at multiple depths.
heatmap()                  — 2-D temperature field (depth × time).
apollo_comparison()        — Model profile vs single Apollo HFE site.
dual_apollo_comparison()   — Both Apollo 15 & 17 side-by-side.
model_comparison()         — Two or more models side-by-side.
sensitivity_sweep()        — Parameter sensitivity: 6-panel summary.
batch_summary()            — Grid of bar/line plots for batch results.
heat_flux_profile()        — Geothermal heat flux Q(z) = k·dT/dz.
amplitude_decay()          — Diurnal amplitude vs depth + skin-depth fit.
combined_heat_flow()       — A15 vs A17 heat-flow bar chart summary.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ── Shared style — publication quality ───────────────────────────────────────
import matplotlib as _mpl

# Use a clean base that renders well in both notebooks and PDFs
plt.style.use('default')
_mpl.rcParams.update({
    # ── Fonts ──────────────────────────────────────────────────────────────
    'font.family':          'DejaVu Sans',
    'font.size':            11,
    'axes.titlesize':       12,
    'axes.titleweight':     'bold',
    'axes.labelsize':       11,
    'axes.labelweight':     'bold',
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
    'legend.fontsize':      9,
    'legend.title_fontsize': 9,
    # ── Axes ───────────────────────────────────────────────────────────────
    'axes.facecolor':       'white',
    'axes.edgecolor':       '#2f2f2f',
    'axes.linewidth':       0.9,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.grid':            True,
    'grid.color':           '#e0e0e0',
    'grid.linewidth':       0.6,
    'grid.linestyle':       '-',
    # ── Figure ─────────────────────────────────────────────────────────────
    'figure.facecolor':     'white',
    'figure.dpi':           130,
    'figure.constrained_layout.use': False,   # we call tight_layout manually
    # ── Lines & markers ────────────────────────────────────────────────────
    'lines.linewidth':      2.0,
    'lines.solid_capstyle': 'round',
    # ── Legend ─────────────────────────────────────────────────────────────
    'legend.framealpha':    0.92,
    'legend.edgecolor':     '#cccccc',
    'legend.borderpad':     0.5,
    # ── Colour cycle — ColorBrewer Set1 (distinguishable at B&W too) ───────
    'axes.prop_cycle': _mpl.cycler(color=[
        '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
        '#FF7F00', '#A65628', '#F781BF', '#999999',
    ]),
    # ── Ticks ──────────────────────────────────────────────────────────────
    'xtick.direction':      'out',
    'ytick.direction':      'out',
    'xtick.major.size':     4,
    'ytick.major.size':     4,
    'xtick.minor.size':     2,
    'ytick.minor.size':     2,
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

# ── Sensor-type marker shapes (consistent across all plots) ────────────────
# TG = gradient-bridge thermocouple → circle
# TR = reference thermocouple        → square
# TC = cable thermocouple            → triangle-up
_STYPE_MARKER = {'TG': 'o', 'TR': 's', 'TC': '^'}
_STYPE_SIZE   = {'TG': 5,   'TR': 5,   'TC': 5}
_STYPE_LABEL  = {
    'TG': 'TG gradient',
    'TR': 'TR reference',
    'TC': 'TC cable',
}

# ── Shared legend keyword defaults ─────────────────────────────────────────
_LEG_KW = dict(framealpha=0.92, edgecolor='#cccccc', borderpad=0.5,
               handlelength=1.8, fontsize=9)

_APOLLO_SITES  = ['Apollo 15', 'Apollo 17']
_APOLLO_COLORS = ['#1A5276', '#7D3C98']   # navy blue, purple


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
                   figsize=(11, 6)):
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
    from lunar.constants import LUNAR_DAY

    # ColorBrewer-inspired sequential palette: surface = warm, deep = cool
    depth_colors = [
        '#D73027', '#FC8D59', '#FEE090',   # warm: surface/shallow
        '#ABD9E9', '#74ADD1', '#4575B4', '#313695', '#1A1A6C',  # cool: deep
    ]

    sorted_cycles = sorted(cycles.items())
    day_h = LUNAR_DAY / 3600.0
    half  = day_h / 2.0

    fig, ax = plt.subplots(figsize=figsize)

    # Night shading (approximate: first/last quarter of day)
    ax.axvspan(0,    half * 0.48, color='#1a1a2e', alpha=0.08, zorder=0, label='_night')
    ax.axvspan(half * 1.52, day_h, color='#1a1a2e', alpha=0.08, zorder=0, label='_night')
    ax.axvline(half, color='#888', lw=0.8, ls='--', alpha=0.6, zorder=1)
    ax.text(half + day_h * 0.01, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 400,
            'Noon', fontsize=8, color='#666', va='top', ha='left')

    for i, (depth, data) in enumerate(sorted_cycles):
        color = depth_colors[i % len(depth_colors)]
        d_cm  = depth * 100
        # Thicker line for surface; thinner for deep
        lw   = 2.5 if d_cm < 1 else max(1.2, 2.5 - i * 0.2)
        zord = 10 - i
        ax.plot(data['time_h'], data['temperature'],
                color=color, linewidth=lw, zorder=zord,
                label=f'{d_cm:.0f} cm' if d_cm >= 1 else 'Surface (0 cm)')

    ax.set_xlabel('Time in lunar day (hours)')
    ax.set_ylabel('Temperature (K)')

    extra = f'  ·  SUNSCALE = {sunscale:.2f}' if sunscale is not None else ''
    ax.set_title(
        f'Diurnal Temperature Cycles — {_subtitle(lat, lon, model_name)}{extra}')

    # Legend: compact, outside right edge so it doesn't cover curves
    ax.legend(title='Depth', loc='center left', bbox_to_anchor=(1.01, 0.5),
              **_LEG_KW)
    ax.set_xlim(0, day_h)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1b. DIURNAL PROBE vs MODELS  — actual Apollo readings vs two model predictions
# ─────────────────────────────────────────────────────────────────────────────

def diurnal_probe_vs_models(probe_diurnal, cycles_model, cycles_hayne,
                             site_name, lat, lon,
                             model_name='This model',
                             hayne_name='Hayne 2017',
                             max_panels=9,
                             amp_threshold=0.05,
                             figsize=None):
    """
    Compare Apollo HFE measured diurnal temperature variations with two model
    predictions (the user's model and the Hayne 2017 model).

    Each sub-panel shows one sensor depth available in the Apollo data.
    Both model diurnal cycles are interpolated to the same depth.
    The three curves are zero-mean normalised (T − <T>) so amplitude and phase
    can be compared without absolute-temperature offsets.

    Parameters
    ----------
    probe_diurnal : dict from hfe_loader.get_probe_diurnal_cycle() —
                    {depth_cm: {'time_h', 'T_anom', 'T_mean', 'sensor', 'stype'}}
    cycles_model  : dict from analysis.get_diurnal_cycles() for the user's model
    cycles_hayne  : dict from analysis.get_diurnal_cycles() for the Hayne model
    site_name     : 'Apollo 15' or 'Apollo 17'  (for axis labels)
    lat, lon      : site coordinates
    model_name    : label for the user's model curve
    hayne_name    : label for the Hayne model curve
    max_panels    : maximum number of depth panels to show (default 9);
                    the depths with the largest diurnal amplitude are kept
    amp_threshold : minimum peak-to-peak anomaly amplitude (K) to include a
                    depth panel; depths below this are silently dropped
    figsize       : figure size; if None, computed from the panel count

    Returns
    -------
    matplotlib.figure.Figure
    """
    import datetime as _dt
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    from lunar.constants import LUNAR_DAY

    if not probe_diurnal:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title('No probe diurnal data available')
        return fig

    day_h = LUNAR_DAY / 3600.0

    # ── Filter to depths with a meaningful diurnal signal ────────────────────
    def _amp(d_cm):
        v = probe_diurnal[d_cm]['T_anom']
        return float(v.max() - v.min())

    all_depths = sorted(probe_diurnal.keys())
    active = [d for d in all_depths if _amp(d) >= amp_threshold]
    # Sort by descending amplitude, keep top max_panels, then re-sort by depth
    active = sorted(active, key=_amp, reverse=True)[:max_panels]
    probe_depths_cm = sorted(active)

    if not probe_depths_cm:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f'No depths with amplitude ≥ {amp_threshold} K')
        return fig

    # ── Layout — size panels to be readable ──────────────────────────────────
    def _closest_model_depth(depth_m, cycles):
        if not cycles:
            return None
        return min(cycles.keys(), key=lambda d: abs(d - depth_m))

    n = len(probe_depths_cm)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (5.5 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              sharex=False, squeeze=False)
    axes_flat = axes.flatten()

    # Helper: extract zero-mean model anomaly at depth depth_m
    def _model_anom(cycles, depth_m):
        if not cycles:
            return None, None
        d_key = _closest_model_depth(depth_m, cycles)
        if d_key is None:
            return None, None
        t_h = cycles[d_key]['time_h']
        T   = np.asarray(cycles[d_key]['temperature'], dtype=float)
        return t_h, T - float(np.mean(T))

    _leg_seen = {}   # {label → handle} — deduplicated across panels for shared legend

    for ax_idx, d_cm in enumerate(probe_depths_cm):
        ax = axes_flat[ax_idx]
        entry  = probe_diurnal[d_cm]
        stype  = entry.get('stype', 'TG')
        sensor = entry.get('sensor', f'{d_cm} cm')
        t_ph   = entry['time_h']
        T_anom = entry['T_anom']

        # Convert phase hours to UTC datetimes if ref_utc is available
        ref_utc = entry.get('ref_utc')
        if ref_utc is not None:
            t_utc_raw = [ref_utc + _dt.timedelta(hours=float(h)) for h in t_ph]
        else:
            t_utc_raw = list(t_ph)   # fallback: plain hours

        # ── Probe data ───────────────────────────────────────────────────────
        # Thin grey scatter for individual readings, bold marker for binned mean
        # Bin into ~24 equal-width bins (one per ~30-hour bin ≈ 1 lunar "hour")
        n_bins    = 48
        bin_edges = np.linspace(0, day_h, n_bins + 1)
        bin_mids  = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_idx   = np.searchsorted(bin_edges, t_ph, side='right') - 1
        bin_idx   = np.clip(bin_idx, 0, n_bins - 1)
        bin_mean  = np.full(n_bins, np.nan)
        bin_std   = np.full(n_bins, np.nan)
        for b in range(n_bins):
            vals = T_anom[bin_idx == b]
            if len(vals) >= 3:
                bin_mean[b] = np.mean(vals)
                bin_std[b]  = np.std(vals)

        # Convert bin_mids to UTC datetimes if possible
        if ref_utc is not None:
            bin_mids_utc = [ref_utc + _dt.timedelta(hours=float(h))
                            for h in bin_mids]
        else:
            bin_mids_utc = list(bin_mids)

        # Raw scatter (very light)
        ax.scatter(t_utc_raw, T_anom, s=1, alpha=0.08, color='#888888',
                   rasterized=True, zorder=1)
        # Binned mean ± std
        good = ~np.isnan(bin_mean)
        mk   = _STYPE_MARKER.get(stype, 'o')
        ms   = _STYPE_SIZE.get(stype, 5)
        good_mids = [bin_mids_utc[i] for i in range(n_bins) if good[i]]
        ax.errorbar(good_mids, bin_mean[good], yerr=bin_std[good],
                    fmt=mk, markersize=ms, color='#2C3E50',
                    ecolor='#888888', elinewidth=0.8, capsize=2,
                    linewidth=1.2, zorder=4,
                    label=f'Apollo {_STYPE_LABEL.get(stype, stype)}')

        # ── Model 1 (user's model) ───────────────────────────────────────────
        t1, A1 = _model_anom(cycles_model, d_cm / 100.0)
        if t1 is not None:
            if ref_utc is not None:
                t1_utc = [ref_utc + _dt.timedelta(hours=float(h)) for h in t1]
            else:
                t1_utc = list(t1)
            ax.plot(t1_utc, A1, lw=2.0, color='#2471A3', zorder=5,
                    label=model_name)

        # ── Model 2 (Hayne) ──────────────────────────────────────────────────
        t2, A2 = _model_anom(cycles_hayne, d_cm / 100.0)
        if t2 is not None:
            if ref_utc is not None:
                t2_utc = [ref_utc + _dt.timedelta(hours=float(h)) for h in t2]
            else:
                t2_utc = list(t2)
            ax.plot(t2_utc, A2, lw=2.0, color='#E67E22', ls='--', zorder=5,
                    label=hayne_name)

        # ── Night shading ────────────────────────────────────────────────────
        half = day_h / 2.0
        if ref_utc is not None:
            night_start_1 = ref_utc
            night_end_1   = ref_utc + _dt.timedelta(hours=half * 0.48)
            night_start_2 = ref_utc + _dt.timedelta(hours=half * 1.52)
            night_end_2   = ref_utc + _dt.timedelta(hours=day_h)
            ax.axvspan(night_start_1, night_end_1,
                       color='#1a1a2e', alpha=0.07, zorder=0)
            ax.axvspan(night_start_2, night_end_2,
                       color='#1a1a2e', alpha=0.07, zorder=0)
        else:
            ax.axvspan(0,           half * 0.48, color='#1a1a2e', alpha=0.07, zorder=0)
            ax.axvspan(half * 1.52, day_h,       color='#1a1a2e', alpha=0.07, zorder=0)
        ax.axhline(0, color='#888', lw=0.7, ls=':')

        ax.set_title(f'{d_cm} cm  —  {sensor}', fontsize=10, weight='bold')
        if ax_idx % ncols == 0:
            ax.set_ylabel('T anomaly (K)', fontsize=9)
        if ax_idx >= (nrows - 1) * ncols:
            ax.set_xlabel('UTC Date' if ref_utc is not None else 'Hours within lunar day',
                          fontsize=9)
        if ref_utc is not None:
            ax.xaxis_date()
            loc = AutoDateLocator(minticks=3, maxticks=5)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right',
                     fontsize=8)
        for _h, _l in zip(*ax.get_legend_handles_labels()):
            if _l not in _leg_seen:
                _leg_seen[_l] = _h

    # Hide unused panels
    for ax_idx in range(n, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(
        f'{site_name} — Diurnal Temperature Cycle: Apollo Probe vs Models\n'
        f'Lat {lat:.2f}°  ·  Lon {lon:.2f}°',
        fontsize=11, weight='bold', y=1.01,
    )
    plt.tight_layout()

    # Shared legend centred below all panels
    if _leg_seen:
        fig.legend(
            list(_leg_seen.values()),
            list(_leg_seen.keys()),
            loc='lower center',
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(len(_leg_seen), 5),
            **{**_LEG_KW, 'fontsize': 9},
        )
        fig.subplots_adjust(bottom=0.10)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1c. DIURNAL ABSOLUTE OVERLAY — single panel, T(K) vs time, Apollo + 2 models
# ─────────────────────────────────────────────────────────────────────────────

def diurnal_absolute_vs_models(probe_diurnal, cycles_model, cycles_hayne,
                                site_name, lat, lon,
                                depths_cm=None,
                                model_name='This model',
                                hayne_name='Hayne 2017',
                                figsize=(11, 6)):
    """
    Single-panel absolute-temperature diurnal overlay.

    Plots T (K) vs time within the lunar day for each selected depth,
    overlaying three curves: Apollo HFE binned mean, the user's model,
    and the Hayne 2017 model.  Apollo readings are phase-aligned to the
    model by matching peak temperatures, removing the arbitrary phase
    offset between the phase-folded observations and the model clock.

    Parameters
    ----------
    probe_diurnal : dict from hfe_loader.get_probe_diurnal_cycle()
    cycles_model  : dict from analysis.get_diurnal_cycles() — user model
    cycles_hayne  : dict from analysis.get_diurnal_cycles() — Hayne model
    site_name     : 'Apollo 15' or 'Apollo 17'
    lat, lon      : site coordinates (degrees)
    depths_cm     : list of sensor depths (cm) to plot; if None the 3
                    shallowest sensors with a diurnal swing > 0.2 K are used
    model_name    : label for the user's model
    hayne_name    : label for the Hayne model
    figsize       : figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    from lunar.constants import LUNAR_DAY

    if not probe_diurnal:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title('No probe diurnal data available')
        return fig

    day_h = LUNAR_DAY / 3600.0

    # ── Choose depths ─────────────────────────────────────────────────────────
    if depths_cm is None:
        # Shallowest sensors with a detectable diurnal swing
        candidates = sorted([
            d for d, v in probe_diurnal.items()
            if (v['T_anom'].max() - v['T_anom'].min()) > 0.2
        ])
        depths_cm = candidates[:3] if candidates else sorted(probe_diurnal.keys())[:3]

    # Depth → colour: warm (red→orange) for shallow, cool (teal→blue) for deep
    _palette = ['#C0392B', '#E67E22', '#27AE60', '#2471A3', '#8E44AD']

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _closest_model_depth(depth_m, cycles):
        if not cycles:
            return None
        return min(cycles.keys(), key=lambda d: abs(d - depth_m))

    def _binned_mean(t_ph, T_ph, n_bins=48):
        """Return (bin_mids, bin_mean, bin_std) over one lunar day."""
        edges = np.linspace(0, day_h, n_bins + 1)
        mids  = 0.5 * (edges[:-1] + edges[1:])
        idx   = np.clip(np.searchsorted(edges, t_ph, side='right') - 1, 0, n_bins - 1)
        mean  = np.full(n_bins, np.nan)
        std   = np.full(n_bins, np.nan)
        for b in range(n_bins):
            vals = T_ph[idx == b]
            if len(vals) >= 3:
                mean[b] = np.mean(vals)
                std[b]  = np.std(vals)
        return mids, mean, std

    import datetime as _dt

    def _phase_shift_xcorr(t_obs_bins, T_obs_mean, t_model, T_model, day_h):
        """
        Return phase shift (hours) to add to Apollo times to align with model.
        Uses cross-correlation on binned data for robustness.
        """
        n = 120  # number of bins for cross-correlation
        bins = np.linspace(0, day_h, n + 1)
        mids = 0.5 * (bins[:-1] + bins[1:])

        # Interpolate model onto uniform grid
        T_mod_grid = np.interp(mids, t_model % day_h, T_model - np.mean(T_model))

        # Resample Apollo binned mean onto same grid (use only good bins)
        good = ~np.isnan(T_obs_mean)
        if good.sum() < 4:
            return 0.0
        T_obs_grid = np.interp(mids, t_obs_bins[good],
                               T_obs_mean[good] - np.nanmean(T_obs_mean[good]))

        # Circular cross-correlation: find lag that maximises correlation
        corr = np.correlate(np.tile(T_mod_grid, 2), T_obs_grid, mode='valid')[:n]
        best_lag = int(np.argmax(corr))
        shift = mids[best_lag]  # hours to add to Apollo times
        # Wrap to (-day_h/2, +day_h/2)
        shift = ((shift + day_h / 2) % day_h) - day_h / 2
        return shift

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    # Determine whether any depth has ref_utc available (use first found)
    _any_ref_utc = None
    for _d in depths_cm:
        if _d in probe_diurnal and probe_diurnal[_d].get('ref_utc') is not None:
            _any_ref_utc = probe_diurnal[_d]['ref_utc']
            break

    half = day_h / 2.0
    if _any_ref_utc is not None:
        ax.axvspan(_any_ref_utc,
                   _any_ref_utc + _dt.timedelta(hours=half * 0.48),
                   color='#1a1a2e', alpha=0.08, zorder=0)
        ax.axvspan(_any_ref_utc + _dt.timedelta(hours=half * 1.52),
                   _any_ref_utc + _dt.timedelta(hours=day_h),
                   color='#1a1a2e', alpha=0.08, zorder=0)
        ax.axvline(_any_ref_utc + _dt.timedelta(hours=half),
                   color='#888', lw=0.8, ls='--', alpha=0.6)
        ax.text(0.52, 0.98, 'Noon',
                transform=ax.get_xaxis_transform(),
                fontsize=8, color='#666', va='top')
    else:
        ax.axvspan(0,           half * 0.48, color='#1a1a2e', alpha=0.08, zorder=0)
        ax.axvspan(half * 1.52, day_h,       color='#1a1a2e', alpha=0.08, zorder=0)
        ax.axvline(half, color='#888', lw=0.8, ls='--', alpha=0.6)
        ax.text(half + day_h * 0.01, 0.98, 'Noon',
                transform=ax.get_xaxis_transform(),
                fontsize=8, color='#666', va='top')

    legend_handles = []

    for i, d_cm in enumerate(depths_cm):
        if d_cm not in probe_diurnal:
            continue
        color  = _palette[i % len(_palette)]
        depth_m = d_cm / 100.0

        entry   = probe_diurnal[d_cm]
        t_ph    = entry['time_h']
        T_raw   = entry['T_raw']
        ref_utc = entry.get('ref_utc')
        stype   = entry.get('stype', 'TG')

        # ── Bin Apollo data ───────────────────────────────────────────────
        mids, bmean, bstd = _binned_mean(t_ph, T_raw)
        good = ~np.isnan(bmean)

        # ── Model at this depth ───────────────────────────────────────────
        d_key_m = _closest_model_depth(depth_m, cycles_model)
        t_mod   = np.asarray(cycles_model[d_key_m]['time_h'])      if cycles_model else None
        T_mod   = np.asarray(cycles_model[d_key_m]['temperature']) if cycles_model else None

        # Phase-align Apollo to model using cross-correlation
        shift = (_phase_shift_xcorr(mids, bmean, t_mod, T_mod, day_h)
                 if t_mod is not None else 0.0)
        t_aligned = (mids + shift) % day_h
        sort_idx  = np.argsort(t_aligned)
        t_al_s    = t_aligned[sort_idx]
        bm_s      = bmean[sort_idx]
        bs_s      = bstd[sort_idx]
        good_s    = ~np.isnan(bm_s)

        # Convert Apollo aligned times to UTC if possible
        if ref_utc is not None:
            t_al_plot = [ref_utc + _dt.timedelta(hours=float(h)) for h in t_al_s]
        else:
            t_al_plot = list(t_al_s)

        label_d = f'{d_cm} cm'

        # Apollo shaded band + mean line
        if good_s.sum() > 1:
            t_al_good = [t_al_plot[j] for j in range(len(t_al_s)) if good_s[j]]
            ax.fill_between(t_al_good,
                            bm_s[good_s] - bs_s[good_s],
                            bm_s[good_s] + bs_s[good_s],
                            color=color, alpha=0.18, zorder=2)
            h_apollo, = ax.plot(t_al_good, bm_s[good_s],
                                color=color, lw=1.6, ls='-',
                                marker=_STYPE_MARKER.get(stype, 'o'), markersize=3, zorder=3,
                                label=f'Apollo {label_d} ({_STYPE_LABEL.get(stype, stype)})')
            legend_handles.append(h_apollo)

        # User model solid line
        if t_mod is not None:
            if ref_utc is not None:
                model_ref_utc = ref_utc + _dt.timedelta(hours=float(shift))
                t_mod_plot = [model_ref_utc + _dt.timedelta(hours=float(h))
                              for h in t_mod]
            else:
                t_mod_plot = list(t_mod)
            h_mod, = ax.plot(t_mod_plot, T_mod,
                             color=color, lw=2.2, ls='-', zorder=5,
                             label=f'{model_name} {label_d}')
            legend_handles.append(h_mod)

        # Hayne dashed line
        if cycles_hayne:
            d_key_h = _closest_model_depth(depth_m, cycles_hayne)
            t_hay   = np.asarray(cycles_hayne[d_key_h]['time_h'])
            T_hay   = np.asarray(cycles_hayne[d_key_h]['temperature'])
            if ref_utc is not None:
                t_hay_plot = [ref_utc + _dt.timedelta(hours=float(h))
                              for h in t_hay]
            else:
                t_hay_plot = list(t_hay)
            h_hay, = ax.plot(t_hay_plot, T_hay,
                             color=color, lw=2.0, ls='--', alpha=0.75, zorder=4,
                             label=f'{hayne_name} {label_d}')
            legend_handles.append(h_hay)

    if _any_ref_utc is not None:
        ax.xaxis_date()
        fig.autofmt_xdate(rotation=30, ha='right')
        ax.set_xlabel('UTC Date')
    else:
        ax.set_xlabel('Time in lunar day (hours)')
        ax.set_xlim(0, day_h)

    ax.set_ylabel('Temperature (K)')
    ax.set_title(
        f'{site_name} — Diurnal Temperature: Apollo vs Models\n'
        f'{lat:.3f}°N, {lon:.3f}°E  ·  solid = {model_name}  ·  '
        f'dashed = {hayne_name}  ·  markers = Apollo HFE',
        fontsize=11,
    )

    # Build a compact legend: group by depth
    ax.legend(handles=legend_handles, loc='upper right',
              ncol=len(depths_cm),
              **{**_LEG_KW, 'fontsize': 8.5, 'handlelength': 2.0})
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEMPERATURE HEATMAP (depth × time)
# ─────────────────────────────────────────────────────────────────────────────

def heatmap(T_profile, t_arr, z_grid, lat, lon, model_name=None,
            depth_limit=1.5, colormap='inferno', show_contours=True,
            figsize=(12, 6)):
    """
    2-D temperature field as a function of depth and time.

    Parameters
    ----------
    depth_limit  : only show the top *depth_limit* metres
    colormap     : matplotlib colormap name ('inferno' is standard for T)
    show_contours: overlay iso-temperature contour lines
    """
    from lunar.constants import LUNAR_DAY

    t_start  = t_arr[-1] - LUNAR_DAY
    idx_t    = np.where(t_arr >= t_start)[0]
    idx_z    = np.where(z_grid <= depth_limit)[0]

    t_hours = (t_arr[idx_t] - t_start) / 3600.0
    z_cm    = z_grid[idx_z] * 100.0
    T_sub   = T_profile[np.ix_(idx_t, idx_z)]

    fig, ax = plt.subplots(figsize=figsize)

    # pcolormesh renders crisper than contourf and avoids level-quantisation
    pm = ax.pcolormesh(t_hours, z_cm, T_sub.T,
                       cmap=colormap, shading='gouraud',
                       vmin=np.nanmin(T_sub), vmax=np.nanmax(T_sub))

    if show_contours:
        # Coarser grid for contours (avoid clutter)
        n_t, n_z = len(t_hours), len(z_cm)
        step_t = max(1, n_t // 200)
        step_z = max(1, n_z // 80)
        cs = ax.contour(t_hours[::step_t], z_cm[::step_z],
                        T_sub[::step_t, ::step_z].T,
                        levels=8, colors='white',
                        linewidths=0.6, alpha=0.45)
        ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d K', use_clabeltext=True)

    cbar = plt.colorbar(pm, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label('Temperature (K)')
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel('Time in lunar day (hours)')
    ax.set_ylabel('Depth (cm)')
    ax.set_title(f'Subsurface Temperature — Depth × Time\n{_subtitle(lat, lon, model_name)}')
    ax.invert_yaxis()
    ax.set_xlim(t_hours[0], t_hours[-1])
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
    _msize = {'TG': 9, 'TR': 8, 'TC': 7}
    _malph = {'TG': 1.0, 'TR': 0.75, 'TC': 0.55}
    _mlbl  = {'TG': 'TG  (gradient bridge)', 'TR': 'TR  (reference TC)',
               'TC': 'TC  (cable, diurnal zone)'}
    _st_arr = np.array(errors.get('apollo_sensor_types', ['TG'] * len(a_depths)))
    for stype in ('TG', 'TR', 'TC'):
        _smask = _st_arr == stype
        if _smask.any():
            ax1.plot(a_temps[_smask], a_depths[_smask] * 100,
                     _STYPE_MARKER.get(stype, 'o'), color='#1A5276',
                     markersize=_msize[stype],
                     markeredgewidth=1.4, markeredgecolor='white',
                     alpha=_malph[stype], zorder=5, label=_mlbl[stype])

    max_meas_cm = float(np.max(a_depths * 100))
    y_max_cm    = min(320.0, max(150.0, max_meas_cm * 1.6))
    mask = z_grid * 100 <= y_max_cm * 1.05

    ax1.set_xlabel('Temperature (K)', fontsize=12, weight='bold')
    ax1.set_ylabel('Depth (cm)', fontsize=12, weight='bold')
    ax1.set_title(f'{site_name} Validation — RMSE: {errors["rmse"]:.2f} K  '
                  f'Bias: {errors["bias"]:+.2f} K  MAE: {errors["mae"]:.2f} K',
                  fontsize=13, weight='bold')
    ax1.legend(**{**_LEG_KW, 'fontsize': 10})
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
                           figsize=(15, 11)):
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
    sites  = _APOLLO_SITES
    colors = _APOLLO_COLORS
    color, ls, label = _model_style(model_name)

    fig = plt.figure(figsize=figsize)
    # 3 rows: profile | residuals | stats bar
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[3.2, 1.6, 0.9],
        hspace=0.55, wspace=0.42,
    )

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

        # ── Depth axis: zoom to include shallowest sensor at top ─────────────
        max_meas_cm = float(np.max(a_depths * 100))
        y_max_cm    = min(320.0, max(160.0, max_meas_cm * 1.55))

        # ─────────────────────────────────────────────────────────────────────
        # Row 0: temperature profile panel
        # ─────────────────────────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])

        # Diurnal zone band (0–80 cm)
        _DIURNAL_ZONE_CM = 80
        ax0.axhspan(0, _DIURNAL_ZONE_CM, color='#FFF3CD', alpha=0.55, zorder=0)
        ax0.axhline(_DIURNAL_ZONE_CM, color='#CC8800', lw=0.9, ls='--',
                    alpha=0.65, zorder=1)
        ax0.text(0.99, _DIURNAL_ZONE_CM - 2,
                 'Diurnal zone  (< 80 cm)',
                 fontsize=6.5, color='#996600', va='bottom', ha='right',
                 transform=ax0.get_yaxis_transform())

        # Model profile + diurnal envelope
        mask = z_grid * 100 <= y_max_cm * 1.05
        ax0.plot(stats['T_mean'][mask], z_grid[mask] * 100,
                 color=color, ls=ls, linewidth=2.2, zorder=5,
                 label=f'{label} (cycle mean)')
        ax0.fill_betweenx(z_grid[mask] * 100,
                          stats['T_min'][mask], stats['T_max'][mask],
                          color=color, alpha=0.14, zorder=4,
                          label='Diurnal range')

        # Sensor markers by type
        sensor_types = errors.get('apollo_sensor_types', ['TG'] * len(a_depths))
        st_arr  = np.array(sensor_types)
        tg_mask = st_arr == 'TG'
        tr_mask = st_arr == 'TR'
        tc_mask = st_arr == 'TC'

        _msize = {'TG': 9, 'TR': 8, 'TC': 7}
        _malph = {'TG': 1.0, 'TR': 0.75, 'TC': 0.55}
        _mlbl  = {
            'TG': 'TG  (gradient bridge, official)',
            'TR': 'TR  (reference thermocouple)',
            'TC': 'TC  (cable, diurnal zone)',
        }
        for stype, smask in [('TG', tg_mask), ('TR', tr_mask), ('TC', tc_mask)]:
            if smask.any():
                ax0.plot(a_temps[smask], a_depths[smask] * 100,
                         _STYPE_MARKER.get(stype, 'o'), color=dot_color,
                         markersize=_msize[stype],
                         markeredgewidth=1.2, markeredgecolor='white',
                         alpha=_malph[stype], zorder=7,
                         label=_mlbl[stype])

        # R² annotation
        r2 = 1.0 - (np.sum(residuals**2) /
                    np.sum((a_temps - np.mean(a_temps))**2))
        ax0.text(0.97, 0.04,
                 f'RMSE = {errors["rmse"]:.2f} K\n'
                 f'Bias  = {errors["bias"]:+.2f} K\n'
                 f'R²    = {r2:.3f}',
                 transform=ax0.transAxes, fontsize=8.5,
                 ha='right', va='bottom', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.45',
                           facecolor='white', edgecolor='#cccccc',
                           alpha=0.95))

        ax0.set_xlabel('Temperature (K)')
        ax0.set_ylabel('Depth (cm)')
        ax0.set_ylim(y_max_cm, 0)
        ax0.set_title(site_name, pad=6)

        # Legend below the axes
        ax0.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.14),
                   borderaxespad=0, **{**_LEG_KW, 'fontsize': 7.5, 'handlelength': 1.5})

        # ─────────────────────────────────────────────────────────────────────
        # Row 1: residual lollipop chart
        # ─────────────────────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])

        d_cm = a_depths * 100
        # Colour by sign: red = warm bias, blue = cold bias
        res_colors = ['#C0392B' if r > 0 else '#2471A3' for r in residuals]
        ax1.hlines(d_cm, 0, residuals,
                   colors=res_colors, linewidth=2.0, alpha=0.85, zorder=2)
        ax1.scatter(residuals, d_cm, s=50,
                    c=res_colors, edgecolors='white',
                    linewidths=0.7, zorder=4)
        ax1.axvline(0, color='#333333', linewidth=1.0, ls='--', alpha=0.6)

        # ±1 K reference band
        ax1.axvspan(-1, 1, color='#27AE60', alpha=0.07, zorder=0)

        ax1.set_xlabel('Residual  (Model − Measured, K)')
        ax1.set_ylabel('Depth (cm)')
        ax1.set_title('Residuals', pad=4)

        # Smart y-limits: show deep sensors (≥80 cm) only
        deep_d = d_cm[d_cm >= _DIURNAL_ZONE_CM]
        if len(deep_d):
            ax1.set_ylim(float(np.max(deep_d)) + 8,
                         float(np.min(deep_d)) - 8)
        else:
            y2max = y_max_cm * 0.58
            ax1.set_ylim(y2max, max(0.0, float(np.min(d_cm)) - 5))
        ax1.invert_yaxis()

    # ─────────────────────────────────────────────────────────────────────────
    # Row 2: combined statistics bar chart (both sites, both metrics)
    # ─────────────────────────────────────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')

    _rows = ['Apollo 15', 'Apollo 17']
    _cols = ['RMSE (K)', 'Bias (K)', 'MAE (K)', 'R²', 'SUNSCALE', 'χ', 'Albedo']
    tbl_data = []
    for s in _rows:
        if s not in apollo_results:
            tbl_data.append(['—'] * len(_cols))
            continue
        e = apollo_results[s]['errors']
        at = e['apollo_temps']
        r  = e['residuals']
        r2 = 1.0 - np.sum(r**2) / np.sum((at - np.mean(at))**2)
        tbl_data.append([
            f'{e["rmse"]:.3f}', f'{e["bias"]:+.3f}',
            f'{e["mae"]:.3f}',  f'{r2:.3f}',
            f'{sunscale:.2f}',  f'{chi:.1f}', f'{albedo:.3f}',
        ])

    tbl = ax_stats.table(
        cellText=tbl_data, rowLabels=_rows, colLabels=_cols,
        cellLoc='center', rowLoc='center', loc='center',
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    # Header style
    for j in range(len(_cols)):
        tbl[(0, j)].set_facecolor('#2C3E50')
        tbl[(0, j)].set_text_props(color='white', weight='bold')
    # Row colours
    for i, clr in enumerate(['#D6EAF8', '#D5F5E3'], start=1):
        for j in range(-1, len(_cols)):
            tbl[(i, j)].set_facecolor(clr)

    # ── Shared super-title ────────────────────────────────────────────────────
    cfg = f'Model: {label}   ·   SUNSCALE {sunscale:.2f}   ·   χ {chi:.1f}   ·   Albedo {albedo:.3f}'
    plt.suptitle(
        f'Apollo 15 & 17 — Thermal Model Validation\n{cfg}',
        fontsize=12, weight='bold', y=1.005)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3b. TEMPERATURE GRADIENT PROFILE  (dT/dz vs depth)
# ─────────────────────────────────────────────────────────────────────────────

def apollo_gradient_profile(apollo_results, model_name, k_Wpm_pK=3.5e-3,
                             figsize=(13, 6)):
    """
    Plot the measured temperature gradient (dT/dz, K/m) at each consecutive
    sensor pair vs the model gradient, for both Apollo sites.

    dT/dz is computed from adjacent equilibrium temperatures; multiplying by
    the thermal conductivity k gives the local heat flux Q = k × dT/dz.
    The Langseth et al. (1976) basal value is ~18 mW/m².

    Parameters
    ----------
    apollo_results : same dict used by dual_apollo_comparison
    model_name     : string for legend
    k_Wpm_pK       : thermal conductivity (W/m/K); default 3.5e-3 W/m/K
    """
    sites  = _APOLLO_SITES
    colors = _APOLLO_COLORS
    _, _, label = _model_style(model_name)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    fig.suptitle(
        f'Temperature Gradient Profile  (dT/dz)  —  {label}\n'
        f'Heat flux Q = k × dT/dz,  k = {k_Wpm_pK*1e3:.1f} mW/m/K',
        fontsize=12, weight='bold', y=1.02)

    for ax, site_name, dot_color in zip(axes, sites, colors):
        if site_name not in apollo_results:
            continue
        errors   = apollo_results[site_name]['errors']
        stats    = apollo_results[site_name]['stats']
        z_grid   = stats['depth']
        a_depths = errors['apollo_depths']
        a_temps  = errors['apollo_temps']

        dz_meas   = np.diff(a_depths)
        dT_meas   = np.diff(a_temps)
        grad_meas = dT_meas / dz_meas           # K/m
        z_mid_cm  = 0.5 * (a_depths[:-1] + a_depths[1:]) * 100
        Q_meas_mW = grad_meas * k_Wpm_pK * 1e3

        mask      = z_grid * 100 <= np.max(a_depths * 100) * 1.6
        z_clip    = z_grid[mask]
        grad_model = np.gradient(stats['T_mean'][mask], z_clip)
        ax.plot(grad_model, z_clip * 100,
                color='#C0392B', lw=2.0, ls='-', label=f'{label} dT/dz')

        ax.scatter(grad_meas, z_mid_cm, s=60, color=dot_color,
                   edgecolors='white', linewidths=1.0, zorder=5,
                   label='Measured (adjacent pairs)')
        for gm, zm, Qm in zip(grad_meas, z_mid_cm, Q_meas_mW):
            ax.annotate(f'{Qm:+.1f} mW/m²',
                        xy=(gm, zm), xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=6, color=dot_color, va='center')

        ax.axvline(0, color='#888', lw=0.8, ls='--')
        ax.invert_yaxis()
        g_range = np.abs(grad_meas).max() * 1.5 + 0.01
        ax.set_xlim(-g_range, g_range)
        ax.set_xlabel('dT/dz  (K/m)', fontsize=11, weight='bold')
        ax.set_ylabel('Depth (cm)',    fontsize=11, weight='bold')
        ax.set_title(site_name, fontsize=12, weight='bold')
        ax.legend(**{**_LEG_KW, 'fontsize': 8})

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3c. SENSOR EQUILIBRATION TIMELINE
# ─────────────────────────────────────────────────────────────────────────────

def sensor_equilibration(site_name, window_days=60, figsize=(14, 5)):
    """
    Show how the rolling-median temperature at each deep sensor (≥80 cm)
    evolves over time.  A flat line = sensor has equilibrated.
    The green band marks the stable window used for model validation.

    Parameters
    ----------
    site_name    : 'Apollo 15' or 'Apollo 17'
    window_days  : rolling window width (days) for the median
    """
    from lunar.hfe_loader import get_timeseries, STABLE_WINDOWS

    probes  = get_timeseries(site_name)
    windows = STABLE_WINDOWS[site_name]

    fig, axes = plt.subplots(1, len(probes), figsize=figsize, sharey=False)
    if len(probes) == 1:
        axes = [axes]

    fig.suptitle(
        f'{site_name} — Sensor Equilibration  '
        f'(rolling {window_days}-day median, deep sensors ≥80 cm)\n'
        'Flat line = equilibrated.  Green band = stable window used for validation.',
        fontsize=11, weight='bold', y=1.03)

    _cmap = plt.cm.viridis_r

    for ax, probe, (win_start, win_end) in zip(axes, probes, windows):
        depths_shown = sorted(
            [(d['depth_cm'], s, d) for s, d in probe.items()
             if d['depth_cm'] >= 80],
            key=lambda x: x[0])

        if not depths_shown:
            ax.set_visible(False)
            continue

        all_depths = [dc for dc, _, _ in depths_shown]
        norm = Normalize(vmin=min(all_depths), vmax=max(all_depths))
        half = window_days / 2

        for d_cm, sensor, data in depths_shown:
            t0    = data['times'][0].timestamp() / 86400
            t_num = np.array([t.timestamp() / 86400 for t in data['times']]) - t0
            temps = data['temps']

            t_centers = np.arange(half, t_num[-1], window_days / 4)
            roll_med = []
            for tc in t_centers:
                win = (t_num >= tc - half) & (t_num <= tc + half)
                if win.sum() >= 3:
                    roll_med.append((tc, float(np.median(temps[win]))))
            if not roll_med:
                continue
            tc_arr, tm_arr = zip(*roll_med)
            ax.plot(tc_arr, tm_arr, lw=1.4, color=_cmap(norm(d_cm)),
                    label=f'{sensor} ({d_cm} cm)')

        ax.axvspan(win_start, win_end, alpha=0.15, color='#2ECC71',
                   label=f'Stable window ({win_start}–{win_end} d)')
        ax.axvline(win_start, color='#1E8449', ls='--', lw=1.0, alpha=0.8)
        ax.axvline(win_end,   color='#1E8449', ls=':',  lw=0.9, alpha=0.6)

        probe_label = next(iter(probe.values()))['probe_label']
        ax.set_title(probe_label, fontsize=11, weight='bold')
        ax.set_xlabel('Days since emplacement', fontsize=10, weight='bold')
        ax.set_ylabel('Rolling median T (K)',   fontsize=10, weight='bold')
        ax.legend(loc='upper right', **{**_LEG_KW, 'fontsize': 7})

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3d. PARAMETER SENSITIVITY HEATMAP  (RMSE vs sunscale × chi)
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_heatmap(rmse_grid, sunscale_vals, chi_vals,
                        site_name='Apollo 15', figsize=(7, 5)):
    """
    2-D heatmap of RMSE vs (sunscale, chi).  Highlights the global minimum.

    Parameters
    ----------
    rmse_grid     : 2-D array shape (len(sunscale_vals), len(chi_vals))
    sunscale_vals : 1-D array of sunscale values  (rows)
    chi_vals      : 1-D array of chi values        (columns)
    site_name     : used in title
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(rmse_grid.T, origin='lower', aspect='auto',
                   cmap='RdYlGn_r',
                   extent=[sunscale_vals[0], sunscale_vals[-1],
                            chi_vals[0],      chi_vals[-1]])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('RMSE (K)', fontsize=10, weight='bold')

    imin, jmin = np.unravel_index(np.argmin(rmse_grid), rmse_grid.shape)
    ax.plot(sunscale_vals[imin], chi_vals[jmin], 'w*', markersize=14,
            label=f'Min RMSE = {rmse_grid[imin,jmin]:.2f} K\n'
                  f'sunscale = {sunscale_vals[imin]:.2f},  χ = {chi_vals[jmin]:.1f}')
    ax.set_xlabel('Sunscale', fontsize=11, weight='bold')
    ax.set_ylabel('χ (radiative conductivity param)', fontsize=11, weight='bold')
    ax.set_title(f'{site_name} — RMSE Sensitivity (sunscale × χ)',
                 fontsize=12, weight='bold')
    ax.legend(loc='upper right', **_LEG_KW)
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
        # Use Apollo data from first model that has it; split by sensor type
        for name in model_names:
            err = (apollo_errors or {}).get(name)
            if err is not None:
                st_arr = np.array(err.get('apollo_sensor_types',
                                          ['TG'] * len(err['apollo_depths'])))
                _msize_mc = {'TG': 9, 'TR': 8, 'TC': 7}
                _malph_mc = {'TG': 1.0, 'TR': 0.85, 'TC': 0.60}
                for stype in ('TC', 'TR', 'TG'):   # TC first so TG sits on top
                    smask = st_arr == stype
                    if smask.any():
                        ax1.plot(err['apollo_temps'][smask],
                                 err['apollo_depths'][smask] * 100,
                                 _STYPE_MARKER.get(stype, 'o'),
                                 color='#1A5276',
                                 markersize=_msize_mc[stype],
                                 markeredgewidth=1.5,
                                 markeredgecolor='white',
                                 alpha=_malph_mc[stype], zorder=10,
                                 label=_STYPE_LABEL.get(stype, stype))
                break

    ax1.set_xlabel('Temperature (K)', fontsize=11, weight='bold')
    ax1.set_ylabel('Depth (cm)', fontsize=11, weight='bold')
    ax1.set_title('Mean Temperature Profiles', fontsize=12, weight='bold')
    ax1.legend(**_LEG_KW)
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
    ax2.legend(**_LEG_KW)
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
    - Sensors styled by instrument type:
        TG (gradient bridge, official)  — solid line, full opacity
        TR (reference TC, supplementary) — dashed line, reduced opacity
        TC (cable TC, shallow/diurnal)   — dotted line, low opacity
    - Sensors coloured by depth (shallow = warm, deep = cool).
    - Green band / dashed lines mark the stable equilibrium window used for
      model validation.
    - Orange bands mark known discrepancy/disturbance regions.

    Parameters
    ----------
    site_name : 'Apollo 15' or 'Apollo 17'

    Returns
    -------
    matplotlib.figure.Figure
    """
    import lunar.hfe_loader as _hfl
    from lunar.hfe_loader import get_timeseries

    # Stable windows and discrepancy regions from hfe_loader (with fallbacks)
    _STABLE_WINDOWS      = getattr(_hfl, '_STABLE_WINDOWS',      None)
    _STABLE_FRACTION     = getattr(_hfl, '_STABLE_FRACTION',     0.25)
    _DISCREPANCY_REGIONS = getattr(_hfl, '_DISCREPANCY_REGIONS', {})

    probes   = get_timeseries(site_name)
    n_probes = len(probes)

    fig, axes = plt.subplots(1, n_probes, figsize=figsize,
                             sharey=False, squeeze=False)
    axes = axes[0]

    # Depth colormap: shallow = warm yellow, deep = cool indigo
    _depth_cmap = plt.get_cmap('viridis_r')

    # Collect overall depth range for shared colorbar (in cm)
    all_depths_global = sorted({d['depth_cm']
                                 for probe in probes
                                 for d in probe.values()})
    g_dmin = all_depths_global[0]
    g_dmax = all_depths_global[-1]

    # Per-probe stable windows
    probe_windows = (
        _STABLE_WINDOWS[site_name]
        if _STABLE_WINDOWS and site_name in _STABLE_WINDOWS
        else [None] * n_probes
    )

    # Per-probe discrepancy regions
    site_discrepancies = _DISCREPANCY_REGIONS.get(site_name, {})

    # Depth threshold below which sensors are excluded from the equilibrium /
    # validation calculation (matches get_equilibrium_temps min_depth_cm=80).
    _VALIDATION_MIN_DEPTH = 80   # cm

    # Sensor-type style lookup  (prefix → (linestyle, linewidth, base_alpha))
    #   TG = official gradient-bridge primary sensors
    #   TR = supplementary reference thermocouples
    #   TC = shallow cable sensors (diurnal zone, not used for heat flow)
    _style = {
        'TG': ('-',  1.8, 0.95),   # solid,  thick,  full
        'TR': ('--', 1.1, 0.65),   # dashed, thin,   medium
        'TC': (':',  0.9, 0.40),   # dotted, thinner, dim
    }

    for probe_idx, (ax, probe, pw) in enumerate(zip(axes, probes, probe_windows)):
        norm = Normalize(vmin=g_dmin, vmax=g_dmax)

        t_stable_final = None
        t_end_final    = None
        probe_t_max    = 0          # track max x for open-ended discrepancy regions

        # ── Plot each sensor ─────────────────────────────────────────────────
        for sensor, data in sorted(probe.items(),
                                   key=lambda kv: kv[1]['depth_cm']):
            times = data['times']
            temps = data['temps']
            d_cm  = data['depth_cm']
            color = _depth_cmap(norm(d_cm))

            t_num = np.array([t.timestamp() / 86400 for t in times])
            t0    = t_num[0]
            t_num = t_num - t0        # days since emplacement

            # Sensor type from two-letter prefix
            prefix = ''.join(c for c in sensor if c.isalpha())[:2]
            ls, lw, alpha = _style.get(prefix, ('-', 1.2, 0.80))

            # Sensors shallower than the validation threshold are excluded from
            # the equilibrium-temperature calculation.  Dim them and flag clearly.
            used_in_validation = d_cm >= _VALIDATION_MIN_DEPTH
            if not used_in_validation:
                alpha *= 0.45          # visually subordinate
                ls     = ':'           # override to dotted regardless of type

            type_labels = {'TG': 'official', 'TR': 'supplementary', 'TC': 'non-official'}
            type_tag = f' [{type_labels.get(prefix, prefix)}]'
            if not used_in_validation:
                type_tag += ' · excl. from validation'

            ax.plot(t_num, temps, lw=lw, ls=ls, color=color, alpha=alpha,
                    label=f'{sensor}  ({d_cm} cm){type_tag}')

            if t_num[-1] > probe_t_max:
                probe_t_max = t_num[-1]

            # Fallback window tracking
            if pw is None:
                n_stable = max(1, int(len(temps) * _STABLE_FRACTION))
                t_s = t_num[-n_stable]
                if t_stable_final is None or t_s < t_stable_final:
                    t_stable_final = t_s
                if t_end_final is None or t_num[-1] > t_end_final:
                    t_end_final = t_num[-1]

        win_start, win_end = pw if pw is not None else (t_stable_final, t_end_final)

        # ── Discrepancy regions (orange bands) ───────────────────────────────
        # Only draw regions adjacent to the stable window (within 300 days).
        disc_regions = site_discrepancies.get(probe_idx, [])
        _disc_label_added = False
        for reg_start, reg_end, reg_desc in disc_regions:
            r_end = reg_end if reg_end is not None else probe_t_max + 50
            if r_end < win_start - 300 or reg_start > win_end + 300:
                continue
            disc_lbl = 'Discrepancy / disturbance' if not _disc_label_added else '_nolegend_'
            _disc_label_added = True
            ax.axvspan(reg_start, r_end,
                       alpha=0.12, color='#E67E22', zorder=1,
                       label=disc_lbl)
            # Boundary tick + day label at start of disturbance
            ax.axvline(reg_start, color='#E67E22', ls='--', lw=0.9, alpha=0.7, zorder=2)
            ax.text(reg_start + 4, 0.98, f'Day {int(reg_start)}\n{reg_desc}',
                    transform=ax.get_xaxis_transform(),
                    fontsize=5.5, va='top', ha='left', color='#B7770D',
                    rotation=0, clip_on=True)

        # ── Stable window (green band) ────────────────────────────────────────
        ax.axvspan(win_start, win_end,
                   alpha=0.14, color='#2ECC71', zorder=2,
                   label=f'Stable window (≥{_VALIDATION_MIN_DEPTH} cm, used for validation)')
        # Left boundary — dashed + day label
        ax.axvline(win_start, color='#1E8449', ls='--', lw=1.4, alpha=0.85, zorder=3)
        ax.text(win_start + 4, 0.02, f'Day {int(win_start)}\n(window start)',
                transform=ax.get_xaxis_transform(),
                fontsize=6, va='bottom', ha='left', color='#1E8449', clip_on=True)
        # Right boundary — dotted + day label
        ax.axvline(win_end, color='#1E8449', ls=':', lw=1.2, alpha=0.65, zorder=3)
        ax.text(win_end - 4, 0.02, f'Day {int(win_end)}\n(window end)',
                transform=ax.get_xaxis_transform(),
                fontsize=6, va='bottom', ha='right', color='#1E8449', clip_on=True)

        probe_label = next(iter(probe.values()))['probe_label']
        ax.set_title(probe_label, fontsize=13, weight='bold', pad=8)
        ax.set_xlabel('Days since emplacement', fontsize=12, weight='bold')
        ax.set_ylabel('Temperature (K)',        fontsize=12, weight='bold')

        leg = ax.legend(ncol=1, loc='upper right',
                        **{**_LEG_KW, 'fontsize': 7.0, 'handlelength': 2.0})
        for line in leg.get_lines():
            line.set_linewidth(2.0)

    # ── Shared colorbar ───────────────────────────────────────────────────────
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.016, 0.68])
    sm = ScalarMappable(cmap=_depth_cmap,
                        norm=Normalize(vmin=g_dmin, vmax=g_dmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Sensor depth (cm)', fontsize=11, weight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        f'{site_name} — HFE Probe Temperature History\n'
        'TG = official (gradient bridge) · TR = supplementary · TC = shallow/diurnal  |  '
        f'Green = stable window (≥{80} cm sensors used for validation) · '
        'Orange = disturbance region  ·  Dotted sensors = excluded from validation',
        fontsize=10, weight='bold', y=1.02,
    )
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
            all_depths_all.append(data['depth_cm'])

    # Build a time-indexed merged view: {timestamp: {depth_cm: T}}
    from collections import defaultdict
    ts_index = defaultdict(dict)
    for probe in probes:
        for sensor, data in probe.items():
            d_cm = data['depth_cm']
            for t, T in zip(data['times'], data['temps']):
                ts_index[t][d_cm] = T

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
        ax_prof.plot(temps, depths,          # depths already in cm
                     'o-', color=color, lw=1.5, ms=5, label=lbl)

    ax_prof.invert_yaxis()
    ax_prof.set_xlabel('Temperature (K)', fontsize=11, weight='bold')
    ax_prof.set_ylabel('Depth (cm)', fontsize=11, weight='bold')
    ax_prof.set_title('T–Depth Profile at Time Snapshots\n'
                      '(disturbance convergence)', fontsize=11, weight='bold')
    ax_prof.legend(**_LEG_KW)
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
            depth_label = f'{dA["depth_cm"]}–{dB["depth_cm"]} cm'
            ax_grad.plot(days, delta, lw=0.9, color=color,
                         label=f'{bridge_name}  ({depth_label})')
            bridge_idx += 1

    ax_grad.axhline(0, color='k', ls='--', lw=0.8)
    ax_grad.set_xlabel('Days since emplacement', fontsize=11, weight='bold')
    ax_grad.set_ylabel('ΔT  (T_A − T_B)  [K]', fontsize=11, weight='bold')
    ax_grad.set_title('Gradient-Bridge ΔT Over Time\n'
                      '(thermal shunting signature)', fontsize=11, weight='bold')
    ax_grad.legend(**{**_LEG_KW, 'fontsize': 8})
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
    ax1.legend(**_LEG_KW)
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
    ax2.legend(**_LEG_KW)
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
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), **_LEG_KW)
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
    ax1.legend(loc='upper right', **_LEG_KW)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Absorbed flux ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(t_hours, sol_flux, alpha=0.4, color='#E74C3C')
    ax2.plot(t_hours, sol_flux, color='#C0392B', lw=1.5, label='Absorbed solar flux')
    ax2.set_ylabel('Absorbed Flux (W/m²)', fontsize=11, weight='bold')
    ax2.set_title('Absorbed Solar Energy', fontsize=12, weight='bold')
    ax2.legend(**_LEG_KW)
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
    ax3.legend(**_LEG_KW)
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
    ax.legend(**_LEG_KW)
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
    ax2.legend(**_LEG_KW)

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
                err    = r['errors']
                st_arr = np.array(err.get('apollo_sensor_types',
                                          ['TG'] * len(err['apollo_depths'])))
                _msize_ss = {'TG': 8, 'TR': 7, 'TC': 6}
                for stype in ('TC', 'TR', 'TG'):
                    smask = st_arr == stype
                    if smask.any():
                        ax1.plot(err['apollo_temps'][smask],
                                 err['apollo_depths'][smask] * 100,
                                 _STYPE_MARKER.get(stype, 'o'),
                                 color='#C0392B',
                                 markersize=_msize_ss[stype],
                                 markeredgewidth=1.5,
                                 markeredgecolor='white',
                                 zorder=10,
                                 label=_STYPE_LABEL.get(stype, stype))
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
    ax2.fill_between(values, T_min, T_max, alpha=0.25, color='#F39C12')
    ax2.plot(values, T_max,  '-o', color='#E74C3C', lw=2, ms=7, label='Maximum')
    ax2.plot(values, T_mean, '-s', color='#5D6D7E', lw=2, ms=7, label='Mean')
    ax2.plot(values, T_min,  '-o', color='#2471A3', lw=2, ms=7, label='Minimum')
    ax2.set_xlabel(param_name, fontsize=11, weight='bold')
    ax2.set_ylabel('Surface Temperature (K)', fontsize=11, weight='bold')
    ax2.set_title('Surface Temperature Response', fontsize=12, weight='bold')
    ax2.legend(**_LEG_KW)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: RMSE / bias vs parameter (or placeholder) ────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if has_apollo:
        rmse_v = [r['errors']['rmse'] for r in sens_results if r['errors']]
        bias_v = [r['errors']['bias'] for r in sens_results if r['errors']]
        v_ap   = [r['value']          for r in sens_results if r['errors']]
        ax3b   = ax3.twinx()
        _RC = '#E74C3C'; _BC = '#2471A3'; _OC = '#27AE60'
        l1,    = ax3.plot(v_ap, rmse_v, '-o', color=_RC, lw=2, ms=7, label='RMSE')
        l2,    = ax3b.plot(v_ap, bias_v, '-s', color=_BC, lw=2, ms=7, label='Bias')
        # Optimal value
        best_i = int(np.argmin(rmse_v))
        ax3.axvline(v_ap[best_i], color=_OC, ls='--', lw=2, alpha=0.7)
        ax3.text(v_ap[best_i], max(rmse_v) * 0.97,
                 f'  Opt={v_ap[best_i]:.3g}',
                 fontsize=9, color=_OC, weight='bold', va='top')
        ax3.set_xlabel(param_name, fontsize=11, weight='bold')
        ax3.set_ylabel('RMSE (K)', fontsize=11, weight='bold', color=_RC)
        ax3b.set_ylabel('Bias (K)', fontsize=11, weight='bold', color=_BC)
        ax3.tick_params(axis='y', labelcolor=_RC)
        ax3b.tick_params(axis='y', labelcolor=_BC)
        ax3.set_title('Accuracy vs Parameter', fontsize=12, weight='bold')
        ax3.legend([l1, l2], ['RMSE', 'Bias'], **_LEG_KW)
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
    ax4.plot(values, T_1m, '-o', color='#27AE60', lw=2.5, ms=9, mew=2, mec='#1A7A46')
    ax4.fill_between(values, min(T_1m), T_1m, alpha=0.2, color='#27AE60')
    ax4.set_xlabel(param_name, fontsize=11, weight='bold')
    ax4.set_ylabel('Temperature at 1 m (K)', fontsize=11, weight='bold')
    ax4.set_title('Deep Temperature Sensitivity', fontsize=12, weight='bold')
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Surface amplitude ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    amp  = [r['stats']['T_amplitude'][0] for r in sens_results]
    ax5.plot(values, amp, '-o', color='#8E44AD', lw=2.5, ms=9, mew=2, mec='#6C3483')
    ax5.fill_between(values, min(amp), amp, alpha=0.2, color='#8E44AD')
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
    ax1.legend(**_LEG_KW)
    ax1.grid(True, alpha=0.3, axis='y')

    # ── Panel 2: Temperature at 0 and 50 cm ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    T_50cm = [float(np.interp(0.5, z_grid, r['stats']['T_mean']))
              for r in batch_results]
    ax2.plot(xs, T_mean_surf, '-o', color='#E74C3C', lw=2, ms=7, label='Surface (0 cm)')
    ax2.plot(xs, T_50cm, '-s', color='#2471A3', lw=2, ms=7, label='Depth 50 cm')
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Mean Temperature (K)', fontsize=11, weight='bold')
    ax2.set_title('Mean Temperature at Key Depths', fontsize=12, weight='bold')
    ax2.legend(**_LEG_KW)
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
        ax3.legend(**_LEG_KW)
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
    ax4.legend(ncol=max(1, n // 5), loc='best', **{**_LEG_KW, 'fontsize': 8})
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()

    plt.suptitle('Batch Processing Summary', fontsize=14, weight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# NEW 1.  HEAT FLUX DEPTH PROFILE   Q(z) = k(z) · dT/dz
# ─────────────────────────────────────────────────────────────────────────────

def heat_flux_profile(apollo_results, model_name,
                      k_surface_mW=1.5, k_deep_mW=3.5,
                      figsize=(13, 6)):
    """
    Compute and plot the vertical heat-flux profile Q(z) = k(z) · dT/dz for
    both Apollo sites, overlaying measured gradient-bridge estimates.

    The thermal conductivity profile follows the two-zone approximation:
        k ≈ k_surface  for z < 0.02 m  (dry powder layer)
        k ≈ k_deep     for z ≥ 0.02 m  (consolidated regolith)
    Values are consistent with Langseth et al. (1976).

    The Langseth basal heat-flow values are annotated:
        Apollo 15: ~21 mW/m²
        Apollo 17: ~16 mW/m²

    Parameters
    ----------
    apollo_results  : same dict used by dual_apollo_comparison
    model_name      : string key ('discrete' or 'hayne_exponential')
    k_surface_mW    : surface-zone conductivity  (mW/m/K)
    k_deep_mW       : deep-zone conductivity     (mW/m/K)
    """
    _LANGSETH = {'Apollo 15': 21.0, 'Apollo 17': 16.0}   # mW/m²
    sites  = _APOLLO_SITES
    colors = _APOLLO_COLORS
    _, _, label = _model_style(model_name)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    fig.suptitle(
        f'Geothermal Heat-Flux Profile  Q(z) = k(z) · dT/dz\n'
        f'Model: {label}   ·   '
        f'k_surface = {k_surface_mW:.1f} mW/m/K  ·  k_deep = {k_deep_mW:.1f} mW/m/K',
        fontsize=11)

    for ax, site_name, dot_color in zip(axes, sites, colors):
        if site_name not in apollo_results:
            ax.set_visible(False)
            continue

        errors  = apollo_results[site_name]['errors']
        stats   = apollo_results[site_name]['stats']
        z_grid  = stats['depth']          # metres
        a_d     = errors['apollo_depths'] # metres

        # Conductivity profile (mW/m/K → W/m/K)
        k_profile = np.where(z_grid < 0.02,
                             k_surface_mW * 1e-3,
                             k_deep_mW    * 1e-3)

        # Model heat flux
        dT_dz   = np.gradient(stats['T_mean'], z_grid)    # K/m
        Q_model = k_profile * dT_dz * 1e3                 # mW/m²

        mask = z_grid * 100 <= float(np.max(a_d * 100)) * 1.6
        ax.plot(Q_model[mask], z_grid[mask] * 100,
                color='#C0392B', lw=2.2, zorder=4,
                label=f'{label} Q(z)')
        ax.fill_betweenx(z_grid[mask] * 100, 0, Q_model[mask],
                         color='#C0392B', alpha=0.10, zorder=3)

        # Measured Q at each adjacent sensor pair (deep sensors only)
        st_arr = np.array(errors.get('apollo_sensor_types', ['TG'] * len(a_d)))
        deep   = (a_d >= 0.80) & ((st_arr == 'TG') | (st_arr == 'TR'))
        a_d_d  = a_d[deep]
        a_T_d  = errors['apollo_temps'][deep]
        if len(a_d_d) >= 2:
            dz_m   = np.diff(a_d_d)
            dT_m   = np.diff(a_T_d)
            grad_m = dT_m / dz_m
            z_mid  = 0.5 * (a_d_d[:-1] + a_d_d[1:])
            k_mid  = np.where(z_mid < 0.02, k_surface_mW, k_deep_mW) * 1e-3
            Q_meas = k_mid * grad_m * 1e3
            ax.scatter(Q_meas, z_mid * 100,
                       s=65, color=dot_color, edgecolors='white',
                       linewidths=1.0, zorder=6,
                       label='Measured  (adjacent pairs)')
            for qm, zm in zip(Q_meas, z_mid * 100):
                ax.annotate(f'{qm:+.1f}',
                            xy=(qm, zm), xytext=(4, 0),
                            textcoords='offset points',
                            fontsize=7.5, color=dot_color, va='center')

        # Langseth reference heat flow
        q_ref = _LANGSETH.get(site_name)
        if q_ref:
            ax.axvline(q_ref, color='#27AE60', ls='--', lw=1.4, alpha=0.8,
                       label=f'Langseth (1976): {q_ref:.0f} mW/m²')
            ax.axvline(0, color='#888', lw=0.7, ls=':')

        ax.invert_yaxis()
        ax.set_xlabel('Heat Flux  Q  (mW/m²)')
        ax.set_ylabel('Depth (cm)')
        ax.set_title(site_name)
        ax.legend(loc='lower right', **{**_LEG_KW, 'fontsize': 8})

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# NEW 2.  DIURNAL AMPLITUDE DECAY  vs DEPTH  (thermal skin depth)
# ─────────────────────────────────────────────────────────────────────────────

def amplitude_decay(stats, z_grid, lat, lon, model_name=None, figsize=(10, 6)):
    """
    Plot the diurnal temperature amplitude A(z) = (T_max − T_min)/2 vs depth
    and fit an exponential decay to extract the thermal skin depth δ.

    The exponential model is A(z) = A₀ · exp(−z / δ), where
        δ = sqrt(κ P / π)   (P = lunar day period, κ = thermal diffusivity)
    Typical lunar values: δ ≈ 0.05 – 0.12 m in the surface layer.

    Parameters
    ----------
    stats      : dict from analysis.extract_stats()
    z_grid     : depth array (m)
    lat, lon   : location (degrees)
    model_name : used in title
    """
    amp   = stats['T_amplitude']          # K, shape (n_z,)
    z_cm  = z_grid * 100.0
    color, ls, label = _model_style(model_name or 'discrete')

    # Fit only to the shallow zone where amplitude > 1 % of surface value
    fit_mask = amp > 0.01 * amp[0]
    z_fit    = z_grid[fit_mask]
    A_fit    = amp[fit_mask]

    # Linear least-squares on log(A) = log(A0) - z/delta
    fit_ok = False
    A0_fit = delta_fit = None
    try:
        log_A = np.log(A_fit)
        # Design matrix [1, z]  →  coefficients [log(A0), -1/delta]
        X = np.column_stack([np.ones_like(z_fit), z_fit])
        coeffs, *_ = np.linalg.lstsq(X, log_A, rcond=None)
        A0_fit    = np.exp(coeffs[0])
        delta_fit = -1.0 / coeffs[1]
        if 0.005 < delta_fit < 1.0:   # sanity bounds: 0.5 cm – 100 cm
            fit_ok = True
    except Exception:
        pass

    fig, (ax_main, ax_log) = plt.subplots(1, 2, figsize=figsize, sharey=False)
    fig.suptitle(
        f'Diurnal Amplitude Decay  —  {_subtitle(lat, lon, model_name)}')

    # ── Left: linear scale ────────────────────────────────────────────────────
    ax_main.plot(amp, z_cm, color=color, lw=2.2, ls=ls, label='Amplitude A(z)')
    ax_main.fill_betweenx(z_cm, 0, amp, color=color, alpha=0.12)

    if fit_ok:
        z_fine  = np.linspace(z_grid[0], z_grid[fit_mask][-1], 400)
        A_fine  = A0_fit * np.exp(-z_fine / delta_fit)
        ax_main.plot(A_fine, z_fine * 100, 'k--', lw=1.6,
                     label=f'Fit: δ = {delta_fit * 100:.1f} cm\n'
                           f'     A₀ = {A0_fit:.1f} K')
        # Skin-depth marker
        ax_main.axhline(delta_fit * 100, color='gray', ls=':',
                        lw=1.0, alpha=0.7)
        ax_main.text(amp[0] * 0.5, delta_fit * 100 + 0.5,
                     f'δ = {delta_fit * 100:.1f} cm',
                     fontsize=8.5, color='gray', va='bottom')

    ax_main.invert_yaxis()
    ax_main.set_xlabel('Amplitude  A(z)  (K)')
    ax_main.set_ylabel('Depth (cm)')
    ax_main.set_title('Linear scale')
    ax_main.legend(**{**_LEG_KW, 'fontsize': 8.5})
    ax_main.set_xlim(left=0)

    # ── Right: log scale — straight line if truly exponential ────────────────
    pos_mask = amp > 0
    ax_log.semilogy(amp[pos_mask], z_cm[pos_mask],
                    color=color, lw=2.2, ls=ls, label='A(z)')
    ax_log.fill_betweenx(z_cm[pos_mask], 1e-3, amp[pos_mask],
                          color=color, alpha=0.10)
    if fit_ok:
        A_fine_pos = A_fine[A_fine > 0]
        z_fine_pos = z_fine[:len(A_fine_pos)]
        ax_log.semilogy(A_fine_pos, z_fine_pos * 100, 'k--', lw=1.6,
                        label=f'Exp. fit  (R² = {1.0:.2f})')

    ax_log.invert_yaxis()
    ax_log.set_xlabel('Amplitude  A(z)  (K, log scale)')
    ax_log.set_ylabel('Depth (cm)')
    ax_log.set_title('Log scale  (straight = exponential decay)')
    ax_log.legend(**{**_LEG_KW, 'fontsize': 8.5})
    ax_log.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# NEW 3.  COMBINED HEAT-FLOW SUMMARY  (A15 vs A17 bar chart + profile)
# ─────────────────────────────────────────────────────────────────────────────

def combined_heat_flow(apollo_results, model_name,
                       k_surface_mW=1.5, k_deep_mW=3.5,
                       figsize=(13, 7)):
    """
    Publication-ready two-panel figure summarising the heat-flow result.

    Left  : Bar chart comparing modelled vs Langseth (1976) heat flow at
            Apollo 15 and Apollo 17, with ±uncertainty band.
    Right : Geothermal gradient dT/dz vs depth for both sites overlaid.

    Parameters
    ----------
    apollo_results  : same dict used by dual_apollo_comparison
    model_name      : model key
    k_surface_mW    : surface conductivity (mW/m/K)
    k_deep_mW       : deep conductivity    (mW/m/K)
    """
    _LANGSETH   = {'Apollo 15': 21.0, 'Apollo 17': 16.0}  # mW/m²
    _LANGSETH_E = {'Apollo 15':  3.0, 'Apollo 17':  2.0}  # ±uncertainty

    sites  = _APOLLO_SITES
    colors = _APOLLO_COLORS
    _, _, label = _model_style(model_name)

    # ── Compute modelled heat flow at each site ───────────────────────────────
    q_model = {}
    grad_data = {}
    for site_name in sites:
        if site_name not in apollo_results:
            continue
        stats  = apollo_results[site_name]['stats']
        z_grid = stats['depth']
        # Use the deep zone (0.5–2 m) where gradient is steady
        deep   = (z_grid >= 0.5) & (z_grid <= 2.0)
        if not deep.any():
            deep = z_grid >= 0.3
        dT_dz_deep = np.gradient(stats['T_mean'][deep], z_grid[deep])
        Q_deep_mW  = float(np.mean(dT_dz_deep)) * k_deep_mW   # mW/m²

        q_model[site_name] = Q_deep_mW

        # Full gradient profile for the right panel
        dT_dz = np.gradient(stats['T_mean'], z_grid)
        grad_data[site_name] = (z_grid, dT_dz)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, (ax_bar, ax_grad) = plt.subplots(1, 2, figsize=figsize,
                                           gridspec_kw={'width_ratios': [1.1, 1.6]})

    # ── Left: bar chart ───────────────────────────────────────────────────────
    x      = np.arange(len(sites))
    width  = 0.35
    q_ref  = [_LANGSETH.get(s, 0) for s in sites]
    q_err  = [_LANGSETH_E.get(s, 0) for s in sites]
    q_mod  = [q_model.get(s, 0) for s in sites]
    clrs   = colors

    bars_ref = ax_bar.bar(x - width / 2, q_ref, width,
                          color='#95A5A6', edgecolor='white', linewidth=0.8,
                          label='Langseth et al. (1976)', zorder=3)
    ax_bar.errorbar(x - width / 2, q_ref, yerr=q_err,
                    fmt='none', color='#2C3E50', lw=1.8, capsize=5, zorder=4)
    bars_mod = ax_bar.bar(x + width / 2, q_mod, width,
                          color=clrs, edgecolor='white', linewidth=0.8,
                          label='This model', zorder=3)

    # Value labels
    for bar, v in zip(bars_ref, q_ref):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                    f'{v:.0f}', ha='center', va='bottom', fontsize=9, color='#333')
    for bar, v in zip(bars_mod, q_mod):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, max(0, v) + 0.4,
                    f'{v:+.1f}', ha='center', va='bottom', fontsize=9,
                    color='#C0392B' if v < 0 else '#1E8449')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(sites, fontsize=10)
    ax_bar.set_ylabel('Heat Flux  (mW/m²)')
    ax_bar.set_title('Modelled vs Measured\nGeothermal Heat Flow')
    ax_bar.legend(**{**_LEG_KW, 'fontsize': 8.5})
    ax_bar.set_ylim(bottom=0)
    ax_bar.axhline(0, color='black', lw=0.6)

    # ── Right: gradient profiles ──────────────────────────────────────────────
    for (site_name, (z_g, dT_dz)), dot_color in zip(grad_data.items(), colors):
        errors = apollo_results[site_name]['errors']
        a_d    = errors['apollo_depths']
        a_T    = errors['apollo_temps']
        max_cm = float(np.max(a_d * 100)) * 1.5
        mask   = z_g * 100 <= max_cm

        ax_grad.plot(dT_dz[mask] * 1000, z_g[mask] * 100,   # mK/m
                     lw=2.0, color=dot_color, label=f'{site_name} (model)')

        # Measured gradient from adjacent deep sensors
        st_arr = np.array(errors.get('apollo_sensor_types', ['TG'] * len(a_d)))
        deep   = a_d >= 0.80
        a_d_d  = a_d[deep]; a_T_d = a_T[deep]
        if len(a_d_d) >= 2:
            dz_m   = np.diff(a_d_d)
            dT_m   = np.diff(a_T_d)
            grad_m = dT_m / dz_m   # K/m
            z_mid  = 0.5 * (a_d_d[:-1] + a_d_d[1:])
            ax_grad.scatter(grad_m * 1000, z_mid * 100,
                            s=60, color=dot_color, edgecolors='white',
                            lw=1.0, zorder=5,
                            label=f'{site_name} (measured)')

    ax_grad.axvline(0, color='#888', lw=0.8, ls='--')
    ax_grad.invert_yaxis()
    ax_grad.set_xlabel('Temperature Gradient  dT/dz  (mK/m)')
    ax_grad.set_ylabel('Depth (cm)')
    ax_grad.set_title('Geothermal Gradient Profile\n(solid = model, circles = measured)')
    ax_grad.legend(ncol=2, **{**_LEG_KW, 'fontsize': 8.5})

    fig.suptitle(
        f'Lunar Geothermal Heat Flow — {label}\n'
        f'k_surface = {k_surface_mW:.1f} mW/m/K  ·  k_deep = {k_deep_mW:.1f} mW/m/K',
        fontsize=11)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BORESTEM / ALBEDO / POLAR DIURNAL — publication-quality figures
# ═══════════════════════════════════════════════════════════════════════════════

# Depth colour ramp shared with diurnal_cycles():
#   surface = warm red → shallow = orange → deep = blue
_DEPTH_COLORS = [
    '#D73027', '#FC8D59', '#FEE090',
    '#ABD9E9', '#74ADD1', '#4575B4', '#313695', '#1A1A6C',
]

# Apollo site colours (navy / purple) — same as _APOLLO_COLORS
_A15_COLOR = '#1A5276'
_A17_COLOR = '#7D3C98'


# ─────────────────────────────────────────────────────────────────────────────
# ALBEDO COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def albedo_comparison(stats_und, cycles_und, stats_dis, cycles_dis,
                      depth=0.0, lat=None, lon=None, figsize=(13, 6)):
    """Side-by-side diurnal surface temperature for undisturbed vs disturbed albedo.

    Parameters
    ----------
    stats_und  : dict — extract_stats() for A = UNDISTURBED_ALBEDO
    cycles_und : dict — get_diurnal_cycles() for A = UNDISTURBED_ALBEDO
    stats_dis  : dict — extract_stats() for A = DISTURBED_ALBEDO
    cycles_dis : dict — get_diurnal_cycles() for A = DISTURBED_ALBEDO
    depth      : float — depth in metres to plot (0.0 = surface)
    lat, lon   : float — site coordinates for subtitle
    """
    from lunar.constants import LUNAR_DAY
    day_h = LUNAR_DAY / 3600.0
    half  = day_h / 2.0

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    pairs = [
        (stats_und, cycles_und, 'Undisturbed  (A = 0.09)', '#2471A3'),   # Hayne blue
        (stats_dis, cycles_dis, 'Disturbed    (A = 0.12)',  '#C0392B'),   # Hayne red
    ]

    for ax, (st, cyc, panel_title, color) in zip(axes, pairs):
        # Night shading
        ax.axvspan(0,           half * 0.48, color='#1a1a2e', alpha=0.08, zorder=0)
        ax.axvspan(half * 1.52, day_h,       color='#1a1a2e', alpha=0.08, zorder=0)
        ax.axvline(half, color='#888', lw=0.8, ls='--', alpha=0.6)
        ax.text(half + day_h * 0.015, 0.98, 'Noon',
                fontsize=8, color='#666', va='top', ha='left',
                transform=ax.get_xaxis_transform())

        # Data
        available = sorted(cyc.keys())
        best_d    = min(available, key=lambda d: abs(d - depth)) if available else None

        if best_d is not None:
            entry = cyc[best_d]
            t_hr  = np.asarray(entry['time_h'])
            T     = np.asarray(entry['temperature'])
            mean_T = np.nanmean(T)
            ax.plot(t_hr, T, lw=2.5, color=color,
                    label=f'{panel_title.strip()}')
            ax.axhline(mean_T, color=color, ls='--', lw=1.2, alpha=0.65,
                       label=f'Mean  {mean_T:.1f} K')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11)

        ax.set_title(panel_title, fontsize=12, weight='bold')
        ax.set_xlabel('Time in lunar day (hours)', fontsize=11, weight='bold')
        ax.set_xlim(0, day_h)
        ax.legend(loc='upper right', **_LEG_KW)

    axes[0].set_ylabel('Temperature (K)', fontsize=11, weight='bold')

    lat_str = f'{lat:.3f}' if lat is not None else '?'
    lon_str = f'{lon:.3f}' if lon is not None else '?'
    depth_cm = depth * 100
    depth_lbl = 'Surface (0 cm)' if depth_cm < 1 else f'{depth_cm:.0f} cm depth'
    fig.suptitle(
        f'Albedo Effect on Diurnal Temperature — {depth_lbl}\n'
        f'{lat_str}°N,  {lon_str}°E',
        fontsize=13, weight='bold')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# POLAR DIURNAL
# ─────────────────────────────────────────────────────────────────────────────

def polar_diurnal(cycles, depths_m=(0.0, 0.35, 1.0, 2.0),
                  lat=None, lon=None, figsize=(9, 9)):
    """Clock-face polar plot of temperature vs lunar local time.

    Angle = local time (0 h at top, clockwise).
    Radius = temperature (K).
    Each ring = one depth; surface is outermost (hottest at noon).

    Parameters
    ----------
    cycles   : dict — get_diurnal_cycles(), keyed by depth (m)
    depths_m : depths to overlay; nearest available depth is used
    lat, lon : site coordinates for title
    """
    if not cycles:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
        ax.set_title('No diurnal cycle data available', fontsize=12)
        return fig

    available = sorted(cycles.keys())
    n         = len(depths_m)
    # Warm-to-cool palette matching depth_colors order
    colors    = [_DEPTH_COLORS[i % len(_DEPTH_COLORS)] for i in range(n)]
    linewidths = np.linspace(2.5, 1.2, n)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
    ax.set_theta_zero_location('N')   # 0 h at top
    ax.set_theta_direction(-1)        # clockwise

    for depth, color, lw in zip(depths_m, colors, linewidths):
        best_d = min(available, key=lambda d: abs(d - depth))
        entry  = cycles[best_d]
        t_hr   = np.asarray(entry['time_h'])
        T      = np.asarray(entry['temperature'])
        actual = entry.get('actual_depth', best_d)

        if t_hr.size < 2:
            continue

        # Angles: map 0…t_end hours onto 0…2π, closed loop
        theta = 2 * np.pi * t_hr / t_hr[-1]
        ang   = np.append(theta, theta[0])
        T_c   = np.append(T, T[0])

        d_lbl = 'Surface (0 cm)' if actual * 100 < 1 else f'{actual * 100:.0f} cm'
        ax.plot(ang, T_c, lw=lw, color=color, label=d_lbl)

    # Tick labels: local time in hours
    n_ticks = 8
    tick_angles = np.linspace(0, 2 * np.pi, n_ticks, endpoint=False)
    tick_labels = [f'{int(round(h))} h'
                   for h in np.linspace(0, available[0] and cycles[available[0]]['time_h'][-1] or 708, n_ticks, endpoint=False)]
    ax.set_xticks(tick_angles)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_ylabel('Temperature (K)', labelpad=45, fontsize=10, weight='bold')
    ax.legend(title='Depth', title_fontsize=9,
              loc='lower right', bbox_to_anchor=(1.30, -0.05),
              **_LEG_KW)

    lat_str = f'{lat:.3f}' if lat is not None else '?'
    lon_str = f'{lon:.3f}' if lon is not None else '?'
    ax.set_title(f'Polar Diurnal Temperature\n{lat_str}°N,  {lon_str}°E',
                 pad=22, fontsize=13, weight='bold')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# BORESTEM CORRECTION PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def borestem_correction_plot(stats, apollo_data, site_name,
                              correction_dT, figsize=(12, 7)):
    """Two-panel figure: mean T profile with borestem correction, and ΔT(z).

    Parameters
    ----------
    stats         : dict with keys 'z_grid', 'T_mean_profile', 'lat', 'lon'
    apollo_data   : dict with keys:
                      'depths'       — array (m)
                      'T_K'          — array (K)
                      'sensor_types' — list of 'TG'/'TR'/'TC' (optional)
    site_name     : '15' or '17'
    correction_dT : array (n,) — total warm bias at each Z_GRID node (K)
    """
    z_grid  = np.asarray(stats.get('z_grid', []))
    T_model = np.asarray(stats.get('T_mean_profile', stats.get('T_mean', [])))
    a_d     = np.asarray(apollo_data.get('depths', []))
    a_T     = np.asarray(apollo_data.get('T_K',    []))
    a_st    = list(apollo_data.get('sensor_types',  ['TG'] * len(a_d)))
    corr    = np.asarray(correction_dT)

    if T_model.size == 0 or z_grid.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title('No model profile data available')
        return fig

    T_corrected = T_model - corr if corr.size == T_model.size else T_model.copy()

    site_color = _A15_COLOR if '15' in str(site_name) else _A17_COLOR

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :2])   # main profile panel (2/3 width)
    ax2 = fig.add_subplot(gs[0, 2])    # correction ΔT panel

    # ── Panel 1: temperature profiles ────────────────────────────────────────
    ax1.plot(T_model,     z_grid * 100, lw=2.5, color='#2471A3',
             label='Model (uncorrected)', zorder=3)
    ax1.plot(T_corrected, z_grid * 100, lw=2.5, color='#C0392B', ls='--',
             label='Model (borestem corrected)', zorder=4)
    if a_d.size:
        _msize = {'TG': 9, 'TR': 8, 'TC': 7}
        _malph = {'TG': 1.0, 'TR': 0.75, 'TC': 0.55}
        _mlbl  = {'TG': 'TG  (gradient bridge)', 'TR': 'TR  (reference TC)',
                   'TC': 'TC  (cable, diurnal zone)'}
        st_arr = np.array(a_st)
        for stype in ('TG', 'TR', 'TC'):
            smask = st_arr == stype
            if smask.any():
                ax1.plot(a_T[smask], a_d[smask] * 100,
                         _STYPE_MARKER.get(stype, 'o'), color=site_color,
                         markersize=_msize[stype],
                         markeredgewidth=1.2, markeredgecolor='white',
                         alpha=_malph[stype], zorder=5,
                         label=_mlbl[stype])

    ax1.invert_yaxis()
    ax1.set_xlabel('Temperature (K)', fontsize=11, weight='bold')
    ax1.set_ylabel('Depth (cm)',      fontsize=11, weight='bold')
    ax1.set_title(f'Apollo {site_name} — Mean Temperature Profile\n'
                  f'Borestem Thermal Correction',
                  fontsize=12, weight='bold')
    ax1.legend(**_LEG_KW)

    # ── Panel 2: correction magnitude ΔT(z) ──────────────────────────────────
    if corr.size == z_grid.size:
        ax2.plot(corr, z_grid * 100, lw=2.5, color='#E67E22', zorder=3)
        ax2.fill_betweenx(z_grid * 100, 0, corr,
                          color='#E67E22', alpha=0.18, zorder=2)
    ax2.axvline(0, color='#333', lw=0.9, ls='--')
    ax2.invert_yaxis()
    ax2.set_xlabel('Warm Bias  ΔT (K)', fontsize=11, weight='bold')
    ax2.set_title('Correction\nΔT(z)', fontsize=12, weight='bold')
    ax2.yaxis.set_ticklabels([])   # depth axis shared with ax1

    fig.suptitle(f'Borestem Fiberglass Thermal Short-Circuit — Apollo {site_name}',
                 fontsize=13, weight='bold', y=1.01)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# BORESTEM 2-D FIELD FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def borestem_2d_field_plot(T_2d, r_grid, z_grid, T_mean, dT_bs,
                            lat=None, lon=None, model_name=None,
                            depth_limit=2.6, figsize=(14, 7)):
    """
    Two-panel publication figure for the 2-D axisymmetric borestem temperature field.

    Left panel  : T(r, z) heatmap (inferno colourmap, z increasing downward)
                  Borestem wall boundaries (r_i, r_o) marked with dashed lines.
                  White iso-temperature contours every ≈ 10 K.

    Right panel : ΔT_bs(z) lollipop chart — warm bias at the sensor axis vs depth.
                  Matches the lollipop style used in apollo_comparison().

    Parameters
    ----------
    T_2d       : (n_z, n_r) temperature field from solve_borestem_2d_steady()
    r_grid     : (n_r,) radial positions (m)
    z_grid     : (n_z,) depth positions (m)
    T_mean     : (n_z,) undisturbed 1-D mean temperature (K)
    dT_bs      : (n_z,) warm bias  T_axis − T_mean  (K)
    lat, lon   : float — site coordinates for subtitle
    model_name : str or None
    depth_limit: float — maximum depth to display (m)
    """
    from lunar.constants import BORESTEM_OUTER_RADIUS_M, BORESTEM_WALL_M, BORESTEM_DEPTH_M

    r_i = BORESTEM_OUTER_RADIUS_M - BORESTEM_WALL_M   # inner wall face
    r_o = BORESTEM_OUTER_RADIUS_M                      # outer wall face

    # ── Restrict to depth_limit ───────────────────────────────────────────────
    i_z  = np.where(z_grid <= depth_limit)[0]
    z_cm = z_grid[i_z] * 100.0
    T_sub = T_2d[i_z, :]
    dT_sub = dT_bs[i_z]

    r_cm = r_grid * 100.0

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.12)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_dT  = fig.add_subplot(gs[0, 1])

    # ── Left: 2-D temperature field ───────────────────────────────────────────
    pm = ax_map.pcolormesh(
        r_cm, z_cm, T_sub,
        cmap='inferno', shading='gouraud',
        vmin=np.nanmin(T_sub), vmax=np.nanmax(T_sub),
    )
    # Iso-temperature contours
    n_r_s = max(1, len(r_cm) // 60)
    n_z_s = max(1, len(z_cm) // 80)
    cs = ax_map.contour(
        r_cm[::n_r_s], z_cm[::n_z_s], T_sub[::n_z_s, ::n_r_s],
        levels=8, colors='white', linewidths=0.6, alpha=0.40,
    )
    ax_map.clabel(cs, inline=True, fontsize=7.5, fmt='%d K', use_clabeltext=True)

    cbar = plt.colorbar(pm, ax=ax_map, pad=0.02, fraction=0.035)
    cbar.set_label('Temperature (K)')
    cbar.ax.tick_params(labelsize=9)

    # Borestem wall boundaries
    bore_depth_cm = min(BORESTEM_DEPTH_M * 100, depth_limit * 100)
    for r_wall, label, ls in [(r_i * 100, f'r_i = {r_i*100:.2f} cm', '--'),
                               (r_o * 100, f'r_o = {r_o*100:.2f} cm', '-.')]:
        ax_map.plot([r_wall, r_wall], [0, bore_depth_cm],
                    color='cyan', lw=1.6, ls=ls,
                    label=f'Borestem wall  ({label})', zorder=5)
    ax_map.axhline(bore_depth_cm, color='cyan', lw=1.2, ls=':', alpha=0.7,
                   label=f'Casing end  ({BORESTEM_DEPTH_M:.1f} m)', zorder=5)

    ax_map.invert_yaxis()
    ax_map.set_xlabel('Radius  r  (cm)', fontsize=11, weight='bold')
    ax_map.set_ylabel('Depth   z  (cm)', fontsize=11, weight='bold')
    ax_map.legend(loc='lower right', **{**_LEG_KW, 'fontsize': 8})

    lat_str = f'{lat:.3f}' if lat is not None else '?'
    lon_str = f'{lon:.3f}' if lon is not None else '?'
    ax_map.set_title(
        f'2-D Borestem Temperature Field  T(r, z)\n'
        f'{_subtitle(lat, lon, model_name) if lat is not None else ""}',
        fontsize=12, weight='bold',
    )

    # ── Right: ΔT lollipop ────────────────────────────────────────────────────
    # Lollipop style matching apollo_comparison() residual panel
    ax_dT.hlines(z_cm, 0, dT_sub,
                 color='#E67E22', linewidth=2.0, alpha=0.85, zorder=3)
    ax_dT.scatter(dT_sub, z_cm,
                  s=55, color='#E67E22', edgecolors='white',
                  linewidths=1.0, zorder=4)
    ax_dT.axvline(0, color='#333', lw=0.9, ls='--')

    # Annotate peak bias
    if dT_sub.size:
        i_peak = np.argmax(np.abs(dT_sub))
        ax_dT.annotate(
            f'{dT_sub[i_peak]:+.2f} K',
            xy=(dT_sub[i_peak], z_cm[i_peak]),
            xytext=(6, 0), textcoords='offset points',
            fontsize=8, color='#A04000', va='center',
        )

    ax_dT.invert_yaxis()
    ax_dT.set_xlabel('Warm bias  ΔT  (K)', fontsize=11, weight='bold')
    ax_dT.set_title('Sensor\nwarm bias', fontsize=12, weight='bold')
    ax_dT.yaxis.set_ticklabels([])
    ax_dT.set_ylim(ax_map.get_ylim())

    fig.suptitle(
        f'Apollo HFE Borestem Correction — 2-D Axisymmetric Steady-State\n'
        f'Fiberglass wall  k = {0.04:.3f} W/m/K  ·  '
        f'Outer radius {r_o*100:.2f} cm  ·  Depth {BORESTEM_DEPTH_M:.1f} m',
        fontsize=13, weight='bold',
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PROBE-TOP RADIATION: WARM BIAS VS DEPTH
# ─────────────────────────────────────────────────────────────────────────────

def probe_radiation_depth_sensitivity(depths_m, delta_T_K, figsize=(9, 5)):
    """Lollipop chart of probe-top solar radiation warm bias vs sensor depth.

    Parameters
    ----------
    depths_m  : array-like — sensor depths (m)
    delta_T_K : array-like — correction at each depth (K, positive = warm bias)
    """
    depths = np.asarray(depths_m)
    dT     = np.asarray(delta_T_K)

    fig, ax = plt.subplots(figsize=figsize)

    # Lollipop style — consistent with residual panels in apollo_comparison()
    ax.hlines(depths * 100, 0, dT,
              color='#E67E22', linewidth=2.0, alpha=0.85, zorder=3)
    ax.scatter(dT, depths * 100,
               s=70, color='#E67E22', edgecolors='white',
               linewidths=1.2, zorder=4)

    # Annotate each dot
    for d, val in zip(depths * 100, dT):
        ax.text(val + max(dT) * 0.02, d, f'+{val:.3f} K',
                va='center', fontsize=9, color='#5D4037')

    ax.axvline(0, color='#333', lw=0.9, ls='--')
    ax.invert_yaxis()
    ax.set_xlabel('Temperature Warm Bias  ΔT (K)', fontsize=11, weight='bold')
    ax.set_ylabel('Depth (cm)',                     fontsize=11, weight='bold')
    ax.set_title('Probe-Top Solar Radiation — Instrumental Warm Bias vs Sensor Depth',
                 fontsize=12, weight='bold')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATED THERMAL WAVE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def thermal_wave_annotated(T_profile, t_arr, z_grid, lat, lon,
                           model_name=None, depth_limit=1.5, figsize=(13, 6)):
    """Heatmap of the subsurface temperature field with skin-depth annotation.

    Adds to the standard heatmap():
    - Cyan dashed skin-depth line (where diurnal amplitude falls to 1/e)
    - Side panel showing amplitude decay with depth + exponential fit

    Parameters
    ----------
    T_profile   : 2-D array (n_snapshots × n_depths) — temperature (K)
    t_arr       : 1-D array — time in seconds (from solver)
    z_grid      : 1-D array — depth in metres
    lat, lon    : float — site coordinates
    model_name  : str or None
    depth_limit : float — maximum depth to show (m)
    """
    from lunar.constants import LUNAR_DAY
    from scipy.optimize import curve_fit

    T = np.asarray(T_profile, dtype=float)
    t = np.asarray(t_arr,     dtype=float)
    z = np.asarray(z_grid,    dtype=float)

    # Ensure shape is (n_snapshots, n_depths)
    if T.ndim == 2 and T.shape == (z.size, t.size):
        T = T.T   # was (n_depths, n_time) — transpose

    # Restrict to final lunar day
    t_start = t[-1] - LUNAR_DAY
    idx_t   = np.where(t >= t_start)[0]
    idx_z   = np.where(z <= depth_limit)[0]

    t_hours = (t[idx_t] - t_start) / 3600.0
    z_cm    = z[idx_z] * 100.0
    T_sub   = T[np.ix_(idx_t, idx_z)]   # (n_t, n_z)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.08)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_amp = fig.add_subplot(gs[0, 1])

    # ── Heatmap ────────────────────────────────────────────────────────────────
    pm = ax_map.pcolormesh(t_hours, z_cm, T_sub.T,
                           cmap='inferno', shading='gouraud',
                           vmin=np.nanmin(T_sub), vmax=np.nanmax(T_sub))
    # Contour lines
    n_t, n_z = len(t_hours), len(z_cm)
    s_t = max(1, n_t // 200); s_z = max(1, n_z // 80)
    cs = ax_map.contour(t_hours[::s_t], z_cm[::s_z],
                        T_sub[::s_t, ::s_z].T,
                        levels=8, colors='white', linewidths=0.6, alpha=0.40)
    ax_map.clabel(cs, inline=True, fontsize=7.5, fmt='%d K', use_clabeltext=True)

    cbar = plt.colorbar(pm, ax=ax_map, pad=0.02, fraction=0.04)
    cbar.set_label('Temperature (K)')
    cbar.ax.tick_params(labelsize=9)

    # ── Skin depth line ────────────────────────────────────────────────────────
    T_amp = 0.5 * (T_sub.max(axis=0) - T_sub.min(axis=0))   # (n_z,)
    if T_amp[0] > 0:
        frac = T_amp / T_amp[0]
        below_inv_e = np.where(frac <= 1.0 / np.e)[0]
        if below_inv_e.size:
            z_skin_cm = float(z_cm[below_inv_e[0]])
            ax_map.axhline(z_skin_cm, color='cyan', lw=1.8, ls='--',
                           label=f'Skin depth  δ ≈ {z_skin_cm:.1f} cm')
            ax_map.legend(loc='lower right', **_LEG_KW)

    ax_map.invert_yaxis()
    ax_map.set_xlabel('Time in lunar day (hours)', fontsize=11, weight='bold')
    ax_map.set_ylabel('Depth (cm)',                fontsize=11, weight='bold')
    ax_map.set_title(
        f'Subsurface Thermal Wave — Depth × Time\n'
        f'{_subtitle(lat, lon, model_name)}',
        fontsize=12, weight='bold')
    ax_map.set_xlim(t_hours[0], t_hours[-1])

    # ── Amplitude decay side panel ────────────────────────────────────────────
    ax_amp.plot(T_amp, z_cm, lw=2.2, color='#E67E22', label='A(z)')

    # Exponential fit A(z) = A0 · exp(-z/δ)
    try:
        def _exp(z, A0, delta): return A0 * np.exp(-z / delta)
        popt, _ = curve_fit(_exp, z_cm, T_amp, p0=[T_amp[0], z_cm[below_inv_e[0]] if below_inv_e.size else 5.0],
                            maxfev=2000)
        z_fit = np.linspace(0, z_cm[-1], 300)
        ax_amp.plot(z_fit, _exp(z_fit, *popt), lw=1.4, ls='--',
                    color='#C0392B', alpha=0.80,
                    label=f'Fit  δ = {popt[1]:.1f} cm')
        ax_amp.legend(**{**_LEG_KW, 'fontsize': 8})
    except Exception:
        pass

    ax_amp.invert_yaxis()
    ax_amp.set_xlabel('Amplitude (K)', fontsize=10, weight='bold')
    ax_amp.set_title('Amp.', fontsize=11, weight='bold')
    ax_amp.set_xlim(left=0)
    ax_amp.yaxis.set_ticklabels([])
    ax_amp.set_ylim(ax_map.get_ylim())

    plt.tight_layout()
    return fig
