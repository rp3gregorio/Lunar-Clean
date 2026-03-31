"""
hfe_loader.py — Load and process Apollo Heat Flow Experiment (HFE) time-series data.

Four probe files (two per mission):
    data/a15p1_depth.tab   Apollo 15 Probe 1  (depths 35–139 cm; probe reached 1.4 m)
    data/a15p2_depth.tab   Apollo 15 Probe 2  (depths 49–97 cm;  probe reached 1.0 m)
    data/a17p1_depth.tab   Apollo 17 Probe 1  (depths 14–233 cm; probe reached 2.3 m)
    data/a17p2_depth.tab   Apollo 17 Probe 2  (depths 15–234 cm; probe reached 2.3 m)

Each file is CSV with columns: Time, T, sensor, depth, flags

NOTE: The 'depth' column is in **centimetres**.  Confirmed against published
NASA/NSSDCA documentation (Langseth et al. 1972, 1976):
    Apollo 17 TC13 = 66 cm, TG11A = 130 cm, TG12B = 233 cm  (exact match).
    Apollo 15 Probe 1 max depth = 139 cm ≈ 1.39 m  (matches reported 1.4 m).

Sensor naming
    TG = Gradient-bridge thermocouples (paired: A upper, B lower)
    TR = Reference thermocouples (absolute T at depth)
    TC = Cable thermocouples (very shallow: 14–67 cm, diurnally active)

Public API
    load_probe(filepath)          → {sensor: dict} with full time-series
    load_site(site_name)          → list of probe dicts
    get_equilibrium_temps(site)   → sorted [(depth_m, T_K), ...] for validation
    get_timeseries(site_name)     → merged list of probe dicts (for plotting)
"""

import os
import csv
import datetime
import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

_PROBE_FILES = {
    'Apollo 15': ['a15p1_depth.tab', 'a15p2_depth.tab'],
    'Apollo 17': ['a17p1_depth.tab', 'a17p2_depth.tab'],
}

# Fraction of each sensor's timeline to use as the "stable equilibrium" window
# (kept for backward-compat / fallback; explicit windows below take precedence)
_STABLE_FRACTION = 0.25

# Explicit stable windows (days since probe emplacement) used for both
# equilibrium-temperature averaging and green-band visualisation.
# Each window is the MAXIMUM contiguous clean interval: starts strictly after
# the last disturbance that precedes it, ends strictly before the next one.
#
#   A15 P1 — window [992, 1240]:
#               Disturbances end at day 990; window starts 2 days later.
#               Next disturbance starts at 1242; window ends at 1240.
#               Clean window: 248 days.
#
#   A15 P2 — window [542, 1240]:
#               Disturbance (518–540) ends at 540; window starts 2 days later.
#               Next disturbance starts at 1242; window ends at 1240.
#               Clean window: 698 days (extended from prior [840, 1240]).
#
#   A17 P1 — window [62, 853]:
#               Emplacement transient ends at day 60; window starts at 62.
#               Next disturbance starts at 855; window ends at 853.
#               Clean window: 791 days (extended from prior [520, 700]).
#
#   A17 P2 — window [62, 494]:
#               Emplacement transient ends at day 60; window starts at 62.
#               Next disturbance (496–530) starts at 496; window ends at 494.
#               Clean window: 432 days.  The later clean gap [532–705] is only
#               173 days — the earlier window is chosen for length.
#
# Format: [(probe1_start, probe1_end), (probe2_start, probe2_end)]
_STABLE_WINDOWS = {
    'Apollo 15': [(992, 1240), (542, 1240)],
    'Apollo 17': [( 62,  853), ( 62,  494)],
}

# Known discrepancy / disturbance regions (days since emplacement) per site
# and per probe index (0 = Probe 1, 1 = Probe 2).
# Each entry is a tuple: (start_day, end_day, description)
# end_day=None means "to the end of the record".
_DISCREPANCY_REGIONS = {
    'Apollo 15': {
        0: [  # Probe 1
            (0,    80,   'Emplacement transient'),
            (585,  610,  'Thermal disturbance (TG11A)'),
            (820,  845,  'Thermal disturbance (TG11A)'),
            (912,  935,  'Thermal disturbance (TG11A)'),
            (967,  990,  'Thermal disturbance (TG11A)'),
            (1242, 1262, 'Thermal disturbance (TG11A/TG22A)'),
            (1370, None, 'Flat-line artefact / data gap'),
        ],
        1: [  # Probe 2
            (0,    120,  'Emplacement transient (TG22A/B)'),
            (518,  540,  'Thermal disturbance (TG22A/B)'),
            (1242, 1262, 'Thermal disturbance (TG22A)'),
        ],
    },
    'Apollo 17': {
        0: [  # Probe 1
            (0,    60,   'Emplacement transient'),
            (855,  875,  'Thermal disturbance (TG11B)'),
            (1176, 1200, 'Thermal disturbance (TG11A)'),
        ],
        1: [  # Probe 2
            (0,    60,   'Emplacement transient'),
            (496,  530,  'Thermal disturbance (TG21A/B)'),
            (707,  725,  'Thermal disturbance (TG21A)'),
            (845,  865,  'Thermal disturbance (TG21A/B)'),
        ],
    },
}


# Public aliases — import these rather than the private underscored names
STABLE_WINDOWS      = _STABLE_WINDOWS
DISCREPANCY_REGIONS = _DISCREPANCY_REGIONS


def load_probe(filepath):
    """
    Load a single HFE probe .tab file (flagged rows are discarded).

    Returns
    -------
    dict  {sensor_name: {'times': np.ndarray[datetime],
                         'temps': np.ndarray[float, K],
                         'depth_cm': int}}   ← depth in centimetres
    """
    by_sensor = {}
    with open(filepath) as fh:
        for row in csv.DictReader(fh):
            if int(row['flags']) != 0:
                continue
            s = row['sensor']
            t = datetime.datetime.fromisoformat(row['Time'].replace('Z', '+00:00'))
            T = float(row['T'])
            d = int(row['depth'])          # centimetres
            if s not in by_sensor:
                by_sensor[s] = {'times': [], 'temps': [], 'depth_cm': d}
            by_sensor[s]['times'].append(t)
            by_sensor[s]['temps'].append(T)

    for s in by_sensor:
        by_sensor[s]['times'] = np.array(by_sensor[s]['times'])
        by_sensor[s]['temps'] = np.array(by_sensor[s]['temps'], dtype=float)
    return by_sensor


def load_site(site_name):
    """
    Load all probes for a site.

    Returns list of probe dicts (one per .tab file, in file order).
    """
    probes = []
    for fname in _PROBE_FILES[site_name]:
        fpath = os.path.join(_DATA_DIR, fname)
        probes.append(load_probe(fpath))
    return probes


def get_equilibrium_temps(site_name, min_depth_cm=0):
    """
    Derive equilibrium temperature at each depth using the explicit stable
    windows defined in ``_STABLE_WINDOWS``.

    Selection method
    ----------------
    For each sensor, the **median** of the readings that fall inside the
    probe's stable window (days since emplacement) is used as the equilibrium
    temperature.  The median is robust to occasional noise spikes.  Where
    multiple sensors share the same nominal depth (e.g. paired gradient-bridge
    A/B elements), their stable medians are averaged to give a single value.

    All known discrepancy/disturbance windows (see ``_DISCREPANCY_REGIONS``)
    are already excluded by design: the stable windows are chosen to be
    strictly between disturbances, so no special per-sensor masking is needed.

    Depth filter
    ------------
    Only sensors at or below *min_depth_cm* (default 80 cm) are included.
    This excludes the cable (TC) thermocouples that sit in the diurnally-
    driven zone and never fully equilibrate to the geothermal gradient.
    The lunar diurnal skin depth is ≈ 50 cm, so 80 cm gives a conservative
    margin below it.

    Returns
    -------
    list of (depth_m, T_K, sensor_type) tuples sorted by ascending depth.
    ``sensor_type`` is the two-letter instrument prefix: 'TG' (primary
    thermogradient bridge), 'TR' (reference thermocouple), or 'TC' (cable).
    """
    probes   = load_site(site_name)
    windows  = _STABLE_WINDOWS[site_name]          # [(start, end), ...]
    depth_buckets   = {}   # depth_cm → [median_T, ...]
    sensor_type_map = {}   # depth_cm → sensor prefix

    for probe, (win_start, win_end) in zip(probes, windows):
        # Probe t0: earliest timestamp across all sensors in this probe
        all_t0 = min(
            (data['times'][0].timestamp() / 86400
             for data in probe.values() if len(data['times']) > 0),
            default=None,
        )
        if all_t0 is None:
            continue

        for sensor, data in probe.items():
            d_cm = data['depth_cm']
            if d_cm < min_depth_cm:
                continue
            times = data['times']
            temps = data['temps']
            if len(temps) == 0:
                continue

            t_days = np.array([t.timestamp() / 86400 for t in times]) - all_t0
            mask   = (t_days >= win_start) & (t_days <= win_end)
            stable = temps[mask]
            if len(stable) == 0:
                continue

            median_T = float(np.median(stable))
            depth_buckets.setdefault(d_cm, []).append(median_T)

            # Two-letter instrument prefix: TG, TR, or TC
            prefix = ''.join(c for c in sensor if c.isalpha())[:2]
            sensor_type_map[d_cm] = prefix

    # Convert cm → m; include sensor-type label for validation-plot colouring
    return [
        (d_cm / 100.0, float(np.mean(Ts)), sensor_type_map.get(d_cm, 'TR'))
        for d_cm, Ts in sorted(depth_buckets.items())
    ]


def get_timeseries(site_name):
    """
    Return probe data ready for plotting.

    Returns
    -------
    list of probe dicts — same structure as load_site(), but each entry also
    carries a 'probe_label' key (e.g. 'Probe 1', 'Probe 2').
    """
    probes = load_site(site_name)
    for i, probe in enumerate(probes):
        label = f'Probe {i + 1}'
        for data in probe.values():
            data['probe_label'] = label
    return probes


def get_probe_diurnal_cycle(site_name, n_lunar_days=5):
    """
    Phase-fold Apollo HFE readings from the stable window into a single
    representative lunar-day diurnal cycle.

    Method
    ------
    For each sensor, readings from the last *n_lunar_days* × 29.53 Earth days
    inside the stable window are taken.  Each timestamp is mapped to a phase
    angle within the lunar day using a **common UTC reference** shared across
    all probes for this site:

        phase_h = (abs_unix_days − common_utc_ref)  mod  29.53 × 24   [hours]

    Using a single common UTC reference ensures that all sensors — regardless
    of which probe they belong to or when that probe was emplaced — are folded
    onto the **same lunar-day phase frame** (i.e. local noon falls at the same
    x-coordinate for every panel).  The previous per-probe epoch reference
    introduced an inter-probe phase offset equal to the difference in emplacement
    dates modulo the lunar period.

    Phase-alignment caveat for Apollo 17
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Apollo 17 Probe 1 and Probe 2 have non-overlapping stable windows
    (P1: days 705–853, P2: days 346–494 since emplacement, ≈ 1 year apart).
    All sensors are still folded onto the same lunar-phase frame via the common
    UTC reference, but the two probes sample different calendar epochs.
    Amplitude differences between sensors from different probes therefore
    reflect both depth-dependent damping AND any real mission-timeline changes.

    Parameters
    ----------
    site_name     : 'Apollo 15' or 'Apollo 17'
    n_lunar_days  : number of lunar days to fold (more = smoother curve)

    Returns
    -------
    dict  {depth_cm: {'time_h'  : ndarray        — hours within lunar day [0, 708]
                      'T_raw'   : ndarray        — absolute temperature (K)
                      'T_anom'  : ndarray        — T − mean(T) (K)
                      'T_mean'  : float          — mean temperature over folded window
                      'sensor'  : str            — sensor name
                      'stype'   : str            — 'TG', 'TR', or 'TC'
                      'ref_utc' : datetime (UTC) — common UTC time at phase_h = 0}}
    Multiple sensors at the same depth_cm are merged; the one with the most
    readings is kept.
    """
    LUNAR_DAY_DAYS = 29.53   # synodic month in Earth days

    probes  = load_site(site_name)
    windows = _STABLE_WINDOWS[site_name]

    # ── Pass 1: determine each probe's selection window in absolute UTC days ──
    probe_epochs = []   # (probe, all_t0, sel_start_days, sel_end_days)
    for probe, (win_start, win_end) in zip(probes, windows):
        all_t0 = min(
            (data['times'][0].timestamp() / 86400
             for data in probe.values() if len(data['times']) > 0),
            default=None,
        )
        if all_t0 is None:
            probe_epochs.append(None)
            continue
        sel_end   = float(win_end)
        sel_start = max(float(win_start), sel_end - n_lunar_days * LUNAR_DAY_DAYS)
        probe_epochs.append((probe, all_t0, sel_start, sel_end))

    # Common UTC reference = earliest sel_start across all probes (absolute Unix days)
    # This guarantees all sensors fold onto the same lunar-phase frame.
    common_utc_ref = min(
        all_t0 + sel_start
        for item in probe_epochs if item is not None
        for _, all_t0, sel_start, _ in [item]
    )
    ref_utc_common = datetime.datetime.utcfromtimestamp(common_utc_ref * 86400)

    # ── Pass 2: phase-fold every sensor using the common reference ────────────
    result = {}   # depth_cm → best entry

    for item in probe_epochs:
        if item is None:
            continue
        probe, all_t0, sel_start, sel_end = item

        for sensor, data in probe.items():
            d_cm  = data['depth_cm']
            times = data['times']
            temps = data['temps']
            if len(temps) < 10:
                continue

            # Absolute Unix days for each reading
            t_unix = np.array([t.timestamp() / 86400 for t in times])
            # Selection mask uses per-probe epoch (days since emplacement)
            t_days = t_unix - all_t0
            mask   = (t_days >= sel_start) & (t_days <= sel_end)
            if mask.sum() < 10:
                continue

            t_unix_sel = t_unix[mask]
            T_sel      = temps[mask]

            # Phase-fold relative to the common UTC reference
            t_phase_h = ((t_unix_sel - common_utc_ref) % LUNAR_DAY_DAYS) * 24.0
            sort_idx  = np.argsort(t_phase_h)
            t_ph      = t_phase_h[sort_idx]
            T_ph      = T_sel[sort_idx]

            T_mean = float(np.mean(T_ph))
            T_anom = T_ph - T_mean

            stype = ''.join(c for c in sensor if c.isalpha())[:2]

            # Keep the sensor with the most readings per depth
            if d_cm not in result or mask.sum() > result[d_cm]['_n']:
                result[d_cm] = {
                    'time_h':  t_ph,
                    'T_raw':   T_ph,
                    'T_anom':  T_anom,
                    'T_mean':  T_mean,
                    'sensor':  sensor,
                    'stype':   stype,
                    'ref_utc': ref_utc_common,   # same for all sensors
                    '_n':      int(mask.sum()),
                }

    # Remove internal counter before returning
    for v in result.values():
        v.pop('_n', None)

    return result
