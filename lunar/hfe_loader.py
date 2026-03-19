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
# Windows are chosen to exclude:
#   A15 — flat-line artefacts that appear in the final ~50 days (~day 1375+)
#   A17 — step-changes and sparse anomalous readings that appear after ~day 700
# Format: [(probe1_start, probe1_end), (probe2_start, probe2_end)]
_STABLE_WINDOWS = {
    'Apollo 15': [(840, 1370), (840, 1370)],
    'Apollo 17': [(520,  700), (520,  700)],
}


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


def get_equilibrium_temps(site_name, min_depth_cm=80):
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
