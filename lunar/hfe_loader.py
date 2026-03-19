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
# (taken from the tail end of the record, after the drilling transient is gone)
_STABLE_FRACTION = 0.25


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


def get_equilibrium_temps(site_name, stable_fraction=_STABLE_FRACTION,
                          min_depth_cm=80):
    """
    Derive equilibrium temperature at each depth from the *stable tail* of
    each sensor's time-series (last *stable_fraction* of readings).

    Selection method
    ----------------
    For each sensor, the **median** of its last *stable_fraction* (25 %) of
    readings is used as the equilibrium temperature.  The median is robust
    to occasional noise spikes in the tail.  Where multiple sensors share
    the same nominal depth (e.g. paired gradient-bridge A/B elements),
    their stable medians are averaged to give a single value per depth.

    Depth filter
    ------------
    Only sensors at or below *min_depth_cm* (default 80 cm) are included.
    This excludes the cable (TC) thermocouples that sit in the diurnally-
    driven zone and never fully equilibrate to the geothermal gradient.
    The lunar diurnal skin depth is ≈ 50 cm, so 80 cm gives a conservative
    margin below it.

    Returns
    -------
    list of (depth_m, T_K) tuples sorted by ascending depth.
    """
    probes = load_site(site_name)
    depth_buckets = {}  # depth_cm → [median_T, ...]

    for probe in probes:
        for sensor, data in probe.items():
            d_cm = data['depth_cm']
            if d_cm < min_depth_cm:
                continue          # skip diurnally-active shallow sensors
            temps = data['temps']
            n = len(temps)
            if n == 0:
                continue
            n_stable = max(1, int(n * stable_fraction))
            median_T = float(np.median(temps[-n_stable:]))
            depth_buckets.setdefault(d_cm, []).append(median_T)

    # Convert cm → m for the returned values
    return [(d_cm / 100.0, float(np.mean(Ts)))
            for d_cm, Ts in sorted(depth_buckets.items())]


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
