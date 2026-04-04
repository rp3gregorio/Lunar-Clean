"""
lunar_lst.py — Compute true Local Solar Time (LST) on the Moon from UTC
               timestamps using NAIF SPICE ephemeris data.

This is the rigorous phase-alignment solution described in Tyler et al. (2016)
and used by studies that compare model output to Apollo HFE data on a common
physical time axis.

Why SPICE is needed
-------------------
The thermal simulation (solver.py) uses an arbitrary t_sec=0 epoch — it has
no connection to any real calendar date.  The Apollo HFE data has real UTC
timestamps.  Without an absolute lunar ephemeris, the two time axes cannot be
placed on the same physical clock.

SPICE provides the sub-solar longitude on the Moon at any UTC instant, which
directly gives the Local Solar Time at any surface point:

    LST (hours) = (site_lon − sub_solar_lon + 180) / 360  ×  LUNAR_DAY_h   (mod LUNAR_DAY_h)

where  LST = 0 h       → midnight  (Sun at antipode, site_lon − ssl_lon = 180)
       LST = day_h/4   → sunrise
       LST = day_h/2   → noon     (Sun directly overhead, site_lon = ssl_lon)
       LST = 3·day_h/4 → sunset

Required kernel files (place in data/ next to notebook)
---------------------------------------------------------
    naif0012.tls                   — leap-second kernel
    pck00010.tpc                   — planetary constants (Moon shape/rotation)
    de430.bsp                      — planetary ephemeris (Sun/Moon positions)
    moon_pa_de421_1900-2050.bpc    — Moon principal-axis orientation
    moon_080317.tf                 — MOON_PA frame definition

All files are freely available from NAIF:
    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/

Public API
----------
    load_kernels(data_dir)          — furnsh all five kernels
    utc_to_lst(utc_times, site_lon_deg) → LST hours array
    get_apollo_lst(site_name)       → diurnal dict with LST instead of phase_h
"""

import os
import numpy as np
import datetime

_LUNAR_DAY_H = 29.53 * 24.0   # hours per synodic month

# SPICE kernel file names (must exist in data_dir)
_KERNEL_FILES = [
    'naif0012.tls',
    'pck00010.tpc',
    'de430.bsp',
    'moon_pa_de421_1900-2050.bpc',
    'moon_080317.tf',
]

_kernels_loaded = False


def load_kernels(data_dir=None):
    """
    Load all required SPICE kernels.

    Parameters
    ----------
    data_dir : path to folder containing the kernel files.
               Defaults to <repo_root>/data/.
    """
    global _kernels_loaded
    import spiceypy as spice

    if _kernels_loaded:
        return

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    data_dir = os.path.abspath(data_dir)

    missing = [f for f in _KERNEL_FILES
               if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        raise FileNotFoundError(
            f'Missing SPICE kernels in {data_dir}:\n  ' +
            '\n  '.join(missing) +
            '\n\nDownload from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/'
        )

    for fname in _KERNEL_FILES:
        spice.furnsh(os.path.join(data_dir, fname))

    _kernels_loaded = True


def utc_to_lst(utc_datetimes, site_lon_deg, data_dir=None):
    """
    Convert an array of UTC datetimes to Local Solar Time hours at a given
    lunar surface longitude.

    Parameters
    ----------
    utc_datetimes : array-like of datetime.datetime (UTC, timezone-aware or naive)
    site_lon_deg  : surface longitude of the site in degrees (0–360 east)
    data_dir      : path to SPICE kernel directory (default: data/)

    Returns
    -------
    lst_hours : ndarray — Local Solar Time in hours, range [0, LUNAR_DAY_H)
                where 0 = midnight, LUNAR_DAY_H/2 = noon
    """
    import spiceypy as spice

    load_kernels(data_dir)

    utc_datetimes = np.asarray(utc_datetimes)
    lst_hours     = np.empty(len(utc_datetimes))

    for i, dt in enumerate(utc_datetimes):
        # Format UTC string for SPICE
        if hasattr(dt, 'utcoffset') and dt.utcoffset() is not None:
            dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        utc_str = dt.strftime('%Y-%m-%dT%H:%M:%S')

        et = spice.utc2et(utc_str)

        # Sub-solar point on Moon (principal-axis body-fixed frame)
        spoint, _, _ = spice.subslr(
            'INTERCEPT/ELLIPSOID', 'MOON', et, 'MOON_PA', 'LT+S', 'SUN')

        _, ssl_lon_rad, _ = spice.reclat(spoint)
        ssl_lon_deg = np.degrees(ssl_lon_rad) % 360.0

        # LST: angular distance from sub-solar point, mapped to hours
        # +180 shift: site_lon - ssl_lon = 0 (Sun overhead) → lst_frac=0.5 → noon
        # site_lon - ssl_lon = 180 (Sun at antipode) → lst_frac=0  → midnight
        lst_frac      = ((site_lon_deg - ssl_lon_deg + 180.0) / 360.0) % 1.0
        lst_hours[i]  = lst_frac * _LUNAR_DAY_H

    return lst_hours


def utc_to_lst_fast(utc_unix_days, site_lon_deg, data_dir=None, batch=500):
    """
    Faster vectorised version of utc_to_lst for large arrays.

    Parameters
    ----------
    utc_unix_days : array of Unix timestamps in *days* (float)
    site_lon_deg  : site longitude (degrees)
    batch         : process this many points at a time (memory control)

    Returns
    -------
    lst_hours : ndarray in [0, LUNAR_DAY_H)
    """
    import spiceypy as spice

    load_kernels(data_dir)

    utc_unix_days = np.asarray(utc_unix_days, dtype=float)
    n             = len(utc_unix_days)
    lst_hours     = np.empty(n)

    # SPICE J2000 epoch = 2000-01-01 12:00:00 TT
    # Unix epoch        = 1970-01-01 00:00:00 UTC
    # Offset (seconds): J2000 - Unix = 946727935.816 s  (accounts for leap seconds)
    _J2000_UNIX_S = 946727935.816

    for start in range(0, n, batch):
        end   = min(start + batch, n)
        chunk = utc_unix_days[start:end]

        # Convert Unix days → ephemeris time (ET seconds past J2000)
        et_arr = chunk * 86400.0 - _J2000_UNIX_S

        for j, et in enumerate(et_arr):
            spoint, _, _ = spice.subslr(
                'INTERCEPT/ELLIPSOID', 'MOON', et, 'MOON_PA', 'LT+S', 'SUN')
            _, ssl_lon_rad, _ = spice.reclat(spoint)
            ssl_lon_deg = np.degrees(ssl_lon_rad) % 360.0
            lst_frac    = ((site_lon_deg - ssl_lon_deg + 180.0) / 360.0) % 1.0
            lst_hours[start + j] = lst_frac * _LUNAR_DAY_H

    return lst_hours


def get_apollo_lst(site_name, n_lunar_days=None, data_dir=None):
    """
    Return Apollo HFE probe diurnal data with the time axis expressed in
    true Local Solar Time (hours) computed via SPICE, replacing the
    arbitrary phase-folded UTC reference used in hfe_loader.

    Parameters
    ----------
    site_name    : 'Apollo 15' or 'Apollo 17'
    n_lunar_days : limit to last N lunar days of each stable window (None = all)
    data_dir     : path to SPICE kernels

    Returns
    -------
    dict  {depth_cm: {'lst_h'  : ndarray  — LST hours [0, LUNAR_DAY_H)
                      'T_raw'  : ndarray  — absolute temperature (K)
                      'T_anom' : ndarray  — T − mean(T) (K)
                      'T_mean' : float    — mean temperature
                      'sensor' : str
                      'stype'  : str      — 'TG', 'TR', or 'TC'}}
    """
    from lunar.hfe_loader import load_site, _STABLE_WINDOWS, _PROBE_FILES
    import os

    LUNAR_DAY_DAYS = 29.53

    probes  = load_site(site_name)
    windows = _STABLE_WINDOWS[site_name]

    # Site longitude
    from lunar.constants import APOLLO_SITES
    site_lon = APOLLO_SITES[site_name]['lon']

    result = {}

    for probe, (win_start, win_end) in zip(probes, windows):
        all_t0 = min(
            (data['times'][0].timestamp() / 86400
             for data in probe.values() if len(data['times']) > 0),
            default=None,
        )
        if all_t0 is None:
            continue

        sel_end   = float(win_end)
        if n_lunar_days is None:
            sel_start = float(win_start)
        else:
            sel_start = max(float(win_start),
                            sel_end - n_lunar_days * LUNAR_DAY_DAYS)

        for sensor, data in probe.items():
            d_cm  = data['depth_cm']
            times = data['times']
            temps = data['temps']
            if len(temps) < 10:
                continue

            t_unix = np.array([t.timestamp() / 86400 for t in times])
            t_days = t_unix - all_t0
            mask   = (t_days >= sel_start) & (t_days <= sel_end)
            if mask.sum() < 10:
                continue

            t_unix_sel = t_unix[mask]
            T_sel      = temps[mask]

            # Compute true LST via SPICE
            lst_h = utc_to_lst_fast(t_unix_sel, site_lon, data_dir=data_dir)

            sort_idx = np.argsort(lst_h)
            lst_s    = lst_h[sort_idx]
            T_s      = T_sel[sort_idx]
            T_mean   = float(np.mean(T_s))
            T_anom   = T_s - T_mean

            stype = ''.join(c for c in sensor if c.isalpha())[:2]

            # Keep sensor with most readings per depth
            if d_cm not in result or mask.sum() > result[d_cm].get('_n', 0):
                result[d_cm] = {
                    'lst_h':  lst_s,
                    'T_raw':  T_s,
                    'T_anom': T_anom,
                    'T_mean': T_mean,
                    'sensor': sensor,
                    'stype':  stype,
                    '_n':     int(mask.sum()),
                }

    for v in result.values():
        v.pop('_n', None)

    return result
