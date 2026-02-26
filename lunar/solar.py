"""
solar.py — Solar geometry and direct solar-flux calculation.

The Moon's obliquity is ~1.54°, giving a small but non-zero seasonal
variation in solar declination.  The dominant effect is the slow rotation
(one revolution per synodic month), which drives the extreme ~300 K
diurnal temperature swings.

Key functions
-------------
solar_geometry()    — Compute sun position (zenith angle, azimuth) at time t.
direct_solar_flux() — Net solar flux absorbed by a sloped surface.
"""

import numpy as np
from numba import njit

from lunar.constants import S0, LUNAR_DAY


@njit(cache=True, fastmath=True, inline='always')
def solar_geometry(lat_deg, lon_deg, t_sec):
    """
    Compute the solar position as seen from (lat, lon) at time t_sec.

    The Moon rotates once per synodic month (29.53 days = LUNAR_DAY seconds).
    Its orbital obliquity (~1.54°) produces a small seasonal wobble in the
    solar declination.

    Parameters
    ----------
    lat_deg : latitude  (degrees, positive = north)
    lon_deg : longitude (degrees, 0–360 east)
    t_sec   : elapsed time since an arbitrary epoch (seconds)

    Returns
    -------
    zenith   : solar zenith angle (radians; 0 = overhead, π/2 = on horizon)
    azimuth  : solar azimuth (radians; 0 = north, increasing clockwise)
    cos_zen  : cosine of zenith angle (handy for flux calculations)
    """
    lat   = np.deg2rad(lat_deg)

    # Solar declination: Moon's obliquity 1.54° oscillating over 27.3-day orbit
    delta = np.deg2rad(1.54) * np.sin(2.0 * np.pi * t_sec / (27.3 * 86400.0))

    # Hour angle from lunar rotation  (lon in radians as initial phase)
    h = (t_sec / LUNAR_DAY) * 2.0 * np.pi + np.deg2rad(lon_deg)

    # Zenith angle
    cos_zen = (np.sin(lat) * np.sin(delta) +
               np.cos(lat) * np.cos(delta) * np.cos(h))
    cos_zen = min(1.0, max(-1.0, cos_zen))
    zenith  = np.arccos(cos_zen)

    # Azimuth
    if abs(np.cos(lat)) < 1e-6:
        azimuth = h % (2.0 * np.pi)
    else:
        sin_az  =  np.sin(h) * np.cos(delta)
        cos_az  =  np.cos(h) * np.sin(lat) * np.cos(delta) - np.sin(delta) * np.cos(lat)
        azimuth = np.arctan2(sin_az, cos_az)
        if azimuth < 0.0:
            azimuth += 2.0 * np.pi

    return zenith, azimuth, cos_zen


@njit(cache=False, fastmath=True, inline='always')
def direct_solar_flux(zenith, azimuth, slope, aspect,
                      sunscale=1.0, albedo=0.09):
    """
    Net solar energy flux absorbed by a sloped surface (W/m²).

    Accounts for:
      • Surface tilt relative to the Sun.
      • Simple Lambertian albedo (fraction of light reflected away).
      • An optional sunscale multiplier for local calibration.

    Parameters
    ----------
    zenith   : solar zenith angle (radians)
    azimuth  : solar azimuth (radians, 0 = north)
    slope    : surface slope angle (radians)
    aspect   : surface aspect, clockwise from north (radians)
    sunscale : solar flux multiplier (1.0 = nominal; >1 = more energy)
    albedo   : Bond albedo (fraction reflected; 0 = absorb all, 1 = reflect all)

    Returns
    -------
    flux : absorbed solar flux (W/m²); 0 if surface faces away from Sun
    """
    # Angle between Sun vector and surface normal
    cos_inc = (np.cos(zenith) * np.cos(slope) +
               np.sin(zenith) * np.sin(slope) * np.cos(aspect - azimuth))

    if cos_inc <= 0.0:
        return 0.0   # Sun behind the surface

    return sunscale * (1.0 - albedo) * S0 * cos_inc
