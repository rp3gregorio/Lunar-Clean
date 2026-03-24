"""
solar.py — Solar geometry and direct solar-flux calculation.

The Moon's obliquity is ~1.54°, giving a small but non-zero seasonal
variation in solar declination.  The dominant effect is the slow rotation
(one revolution per synodic month), which drives the extreme ~300 K
diurnal temperature swings.

Key functions
-------------
solar_geometry()           — Compute sun position (zenith angle, azimuth) at time t.
direct_solar_flux()        — Net solar flux absorbed by a sloped surface.
heliocentric_flux_factor() — Earth's orbital eccentricity correction to S0.

Heliocentric distance correction
---------------------------------
Earth's orbital eccentricity (e ≈ 0.0167) causes the Earth-Moon system to
move ±1.67 % in heliocentric distance over the year, producing a ±3.3 %
variation in solar flux (±45 W/m²).  Over a 5-year dataset (Apollo 1971–1977)
this can shift modelled surface temperatures by ±5–8 K at local noon.

The main solver (solve_thermal_model) uses a fixed S0 for simplicity and
speed.  For studies requiring seasonal accuracy, scale the sunscale parameter:

    from lunar.solar import heliocentric_flux_factor
    sunscale = heliocentric_flux_factor(epoch_jd, t_sec)

where epoch_jd is the Julian Date of the simulation start (t_sec = 0).
"""

import numpy as np
from numba import njit

from lunar.constants import S0, LUNAR_DAY

# Julian Date of perihelion 2000 Jan 3 (J2000 + 3 days ≈ 2451547.0)
# and Earth's mean orbital period.
_JD_PERIHELION_2000 = 2451547.0   # JD of perihelion closest to J2000
_EARTH_YEAR_DAYS    = 365.25


def heliocentric_flux_factor(epoch_jd, t_sec=0.0):
    """
    Solar flux correction factor for Earth's orbital eccentricity.

    Returns the ratio (actual solar flux) / S0 at the given time, accounting
    for the variation in heliocentric distance over Earth's elliptical orbit.

    Uses the low-eccentricity approximation:
        r ≈ a (1 − e cos(M))    →    S ∝ 1/r² ≈ 1 + 2e cos(M)
    where M is the mean anomaly (zero at perihelion), e = 0.0167.

    Parameters
    ----------
    epoch_jd : Julian Date at t_sec = 0 (start of simulation).
               Examples:
                 Apollo 15 landing: JD 2441495.7  (1971 Jul 31)
                 Apollo 17 landing: JD 2441680.6  (1972 Dec 11)
    t_sec    : elapsed time since epoch_jd (seconds); default 0.

    Returns
    -------
    factor : float — multiply S0 by this to get the corrected solar constant.
             Range ≈ 0.967 – 1.034 (perihelion Jan, aphelion Jul).

    Example
    -------
    >>> # Correct sunscale for Apollo 17 epoch
    >>> from lunar.solar import heliocentric_flux_factor
    >>> factor = heliocentric_flux_factor(2441680.6, t_sec=0.0)
    >>> # Pass factor*sunscale into solve_thermal_model as sunscale
    """
    e = 0.0167   # Earth orbital eccentricity
    jd_now = epoch_jd + t_sec / 86400.0
    # Mean anomaly (radians): 0 at perihelion, advances 2π per year
    M = 2.0 * np.pi * ((jd_now - _JD_PERIHELION_2000) % _EARTH_YEAR_DAYS) / _EARTH_YEAR_DAYS
    # Flux ∝ 1/r² ≈ (1 + e cos M)²  ≈  1 + 2e cos M  for small e
    return (1.0 + e * np.cos(M)) ** 2


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
