"""
horizon.py — Horizon-profile computation for topographic shadowing.

For a given pixel on the DEM, we scan outward in every azimuth direction and
record the maximum elevation angle seen.  If the Sun is below that angle for
a given direction, the surface is in shadow and solar flux is zero.

Key functions
-------------
compute_horizon_profile()   — Scan all azimuths; returns horizon angles.
check_illumination()        — Is the sun above the horizon?
compute_sky_view_factor()   — Sky-view factor (SVF) from horizon profile.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def compute_horizon_profile(row, col, elev_m, pixel_m, az_angles,
                             max_range_px=3000):
    """
    Compute the horizon elevation angle in every azimuth direction.

    Scans outward from (row, col) along each azimuth ray, recording the
    maximum angular elevation of any terrain obstacle.

    Parameters
    ----------
    row, col      : centre pixel
    elev_m        : full DEM array (metres)
    pixel_m       : pixel size (metres)
    az_angles     : 1-D array of azimuths to sample (radians, 0 = north,
                    increasing clockwise)
    max_range_px  : maximum search distance (pixels)

    Returns
    -------
    horizons : 1-D float32 array, same length as az_angles
               — elevation angle of the horizon (radians) in each direction
    """
    H, W   = elev_m.shape
    elev0  = elev_m[row, col]
    n_az   = len(az_angles)
    horizons = np.full(n_az, -np.pi / 2.0, dtype=np.float32)

    for k in range(n_az):
        az   = az_angles[k]
        dx   =  np.sin(az)   # east component
        dy   =  np.cos(az)   # north component

        x, y = float(col), float(row)
        max_angle = -np.pi / 2.0
        dist      = 0.0
        step      = 2.0

        while dist < max_range_px:
            x   += dx * step
            y   -= dy * step   # rows increase southward
            dist += step

            jj = int(round(x))
            ii = int(round(y))
            if ii < 0 or ii >= H or jj < 0 or jj >= W:
                break

            horiz_dist = np.hypot(x - col, y - row) * pixel_m
            dz         = elev_m[ii, jj] - elev0
            angle      = np.arctan2(dz, horiz_dist + 1e-6)

            if angle > max_angle:
                max_angle = angle

            # Coarsen the step at large distances (faster, negligible accuracy loss)
            if dist > 500  and step < 4.0:
                step = 4.0
            elif dist > 1500 and step < 8.0:
                step = 8.0

        horizons[k] = max_angle

    return horizons


@njit(cache=True, fastmath=True, inline='always')
def check_illumination(solar_zenith, solar_azimuth, horizons, az_angles):
    """
    Return True if the Sun is above the local horizon (surface is lit).

    Parameters
    ----------
    solar_zenith   : solar zenith angle (radians); π/2 = on horizon, >π/2 = below
    solar_azimuth  : solar azimuth (radians, 0 = north, clockwise)
    horizons       : output of compute_horizon_profile
    az_angles      : same az_angles array used in compute_horizon_profile
    """
    if solar_zenith >= np.pi / 2.0:
        return False   # Sun below geometric horizon

    solar_elev = np.pi / 2.0 - solar_zenith
    n_az       = len(az_angles)
    idx        = int(round(solar_azimuth / (2.0 * np.pi) * n_az)) % n_az

    return solar_elev > horizons[idx]


@njit(cache=True, fastmath=True)
def compute_sky_view_factor(horizons):
    """
    Compute the sky-view factor (SVF) from a horizon profile.

    SVF = 1 means the full sky hemisphere is visible (flat terrain).
    SVF < 1 means surrounding terrain blocks part of the sky.

    Uses the analytical integral:
        SVF = (1 / 2π) ∫ cos³(h(φ)) dφ
    where h(φ) is the horizon elevation angle at azimuth φ.
    """
    n_az = len(horizons)
    daz  = 2.0 * np.pi / n_az
    svf  = 0.0
    for k in range(n_az):
        h    = horizons[k]
        svf += np.cos(h) ** 3 * daz / (2.0 * np.pi)
    return svf
