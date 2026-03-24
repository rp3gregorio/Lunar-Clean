"""
borestem.py — Thermal corrections for Apollo HFE borestem (fiberglass casing)
              and probe-top solar radiation heating.

Physical background
-------------------
Two instrumental effects cause the Apollo Heat Flow Experiment sensors to read
slightly *warmer* than the true undisturbed regolith at the same depth:

1. BORESTEM THERMAL SHORT-CIRCUIT
   The Apollo HFE drill holes were lined with a hollow fiberglass "borestem"
   casing (~2.5 cm outer diameter).  Fiberglass has k_f ≈ 0.04 W/m/K, roughly
   40× higher than dry regolith surface conductivity (k_s ≈ 0.001–0.003 W/m/K).
   This creates a preferential thermal path from the hot near-surface regolith
   layer down to the sensors.  The effect is strongest at shallow depths where
   the vertical temperature gradient is steepest, and diminishes below ~1 m.

   Estimated magnitude: +0.5 K to +3 K for sensors at 0.3–1.4 m depth.

2. PROBE-TOP SOLAR RADIATION
   The top of the Apollo probe assembly (connector plug + cable) was exposed at
   the surface and absorbed direct solar radiation.  This energy was conducted
   downward through the probe body and cable, raising temperatures at the sensors
   above the true regolith value.  The cable ran the full length of the probe
   (up to ~2.3 m) and was in direct thermal contact with the surrounding regolith.

   Estimated magnitude: +0.2 K to +1.5 K (largest during local noon, at
   shallower sensor depths).

References
----------
Langseth M.G. et al. 1976, J. Geophys. Res., 81, 3143–3161
Warren J.L. 1969, "Apollo Lunar Surface Experiments Package — HFE description"
Grott M. et al. 2010, J. Geophys. Res. (Mars InSight analogy for borestem effect)
Kiefer W.S. & Macke R.J. 2018, Lunar & Planet. Sci. Conf.

Key functions
-------------
borestem_k_effective(k_regolith, z_grid)
    Compute the effective conductivity of the borestem-disturbed zone.

borestem_temperature_correction(z_grid, T_mean, k_profile, T_surf_mean)
    Steady-state temperature perturbation from the fiberglass borestem.

probe_top_radiation_correction(Q_solar_day, z_sensor_m)
    Steady-state temperature offset from probe-top solar heating.

apply_all_corrections(T_mean, z_grid, k_profile, T_surf_mean, Q_solar_mean)
    Apply both corrections and return corrected profile + breakdown.
"""

import numpy as np

from lunar.constants import (
    BORESTEM_OUTER_RADIUS_M,
    BORESTEM_WALL_M,
    K_FIBERGLASS,
    BORESTEM_DEPTH_M,
    BORESTEM_DISTURBED_RADIUS,
    PROBE_TOP_AREA_CM2,
    PROBE_TOP_ALBEDO,
    PROBE_CABLE_K_EFF,
    PROBE_CABLE_AREA_M2,
    Q_basal,
    S0,
)


# ─────────────────────────────────────────────────────────────────────────────
# BORESTEM GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def borestem_fiberglass_area() -> float:
    """Cross-sectional area of the fiberglass annulus (m²)."""
    r_o = BORESTEM_OUTER_RADIUS_M
    r_i = r_o - BORESTEM_WALL_M
    return np.pi * (r_o**2 - r_i**2)


def borestem_disturbed_area() -> float:
    """Total cross-sectional area of the mechanically disturbed zone (m²)."""
    return np.pi * BORESTEM_DISTURBED_RADIUS**2


def borestem_area_fraction() -> float:
    """Fraction of the disturbed-zone cross-section occupied by fiberglass."""
    return borestem_fiberglass_area() / borestem_disturbed_area()


# ─────────────────────────────────────────────────────────────────────────────
# BORESTEM EFFECTIVE CONDUCTIVITY
# ─────────────────────────────────────────────────────────────────────────────

def borestem_k_effective(k_regolith: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    """
    Effective thermal conductivity of the borestem-disturbed column (W/m/K).

    Uses a parallel heat-flow composite rule within the disturbed zone:

        k_eff(z) = k_reg(z) · (1 − f) + k_fg · f

    where f = A_fiberglass / A_disturbed_total (area fraction, ~0.013).

    Below BORESTEM_DEPTH_M the conductivity reverts to undisturbed regolith.

    Parameters
    ----------
    k_regolith : array (n_depths,) — regolith k at each grid depth (W/m/K)
    z_grid     : array (n_depths,) — depths (m)

    Returns
    -------
    k_eff : array (n_depths,) — effective k in borestem-disturbed zone
    """
    f = borestem_area_fraction()
    k_reg = np.asarray(k_regolith, dtype=float)
    k_eff = k_reg * (1.0 - f) + K_FIBERGLASS * f

    # Beyond borestem depth, revert to undisturbed regolith
    below = z_grid > BORESTEM_DEPTH_M
    k_eff[below] = k_reg[below]

    return k_eff


# ─────────────────────────────────────────────────────────────────────────────
# BORESTEM TEMPERATURE CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def borestem_temperature_correction(
    z_grid: np.ndarray,
    T_mean: np.ndarray,
    k_profile: np.ndarray,
    T_surf_mean: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the steady-state temperature offset caused by the fiberglass borestem.

    Approach — thermal-resistance perturbation:
    In steady 1-D heat conduction, the temperature at depth z is related to the
    surface temperature and the cumulative thermal resistance R(z) = ∫₀ᶻ dz/k(z).

    With borestem:   T_bs(z)  = T_surf − Q_surf · R_eff(z)
    Without:         T_reg(z) = T_surf − Q_surf · R_reg(z)

    Difference:  ΔT_bs(z) = Q_surf · (R_reg(z) − R_eff(z))

    Because k_eff > k_reg at every depth, R_eff < R_reg, so ΔT_bs > 0 (warmer).

    Parameters
    ----------
    z_grid      : array (n,) — depth grid (m)
    T_mean      : array (n,) — model mean temperature profile (K)
    k_profile   : array (n,) — model conductivity profile (W/m/K)
    T_surf_mean : float — mean surface temperature (K)

    Returns
    -------
    dT_borestem : array (n,) — temperature correction, K (positive = warmer)
    k_eff       : array (n,) — effective borestem conductivity (W/m/K)
    """
    k_reg = np.asarray(k_profile, dtype=float)
    k_eff = borestem_k_effective(k_reg, z_grid)

    # Cumulative thermal resistance (K/W · m²) from surface to each depth
    dz   = np.gradient(z_grid)
    R_reg = np.cumsum(dz / k_reg)
    R_eff = np.cumsum(dz / k_eff)

    # Estimate the downward surface heat flux driving the borestem correction.
    # The near-surface gradient (node 0–1) is dominated by the diurnal wave
    # and can be 10–100× larger than the quasi-steady conductive flux.  Using
    # a deeper gradient (below the diurnal skin depth, z > 0.3 m) gives a
    # physically correct estimate of the flux driving the steady-state bias.
    skin_depth = 0.30   # below this, diurnal amplitude < ~1% of surface swing
    iz_deep = 1
    for _i in range(len(z_grid)):
        if z_grid[_i] >= skin_depth:
            iz_deep = _i
            break
    if iz_deep >= len(z_grid) - 1:
        iz_deep = len(z_grid) // 2

    if len(z_grid) > iz_deep + 1 and z_grid[iz_deep + 1] > z_grid[iz_deep]:
        dz_deep  = z_grid[iz_deep + 1] - z_grid[iz_deep]
        k_mid    = 0.5 * (k_reg[iz_deep] + k_reg[iz_deep + 1])
        Q_surf   = abs(k_mid * (float(T_mean[iz_deep + 1]) - float(T_mean[iz_deep]))
                       / dz_deep)
        # Sanity check: Q_surf should be close to Q_basal at depth; fall back
        # to Q_basal if the estimate is unreasonably small (< 0.1 mW/m²).
        if Q_surf < 1e-4:
            Q_surf = abs(Q_basal)
    else:
        Q_surf = abs(Q_basal)

    dT_borestem = Q_surf * (R_reg - R_eff)

    # Zero-out below borestem depth (no casing there)
    dT_borestem[z_grid > BORESTEM_DEPTH_M] = 0.0

    return dT_borestem, k_eff


# ─────────────────────────────────────────────────────────────────────────────
# PROBE-TOP RADIATION CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def probe_top_radiation_correction(
    Q_solar_mean: float,
    z_sensor_m: float,
    probe_length_m: float = 2.5,
    k_regolith: float = K_FIBERGLASS,
) -> float:
    """
    Steady-state temperature offset at a sensor from solar heating of the
    probe-top hardware (connector plug, cable head).

    The probe top absorbs solar power and conducts it into the borestem casing.
    Heat spreads radially from the probe-top point source using the half-space
    Green's function (Carslaw & Jaeger §10.2):

        ΔT(z) = Q_top / (2π · k_eff · z)

    where Q_top (W) is the mean absorbed power and k_eff is the effective
    thermal conductivity of the primary conduction path.  Because heat travels
    preferentially through the high-conductivity fiberglass casing before
    spreading into the regolith, k_eff ≈ K_FIBERGLASS = 0.04 W/m/K gives
    physically correct corrections of 0.1–1 K (Langseth et al. 1976).

    Parameters
    ----------
    Q_solar_mean  : mean absorbed solar flux at the surface (W/m²)
    z_sensor_m    : sensor depth (m) — positive downward
    probe_length_m: unused, kept for API compatibility
    k_regolith    : effective conductivity (W/m/K); default = K_FIBERGLASS.

    Returns
    -------
    dT_probe_top : scalar temperature offset (K) — positive = warmer
    """
    # The shallowest Apollo sensors are at ~35 cm; below ~3 cm the probe-top
    # hardware itself occupies the hole, so the correction is not meaningful.
    _Z_MIN = 0.03   # 3 cm — probe-top hardware height
    if z_sensor_m <= _Z_MIN:
        return 0.0

    # Heat absorbed by probe-top hardware (W)
    A_top_m2 = PROBE_TOP_AREA_CM2 * 1e-4                         # cm² → m²
    Q_top_W  = (1.0 - PROBE_TOP_ALBEDO) * Q_solar_mean * A_top_m2

    # Half-space point-source spreading: R = 1 / (2π k z)
    dT = Q_top_W / (2.0 * np.pi * k_regolith * z_sensor_m)

    return float(dT)


def probe_top_correction_profile(
    Q_solar_mean: float,
    z_grid: np.ndarray,
    probe_length_m: float = 2.5,
    k_regolith=None,
) -> np.ndarray:
    """
    Probe-top radiation temperature correction at every depth in z_grid.

    Parameters
    ----------
    Q_solar_mean : mean absorbed solar flux at the surface (W/m²)
    z_grid       : depth array (m)
    probe_length_m: unused, kept for API compatibility
    k_regolith   : scalar or array (n,) — regolith conductivity (W/m/K).
                   If None, uses the default 0.02 W/m/K.

    Returns
    -------
    dT_profile : array (n,) — correction in K at each depth
    """
    z_grid = np.asarray(z_grid, dtype=float)
    if k_regolith is None:
        k_arr = np.full(len(z_grid), 0.02)
    elif np.isscalar(k_regolith):
        k_arr = np.full(len(z_grid), float(k_regolith))
    else:
        k_arr = np.asarray(k_regolith, dtype=float)

    dT = np.array([
        probe_top_radiation_correction(Q_solar_mean, float(z), probe_length_m, float(k))
        for z, k in zip(z_grid, k_arr)
    ])
    return dT


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def apply_all_corrections(
    T_mean: np.ndarray,
    z_grid: np.ndarray,
    k_profile: np.ndarray,
    T_surf_mean: float,
    Q_solar_mean: float,
    apply_borestem: bool = True,
    apply_probe_top: bool = True,
    probe_length_m: float = 2.5,
    use_2d_borestem: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Apply borestem and probe-top corrections to a mean temperature profile.

    Parameters
    ----------
    T_mean           : array (n,) — uncorrected mean temperature at each depth (K)
    z_grid           : array (n,) — depth grid (m)
    k_profile        : array (n,) — conductivity profile (W/m/K)
    T_surf_mean      : float — mean surface temperature (K)
    Q_solar_mean     : float — mean absorbed solar flux (W/m²)  [daytime average]
    apply_borestem   : if True, include fiberglass borestem correction
    apply_probe_top  : if True, include probe-top solar radiation correction
    probe_length_m   : length of probe for taper calculation (m)
    use_2d_borestem  : if True (default), use the physically correct 2-D
                       axisymmetric FD solver (borestem2d); if False, fall back
                       to the faster 1-D parallel-composite approximation.

    Returns
    -------
    T_corrected : array (n,) — corrected temperature profile (K)
    breakdown   : dict with keys:
        'borestem'     — correction from fiberglass casing (K)
        'probe_top'    — correction from solar heating of probe top (K)
        'total'        — sum of all corrections (K)
        'k_eff'        — borestem-effective conductivity (W/m/K) or None
        'borestem_method' — '2d_cylindrical' or '1d_composite'
    """
    T_corr = np.array(T_mean, dtype=float)
    zeros  = np.zeros_like(z_grid)
    k_eff_out = None

    if apply_borestem:
        if use_2d_borestem:
            from lunar.borestem2d import borestem_2d_correction
            dT_bs = borestem_2d_correction(z_grid, T_mean, k_profile, T_surf_mean)
            method = '2d_cylindrical'
        else:
            dT_bs, k_eff_out = borestem_temperature_correction(
                z_grid, T_mean, k_profile, T_surf_mean
            )
            method = '1d_composite'
        T_corr += dT_bs
    else:
        dT_bs  = zeros.copy()
        method = 'none'

    if apply_probe_top:
        dT_pt = probe_top_correction_profile(Q_solar_mean, z_grid, probe_length_m)
        T_corr += dT_pt
    else:
        dT_pt = zeros.copy()

    return T_corr, {
        'borestem':        dT_bs,
        'probe_top':       dT_pt,
        'total':           dT_bs + dT_pt,
        'k_eff':           k_eff_out,
        'borestem_method': method,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: ESTIMATE MEAN DAYTIME SOLAR FLUX
# ─────────────────────────────────────────────────────────────────────────────

def mean_daytime_solar_flux(lat_deg: float, albedo: float = 0.09,
                             sunscale: float = 1.0) -> float:
    """
    Approximate mean absorbed solar flux over the lit half of a lunar day.

    Uses a simple cosine-weighted integral over hour angles ±π/2:

        <Q> = (S0 · (1−A) · sunscale · cos(lat)) / π

    This is the Lambert-sphere mean for a flat surface at the given latitude.

    Parameters
    ----------
    lat_deg  : latitude in degrees
    albedo   : Bond albedo
    sunscale : solar flux multiplier

    Returns
    -------
    Q_mean : mean absorbed solar flux (W/m²)
    """
    lat_rad = np.deg2rad(lat_deg)
    Q_mean  = S0 * (1.0 - albedo) * sunscale * np.cos(lat_rad) / np.pi
    return float(Q_mean)
