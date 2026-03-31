"""
models.py — Density and thermal-property models for lunar regolith.

Three density models are available:

  'discrete'          — Sharp layers based on Apollo drill-core measurements.
  'hayne_exponential' — Smooth exponential compaction (Hayne et al. 2017,
                        doi:10.1002/2017JE005387), using Diviner global data.
  'custom'            — Template for a user-defined profile; edit the
                        density_custom() / k_solid_custom() functions below.

Usage
-----
    from lunar.models import (
        thermal_conductivity, heat_capacity,
        MODEL_ID_MAP,
        set_hayne_h, set_layer1_h,
    )
    # Choose a model once
    model_id = MODEL_ID_MAP['discrete']
    # Compute k at any depth z (m) and temperature T (K)
    k = thermal_conductivity(T=250.0, z=0.1, chi=2.7, model_id=model_id)

The H-parameter (scale height in the Hayne model, or Layer-1 thickness in
the discrete model) can be changed at runtime with set_hayne_h() /
set_layer1_h().  Note: numba functions read these globals at *compile* time,
so the pure-Python versions (density_hayne_py, k_solid_hayne_py, etc.) must
be used when sweeping H during sensitivity analysis.
"""

import numpy as np
from numba import njit

# ── Module-level globals for dynamic parameters ───────────────────────────────
# Use the set_*() functions below to change these before model runs.
_HAYNE_H     = 0.07    # Hayne exponential scale height (m) — default 7 cm
_H_LAYER1    = 0.07    # Discrete Layer-1 boundary (m)      — default 7 cm
_RHO_SURFACE = 1100.0  # Top-layer surface density (kg/m³)  — default 1100


def set_hayne_h(h_m: float):
    """Set the Hayne e-folding scale height (metres) for subsequent runs."""
    global _HAYNE_H
    _HAYNE_H = float(h_m)


def set_layer1_h(h_m: float):
    """Set the discrete Layer-1 thickness (metres) for subsequent runs."""
    global _H_LAYER1
    _H_LAYER1 = float(h_m)


def set_rho_surface(rho_kg_m3: float):
    """
    Set the top-layer regolith density (kg/m³) for pure-Python model runs.

    Typical range: 800 (very fluffy, freshly disturbed) – 1400 (compact).
    Apollo drill-core measurements suggest ~1100 kg/m³ for the uppermost dust.

    Note: the Numba-compiled solver uses a compile-time constant (1100 kg/m³).
    This setter affects the pure-Python functions (density_*_py) used for
    sensitivity sweeps and the interactive density-profile viewer.
    """
    global _RHO_SURFACE
    _RHO_SURFACE = float(rho_kg_m3)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — DISCRETE LAYERS  (Apollo drill-core based)
# ─────────────────────────────────────────────────────────────────────────────
# Layer boundaries:
#   Layer 1 : 0 → _H_LAYER1      — fluffy surface dust (ρ = 1100 kg/m³)
#   Layer 2 : _H_LAYER1 → 20 cm  — transitional compaction (linear ramp)
#   Layer 3 : > 20 cm            — consolidated regolith (slow ramp)
#
# Conductivity values:
#   k_surface = 1.0e-3 W/m/K — unconsolidated dust; from Langseth et al. 1976
#               (Apollo 15 & 17 HFE fit to near-surface thermal gradient)
#   k_deep    = 1.2e-2 W/m/K — consolidated regolith at > 20 cm.
#               Derived from the Apollo heat flow measurement itself:
#               k = Q_basal / (dT/dz)  ≈  0.018 W/m² / 1.5 K/m  ≈  1.2e-2 W/m/K
#               where dT/dz ≈ 1.5 K/m is the measured deep gradient (Langseth 1973).
#
# IMPORTANT — calibration basis differs between models:
#   Discrete model k_deep is calibrated to Apollo HFE geothermal gradient data
#   (steady-state heat flow at 0.3–2.3 m depth).
#   Hayne model k_deep = 3.8e-3 is calibrated to Diviner surface brightness
#   temperatures (diurnal skin depth, topmost ~20 cm).
#   These represent different depth ranges and physical processes; the factor
#   of ~3 difference between models is physically meaningful, not an error.
#   Use the discrete model when comparing against Apollo deep-temperature data;
#   use the Hayne model when comparing against Diviner surface observations.

@njit(cache=True, fastmath=True, inline='always')
def density_discrete(z):
    """Density (kg/m³) for the discrete-layer model.

    Based on Apollo 15 and 17 drill-core measurements (Carrier et al. 1991,
    Lunar Sourcebook, Table 9.1).
    """
    H     = 0.07   # Layer-1 boundary — COMPILE-TIME constant
    L2    = 0.20   # Layer-2 boundary (fixed at 20 cm)
    if z < H:
        return 1100.0
    elif z < L2:
        return 1100.0 + (1700.0 - 1100.0) * (z - H) / (L2 - H)
    else:
        return 1700.0 + (1800.0 - 1700.0) * min(1.0, (z - L2) / 2.80)


@njit(cache=True, fastmath=True, inline='always')
def k_solid_discrete(z):
    """Solid (contact) thermal conductivity (W/m/K) for discrete-layer model.

    Surface value (1e-3) from Langseth et al. 1976 HFE analysis.
    Deep value (1.2e-2) derived from measured Apollo heat flow Q ≈ 18–21 mW/m²
    and observed deep temperature gradient dT/dz ≈ 1.5 K/m (Langseth 1973).
    Calibrated to geothermal gradient data, not surface thermal inertia.
    """
    H  = 0.07
    L2 = 0.20
    if z < H:
        return 1.0e-3
    elif z < L2:
        return 1.0e-3 + (1.0e-2 - 1.0e-3) * (z - H) / (L2 - H)
    else:
        return 1.2e-2


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — HAYNE ET AL. 2017  (Diviner global mapping)
# ─────────────────────────────────────────────────────────────────────────────
# ρ(z) = ρ_d − (ρ_d − ρ_s) · exp(−z / H)
#   ρ_s = 1100 kg/m³  (surface)
#   ρ_d = 1800 kg/m³  (deep)
#   H   = 0.07 m      (7 cm global average from Diviner)
# Conductivity is scaled linearly from surface to deep values.

@njit(cache=True, fastmath=True, inline='always')
def density_hayne(z):
    """Density (kg/m³) for the Hayne 2017 exponential model (H = 7 cm fixed).

    ρ(z) = ρ_d − (ρ_d − ρ_s) · exp(−z / H)
    Reference: Hayne et al. 2017, JGR Planets, doi:10.1002/2017JE005387

    H=0.07 m matches the discrete model's Layer-1 depth for a fair comparison;
    the Hayne et al. (2017) global best-fit from Diviner is H=0.06 m.
    """
    rho_s = 1100.0
    rho_d = 1800.0
    H     = 0.07  # H=0.07 m here; Hayne 2017 global best-fit is H=0.06 m
    return rho_d - (rho_d - rho_s) * np.exp(-z / H)


@njit(cache=True, fastmath=True, inline='always')
def k_solid_hayne(z):
    """Solid conductivity (W/m/K) for Hayne 2017 model.

    k_surf = 7.4e-4 W/m/K, k_deep = 3.8e-3 W/m/K from Hayne et al. 2017
    Table 1 (fit to Diviner surface brightness temperatures, diurnal skin
    depth regime).

    CALIBRATION NOTE: these k values are fit to Diviner surface observations
    (topmost ~20 cm, diurnal timescale) rather than to the Apollo deep
    geothermal gradient.  The asymptotic k_deep = 3.8e-3 W/m/K implies a
    temperature gradient dT/dz = Q_basal / k ≈ 4.7 K/m at depth, which is
    steeper than the observed Apollo gradient (~1.5 K/m).  This reflects the
    different calibration basis of the two models — use the discrete model for
    geothermal comparisons and the Hayne model for surface thermal inertia
    comparisons with Diviner data.
    """
    rho_s = 1100.0
    rho_d = 1800.0
    H     = 0.07
    rho   = rho_d - (rho_d - rho_s) * np.exp(-z / H)
    k_surf = 7.4e-4   # W/m/K at surface   (Hayne et al. 2017, Table 1)
    k_deep = 3.8e-3   # W/m/K at ~1 m depth (Hayne et al. 2017, Table 1)
    frac   = (rho - rho_s) / (rho_d - rho_s)
    return k_surf + (k_deep - k_surf) * frac


@njit(cache=True, fastmath=True, inline='always')
def density_hayne_h(z, H):
    """Density (kg/m³) for Hayne 2017 model with variable scale height H (m)."""
    rho_s = 1100.0
    rho_d = 1800.0
    return rho_d - (rho_d - rho_s) * np.exp(-z / H)


@njit(cache=True, fastmath=True, inline='always')
def k_solid_hayne_h(z, H):
    """Solid conductivity (W/m/K) for Hayne 2017 model with variable H (m)."""
    rho_s  = 1100.0
    rho_d  = 1800.0
    rho    = rho_d - (rho_d - rho_s) * np.exp(-z / H)
    k_surf = 7.4e-4
    k_deep = 3.4e-3
    frac   = (rho - rho_s) / (rho_d - rho_s)
    return k_surf + (k_deep - k_surf) * frac


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — CUSTOM  (user-defined template)
# ─────────────────────────────────────────────────────────────────────────────
# Edit density_custom() and k_solid_custom() to implement your own profile.
# Examples:
#   Power law  : return 1100.0 + 700.0 * (z / 3.0) ** 0.5
#   Two-layer  : return 1100.0 if z < 0.05 else 1800.0
#   Data-driven: use np.interp inside a non-njit wrapper

@njit(cache=True, fastmath=True, inline='always')
def density_custom(z):
    """Custom density model — edit this function."""
    return density_discrete(z)   # placeholder: same as discrete


@njit(cache=True, fastmath=True, inline='always')
def k_solid_custom(z):
    """Custom solid conductivity — edit this function."""
    return k_solid_discrete(z)   # placeholder: same as discrete


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID_MAP = {
    'discrete':           0,
    'hayne_exponential':  1,
    'custom':             2,
}

MODEL_NAMES = {v: k for k, v in MODEL_ID_MAP.items()}   # reverse lookup


@njit(cache=True, fastmath=True, inline='always')
def get_density(z, model_id, H_param):
    """Return density (kg/m³) at depth z for the selected model."""
    if model_id == 0:
        return density_discrete(z)
    elif model_id == 1:
        return density_hayne_h(z, H_param)
    else:
        return density_custom(z)


@njit(cache=True, fastmath=True, inline='always')
def get_k_solid(z, model_id, H_param):
    """Return solid conductivity (W/m/K) at depth z for the selected model."""
    if model_id == 0:
        return k_solid_discrete(z)
    elif model_id == 1:
        return k_solid_hayne_h(z, H_param)
    else:
        return k_solid_custom(z)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED THERMAL PROPERTIES
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True, inline='always')
def heat_capacity(T):
    """
    Specific heat capacity (J/kg/K) — temperature-dependent polynomial.

    Reference: Hemingway et al. (1973), Lunar Sci. Conf., Table 1.
    Valid for T ∈ [20, 420] K; positive throughout this range.
    Returns a minimum of 10 J/kg/K as a physical guard below ~10 K.
    """
    c = np.array([-3.6125, 2.7431, 2.3616e-3, -1.234e-5, 8.9093e-9])
    cp = c[0] + T * (c[1] + T * (c[2] + T * (c[3] + T * c[4])))
    # Guard: polynomial can theoretically go negative below ~10 K
    return cp if cp > 10.0 else 10.0


@njit(cache=True, fastmath=True, inline='always')
def thermal_conductivity(T, z, chi, model_id, H_param):
    """
    Total thermal conductivity (W/m/K).

    k_total = k_solid · (1 + chi · (T / T_ref)³)

    The radiative (T³) term represents grain-to-grain radiation and grows
    strongly with temperature (important on the hot dayside surface).

    Parameters
    ----------
    T        : temperature (K)
    z        : depth (m)
    chi      : radiative conductivity parameter (typical range 1.5–4)
    model_id : 0 = discrete, 1 = hayne, 2 = custom
    H_param  : Hayne scale height (m); used only when model_id == 1
    """
    k_s   = get_k_solid(z, model_id, H_param)
    T_ref = 350.0
    return k_s * (1.0 + chi * (T / T_ref) ** 3)


# ─────────────────────────────────────────────────────────────────────────────
# PURE-PYTHON VERSIONS — used for H-parameter sensitivity sweeps
# ─────────────────────────────────────────────────────────────────────────────
# numba functions capture globals at compile time, so H cannot be changed
# at runtime inside @njit.  These plain-Python versions read the module
# globals (_HAYNE_H, _H_LAYER1) on every call.

def density_hayne_py(z, H=None, rho_surface=None):
    """Pure-Python Hayne density (reads _HAYNE_H / _RHO_SURFACE if not given)."""
    h     = H            if H            is not None else _HAYNE_H
    rho_s = rho_surface  if rho_surface  is not None else _RHO_SURFACE
    rho_d = 1800.0
    return rho_d - (rho_d - rho_s) * np.exp(-z / h)


def k_solid_hayne_py(z, H=None):
    """Pure-Python Hayne conductivity."""
    h = H if H is not None else _HAYNE_H
    rho_s, rho_d = 1100.0, 1800.0
    rho   = rho_d - (rho_d - rho_s) * np.exp(-z / h)
    k_surf, k_deep = 7.4e-4, 3.8e-3   # k_deep from Hayne et al. 2017, Table 1
    return k_surf + (k_deep - k_surf) * (rho - rho_s) / (rho_d - rho_s)


def density_discrete_py(z, H=None, rho_surface=None):
    """Pure-Python discrete density (reads _H_LAYER1 / _RHO_SURFACE if not given)."""
    h     = H            if H            is not None else _H_LAYER1
    rho_s = rho_surface  if rho_surface  is not None else _RHO_SURFACE
    L2    = 0.20
    if z < h:
        return rho_s
    elif z < L2:
        return rho_s + (1700.0 - rho_s) * (z - h) / (L2 - h)
    else:
        return 1700.0 + (1800.0 - 1700.0) * min(1.0, (z - L2) / 2.80)


def k_solid_discrete_py(z, H=None):
    """Pure-Python discrete conductivity."""
    h  = H if H is not None else _H_LAYER1
    L2 = 0.20
    if z < h:
        return 1.0e-3
    elif z < L2:
        return 1.0e-3 + (1.0e-2 - 1.0e-3) * (z - h) / (L2 - h)
    else:
        return 1.2e-2
