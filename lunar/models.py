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

# ── Module-level globals for dynamic H-parameter ──────────────────────────────
# Use set_hayne_h() / set_layer1_h() to change these before model runs.
_HAYNE_H     = 0.07   # Hayne exponential scale height (m) — default 7 cm
_H_LAYER1    = 0.07   # Discrete Layer-1 boundary (m)      — default 7 cm


def set_hayne_h(h_m: float):
    """Set the Hayne e-folding scale height (metres) for subsequent runs."""
    global _HAYNE_H
    _HAYNE_H = float(h_m)


def set_layer1_h(h_m: float):
    """Set the discrete Layer-1 thickness (metres) for subsequent runs."""
    global _H_LAYER1
    _H_LAYER1 = float(h_m)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — DISCRETE LAYERS  (Apollo drill-core based)
# ─────────────────────────────────────────────────────────────────────────────
# Layer boundaries:
#   Layer 1 : 0 → _H_LAYER1      — fluffy surface dust (ρ = 1100 kg/m³)
#   Layer 2 : _H_LAYER1 → 20 cm  — transitional compaction (linear ramp)
#   Layer 3 : > 20 cm            — consolidated regolith (slow ramp)

@njit(cache=True, fastmath=True, inline='always')
def density_discrete(z):
    """Density (kg/m³) for the discrete-layer model."""
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
    """Solid (contact) thermal conductivity (W/m/K) for discrete-layer model."""
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
    """Density (kg/m³) for the Hayne 2017 exponential model (H = 7 cm fixed)."""
    rho_s = 1100.0
    rho_d = 1800.0
    H     = 0.07
    return rho_d - (rho_d - rho_s) * np.exp(-z / H)


@njit(cache=True, fastmath=True, inline='always')
def k_solid_hayne(z):
    """Solid conductivity (W/m/K) for Hayne 2017 model."""
    rho_s = 1100.0
    rho_d = 1800.0
    H     = 0.07
    rho   = rho_d - (rho_d - rho_s) * np.exp(-z / H)
    k_surf = 7.4e-4   # W/m/K at surface   (from Hayne et al. 2017)
    k_deep = 3.4e-3   # W/m/K at ~1 m depth
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
def get_density(z, model_id):
    """Return density (kg/m³) at depth z for the selected model."""
    if model_id == 0:
        return density_discrete(z)
    elif model_id == 1:
        return density_hayne(z)
    else:
        return density_custom(z)


@njit(cache=True, fastmath=True, inline='always')
def get_k_solid(z, model_id):
    """Return solid conductivity (W/m/K) at depth z for the selected model."""
    if model_id == 0:
        return k_solid_discrete(z)
    elif model_id == 1:
        return k_solid_hayne(z)
    else:
        return k_solid_custom(z)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED THERMAL PROPERTIES
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True, inline='always')
def heat_capacity(T):
    """
    Specific heat capacity (J/kg/K) — temperature-dependent polynomial.
    From Hemingway et al. (1973).
    """
    c = np.array([-3.6125, 2.7431, 2.3616e-3, -1.234e-5, 8.9093e-9])
    return c[0] + T * (c[1] + T * (c[2] + T * (c[3] + T * c[4])))


@njit(cache=True, fastmath=True, inline='always')
def thermal_conductivity(T, z, chi, model_id):
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
    """
    k_s   = get_k_solid(z, model_id)
    T_ref = 350.0
    return k_s * (1.0 + chi * (T / T_ref) ** 3)


# ─────────────────────────────────────────────────────────────────────────────
# PURE-PYTHON VERSIONS — used for H-parameter sensitivity sweeps
# ─────────────────────────────────────────────────────────────────────────────
# numba functions capture globals at compile time, so H cannot be changed
# at runtime inside @njit.  These plain-Python versions read the module
# globals (_HAYNE_H, _H_LAYER1) on every call.

def density_hayne_py(z, H=None):
    """Pure-Python Hayne density (reads _HAYNE_H if H is None)."""
    h = H if H is not None else _HAYNE_H
    rho_s, rho_d = 1100.0, 1800.0
    return rho_d - (rho_d - rho_s) * np.exp(-z / h)


def k_solid_hayne_py(z, H=None):
    """Pure-Python Hayne conductivity."""
    h = H if H is not None else _HAYNE_H
    rho_s, rho_d = 1100.0, 1800.0
    rho   = rho_d - (rho_d - rho_s) * np.exp(-z / h)
    k_surf, k_deep = 7.4e-4, 3.4e-3
    return k_surf + (k_deep - k_surf) * (rho - rho_s) / (rho_d - rho_s)


def density_discrete_py(z, H=None):
    """Pure-Python discrete density (reads _H_LAYER1 if H is None)."""
    h  = H if H is not None else _H_LAYER1
    L2 = 0.20
    if z < h:
        return 1100.0
    elif z < L2:
        return 1100.0 + (1700.0 - 1100.0) * (z - h) / (L2 - h)
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
