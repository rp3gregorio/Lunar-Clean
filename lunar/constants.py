"""
constants.py — Physical constants, depth-grid settings, and Apollo data.

Everything here is read-only.  Import with:
    from lunar.constants import *
or pick what you need:
    from lunar.constants import sigma, LUNAR_DAY, APOLLO_DATA
"""

import numpy as np

# ── Physical constants ─────────────────────────────────────────────────────────
R_MOON     = 1_737_400.0        # Moon mean radius (m)
sigma      = 5.670374419e-8     # Stefan-Boltzmann constant (W m⁻² K⁻⁴)
S0         = 1361.0             # Solar constant at 1 AU (W m⁻²)
Q_basal    = 18e-3              # Basal heat flux (W m⁻²) — from Apollo HFE
LUNAR_DAY  = 29.53 * 86400.0   # Synodic lunar day (s)

# ── Default surface properties (can be overridden in config) ───────────────────
DEFAULT_ALBEDO     = 0.09   # Bond albedo — typical lunar regolith
DEFAULT_EMISSIVITY = 0.95   # Infrared emissivity — typical lunar regolith

# ── Depth-grid configuration ───────────────────────────────────────────────────
# The model uses a non-uniform grid: fine near the surface (where most variation
# occurs and where Apollo drills reached), coarser at depth.
DZ_FINE    = 0.005   # 5 mm spacing for the top layer
DZ_COARSE  = 0.02    # 20 mm spacing below DEPTH_FINE
DEPTH_FINE = 0.10    # Fine grid down to 10 cm
DEPTH_MAX  = 3.0     # Total model depth (m)

# ── Apollo 15 & 17 Heat Flow Experiment (HFE) validation data ────────────────
# Equilibrium temperatures are derived from the *stable tail* of the actual
# HFE probe time-series (data/a15p*_depth.tab, data/a17p*_depth.tab).
# The last 25 % of each sensor's record is used (after the drilling transient
# has fully dissipated).  Source: hfe_loader.get_equilibrium_temps()
#
# If the data files are unavailable the fallback tuples below are used.
_APOLLO_15_FALLBACK = [
    (0.035, 252.8), (0.045, 252.8), (0.049, 252.9), (0.059, 252.9),
    (0.073, 252.9), (0.084, 253.0), (0.087, 252.9), (0.091, 253.2),
    (0.097, 253.2), (0.101, 253.2), (0.129, 253.3), (0.139, 253.5),
]
_APOLLO_17_FALLBACK = [
    (0.014, 255.0), (0.015, 255.0), (0.066, 255.8), (0.067, 255.8),
    (0.130, 256.1), (0.131, 256.1), (0.140, 256.2), (0.167, 256.3),
    (0.169, 256.3), (0.177, 256.4), (0.178, 256.4), (0.185, 256.4),
    (0.186, 256.4), (0.195, 256.5), (0.196, 256.5), (0.223, 256.6),
    (0.224, 256.6), (0.233, 256.7), (0.234, 256.7),
]

def _load_apollo_data():
    """Load equilibrium temperatures from HFE files; fall back to literals."""
    try:
        from lunar.hfe_loader import get_equilibrium_temps
        a15 = get_equilibrium_temps('Apollo 15')
        a17 = get_equilibrium_temps('Apollo 17')
        return a15, a17
    except Exception:
        return _APOLLO_15_FALLBACK, _APOLLO_17_FALLBACK

_a15_data, _a17_data = _load_apollo_data()

# True landing-site coordinates (NSSDCA / Lunar and Planetary Institute)
APOLLO_SITES = {
    'Apollo 15': {'lat': 26.1322, 'lon':  3.6339},
    'Apollo 17': {'lat': 20.1908, 'lon': 30.7717},
}

# Convenience dict: name → numpy arrays of (depths_m, temps_K)
APOLLO_DATA = {
    'Apollo 15': {
        'depths': np.array([d for d, _ in _a15_data]),
        'temps':  np.array([t for _, t in _a15_data]),
    },
    'Apollo 17': {
        'depths': np.array([d for d, _ in _a17_data]),
        'temps':  np.array([t for _, t in _a17_data]),
    },
}
