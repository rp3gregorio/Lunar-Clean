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

# ── Apollo 15 & 17 Heat Flow Experiment (HFE) data ───────────────────────────
# Each tuple is (depth_m, temperature_K) measured in the drill holes.
# Source: Apollo HFE publications.
APOLLO_15_DATA = [
    (0.40, 253.0), (0.54, 252.9), (0.64, 252.8), (0.73, 252.8),
    (0.91, 252.9), (1.01, 252.8), (1.29, 253.1), (1.39, 253.3),
]

APOLLO_17_DATA = [
    (0.15, 255.5), (0.67, 256.3), (1.31, 256.4), (1.40, 256.2),
    (1.69, 256.4), (1.78, 256.6), (1.86, 256.6), (1.96, 256.6),
    (2.24, 256.8), (2.34, 256.9),
]

# Coordinates of the two Apollo sites used for validation
APOLLO_SITES = {
    'Apollo 15': {'lat': 26.1323, 'lon':  3.6285, 'data': APOLLO_15_DATA},
    'Apollo 17': {'lat': 20.1911, 'lon': 30.7723, 'data': APOLLO_17_DATA},
}

# Convenience dict: name → numpy arrays
APOLLO_DATA = {}
for _name, _info in APOLLO_SITES.items():
    _d = np.array([row[0] for row in _info['data']])
    _t = np.array([row[1] for row in _info['data']])
    APOLLO_DATA[_name] = {'depths': _d, 'temps': _t}
