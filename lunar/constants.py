"""
constants.py — Physical constants, depth-grid settings, and Apollo data.

Everything here is read-only.  Import with:
    from lunar.constants import *
or pick what you need:
    from lunar.constants import sigma, LUNAR_DAY, APOLLO_DATA
"""

import warnings
import numpy as np

# ── Physical constants ─────────────────────────────────────────────────────────
R_MOON     = 1_737_400.0        # Moon mean radius (m)
sigma      = 5.670374419e-8     # Stefan-Boltzmann constant (W m⁻² K⁻⁴)
S0         = 1361.0             # Solar constant at 1 AU (W m⁻²)
Q_basal    = 18e-3              # Basal heat flux (W m⁻²) — average of A15 & A17

# ── Site-specific basal heat flux (Langseth et al. 1976, revised values) ──────
# Published Langseth 1976 revised values after borestem + transient correction:
#   Apollo 15: Q = 21 mW/m²
#   Apollo 17: Q = 16 mW/m²
# The shared Q_basal = 18 mW/m² above is their mean — a common simplification.
# Systematic bias if used per-site: A15 is ~3 mW/m² too low (→ ~0.4 K cooler at 2 m);
#                                    A17 is ~2 mW/m² too high (→ ~0.3 K warmer at 2 m).
# Reference: Langseth, Keihm & Peters (1976), J. Geophys. Res., 81, 5765–5778.
Q_BASAL_A15 = 21e-3             # Apollo 15 revised heat flux (W m⁻²)
Q_BASAL_A17 = 16e-3             # Apollo 17 revised heat flux (W m⁻²)
Q_BASAL_SITES = {
    'Apollo 15': Q_BASAL_A15,
    'Apollo 17': Q_BASAL_A17,
}
LUNAR_DAY  = 29.53 * 86400.0   # Synodic lunar day (s)

# ── Default surface properties (can be overridden in config) ───────────────────
DEFAULT_ALBEDO     = 0.09   # Bond albedo — typical undisturbed, space-weathered regolith
DEFAULT_EMISSIVITY = 0.95   # Infrared emissivity — typical lunar regolith

# ── Albedo variants: disturbed vs undisturbed regolith ─────────────────────────
# Space weathering (solar wind, micrometeorite gardening) darkens the surface
# by creating agglutinates and nanophase iron → A ≈ 0.07–0.09.
# When astronauts disturb the surface they expose fresher, less-weathered
# material that reflects more light → A ≈ 0.12–0.15.
# This has a measurable effect on surface temperature: ΔT_surf ≈ 5–12 K.
# References: Lucey et al. 2000 (J. Geophys. Res.); Hapke 2001; Pieters & Englert 1993
UNDISTURBED_ALBEDO    = 0.09   # Same as DEFAULT_ALBEDO (space-weathered baseline)
DISTURBED_ALBEDO      = 0.12   # Freshly disturbed regolith — drill site / boot tracks
FRESH_HIGHLAND_ALBEDO = 0.15   # Very fresh highland material (crater rays, etc.)

# ── Apollo borestem (fiberglass drill casing) thermal properties ──────────────
# The Apollo HFE probes were lowered into bore holes lined with a hollow
# fiberglass "borestem" casing (roughly 2.5 cm outer diameter).
# Fiberglass k_f ≈ 0.04 W/m/K — about 40× higher than dry regolith surface
# conductivity (~0.001 W/m/K). This creates a "thermal short-circuit" that
# preferentially conducts heat from the warm surface layer down to the sensors,
# causing readings to be slightly warmer than the true undisturbed regolith.
# References: Langseth et al. 1976; Warren 1969 (HFE hardware description)
BORESTEM_OUTER_RADIUS_M  = 0.0125   # 2.5 cm outer diameter → 1.25 cm radius
BORESTEM_WALL_M          = 0.003    # ~3 mm fiberglass wall thickness
K_FIBERGLASS             = 0.04     # W/m/K — typical E-glass / epoxy composite
BORESTEM_DEPTH_M         = 2.5      # Depth of fiberglass casing (m)
BORESTEM_DISTURBED_RADIUS = 0.05    # Radius of mechanically disturbed zone (m)

# ── Probe-top solar radiation constants ────────────────────────────────────────
# The Apollo probe cable and connector hardware at the surface absorbed direct
# solar radiation. This heat was conducted downward through the probe body and
# cable, raising temperatures at the sensors above the true regolith value.
# Effect is largest near the surface and for shallower sensors.
# Estimated from Langseth et al. 1976 thermal disturbance analysis.
PROBE_TOP_AREA_CM2  = 3.0    # Effective solar-absorbing area of probe top (cm²)
PROBE_TOP_ALBEDO    = 0.25   # Polished metallic hardware — lower absorption than regolith
PROBE_CABLE_K_EFF   = 0.02   # Effective axial conductance of probe cable (W/m/K)
PROBE_CABLE_AREA_M2 = 5.0e-6 # Cable cross-sectional area (m²) — ~2.5 mm diameter

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
# Depths are in metres.  The 'depth' column in the .tab files is in centimetres;
# confirmed against NASA/NSSDCA documentation (Langseth et al. 1972, 1976):
#   A15 Probe 1 reached 1.39 m, Probe 2 reached 0.97 m.
#   A17 both probes reached ≈ 2.34 m.
#
# If the data files are unavailable the fallback tuples below are used.
_APOLLO_15_FALLBACK = [
    # (depth_m, T_K) — depths from published sensor positions (Langseth 1976)
    # Shallow sensors (< 0.80 m) are excluded by min_depth_cm=80 filter
    (0.35, 252.8), (0.45, 252.8), (0.49, 252.9), (0.59, 252.9),
    (0.73, 252.9), (0.84, 253.0), (0.87, 252.9), (0.91, 253.2),
    (0.97, 253.2), (1.01, 253.2), (1.29, 253.3), (1.39, 253.5),
]
_APOLLO_17_FALLBACK = [
    # (depth_m, T_K) — depths from published sensor positions (Langseth 1976)
    # Shallow cable sensors (< 0.80 m) excluded by min_depth_cm=80 filter
    (0.14, 255.0), (0.15, 255.0), (0.66, 255.8), (0.67, 255.8),
    (1.30, 256.1), (1.31, 256.1), (1.40, 256.2), (1.67, 256.3),
    (1.69, 256.3), (1.77, 256.4), (1.78, 256.4), (1.85, 256.4),
    (1.86, 256.4), (1.95, 256.5), (1.96, 256.5), (2.23, 256.6),
    (2.24, 256.6), (2.33, 256.7), (2.34, 256.7),
]

def _load_apollo_data():
    """Load equilibrium temperatures from HFE files; fall back to literals."""
    def _normalise(rows):
        """Ensure every row is a (depth_m, T_K, sensor_type) 3-tuple.
        Old versions of get_equilibrium_temps returned 2-tuples; pad those."""
        return [r if len(r) == 3 else (*r, 'TG') for r in rows]

    try:
        from lunar.hfe_loader import get_equilibrium_temps
        # min_depth_cm=80 excludes shallow TC cable sensors (z < 50 cm):
        #   - TC sensors are diurnally active (within the ~50 cm skin depth)
        #   - TC sensors are attached to probe hardware that absorbs direct solar
        #     radiation, causing readings to be biased warm in daytime data
        # The 80 cm cutoff matches hfe_loader documentation ("gives a conservative
        # margin below the diurnal skin depth"). Including TC sensors at 14–67 cm
        # in RMSE validation inflates apparent model error by ~2–4 K.
        a15 = _normalise(get_equilibrium_temps('Apollo 15', min_depth_cm=80))
        a17 = _normalise(get_equilibrium_temps('Apollo 17', min_depth_cm=80))
        return a15, a17
    except Exception as exc:
        warnings.warn(
            f"HFE data files could not be loaded ({exc}). "
            "Falling back to hardcoded equilibrium temperatures from Langseth 1976. "
            "Place data/a15p*.tab and data/a17p*.tab in the project root to use "
            "real Apollo measurements.",
            UserWarning,
            stacklevel=2,
        )
        # Fallback literals: tag everything as 'TG' (conservative)
        a15 = [(d, T, 'TG') for d, T in _APOLLO_15_FALLBACK]
        a17 = [(d, T, 'TG') for d, T in _APOLLO_17_FALLBACK]
        return a15, a17

_a15_data, _a17_data = _load_apollo_data()

# True landing-site coordinates (NSSDCA / Lunar and Planetary Institute)
APOLLO_SITES = {
    'Apollo 15': {'lat': 26.1322, 'lon':  3.6339},
    'Apollo 17': {'lat': 20.1908, 'lon': 30.7717},
}

# Convenience dict: name → numpy arrays of (depths_m, temps_K) + sensor types
APOLLO_DATA = {
    'Apollo 15': {
        'depths':       np.array([d  for d, _, _  in _a15_data]),
        'temps':        np.array([T  for _, T, _  in _a15_data]),
        'sensor_types': [st          for _, _, st in _a15_data],
    },
    'Apollo 17': {
        'depths':       np.array([d  for d, _, _  in _a17_data]),
        'temps':        np.array([T  for _, T, _  in _a17_data]),
        'sensor_types': [st          for _, _, st in _a17_data],
    },
}
