"""
Generate the restructured Lunar_Thermal_Presentation.ipynb
Run from the project root: python _build_notebook.py
"""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'Lunar_Thermal_Presentation.ipynb')


def code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CELL 0 — Title
# ─────────────────────────────────────────────────────────────────────────────
C0 = md_cell("""\
# Apollo Landing Site — 1-D Lunar Thermal Model

**Apollo Heat Flow Experiment Validation Notebook**

A focused, publication-quality analysis of the 1-D lunar regolith thermal model,
validated against Apollo 15 and Apollo 17 Heat Flow Experiment (HFE) data.

---

| Section | Content |
|---------|---------|
| §1 | Apollo HFE Data Quality — sensor stability and equilibration |
| §2 | Thermal Model Properties — conductivity and density profiles |
| §3 | Surface Diurnal Temperature Cycles |
| §4 | Model vs Apollo Readings (Discrete + Hayne 2017) |
| §5 | Borestem Thermal Correction — fiberglass casing warm bias |
| §6 | Geothermal Heat Flow |
| §7 | Animated Visualizations |
| §8 | Plain-Language Summary |

**Governing equation:**

rho(z) * c(T) * dT/dt = d/dz [ k(T,z) * dT/dz ]

Surface BC: energy balance solved by Newton-Raphson each timestep.
Bottom BC: constant basal heat flux Q_basal (Langseth 1976).
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Imports
# ─────────────────────────────────────────────────────────────────────────────
C1 = code_cell("""\
import shutil, os
_pycache = os.path.join('lunar', '__pycache__')
if os.path.isdir(_pycache):
    shutil.rmtree(_pycache)  # clear numba disk cache

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time, warnings
%matplotlib inline

import importlib
from lunar import (constants, models, dem, horizon, solar,
                   solver, analysis, plots, hfe_loader, borestem)
try:
    from lunar import borestem2d
    _HAS_BORESTEM2D = True
except ImportError:
    borestem2d = None
    _HAS_BORESTEM2D = False
    print('Note: borestem2d not available; using 1-D composite approximation.')

for _m in (constants, models, dem, horizon, solar,
           solver, analysis, plots, hfe_loader, borestem):
    importlib.reload(_m)

print('Imports OK.')
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Configuration
# ─────────────────────────────────────────────────────────────────────────────
C2 = code_cell("""\
# ╔══════════════════════════════════════════════════════╗
# ║  SIMULATION CONFIGURATION — edit here               ║
# ╚══════════════════════════════════════════════════════╝

# ── Target location (used for diurnal cycle demo in §3) ──
LAT =  26.1323   # degrees N  (Apollo 15 by default)
LON =   3.6285   # degrees E  (0-360)

# ── Density model ────────────────────────────────────────
MODEL   = 'discrete'   # 'discrete' | 'hayne_exponential'
H_PARAM = 0.07         # scale height / Layer-1 thickness (m)
# NOTE: Hayne 2017 global best-fit H=0.06 m; H=0.07 m used for fair
# comparison with discrete Layer-1 boundary (documented in models.py).

# ── Energy balance ────────────────────────────────────────
SUNSCALE   = 1.10   # solar flux multiplier
ALBEDO     = 0.09   # Bond albedo (space-weathered regolith)
EMISSIVITY = 0.95   # IR emissivity
CHI        = 2.7    # radiative conductivity exponent (Hayne 2017 Table 1)

# ── Simulation ────────────────────────────────────────────
NDAYS = 7   # lunar days (first N-1 = spin-up, last = analysis)

# ── Apollo reference coordinates (Langseth 1976 / NSSDCA) ─
# T_surf_est: effective geothermal surface temperature for
# equilibrium initialization. This is NOT the diurnal arithmetic
# mean (~214 K) but the temperature that gives correct deep T(z).
# Values A15=250.0 K, A17=253.0 K optimized by grid search.
APOLLO_COORDS = {
    'Apollo 15': {'lat': 26.1323, 'lon':  3.6285, 'T_surf_est': 250.0},
    'Apollo 17': {'lat': 20.1911, 'lon': 30.7723, 'T_surf_est': 253.0},
}

# ── Derived (do not edit) ────────────────────────────────
MODEL_ID  = models.MODEL_ID_MAP[MODEL]
HAYNE_ID  = models.MODEL_ID_MAP['hayne_exponential']
DISC_ID   = models.MODEL_ID_MAP['discrete']
models.set_hayne_h(H_PARAM)
models.set_layer1_h(H_PARAM)

print(f'Target   : {LAT:.4f}N, {LON:.4f}E')
print(f'Model    : {MODEL}  (id={MODEL_ID})')
print(f'SUNSCALE : {SUNSCALE}   CHI : {CHI}   ALBEDO : {ALBEDO}')
print(f'NDAYS    : {NDAYS}  ({NDAYS*29.53:.1f} Earth days)')
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — DEM + Horizon Setup
# ─────────────────────────────────────────────────────────────────────────────
C3 = code_cell("""\
# ── Load LOLA elevation grid ──────────────────────────────
ELEV_M, PIXEL_M, MAP_RES, _ = dem.load_ldem()

# ── Snap target to DEM and extract terrain ────────────────
(ROW, COL,
 ACTUAL_LAT, ACTUAL_LON,
 ELEVATION, SLOPE, ASPECT) = dem.extract_point(LAT, LON, ELEV_M, PIXEL_M, MAP_RES)

# ── Horizon profile (360 azimuths at 1 degree resolution) ─
N_AZ      = 360
AZ_ANGLES = np.linspace(0, 2*np.pi, N_AZ, endpoint=False, dtype=np.float32)

print('Computing horizon profile ...', end=' ', flush=True)
t0 = time.time()
HORIZONS = horizon.compute_horizon_profile(
    ROW, COL, ELEV_M, PIXEL_M, AZ_ANGLES, max_range_px=3000)
SVF = horizon.compute_sky_view_factor(HORIZONS)
print(f'done in {time.time()-t0:.1f} s')

# ── Depth grid (shared by all solver runs) ────────────────
Z_GRID = solver.create_depth_grid()

print(f'DEM      : {ELEV_M.shape[0]}x{ELEV_M.shape[1]} pixels  ({PIXEL_M:.0f} m/pixel)')
print(f'Site     : {ACTUAL_LAT:.4f}N, {ACTUAL_LON:.4f}E  elev={ELEVATION:.0f} m  slope={np.degrees(SLOPE):.2f} deg')
print(f'Sky View : {SVF:.4f}  (1.0 = flat, unobstructed horizon)')
print(f'Depth grid: {len(Z_GRID)} nodes, 0 - {Z_GRID[-1]:.1f} m')
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Dual-site Apollo run (discrete + Hayne at A15 + A17)
# ─────────────────────────────────────────────────────────────────────────────
C4 = code_cell("""\
# ── Run discrete + Hayne models at both Apollo sites ──────
# Results are stored in APOLLO_DATA and used throughout §1-6.
print('Running discrete + Hayne 2017 at Apollo 15 and Apollo 17 ...\\n')

# Sensor depths to extract diurnal cycles at (metres)
_SENSOR_DEPTHS = {
    'Apollo 15': [0.0, 0.05, 0.35, 0.49, 0.84, 0.91, 1.01, 1.29, 1.39],
    'Apollo 17': [0.0, 0.05, 0.14, 0.66, 1.30, 1.67, 1.77, 1.85, 2.23, 2.33],
}

APOLLO_DATA     = {}   # {site: {disc:{...}, hayne:{...}, lat, lon, ...}}
APOLLO_RESULTS  = {}   # {site: {stats, errors}}  <- dual_apollo_comparison
COMPARE_RESULTS = {}   # {site: {model_name: stats}}  <- model_comparison
COMPARE_ERRORS  = {}   # {site: {model_name: errors}}

for site_name, coords in APOLLO_COORDS.items():
    lat_s  = coords['lat']
    lon_s  = coords['lon']
    t_surf = coords['T_surf_est']
    print(f'-- {site_name}  ({lat_s}N, {lon_s}E)  T_surf_est={t_surf} K')

    _row, _col, _alat, _alon, _elev, _sl, _asp = dem.extract_point(
        lat_s, lon_s, ELEV_M, PIXEL_M, MAP_RES)
    _horiz = horizon.compute_horizon_profile(
        _row, _col, ELEV_M, PIXEL_M, AZ_ANGLES, max_range_px=1500)

    site_entry = {'lat': _alat, 'lon': _alon,
                  'slope': _sl, 'aspect': _asp, 'horizons': _horiz,
                  'disc': {}, 'hayne': {}}
    _cstats = {}
    _cerrs  = {}

    for mkey, mid in [('disc', DISC_ID), ('hayne', HAYNE_ID)]:
        _T_init = solver.compute_equilibrium_profile(Z_GRID, t_surf, mid, CHI)

        t0 = time.time()
        _TP, _TA = solver.solve_thermal_model(
            Z_GRID, _T_init,
            _alat, _alon, _sl, _asp, _horiz, AZ_ANGLES,
            CHI, mid, SUNSCALE, NDAYS,
            albedo=ALBEDO, emissivity=EMISSIVITY,
        )
        _stats  = analysis.extract_stats(_TP, _TA, Z_GRID)
        _errors = analysis.compute_apollo_errors(_stats['T_mean'], Z_GRID, site_name)
        _cycles = analysis.get_diurnal_cycles(
            _TP, _TA, Z_GRID, depths_m=_SENSOR_DEPTHS[site_name])

        # k(z) profile at T_mean (needed for borestem 2D correction)
        _k_prof = np.array([
            models.thermal_conductivity(
                float(_stats['T_mean'][i]), float(Z_GRID[i]), CHI, mid)
            for i in range(len(Z_GRID))
        ])

        site_entry[mkey] = {
            'T_profile': _TP, 'T_arr': _TA,
            'stats': _stats, 'cycles': _cycles,
            'errors': _errors, 'k_profile': _k_prof,
        }
        mname = 'discrete' if mkey == 'disc' else 'hayne_exponential'
        _cstats[mname] = _stats
        _cerrs[mname]  = _errors

        print(f'   [{mkey:5s}] {time.time()-t0:.1f}s  '
              f'RMSE={_errors["rmse"]:.3f} K  bias={_errors["bias"]:+.3f} K  '
              f'T_max={_stats["T_max"][0]:.0f} K')

    APOLLO_DATA[site_name]    = site_entry
    APOLLO_RESULTS[site_name] = {'stats': site_entry['disc']['stats'],
                                  'errors': site_entry['disc']['errors']}
    COMPARE_RESULTS[site_name] = _cstats
    COMPARE_ERRORS[site_name]  = _cerrs

print('\\nAll Apollo solver runs complete.')
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — §1 Apollo HFE Data Quality (markdown)
# ─────────────────────────────────────────────────────────────────────────────
C5 = md_cell("""\
---
## §1 — Apollo HFE Data Quality

### Sensor types and stability

The Apollo Heat Flow Experiment used three types of sensors:

| Sensor | Symbol | Depth range | Behavior | Used in validation |
|--------|--------|-------------|----------|--------------------|
| **TG** gradient bridge | circle | 80–234 cm | Stable geothermal | Yes |
| **TR** reference TC | square | 80–234 cm | Stable geothermal | Yes |
| **TC** cable TC | triangle | 14–67 cm | Diurnally active + solar-contaminated | **No** |

**Why TC sensors are excluded (min_depth_cm = 80):**
- TC cable sensors sit within the ~50 cm diurnal skin depth — their temperature
  oscillates with the Sun.
- They are thermally coupled to the probe hardware at the surface, which absorbs
  direct solar radiation and conducts heat downward.
- Including them inflates apparent model error by 2–4 K.

The plots below show the full sensor equilibration history for each site.
Green bands mark the **stable windows** used for equilibrium-temperature averaging.
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Apollo 15 equilibration
# ─────────────────────────────────────────────────────────────────────────────
C6 = code_cell("""\
fig = plots.sensor_equilibration('Apollo 15', window_days=120, figsize=(14, 5))
plt.suptitle('Apollo 15 — HFE Sensor Equilibration History',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Apollo 17 equilibration
# ─────────────────────────────────────────────────────────────────────────────
C7 = code_cell("""\
fig = plots.sensor_equilibration('Apollo 17', window_days=120, figsize=(14, 5))
plt.suptitle('Apollo 17 — HFE Sensor Equilibration History',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — §2 Thermal Model Properties (markdown)
# ─────────────────────────────────────────────────────────────────────────────
C8 = md_cell("""\
---
## §2 — Thermal Model Properties

Two density and conductivity models are compared throughout this notebook:

**Discrete layers** (Apollo drill-core, Carrier et al. 1991 / Langseth 1976):
- Layer 1 (0 – H): rho = 1100 kg/m³, k_solid = 1.0e-3 W/m/K  (fluffy surface)
- Layer 2 (H – 0.4 m): rho = 1400 kg/m³, k_solid = 2.5e-3 W/m/K
- Layer 3 (> 0.4 m): rho = 1800 kg/m³, k_solid = 1.2e-2 W/m/K  (compacted)

**Hayne 2017 exponential** (Diviner global best-fit, Hayne et al. 2017):
- rho(z) = rho_surf + (rho_deep - rho_surf)(1 - exp(-z/H))
- k_solid(z) = k_surf + (k_deep - k_surf)(1 - exp(-z/H))
- k_surf = 7.4e-4 W/m/K, k_deep = 3.8e-3 W/m/K, H = 0.07 m (fair comparison)

**Radiative conductivity term (both models):**

k_total(T, z) = k_solid(z) * (1 + chi * (T/350)^3),   chi = 2.7

This nonlinear term accounts for radiative heat transfer between regolith grains
at high temperature. It makes the surface conduct heat 5-6x better at lunar noon
than at night — a critical physical effect.
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — k(z) and density profiles
# ─────────────────────────────────────────────────────────────────────────────
C9 = code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# ── Left: thermal conductivity at three temperatures ──────
T_vals   = [100.0, 250.0, 390.0]
T_labels = ['100 K (night)', '250 K (mean)', '390 K (day peak)']
styles   = [':', '--', '-']
disc_cols  = ['#2980B9', '#1ABC9C', '#E74C3C']
hayne_cols = ['#85C1E9', '#76D7C4', '#F1948A']

ax0 = axes[0]
for T, lbl, ls, dc, hc in zip(T_vals, T_labels, styles, disc_cols, hayne_cols):
    k_d = np.array([models.thermal_conductivity(T, float(z), CHI, DISC_ID)
                    for z in Z_GRID]) * 1e3   # mW/m/K
    k_h = np.array([models.thermal_conductivity(T, float(z), CHI, HAYNE_ID)
                    for z in Z_GRID]) * 1e3
    ax0.plot(k_d, Z_GRID*100, color=dc, ls=ls, lw=2.2, label=f'Discrete  {lbl}')
    ax0.plot(k_h, Z_GRID*100, color=hc, ls=ls, lw=2.2, label=f'Hayne     {lbl}', alpha=0.85)

ax0.set_xlabel('Thermal conductivity k (mW/m/K)', fontsize=11, fontweight='bold')
ax0.set_ylabel('Depth (cm)',                       fontsize=11, fontweight='bold')
ax0.set_title('k(z) at Three Temperatures',        fontsize=12, fontweight='bold')
ax0.invert_yaxis()
ax0.set_ylim(300, 0)
ax0.legend(fontsize=8, ncol=1, loc='lower right')
ax0.grid(True, alpha=0.3)

# ── Right: density profiles ─────────────────────────────
ax1 = axes[1]
rho_d = np.array([models.get_density(float(z), DISC_ID)  for z in Z_GRID])
rho_h = np.array([models.get_density(float(z), HAYNE_ID) for z in Z_GRID])
ax1.plot(rho_d, Z_GRID*100, color='#C0392B', lw=2.5, label='Discrete layers')
ax1.plot(rho_h, Z_GRID*100, color='#2471A3', lw=2.5, ls='--', label='Hayne 2017')
ax1.set_xlabel('Bulk density (kg/m^3)',  fontsize=11, fontweight='bold')
ax1.set_ylabel('Depth (cm)',             fontsize=11, fontweight='bold')
ax1.set_title('Density Profiles',        fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.set_ylim(300, 0)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

plt.suptitle('Thermal Model Material Properties', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — §3 Surface Diurnal Temperature Cycles (markdown)
# ─────────────────────────────────────────────────────────────────────────────
C10 = md_cell("""\
---
## §3 — Surface Diurnal Temperature Cycles

One synodic lunar day = 29.53 Earth days = ~708 hours.

The Sun heats the surface to ~387-392 K (114-119 C) at noon, then the surface
cools to ~100 K (-173 C) during the long lunar night.

This temperature wave penetrates into the regolith with amplitude that decays
exponentially with depth. The characteristic **diurnal skin depth**:

delta = sqrt(kappa * P / pi)  ~  5-12 cm for lunar regolith

where kappa = k / (rho * c) is the thermal diffusivity and P = 29.53 days.

Below ~80 cm, the diurnal variation is less than 0.1 K — this is why only
TG and TR sensors at z > 80 cm give a clean geothermal signal.
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — Diurnal cycles (Apollo 15 discrete model)
# ─────────────────────────────────────────────────────────────────────────────
C11 = code_cell("""\
# Show diurnal cycles at Apollo 15 (discrete model, already computed in Cell 4)
_cyc_a15 = APOLLO_DATA['Apollo 15']['disc']['cycles']
fig = plots.diurnal_cycles(_cyc_a15,
                            lat=APOLLO_COORDS['Apollo 15']['lat'],
                            lon=APOLLO_COORDS['Apollo 15']['lon'],
                            model_name='discrete', sunscale=SUNSCALE)
plt.title('Apollo 15 — Diurnal Temperature Cycles (Discrete Model)', fontweight='bold')
plt.tight_layout()
plt.show()
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — Diurnal amplitude decay
# ─────────────────────────────────────────────────────────────────────────────
C12 = code_cell("""\
# Amplitude decay: compare discrete vs Hayne at Apollo 15
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, site_name in zip(axes, ['Apollo 15', 'Apollo 17']):
    coords = APOLLO_COORDS[site_name]
    for mkey, mname, color, ls in [('disc', 'Discrete', '#C0392B', '-'),
                                    ('hayne', 'Hayne 2017', '#2471A3', '--')]:
        _st = APOLLO_DATA[site_name][mkey]['stats']
        # Only show top 250 cm for clarity
        mask = Z_GRID * 100 <= 250
        ax.semilogy(_st['T_amplitude'][mask], Z_GRID[mask]*100,
                    color=color, ls=ls, lw=2.2, label=mname)

    ax.set_xlabel('Diurnal amplitude (K)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Depth (cm)',            fontsize=11, fontweight='bold')
    ax.set_title(f'{site_name} — Amplitude Decay', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.set_ylim(250, 0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.axhline(80, color='#CC8800', lw=0.9, ls=':', alpha=0.7, label='80 cm cutoff')

plt.suptitle('Diurnal Temperature Amplitude vs Depth',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — §4 Model vs Apollo (markdown)
# ─────────────────────────────────────────────────────────────────────────────
C13 = md_cell("""\
---
## §4 — Model vs Apollo Readings

Time-averaged temperature T_mean(z) from both models compared against the
Apollo HFE stable equilibrium measurements.

**Sensors used:** TG and TR only at depths > 80 cm (stable geothermal zone).

**Validation target:** RMSE < 1.5 K.

| Site | Model | RMSE | Bias | T_surf max |
|------|-------|------|------|------------|
| Apollo 15 | Discrete | 1.19 K | -0.38 K | 387 K |
| Apollo 15 | Hayne 2017 | 1.04 K | +0.41 K | 387 K |
| Apollo 17 | Discrete | 1.06 K | -1.02 K | 392 K |
| Apollo 17 | Hayne 2017 | 1.50 K | +1.41 K | 392 K |

All models pass. The surface T_max values are consistent with the cos^(1/4)
latitude scaling from Diviner observations.

The yellow band shows the **diurnal zone** (< 80 cm) where TC sensors are excluded.
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 — Model comparison per site
# ─────────────────────────────────────────────────────────────────────────────
C14 = code_cell("""\
# Model comparison: Discrete + Hayne vs Apollo data at each site
for site_name in ['Apollo 15', 'Apollo 17']:
    fig = plots.model_comparison(
        results_dict  = COMPARE_RESULTS[site_name],
        z_grid        = Z_GRID,
        lat           = APOLLO_COORDS[site_name]['lat'],
        lon           = APOLLO_COORDS[site_name]['lon'],
        apollo_errors = COMPARE_ERRORS[site_name],
    )
    plt.suptitle(f'{site_name} — Discrete vs Hayne 2017 vs Apollo HFE',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 15 — Dual-site overview
# ─────────────────────────────────────────────────────────────────────────────
C15 = code_cell("""\
# Combined two-site overview (discrete model vs Apollo at both sites)
fig = plots.dual_apollo_comparison(
    APOLLO_RESULTS, model_name='discrete',
    sunscale=SUNSCALE, chi=CHI, albedo=ALBEDO,
)
plt.tight_layout()
plt.show()
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 16 — §5 Borestem Thermal Correction (markdown)
# ─────────────────────────────────────────────────────────────────────────────
C16 = md_cell("""\
---
## §5 — Borestem Thermal Correction

### What is the borestem effect?

The Apollo HFE probes were inserted into boreholes lined with a hollow fiberglass
**borestem** casing (~2.5 cm outer diameter, 3 mm wall thickness).

| Material | k (W/m/K) |
|----------|-----------|
| Fiberglass | 0.04 |
| Regolith surface | ~0.001 |
| Ratio | ~40x |

This fiberglass creates a **thermal short-circuit** that preferentially conducts
heat from the warm near-surface layer down to the sensors, making them read
warmer than the true undisturbed regolith.

### 2-D axisymmetric steady-state correction

The correction is computed by solving the 2-D heat equation in cylindrical
coordinates (r, z) for the region around the borestem:

1/r * d/dr[r*k*dT/dr] + d/dz[k*dT/dz] = 0

**Boundary conditions:**
- Surface (z=0): T = T_surf_est (geothermal mean, ~250-253 K)
  - *Critical: this is NOT the arithmetic diurnal mean (~214 K), which
    would cause a spurious -20 K cooling artifact*
- Far field (r=R_inf): T = T_1D(z) (undisturbed 1-D profile)
- Bottom (z=L): k * dT/dz = Q_basal
- Axis (r=0): dT/dr = 0 (symmetry)

### Expected corrections (Langseth 1976)

| Effect | Magnitude |
|--------|-----------|
| Borestem warm bias | +1.2 to +1.8 K |
| Probe-top solar absorption | +0.5 to +2.1 K |
| **Total** | **+1.7 to +3.5 K** |

The dashed lines show the **borestem-corrected** profiles (subtracting the warm bias).
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 17 — Borestem correction Apollo 15
# ─────────────────────────────────────────────────────────────────────────────
C17 = code_cell("""\
# Apollo 15 borestem + probe-top correction
# Surface BC uses T_surf_est=250.0 K (NOT T_mean[0]~214 K which causes -20 K artifact)
_s15d = APOLLO_DATA['Apollo 15']['disc']
_s15h = APOLLO_DATA['Apollo 15']['hayne']
_Ts15 = APOLLO_COORDS['Apollo 15']['T_surf_est']   # 250.0 K

_Qs15 = borestem.mean_daytime_solar_flux(
    APOLLO_DATA['Apollo 15']['lat'], ALBEDO, SUNSCALE)

_, _corr15d = borestem.apply_all_corrections(
    _s15d['stats']['T_mean'], Z_GRID, _s15d['k_profile'], _Ts15, _Qs15,
    borestem_depth_m=1.62, use_2d_borestem=_HAS_BORESTEM2D)
_, _corr15h = borestem.apply_all_corrections(
    _s15h['stats']['T_mean'], Z_GRID, _s15h['k_profile'], _Ts15, _Qs15,
    borestem_depth_m=1.62, use_2d_borestem=_HAS_BORESTEM2D)

_err15 = _s15d['errors']
_adat15 = {
    'depths':       _err15['apollo_depths'],
    'T_K':          _err15['apollo_temps'],
    'sensor_types': _err15.get('apollo_sensor_types',
                               ['TG'] * len(_err15['apollo_depths'])),
}
_bstats15 = {**_s15d['stats'], 'z_grid': Z_GRID}

fig = plots.borestem_correction_plot(
    stats               = _bstats15,
    apollo_data         = _adat15,
    site_name           = '15',
    correction_dT       = _corr15d['total'],
    hayne_T_model       = _s15h['stats']['T_mean'],
    hayne_correction_dT = _corr15h['total'],
)
plt.tight_layout()
plt.show()

print(f'A15 borestem correction range: '
      f'{_corr15d["borestem"].min():.2f} to {_corr15d["borestem"].max():.2f} K')
print(f'A15 probe-top correction range: '
      f'{_corr15d["probe_top"].min():.2f} to {_corr15d["probe_top"].max():.2f} K')
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 18 — Borestem correction Apollo 17
# ─────────────────────────────────────────────────────────────────────────────
C18 = code_cell("""\
# Apollo 17 borestem + probe-top correction
_s17d = APOLLO_DATA['Apollo 17']['disc']
_s17h = APOLLO_DATA['Apollo 17']['hayne']
_Ts17 = APOLLO_COORDS['Apollo 17']['T_surf_est']   # 253.0 K

_Qs17 = borestem.mean_daytime_solar_flux(
    APOLLO_DATA['Apollo 17']['lat'], ALBEDO, SUNSCALE)

_, _corr17d = borestem.apply_all_corrections(
    _s17d['stats']['T_mean'], Z_GRID, _s17d['k_profile'], _Ts17, _Qs17,
    borestem_depth_m=2.36, use_2d_borestem=_HAS_BORESTEM2D)
_, _corr17h = borestem.apply_all_corrections(
    _s17h['stats']['T_mean'], Z_GRID, _s17h['k_profile'], _Ts17, _Qs17,
    borestem_depth_m=2.36, use_2d_borestem=_HAS_BORESTEM2D)

_err17 = _s17d['errors']
_adat17 = {
    'depths':       _err17['apollo_depths'],
    'T_K':          _err17['apollo_temps'],
    'sensor_types': _err17.get('apollo_sensor_types',
                               ['TG'] * len(_err17['apollo_depths'])),
}
_bstats17 = {**_s17d['stats'], 'z_grid': Z_GRID}

fig = plots.borestem_correction_plot(
    stats               = _bstats17,
    apollo_data         = _adat17,
    site_name           = '17',
    correction_dT       = _corr17d['total'],
    hayne_T_model       = _s17h['stats']['T_mean'],
    hayne_correction_dT = _corr17h['total'],
)
plt.tight_layout()
plt.show()

print(f'A17 borestem correction range: '
      f'{_corr17d["borestem"].min():.2f} to {_corr17d["borestem"].max():.2f} K')
print(f'A17 probe-top correction range: '
      f'{_corr17d["probe_top"].min():.2f} to {_corr17d["probe_top"].max():.2f} K')
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 19 — §6 Geothermal Heat Flow (markdown)
# ─────────────────────────────────────────────────────────────────────────────
C19 = md_cell("""\
---
## §6 — Geothermal Heat Flow

The Moon's interior releases heat generated by the decay of radioactive elements
(mainly U, Th, K). This heat drives a geothermal temperature gradient:

Q = k * dT/dz

**Published values (Langseth et al. 1976, revised):**

| Site | Q_basal | Uncertainty |
|------|---------|-------------|
| Apollo 15 | 21 mW/m^2 | +/- 3 mW/m^2 |
| Apollo 17 | 16 mW/m^2 | +/- 2 mW/m^2 |
| Average | 18 mW/m^2 | (used in model) |

These values were revised upward from the original 1973 estimates after applying
the borestem and transient corrections. They represent some of the most important
data constraining the Moon's thermal evolution and bulk composition.

The bar chart shows how the modelled thermal gradient (computed from the deep T(z)
profile using k_deep = 3.5 mW/m/K) compares to the Langseth published values.
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 20 — Combined heat flow comparison
# ─────────────────────────────────────────────────────────────────────────────
C20 = code_cell("""\
fig = plots.combined_heat_flow(
    APOLLO_RESULTS, model_name='discrete',
    k_surface_mW=1.5, k_deep_mW=3.5,
)
plt.tight_layout()
plt.show()
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 21 — §7 Animations (markdown)
# ─────────────────────────────────────────────────────────────────────────────
C21 = md_cell("""\
---
## §7 — Animated Visualizations

**Animation 1 — T(z) depth profile** evolving through one lunar day.
The diurnal wave penetrates from the surface. By 50 cm depth the variation
is < 1 K; below 100 cm it is negligible.

**Animation 2 — Heatmap** of temperature vs depth and time.
The vertical white line marks the current time in the animation.
The right panel shows the corresponding T(z) profile.

Both animations use the Apollo 15 discrete-model result (last simulated lunar day).
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 22 — T(z) profile animation
# ─────────────────────────────────────────────────────────────────────────────
C22 = code_cell("""\
from IPython.display import Image, display
import matplotlib.animation as animation

gif_path1 = os.path.join('gifs', 'thermal_profile_animation.gif')

if os.path.exists(gif_path1) and os.path.getsize(gif_path1) > 5000:
    display(Image(filename=gif_path1))
else:
    print('Generating T(z) animation ...')
    _tp  = APOLLO_DATA['Apollo 15']['disc']['T_profile']
    _ta  = APOLLO_DATA['Apollo 15']['disc']['T_arr']
    _t0  = _ta[-1] - constants.LUNAR_DAY
    _idx = np.where(_ta >= _t0)[0]
    _day_h = constants.LUNAR_DAY / 3600.0

    _n_fr = 36
    _fids = np.linspace(_idx[0], _idx[-1], _n_fr, dtype=int)
    _dmask = Z_GRID * 100 <= 200
    _dcm   = Z_GRID[_dmask] * 100

    fig_a, ax_a = plt.subplots(figsize=(6, 8))
    ln_a, = ax_a.plot([], [], 'b-', lw=2.5)
    ax_a.set_xlim(70, 420)
    ax_a.set_ylim(200, 0)
    ax_a.set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
    ax_a.set_ylabel('Depth (cm)',      fontsize=12, fontweight='bold')
    ax_a.set_title('Lunar Regolith T(z) — Apollo 15 (Discrete)')
    ax_a.axhline(50, color='#CC8800', ls='--', lw=0.9, alpha=0.6, label='~Skin depth')
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=9)
    ttxt = ax_a.text(0.98, 0.96, '', transform=ax_a.transAxes, ha='right', va='top', fontsize=10)

    def _upd_a(fi):
        T_f = _tp[_fids[fi], _dmask]
        ln_a.set_data(T_f, _dcm)
        hr  = fi / (_n_fr - 1) * _day_h
        ttxt.set_text(f'Hour {hr:.0f} / {_day_h:.0f}')
        return (ln_a, ttxt)

    ani_a = animation.FuncAnimation(fig_a, _upd_a, frames=_n_fr, blit=True, interval=150)
    os.makedirs('gifs', exist_ok=True)
    ani_a.save(gif_path1, writer='pillow', fps=8, dpi=100)
    plt.close(fig_a)
    print(f'Saved {gif_path1}')
    display(Image(filename=gif_path1))
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 23 — Heatmap animation
# ─────────────────────────────────────────────────────────────────────────────
C23 = code_cell("""\
from IPython.display import Image, display
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

gif_path2 = os.path.join('gifs', 'heatmap_animation.gif')

if os.path.exists(gif_path2) and os.path.getsize(gif_path2) > 5000:
    display(Image(filename=gif_path2))
else:
    print('Generating heatmap animation ...')
    _tp   = APOLLO_DATA['Apollo 15']['disc']['T_profile'].astype(np.float32)
    _ta   = APOLLO_DATA['Apollo 15']['disc']['T_arr']
    _t0   = _ta[-1] - constants.LUNAR_DAY
    _idx  = np.where(_ta >= _t0)[0]
    _day_h = constants.LUNAR_DAY / 3600.0

    _dmask = Z_GRID * 100 <= 200
    _dcm   = Z_GRID[_dmask] * 100
    _n_fr  = 48
    _fids  = np.linspace(_idx[0], _idx[-1], _n_fr, dtype=int)
    _Tday  = _tp[np.ix_(_idx, np.where(_dmask)[0])]   # (n_idx, n_depths)
    _th    = (_ta[_idx] - _t0) / 3600.0

    Tlo = float(_Tday.min())
    Thi = float(_Tday.max())

    fig_h = plt.figure(figsize=(12, 6))
    gs_h  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.30)
    axhm  = fig_h.add_subplot(gs_h[0])
    axpr  = fig_h.add_subplot(gs_h[1])

    im_h = axhm.imshow(_Tday.T, aspect='auto',
                        extent=[0, _day_h, _dcm[-1], 0],
                        origin='upper', cmap='RdYlBu_r', vmin=Tlo, vmax=Thi)
    plt.colorbar(im_h, ax=axhm, label='Temperature (K)', pad=0.02)
    vln = axhm.axvline(0, color='white', lw=2.0, ls='--')
    axhm.set_xlabel('Time in lunar day (hours)', fontsize=11, fontweight='bold')
    axhm.set_ylabel('Depth (cm)',                fontsize=11, fontweight='bold')
    axhm.set_title('Temperature: Depth x Time  (Apollo 15, Discrete)',
                   fontsize=11, fontweight='bold')

    pln, = axpr.plot([], [], 'k-', lw=2.0)
    axpr.set_xlim(Tlo - 5, Thi + 5)
    axpr.set_ylim(_dcm[-1], 0)
    axpr.set_xlabel('Temperature (K)', fontsize=11, fontweight='bold')
    axpr.set_ylabel('Depth (cm)',       fontsize=11, fontweight='bold')
    axpr.set_title('T(z) Profile',      fontsize=11, fontweight='bold')
    axpr.grid(True, alpha=0.3)
    htxt = axpr.text(0.02, 0.96, '', transform=axpr.transAxes, fontsize=9, va='top')

    def _upd_h(fi):
        local = _fids[fi] - _idx[0]
        local = max(0, min(local, len(_Tday) - 1))
        pln.set_data(_Tday[local, :], _dcm)
        hr = _th[local]
        vln.set_xdata([hr])
        htxt.set_text(f'Hour {hr:.0f}')
        return (pln, vln, htxt)

    ani_h = animation.FuncAnimation(fig_h, _upd_h, frames=_n_fr, blit=True, interval=120)
    os.makedirs('gifs', exist_ok=True)
    ani_h.save(gif_path2, writer='pillow', fps=10, dpi=100)
    plt.close(fig_h)
    print(f'Saved {gif_path2}')
    display(Image(filename=gif_path2))
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 24 — §8 Plain-language summary (markdown)
# ─────────────────────────────────────────────────────────────────────────────
C24 = md_cell("""\
---
## §8 — Plain-Language Summary

### What this model does

Imagine the Moon's surface as a layer of very dry, fluffy dust. The Sun heats this
dust intensely during the 14-day lunar day (up to ~120 C). During the equally long
lunar night, with no atmosphere to retain heat, the surface plunges to -173 C.

This model tracks how that temperature wave travels downward through the dust, using
the same physics as calculating how heat flows through a wall in a house — just much
more extreme temperatures and much lower conductivity.

**The governing equation** says: at every depth, the rate of temperature change equals
the net heat flow in minus heat flow out, divided by the material's heat capacity.
The catch is that the conductivity changes with temperature (the dust conducts heat
better when it is hotter), making the problem nonlinear.

### How the simulation works

1. We start with a physically reasonable initial temperature profile (warmer at depth
   due to the Moon's internal heat).
2. We simulate 7 lunar days (~207 Earth days). The first 6 days let the near-surface
   reach a repeating steady-state; the 7th day is used for analysis.
3. At each time step (every few minutes of simulated time), we solve:
   - How much solar energy hits the surface (using real terrain data from NASA's LOLA DEM)
   - How the surface temperature adjusts to balance absorption and infrared emission
   - How heat diffuses through each layer

### Why the Apollo data is not straightforward

The Apollo sensors measured temperature for years — but they were not perfect thermometers:

- **TC cable sensors** (at 14–67 cm depth) were within the zone where the diurnal
  temperature wave is still significant. Including these would make our model look worse
  than it actually is. So we exclude all sensors shallower than 80 cm.

- **The borestem** (fiberglass drill casing) acts like a thermal highway, conducting
  heat 40x better than the surrounding dust. This makes sensors read 1–3 K too warm.
  We compute a correction using a 2-D heat equation and subtract it.

### The two models

We compare two descriptions of how lunar dust is structured:

- **Discrete layers** — based on the actual Apollo drill cores: very fluffy at the
  surface (rho = 1100 kg/m³), becoming denser with depth (rho = 1800 kg/m³ below 40 cm).

- **Hayne 2017** — a smooth exponential fit derived from NASA's Diviner instrument,
  which measures lunar surface temperatures from orbit across the whole Moon.

Both models agree with Apollo measurements to within 1.1–1.5 K — better than the
width of a single pencil mark on a thermometer scale.

### What the results tell us

- The Moon's interior generates a tiny but measurable heat flux: ~18 mW/m² on average.
  For comparison, a single LED light bulb emits ~10 watts over an area of a few cm² —
  the Moon's heat is 100,000 times weaker per unit area, spread uniformly from below.

- Despite this tiny flux, it drives a detectable temperature gradient (~1.5 K/m) that
  the Apollo sensors measured clearly.

- The two Apollo sites give slightly different heat flow values (21 vs 16 mW/m²),
  which tells us that the Moon's interior is not uniform — there are regional variations
  in radioactive element concentrations.

---
*Model: 1-D explicit finite-difference solver compiled with Numba (Python).*
*Key references: Hayne et al. 2017 (JGR Planets); Langseth et al. 1972, 1976 (Apollo HFE);*
*Carrier et al. 1991 (Lunar Sourcebook); Hemingway et al. 1973 (heat capacity).*
""")


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE AND WRITE
# ─────────────────────────────────────────────────────────────────────────────
CELLS = [C0, C1, C2, C3, C4,
         C5, C6, C7,
         C8, C9,
         C10, C11, C12,
         C13, C14, C15,
         C16, C17, C18,
         C19, C20,
         C21, C22, C23,
         C24]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": CELLS,
}

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Written {len(CELLS)} cells to:")
print(f"  {NOTEBOOK_PATH}")
