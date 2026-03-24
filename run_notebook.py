#!/usr/bin/env python
"""
Direct notebook execution — runs all cells in order without using Jupyter.
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import shutil as _shutil, os as _os
_pycache = _os.path.join('lunar', '__pycache__')
if _os.path.isdir(_pycache):
    _shutil.rmtree(_pycache)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, time, warnings
import importlib
from lunar import (constants, models, dem, horizon, solar,
                   solver, analysis, plots, hfe_loader, borestem)
try:
    from lunar import borestem2d
    _HAS_BORESTEM2D = True
except ImportError:
    borestem2d = None
    _HAS_BORESTEM2D = False
    print('Note: borestem2d not available.')

for _m in (constants, models, dem, horizon, solar,
           solver, analysis, plots, hfe_loader, borestem):
    importlib.reload(_m)

plt.ioff()  # non-interactive mode
mpl.rcParams['figure.dpi']   = 130
mpl.rcParams['savefig.dpi']  = 300
mpl.rcParams['savefig.bbox'] = 'tight'

warnings.filterwarnings('ignore', category=UserWarning)

# Create output directory for figures
os.makedirs('figures', exist_ok=True)
fig_counter = {'count': 0}
print('✓ All modules loaded\n')

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
LAT =  26.1323
LON =   3.6285
MODEL   = 'discrete'
H_PARAM = 0.07
SUNSCALE   = 1.10
ALBEDO     = 0.09
EMISSIVITY = 0.95
CHI        = 2.7
NDAYS = 3

APOLLO_COORDS = {
    'Apollo 15': {'lat': 26.1323, 'lon':  3.6285, 'T_surf_est': 250.0},
    'Apollo 17': {'lat': 20.1911, 'lon': 30.7723, 'T_surf_est': 252.0},
}

MODEL_ID  = models.MODEL_ID_MAP[MODEL]
HAYNE_ID  = models.MODEL_ID_MAP['hayne_exponential']
DISC_ID   = models.MODEL_ID_MAP['discrete']
models.set_hayne_h(H_PARAM)
models.set_layer1_h(H_PARAM)

print(f'Target   : {LAT:.4f}°N, {LON:.4f}°E')
print(f'Model    : {MODEL}  (id={MODEL_ID})')
print(f'SUNSCALE : {SUNSCALE}   CHI : {CHI}   ALBEDO : {ALBEDO}')
print(f'NDAYS    : {NDAYS}  ({NDAYS*29.53:.1f} Earth days)\n')

# ─────────────────────────────────────────────────────────────────────────────
# Load DEM
# ─────────────────────────────────────────────────────────────────────────────
print('Loading LOLA DEM...')
ELEV_M, PIXEL_M, MAP_RES, _ = dem.load_ldem()

(ROW, COL,
 ACTUAL_LAT, ACTUAL_LON,
 ELEVATION, SLOPE, ASPECT) = dem.extract_point(LAT, LON, ELEV_M, PIXEL_M, MAP_RES)

N_AZ      = 360
AZ_ANGLES = np.linspace(0, 2*np.pi, N_AZ, endpoint=False, dtype=np.float32)

print('Computing horizon profile …', end=' ', flush=True)
t0 = time.time()
HORIZONS = horizon.compute_horizon_profile(
    ROW, COL, ELEV_M, PIXEL_M, AZ_ANGLES, max_range_px=3000)
SVF = horizon.compute_sky_view_factor(HORIZONS)
print(f'done in {time.time()-t0:.1f} s')

print(f'  DEM      : {ELEV_M.shape[0]}×{ELEV_M.shape[1]} px, {MAP_RES} pix/deg')
print(f'  Snapped  : {ACTUAL_LAT:.4f}°N, {ACTUAL_LON:.4f}°E')
print(f'  Elevation: {ELEVATION:.0f} m')
print(f'  Slope    : {np.degrees(SLOPE):.2f}°   Aspect: {np.degrees(ASPECT):.1f}°')
print(f'  SVF      : {SVF:.3f}   Max horizon: {np.degrees(HORIZONS.max()):.1f}°\n')

# ─────────────────────────────────────────────────────────────────────────────
# Run Thermal Model at Target Site
# ─────────────────────────────────────────────────────────────────────────────
print('Running thermal model...')
Z_GRID = solver.create_depth_grid()
print(f'Depth grid: {len(Z_GRID)} nodes, 0 – {Z_GRID[-1]:.1f} m')

T_INIT = solver.compute_equilibrium_profile(
    Z_GRID, T_surf_mean=250.0, model_id=MODEL_ID, chi=CHI)

print(f'Running {NDAYS} lunar day(s) at target site …', end=' ', flush=True)
t0 = time.time()
T_PROFILE, T_ARR = solver.solve_thermal_model(
    Z_GRID, T_INIT,
    ACTUAL_LAT, ACTUAL_LON, SLOPE, ASPECT, HORIZONS, AZ_ANGLES,
    CHI, MODEL_ID, SUNSCALE, NDAYS,
    albedo=ALBEDO, emissivity=EMISSIVITY,
)
print(f'done in {time.time()-t0:.1f} s')
print(f'  Shape : {T_PROFILE.shape[0]} snapshots × {T_PROFILE.shape[1]} depth nodes')
print(f'  T range: {T_PROFILE.min():.0f} – {T_PROFILE.max():.0f} K')

STATS  = analysis.extract_stats(T_PROFILE, T_ARR, Z_GRID)
CYCLES = analysis.get_diurnal_cycles(
    T_PROFILE, T_ARR, Z_GRID,
    depths_m=[0.0, 0.05, 0.10, 0.35, 0.70, 1.50])

print(f'  Surface : min={STATS["T_min"][0]:.0f} K  '
      f'max={STATS["T_max"][0]:.0f} K  '
      f'mean={STATS["T_mean"][0]:.0f} K\n')

# ─────────────────────────────────────────────────────────────────────────────
# Run Both Density Models at Both Apollo Sites
# ─────────────────────────────────────────────────────────────────────────────
print('Running discrete + Hayne models at Apollo 15 and Apollo 17 …\n')

APOLLO_DATA     = {}
APOLLO_RESULTS  = {}
COMPARE_RESULTS = {}
COMPARE_ERRORS  = {}

for site_name, coords in APOLLO_COORDS.items():
    lat_s   = coords['lat']
    lon_s   = coords['lon']
    t_surf  = coords['T_surf_est']
    print(f'── {site_name}  ({lat_s}°N, {lon_s}°E)')

    _row, _col, _alat, _alon, _elev, _sl, _asp = dem.extract_point(
        lat_s, lon_s, ELEV_M, PIXEL_M, MAP_RES)

    _horiz = horizon.compute_horizon_profile(
        _row, _col, ELEV_M, PIXEL_M, AZ_ANGLES, max_range_px=1500)

    site_entry = {
        'lat': _alat, 'lon': _alon,
        'slope': _sl, 'aspect': _asp, 'horizons': _horiz,
        'disc': {}, 'hayne': {},
    }
    _compare_stats  = {}
    _compare_errors = {}

    for model_key, mid in [('disc', DISC_ID), ('hayne', HAYNE_ID)]:
        _T_init = solver.compute_equilibrium_profile(
            Z_GRID, t_surf, mid, CHI)

        t0 = time.time()
        _TP, _TA = solver.solve_thermal_model(
            Z_GRID, _T_init,
            _alat, _alon, _sl, _asp, _horiz, AZ_ANGLES,
            CHI, mid, SUNSCALE, NDAYS,
            albedo=ALBEDO, emissivity=EMISSIVITY,
        )
        _stats  = analysis.extract_stats(_TP, _TA, Z_GRID)
        _errors = analysis.compute_apollo_errors(
            _stats['T_mean'], Z_GRID, site_name)
        _cycles = analysis.get_diurnal_cycles(
            _TP, _TA, Z_GRID,
            depths_m=[0.0, 0.05, 0.10, 0.35, 0.70, 1.50])

        _k_prof = np.array([
            models.thermal_conductivity(
                float(_stats['T_mean'][i]), float(Z_GRID[i]), CHI, mid)
            for i in range(len(Z_GRID))
        ])

        site_entry[model_key] = {
            'T_profile': _TP, 'T_arr': _TA,
            'stats': _stats, 'cycles': _cycles,
            'errors': _errors, 'k_profile': _k_prof,
        }
        model_str = 'discrete' if model_key == 'disc' else 'hayne_exponential'
        _compare_stats[model_str]  = _stats
        _compare_errors[model_str] = _errors

        print(f'   [{model_key}] {time.time()-t0:.1f}s  '
              f'RMSE={_errors["rmse"]:.2f} K  '
              f'bias={_errors["bias"]:+.2f} K')

    APOLLO_DATA[site_name]    = site_entry
    APOLLO_RESULTS[site_name] = {
        'stats':  site_entry['disc']['stats'],
        'errors': site_entry['disc']['errors'],
    }
    COMPARE_RESULTS[site_name] = _compare_stats
    COMPARE_ERRORS[site_name]  = _compare_errors
    print()

print('✓ All Apollo model runs complete.\n')

# ─────────────────────────────────────────────────────────────────────────────
# Generate Plots
# ─────────────────────────────────────────────────────────────────────────────
print('Generating plots...\n')

# §1 Diurnal Cycles
print('§1 Diurnal Cycles')
fig = plots.diurnal_cycles(
    CYCLES, ACTUAL_LAT, ACTUAL_LON,
    model_name=MODEL, sunscale=SUNSCALE,
)
fig_counter['count'] += 1
plt.savefig(f'figures/01_diurnal_cycles.png')
plt.close(fig)

# Amplitude Decay
fig = plots.amplitude_decay(
    STATS, Z_GRID, ACTUAL_LAT, ACTUAL_LON,
    model_name=MODEL,
)
fig_counter['count'] += 1
plt.savefig(f'figures/02_amplitude_decay.png')
plt.close(fig)

# Polar Diurnal
fig = plots.polar_diurnal(
    CYCLES,
    depths_m=(0.0, 0.05, 0.35, 0.70, 1.50),
    lat=ACTUAL_LAT, lon=ACTUAL_LON,
)
fig_counter['count'] += 1
plt.savefig(f'figures/03_polar_diurnal.png')
plt.close(fig)

# §2 Heatmap
print('§2 Subsurface Temperature Heatmap')
fig = plots.heatmap(
    T_PROFILE, T_ARR, Z_GRID,
    ACTUAL_LAT, ACTUAL_LON,
    model_name=MODEL,
    depth_limit=1.5,
    zoom_depth_cm=30,
    colormap='inferno',
    show_contours=True,
)
fig_counter['count'] += 1
plt.savefig(f'figures/04_heatmap.png')
plt.close(fig)

# §3 Surface Temperature Map
print('§3 Analytical Surface Temperature Map')
T_sim_max = float(T_PROFILE[:, 0].max())
print(f'Full-model surface T_max = {T_sim_max:.1f} K')

fig = plots.surface_temperature_map(
    ELEV_M, MAP_RES,
    target_lat=ACTUAL_LAT,
    target_lon=ACTUAL_LON,
    albedo=ALBEDO,
    emissivity=EMISSIVITY,
    window_deg=5,
    T_simulated_max=T_sim_max,
)
fig_counter['count'] += 1
plt.savefig(f'figures/05_surface_temp_map.png')
plt.close(fig)

# §4 Apollo Sensor Analysis
print('§4 Apollo Sensor Equilibration')
fig = plots.sensor_equilibration('Apollo 15')
fig_counter['count'] += 1
plt.savefig(f'figures/06_apollo15_equilibration.png')
plt.close(fig)

fig = plots.sensor_equilibration('Apollo 17')
fig_counter['count'] += 1
plt.savefig(f'figures/07_apollo17_equilibration.png')
plt.close(fig)

# §5 Apollo HFE Validation
print('§5 Apollo HFE Validation')
fig = plots.dual_apollo_comparison(
    APOLLO_RESULTS,
    model_name=MODEL,
    sunscale=SUNSCALE,
    chi=CHI,
    albedo=ALBEDO,
)
fig_counter['count'] += 1
plt.savefig(f'figures/08_dual_apollo_comparison.png')
plt.close(fig)

fig = plots.combined_heat_flow(
    APOLLO_RESULTS,
    model_name=MODEL,
)
fig_counter['count'] += 1
plt.savefig(f'figures/09_combined_heat_flow.png')
plt.close(fig)

print('\n✓ All sections complete!')
print(f'Generated {fig_counter["count"]} figures in figures/ directory')
print('Notebook execution finished successfully.')
