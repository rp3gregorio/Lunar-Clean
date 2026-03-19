"""
analysis.py — Post-processing helpers: extract statistics, run sensitivity
              sweeps, and process batches of locations.

Key functions
-------------
extract_stats()          — Min/max/mean temperature vs depth for the final day.
get_diurnal_cycles()     — Temperature vs time at selected depths (final day).
find_apollo_site()       — Check if coordinates are near an Apollo HFE site.
compute_apollo_errors()  — RMSE, bias, MAE vs Apollo measurements.
run_sensitivity()        — Sweep one parameter, collect stats for each value.
run_batch()              — Process a list of locations, return summary table.
"""

import time
import numpy as np

from lunar.constants import LUNAR_DAY, APOLLO_DATA, APOLLO_SITES, DEFAULT_ALBEDO, DEFAULT_EMISSIVITY
from lunar.dem       import extract_point, latlon_to_pixel, compute_slope_aspect
from lunar.horizon   import compute_horizon_profile, compute_sky_view_factor
from lunar.solver    import solve_thermal_model, create_depth_grid, solve_with_h
from lunar.models    import (MODEL_ID_MAP,
                              density_hayne_py, k_solid_hayne_py,
                              density_discrete_py, k_solid_discrete_py,
                              set_rho_surface)


# ─────────────────────────────────────────────────────────────────────────────
# TEMPERATURE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def extract_stats(T_profile, t_arr, z_grid):
    """
    Extract temperature statistics from the *final* simulated lunar day.

    Parameters
    ----------
    T_profile : (n_snapshots, n_depths) float32
    t_arr     : (n_snapshots,) seconds
    z_grid    : (n_depths,) metres

    Returns
    -------
    dict with keys:
        depth       — z_grid (m)
        T_min       — minimum temperature at each depth (K)
        T_max       — maximum temperature at each depth (K)
        T_mean      — time-mean temperature at each depth (K)
        T_amplitude — half peak-to-peak amplitude (K)
    """
    t_start  = t_arr[-1] - LUNAR_DAY
    idx      = np.where(t_arr >= t_start)[0]
    if len(idx) == 0:
        idx = np.arange(len(t_arr))

    T_day = T_profile[idx, :]

    return {
        'depth':       z_grid,
        'T_min':       np.min(T_day,  axis=0),
        'T_max':       np.max(T_day,  axis=0),
        'T_mean':      np.mean(T_day, axis=0),
        'T_amplitude': (np.max(T_day, axis=0) - np.min(T_day, axis=0)) / 2.0,
    }


def get_diurnal_cycles(T_profile, t_arr, z_grid, depths_m=None):
    """
    Extract temperature vs time at selected depths for the final lunar day.

    Parameters
    ----------
    depths_m : list of depths in metres; defaults to [0, 0.1, 0.5, 1.0]

    Returns
    -------
    dict: depth → {'time_h': ..., 'temperature': ..., 'actual_depth': ...}
    """
    if depths_m is None:
        depths_m = [0.0, 0.1, 0.5, 1.0]

    t_start  = t_arr[-1] - LUNAR_DAY
    idx      = np.where(t_arr >= t_start)[0]
    t_hours  = (t_arr[idx] - t_start) / 3600.0

    cycles = {}
    for d in depths_m:
        if d > z_grid[-1]:
            continue
        iz = int(np.argmin(np.abs(z_grid - d)))
        cycles[d] = {
            'time_h':       t_hours,
            'temperature':  T_profile[idx, iz],
            'actual_depth': float(z_grid[iz]),
        }
    return cycles


# ─────────────────────────────────────────────────────────────────────────────
# APOLLO SITE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def find_apollo_site(lat_deg, lon_deg, tolerance_deg=0.1):
    """
    Return Apollo site name if (lat, lon) is within *tolerance_deg* of a
    known site, otherwise return None.
    """
    for name, info in APOLLO_SITES.items():
        if (abs(lat_deg - info['lat']) < tolerance_deg and
                abs(lon_deg - info['lon']) < tolerance_deg):
            return name
    return None


def compute_apollo_errors(model_T_mean, z_grid, site_name):
    """
    Compare mean model temperatures with Apollo HFE measurements.

    Parameters
    ----------
    model_T_mean : 1-D array of mean temperature vs depth (K)
    z_grid       : depth array matching model_T_mean (m)
    site_name    : 'Apollo 15' or 'Apollo 17'

    Returns
    -------
    dict with:
        rmse, bias, mae : scalar metrics (K)
        residuals       : model − measured at each Apollo depth
        apollo_depths   : Apollo measurement depths (m)
        apollo_temps    : Apollo measured temperatures (K)
        model_at_apollo : interpolated model temps at Apollo depths (K)
    """
    apollo      = APOLLO_DATA[site_name]
    a_depths    = apollo['depths']
    a_temps     = apollo['temps']
    m_at_apollo = np.interp(a_depths, z_grid, model_T_mean)
    residuals   = m_at_apollo - a_temps

    return {
        'rmse':               float(np.sqrt(np.mean(residuals ** 2))),
        'bias':               float(np.mean(residuals)),
        'mae':                float(np.mean(np.abs(residuals))),
        'residuals':          residuals,
        'apollo_depths':      a_depths,
        'apollo_temps':       a_temps,
        'apollo_sensor_types': apollo.get('sensor_types',
                                          ['TG'] * len(a_depths)),
        'model_at_apollo':    m_at_apollo,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity(
    param_name,
    param_values,
    z_grid, T_init,
    lat_deg, lon_deg,
    slope, aspect,
    horizons, az_angles,
    chi, model_id, sunscale, ndays,
    albedo=DEFAULT_ALBEDO,
    emissivity=DEFAULT_EMISSIVITY,
    baseline_h=0.07,
    baseline_rho_surface=1100.0,
    apollo_site=None,
    verbose=True,
):
    """
    Vary one parameter across *param_values* and record temperature statistics.

    Parameters
    ----------
    param_name  : one of 'sunscale', 'albedo', 'emissivity', 'chi',
                  'h_parameter', 'rho_surface'
    param_values: 1-D array of values to test
    ... (same geometry args as solve_thermal_model) ...
    apollo_site : name of Apollo site for error metrics, or None
    verbose     : print progress to console

    Returns
    -------
    list of dicts, one per parameter value:
        {
          'value'  : float,
          'stats'  : dict from extract_stats(),
          'errors' : dict from compute_apollo_errors() or None,
        }
    """
    results = []

    for i, val in enumerate(param_values):
        if verbose:
            print(f'  [{i+1}/{len(param_values)}] {param_name} = {val:.4f} ...', end=' ')
        t0 = time.time()

        # ── Set per-run parameters ─────────────────────────────────────────────
        _sunscale   = val if param_name == 'sunscale'    else sunscale
        _albedo     = val if param_name == 'albedo'      else albedo
        _emissivity = val if param_name == 'emissivity'  else emissivity
        _chi        = val if param_name == 'chi'         else chi
        _h          = val if param_name == 'h_parameter' else baseline_h
        _rho_s      = val if param_name == 'rho_surface' else baseline_rho_surface

        # ── Run solver ────────────────────────────────────────────────────────
        needs_py_path = param_name in ('h_parameter', 'rho_surface')

        if needs_py_path:
            # Must use pure-Python density functions: numba can't see runtime globals
            if model_id == MODEL_ID_MAP['hayne_exponential']:
                den_fn = lambda z, H: density_hayne_py(z, H=H, rho_surface=_rho_s)
                kso_fn = lambda z, H: k_solid_hayne_py(z, H=H)
            else:
                den_fn = lambda z, H: density_discrete_py(z, H=H, rho_surface=_rho_s)
                kso_fn = lambda z, H: k_solid_discrete_py(z, H=H)

            T_profile, t_arr = solve_with_h(
                z_grid, T_init,
                lat_deg, lon_deg,
                slope, aspect,
                horizons, az_angles,
                _chi, model_id, _sunscale, ndays,
                H_param=_h,
                albedo=_albedo, emissivity=_emissivity,
                density_fn=den_fn, k_solid_fn=kso_fn,
            )
        else:
            T_profile, t_arr = solve_thermal_model(
                z_grid, T_init,
                lat_deg, lon_deg,
                slope, aspect,
                horizons, az_angles,
                _chi, model_id, _sunscale, ndays,
                albedo=_albedo, emissivity=_emissivity,
            )

        stats  = extract_stats(T_profile, t_arr, z_grid)
        errors = compute_apollo_errors(stats['T_mean'], z_grid, apollo_site) \
                 if apollo_site else None

        if verbose:
            print(f'{time.time()-t0:.1f}s')

        results.append({'value': float(val), 'stats': stats, 'errors': errors})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    locations,
    elev_m, pixel_m, map_res,
    z_grid, T_init,
    chi, model_id, sunscale, ndays,
    albedo=DEFAULT_ALBEDO,
    emissivity=DEFAULT_EMISSIVITY,
    n_az=72,
    max_range_px=1500,
    verbose=True,
):
    """
    Run the thermal model for a list of (name, lat, lon) locations.

    For each location the function:
      1. Snaps to the DEM pixel and extracts slope / aspect.
      2. Computes the horizon profile and sky-view factor.
      3. Runs solve_thermal_model().
      4. Extracts statistics (min / max / mean temperature vs depth).
      5. Checks for an Apollo site match and computes errors if found.

    Parameters
    ----------
    locations  : list of dicts with keys 'name', 'lat', 'lon'
    elev_m     : DEM elevation grid (from load_ldem)
    pixel_m    : pixel size (metres)
    map_res    : pixels per degree
    z_grid     : depth grid
    T_init     : initial temperature (K)
    ...        : same physics parameters as solve_thermal_model
    n_az       : azimuth resolution for horizon computation (72 = 5° steps)
    max_range_px : maximum horizon-scan distance (pixels)
    verbose    : print progress

    Returns
    -------
    list of dicts, one per location:
        {
          'name', 'lat_req', 'lon_req', 'lat_act', 'lon_act',
          'elevation', 'slope_deg', 'aspect_deg', 'svf',
          'stats'   : dict from extract_stats(),
          'errors'  : dict from compute_apollo_errors() or None,
          'runtime' : seconds for this location,
        }
    """
    az_angles = np.linspace(0, 2 * np.pi, n_az, endpoint=False, dtype=np.float32)
    results   = []

    for k, loc in enumerate(locations):
        name      = loc.get('name', f'Location {k}')
        lat_req   = loc['lat']
        lon_req   = loc['lon']

        if verbose:
            print(f'[{k+1}/{len(locations)}] {name}  ({lat_req:.3f}°N, {lon_req:.3f}°E) ...',
                  end=' ', flush=True)
        t0 = time.time()

        # Topography
        row, col, lat_act, lon_act, elev, slope, aspect = extract_point(
            lat_req, lon_req, elev_m, pixel_m, map_res
        )

        # Horizon
        horizons = compute_horizon_profile(row, col, elev_m, pixel_m,
                                           az_angles, max_range_px)
        svf      = compute_sky_view_factor(horizons)

        # Thermal model
        T_profile, t_arr = solve_thermal_model(
            z_grid, T_init,
            lat_act, lon_act,
            slope, aspect,
            horizons, az_angles,
            chi, model_id, sunscale, ndays,
            albedo=albedo, emissivity=emissivity,
        )

        stats = extract_stats(T_profile, t_arr, z_grid)

        # Apollo comparison
        apollo_site = find_apollo_site(lat_act, lon_act)
        errors      = compute_apollo_errors(stats['T_mean'], z_grid, apollo_site) \
                      if apollo_site else None

        elapsed = time.time() - t0
        if verbose:
            tsurf = f"{stats['T_min'][0]:.0f}–{stats['T_max'][0]:.0f} K"
            print(f'{elapsed:.1f}s  (surface {tsurf})')

        results.append({
            'name':       name,
            'lat_req':    lat_req,
            'lon_req':    lon_req,
            'lat_act':    lat_act,
            'lon_act':    lon_act,
            'elevation':  elev,
            'slope_deg':  float(np.degrees(slope)),
            'aspect_deg': float(np.degrees(aspect)),
            'svf':        float(svf),
            'stats':      stats,
            'errors':     errors,
            'runtime':    elapsed,
        })

    return results


def batch_to_table(batch_results, z_eval=None):
    """
    Convert batch results to a simple printable summary dict-of-lists.

    z_eval : depths (m) at which to record T_min, T_max, T_mean.
             Defaults to [0.0, 0.5, 1.0].
    """
    if z_eval is None:
        z_eval = [0.0, 0.5, 1.0]

    table = {
        'name': [], 'lat': [], 'lon': [],
        'elevation_m': [], 'slope_deg': [], 'svf': [],
    }
    for d in z_eval:
        table[f'T_min_{int(d*100)}cm']  = []
        table[f'T_max_{int(d*100)}cm']  = []
        table[f'T_mean_{int(d*100)}cm'] = []

    table['RMSE_K'] = []
    table['bias_K'] = []

    for r in batch_results:
        table['name'].append(r['name'])
        table['lat'].append(round(r['lat_act'], 4))
        table['lon'].append(round(r['lon_act'], 4))
        table['elevation_m'].append(round(r['elevation'], 1))
        table['slope_deg'].append(round(r['slope_deg'], 2))
        table['svf'].append(round(r['svf'], 3))

        z_grid = r['stats']['depth']
        for d in z_eval:
            tag = int(d * 100)
            table[f'T_min_{tag}cm'].append(
                round(float(np.interp(d, z_grid, r['stats']['T_min'])), 1))
            table[f'T_max_{tag}cm'].append(
                round(float(np.interp(d, z_grid, r['stats']['T_max'])), 1))
            table[f'T_mean_{tag}cm'].append(
                round(float(np.interp(d, z_grid, r['stats']['T_mean'])), 1))

        err = r.get('errors')
        table['RMSE_K'].append(round(err['rmse'], 3) if err else None)
        table['bias_K'].append(round(err['bias'], 3) if err else None)

    return table
