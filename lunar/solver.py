"""
solver.py — 1-D finite-difference thermal solver for lunar regolith.

The model discretises the heat-conduction equation:

    ρ(z) · c(T) · ∂T/∂t = ∂/∂z [ k(T,z) · ∂T/∂z ]

on a non-uniform depth grid, with:

  • Surface boundary : energy balance  k ∂T/∂z + Q_solar = ε σ T⁴
    solved by Newton-Raphson at every timestep.
  • Bottom boundary  : constant basal heat flux Q_basal (Apollo HFE value).
  • Initial condition: uniform temperature T_init.

The solver is compiled with Numba for speed (~10× faster than pure Python).

Key functions
-------------
create_depth_grid()   — Build the non-uniform depth array.
solve_thermal_model() — Run the simulation; return T(depth, time).
"""

import numpy as np
from numba import njit

from lunar.constants import (
    sigma, Q_basal, LUNAR_DAY,
    DZ_FINE, DZ_COARSE, DEPTH_FINE, DEPTH_MAX,
    DEFAULT_ALBEDO, DEFAULT_EMISSIVITY,
)
from lunar.models   import (
    get_density, get_k_solid, heat_capacity, thermal_conductivity,
)
from lunar.solar    import solar_geometry, direct_solar_flux
from lunar.horizon  import check_illumination


# ─────────────────────────────────────────────────────────────────────────────
# DEPTH GRID
# ─────────────────────────────────────────────────────────────────────────────

def create_depth_grid(depth_max=DEPTH_MAX,
                      dz_fine=DZ_FINE,
                      dz_coarse=DZ_COARSE,
                      depth_fine=DEPTH_FINE):
    """
    Build a non-uniform depth array (metres).

    Fine spacing (dz_fine) near the surface captures the steep near-surface
    temperature gradients and matches the depths where Apollo drills measured.
    Coarser spacing (dz_coarse) below depth_fine saves computation while
    keeping the bottom boundary far from the region of interest.

    Returns
    -------
    z_grid : 1-D float32 array of node depths (m), from 0 to depth_max
    """
    z_fine   = np.arange(0,           depth_fine,            dz_fine)
    z_coarse = np.arange(depth_fine + dz_coarse, depth_max + dz_coarse, dz_coarse)
    z        = np.concatenate([z_fine, z_coarse])
    if z[-1] < depth_max:
        z = np.append(z, depth_max)
    return z.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SURFACE BOUNDARY CONDITION — Newton-Raphson solver
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True, inline='always')
def _surface_bc(T_below, dz, Q_solar, k_surf, emissivity):
    """
    Solve the surface energy balance for the surface temperature T0.

    Equation:  k (T_below − T0) / dz + Q_solar − ε σ T0⁴ = 0

    Parameters
    ----------
    T_below    : temperature at the node just below the surface (K)
    dz         : depth spacing between surface and first sub-surface node (m)
    Q_solar    : absorbed solar flux (W/m²)
    k_surf     : thermal conductivity at the surface (W/m/K)
    emissivity : IR emissivity (dimensionless)
    """
    T0 = max(40.0, T_below)   # initial guess

    for _ in range(10):       # Newton-Raphson iterations (converges in ~3)
        f  = k_surf * (T_below - T0) / dz + Q_solar - emissivity * sigma * T0 ** 4
        df = -k_surf / dz - 4.0 * emissivity * sigma * T0 ** 3

        if abs(df) < 1e-10:
            break

        T0_new = T0 - f / df
        if abs(T0_new - T0) < 1e-5:   # converged
            break
        T0 = max(20.0, min(450.0, T0_new))   # clamp to physical range

    return T0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN THERMAL SOLVER
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=False, fastmath=True)
def solve_thermal_model(
    z_grid,
    T_init,
    lat_deg, lon_deg,
    slope, aspect,
    horizons, az_angles,
    chi,
    model_id,
    sunscale,
    ndays,
    H_param    = 0.07,
    dt_frac    = 0.01,
    albedo     = DEFAULT_ALBEDO,
    emissivity = DEFAULT_EMISSIVITY,
):
    """
    Run the 1-D thermal diffusion model at a single surface point.

    The simulation covers *ndays* complete lunar days.  The first (ndays − 1)
    days act as spin-up so the temperature field reaches a periodic steady
    state; analysis is done on the final day.

    Parameters
    ----------
    z_grid     : depth grid (m), from create_depth_grid()
    T_init     : uniform initial temperature (K)
    lat_deg    : latitude (degrees, positive = north)
    lon_deg    : longitude (degrees, 0–360 east)
    slope      : surface slope (radians)
    aspect     : surface aspect, clockwise from north (radians)
    horizons   : horizon elevation array (from compute_horizon_profile)
    az_angles  : corresponding azimuth array (radians)
    chi        : radiative conductivity parameter (typically 2–4)
    model_id   : density model — 0 = discrete, 1 = hayne, 2 = custom
    sunscale   : solar flux multiplier (1.0 = nominal)
    ndays      : number of lunar days to simulate
    dt_frac    : timestep as a fraction of the numerical stability limit
    albedo     : Bond albedo (fraction of sunlight reflected)
    H_param    : Hayne exponential scale height (m); used only when model_id == 1
    emissivity : IR emissivity

    Returns
    -------
    T_profile : (n_snapshots, n_depths) float32 — temperature (K) vs time and depth
    t_arr     : (n_snapshots,) float32          — times of each snapshot (seconds)
    """
    nz = len(z_grid)
    T  = np.full(nz, T_init, dtype=np.float32)

    # ── Timestep from stability criterion ─────────────────────────────────────
    # dt < 0.5 · ρ · c · Δz² / k  (explicit finite-difference stability)
    dz_min    = np.min(np.diff(z_grid))
    dt_stable = 0.5 * 1800.0 * 1000.0 * dz_min ** 2 / 0.02
    dt        = dt_frac * dt_stable

    nt_total      = int(np.ceil(ndays * LUNAR_DAY / dt))
    save_interval = max(1, nt_total // 10_000)
    nt_save       = nt_total // save_interval + 1

    T_profile = np.zeros((nt_save, nz), dtype=np.float32)
    t_arr     = np.zeros(nt_save, dtype=np.float32)

    dz      = np.diff(z_grid)

    # Precompute depth-only quantities — these never change during the simulation
    rho  = np.array([get_density(z_grid[iz], model_id, H_param) for iz in range(nz)],
                    dtype=np.float32)
    dz_c = np.array([0.5 * (dz[iz - 1] + dz[iz]) for iz in range(1, nz - 1)],
                    dtype=np.float32)

    T_profile[0, :] = T
    save_idx = 1

    for it in range(nt_total):
        t = it * dt

        # Solar geometry
        zenith, azimuth, _ = solar_geometry(lat_deg, lon_deg, t)

        # Shadowing check
        lit = check_illumination(zenith, azimuth, horizons, az_angles)
        Q_solar = (direct_solar_flux(zenith, azimuth, slope, aspect,
                                     sunscale, albedo)
                   if lit else 0.0)

        # Material properties (k and cp depend on temperature — must stay in loop)
        k  = np.empty(nz, dtype=np.float32)
        cp = np.empty(nz, dtype=np.float32)

        for iz in range(nz):
            k[iz]  = thermal_conductivity(T[iz], z_grid[iz], chi, model_id, H_param)
            cp[iz] = heat_capacity(T[iz])

        # Surface BC
        T[0] = _surface_bc(T[1], dz[0], Q_solar, k[0], emissivity)

        # Interior — explicit finite difference
        T_new = T.copy()
        for iz in range(1, nz - 1):
            k_m  = 0.5 * (k[iz - 1] + k[iz])
            k_p  = 0.5 * (k[iz]     + k[iz + 1])
            q_m  = k_m * (T[iz] - T[iz - 1]) / dz[iz - 1]
            q_p  = k_p * (T[iz + 1] - T[iz]) / dz[iz]
            T_new[iz] = T[iz] + dt * (q_p - q_m) / (dz_c[iz - 1] * rho[iz] * cp[iz])

        # Bottom BC — constant basal heat flux
        T_new[-1] = T[-2] + Q_basal * dz[-1] / k[-1]

        T = T_new

        # Save snapshot
        if (it + 1) % save_interval == 0 and save_idx < nt_save:
            T_profile[save_idx, :] = T
            t_arr[save_idx]        = (it + 1) * dt
            save_idx += 1

    return T_profile, t_arr


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER — pure-Python H-parameter version
# ─────────────────────────────────────────────────────────────────────────────

def solve_with_h(
    z_grid, T_init,
    lat_deg, lon_deg,
    slope, aspect,
    horizons, az_angles,
    chi, model_id, sunscale, ndays,
    H_param=0.07,
    dt_frac=0.01,
    albedo=DEFAULT_ALBEDO,
    emissivity=DEFAULT_EMISSIVITY,
    density_fn=None,
    k_solid_fn=None,
):
    """
    Run the solver with a custom density/conductivity pair (pure Python).

    This is the right function to use when sweeping the H-parameter, because
    numba-compiled functions cannot read Python globals that change at runtime.
    Pass density_fn and k_solid_fn as callables (z, H) → scalar.

    If density_fn / k_solid_fn are None, falls back to the compiled solver.
    """
    if density_fn is None and k_solid_fn is None:
        return solve_thermal_model(
            z_grid, T_init,
            lat_deg, lon_deg,
            slope, aspect,
            horizons, az_angles,
            chi, model_id, sunscale, ndays,
            H_param, dt_frac, albedo, emissivity,
        )

    # model_id 0 (discrete) and 1 (hayne) are fully supported by the compiled
    # solver — route them through Numba regardless of whether callables were passed.
    if model_id in (0, 1):
        return solve_thermal_model(
            z_grid, T_init,
            lat_deg, lon_deg,
            slope, aspect,
            horizons, az_angles,
            chi, model_id, sunscale, ndays,
            H_param, dt_frac, albedo, emissivity,
        )

    # Pure-Python fallback — only reached for model_id == 2 (custom) with callables
    from lunar.models   import heat_capacity as cp_fn
    from lunar.solar    import solar_geometry as sol_geo
    from lunar.horizon  import check_illumination as chk_ill
    from lunar.solar    import direct_solar_flux as dflux
    from lunar.constants import sigma as _sigma, Q_basal as _Qb, LUNAR_DAY as _LD

    nz        = len(z_grid)
    T         = np.full(nz, float(T_init))
    dz        = np.diff(z_grid)
    dz_min    = float(np.min(dz))
    dt_stable = 0.5 * 1800.0 * 1000.0 * dz_min ** 2 / 0.02
    dt        = dt_frac * dt_stable
    nt_total  = int(np.ceil(ndays * _LD / dt))
    save_int  = max(1, nt_total // 5_000)
    nt_save   = nt_total // save_int + 1
    T_profile = np.zeros((nt_save, nz), dtype=np.float32)
    t_arr     = np.zeros(nt_save, dtype=np.float32)
    T_profile[0, :] = T
    sidx = 1

    # rho and ks depend only on z_grid and H_param — compute once outside the loop
    rho = np.array([density_fn(float(z), H_param) for z in z_grid])
    ks  = np.array([k_solid_fn(float(z), H_param) for z in z_grid])

    for it in range(nt_total):
        t_now = it * dt
        zen, az, _ = sol_geo(lat_deg, lon_deg, t_now)
        lit = bool(chk_ill(zen, az, horizons, az_angles))
        Q = (dflux(zen, az, slope, aspect, sunscale, albedo)
             if lit else 0.0)

        cp  = np.array([cp_fn(t_)  for t_ in T])
        k   = ks * (1.0 + chi * (T / 350.0) ** 3)

        # Surface BC (Newton)
        T0 = max(40.0, float(T[1]))
        for _ in range(10):
            f  = k[0] * (T[1] - T0) / dz[0] + Q - emissivity * _sigma * T0 ** 4
            df = -k[0] / dz[0] - 4.0 * emissivity * _sigma * T0 ** 3
            if abs(df) < 1e-10:
                break
            T0n = T0 - f / df
            if abs(T0n - T0) < 1e-5:
                break
            T0 = max(20.0, min(450.0, T0n))
        T[0] = T0

        T_new = T.copy()
        for iz in range(1, nz - 1):
            km   = 0.5 * (k[iz - 1] + k[iz])
            kp   = 0.5 * (k[iz]     + k[iz + 1])
            qm   = km * (T[iz] - T[iz - 1]) / dz[iz - 1]
            qp   = kp * (T[iz + 1] - T[iz]) / dz[iz]
            dzc  = 0.5 * (dz[iz - 1] + dz[iz])
            T_new[iz] = T[iz] + dt * (qp - qm) / (dzc * rho[iz] * cp[iz])
        T_new[-1] = T[-2] + _Qb * dz[-1] / k[-1]
        T = T_new

        if (it + 1) % save_int == 0 and sidx < nt_save:
            T_profile[sidx, :] = T.astype(np.float32)
            t_arr[sidx]        = (it + 1) * dt
            sidx += 1

    return T_profile, t_arr
