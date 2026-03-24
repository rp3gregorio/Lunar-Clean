"""
solver.py — 1-D finite-difference thermal solver for lunar regolith.

The model discretises the heat-conduction equation:

    ρ(z) · c(T) · ∂T/∂t = ∂/∂z [ k(T,z) · ∂T/∂z ]

on a non-uniform depth grid, with:

  • Surface boundary : energy balance  k ∂T/∂z + Q_solar = ε σ T⁴
    solved by Newton-Raphson at every timestep.
  • Bottom boundary  : constant basal heat flux Q_basal (Apollo HFE value).
  • Initial condition: uniform T_init scalar, or a pre-computed profile from
    compute_equilibrium_profile() (recommended for accurate deep temperatures).

The solver is compiled with Numba for speed (~10–20× faster than pure Python).

Key functions
-------------
create_depth_grid()          — Build the non-uniform depth array.
compute_equilibrium_profile() — Geothermal steady-state T(z) for initialisation.
solve_thermal_model()        — Run the simulation; return T(depth, time).

Numerical notes
---------------
Interface conductivities use the **harmonic mean** (physically correct for
materials in series; preserves flux continuity across layer boundaries).

Computation is performed in float64 for accuracy; only the saved T_profile
snapshots are downcast to float32 to reduce memory.

The timestep stability criterion is derived from the maximum thermal
diffusivity α_max = k_max / (ρ_min · c_min) across the grid.  The default
dt_frac = 0.20 gives a 2.5× safety margin below the Von Neumann limit while
running ~15× faster than the previous dt_frac = 0.01 default.
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
    z_grid : 1-D float64 array of node depths (m), from 0 to depth_max
    """
    z_fine   = np.arange(0,           depth_fine,            dz_fine)
    z_coarse = np.arange(depth_fine + dz_coarse, depth_max + dz_coarse, dz_coarse)
    z        = np.concatenate([z_fine, z_coarse])
    if z[-1] < depth_max:
        z = np.append(z, depth_max)
    return z.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# EQUILIBRIUM PROFILE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_equilibrium_profile(z_grid, T_surf_mean, model_id, chi=2.7):
    """
    Compute the geothermal steady-state temperature profile for initialisation.

    For depths below ~0.5 m, the thermal equilibration timescale exceeds one
    year, so a short simulation (< 10 lunar days) started from a uniform
    T_init never converges at depth.  Initialising from this profile instead
    gives physically correct deep temperatures from the very first timestep.

    Method
    ------
    In steady state, Q_basal is constant with depth (no internal heat sources):
        T(z) = T_surf_mean + Q_basal · R(z)
    where R(z) = ∫₀ᶻ dz' / k(z', T_mean) is the cumulative thermal resistance.
    k is evaluated at T ≈ T_surf_mean (a good approximation at depth where
    the diurnal swing is negligible).

    Parameters
    ----------
    z_grid      : (n,) depth array (m) from create_depth_grid()
    T_surf_mean : mean surface temperature (K) — use ~250 K for A15 lat, ~252 K for A17
    model_id    : 0 = discrete, 1 = hayne, 2 = custom
    chi         : radiative conductivity parameter (same as used in the solver)

    Returns
    -------
    T_profile : (n,) float64 — equilibrium temperature at each depth (K)

    Notes
    -----
    The diurnal skin depth on the Moon is ≈ 0.05–0.12 m, so above ~0.3 m
    the equilibrium assumption is an approximation.  The solver's spin-up
    (first ndays-1 lunar days) corrects the near-surface quickly.
    """
    from lunar.models import thermal_conductivity as tc_fn, MODEL_ID_MAP
    z  = np.asarray(z_grid, dtype=np.float64)
    dz = np.diff(z)

    # Evaluate k at a representative subsurface temperature
    T_ref = T_surf_mean   # a good approximation for z > 0.3 m
    k_arr = np.array([
        tc_fn(T_ref, float(zi), chi, model_id)
        for zi in z
    ])

    # Cumulative thermal resistance from surface (trapezoidal on the grid)
    R = np.zeros(len(z))
    for i in range(1, len(z)):
        k_face = 2.0 * k_arr[i-1] * k_arr[i] / (k_arr[i-1] + k_arr[i])  # harmonic mean
        R[i]   = R[i-1] + dz[i-1] / k_face

    # Geothermal profile: T increases downward due to basal heat flux
    T_eq = T_surf_mean + Q_basal * R
    return T_eq


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
        if abs(T0_new - T0) < 1e-6:   # converged to 1 mK
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
    dt_frac    = 0.20,
    albedo     = DEFAULT_ALBEDO,
    emissivity = DEFAULT_EMISSIVITY,
):
    """
    Run the 1-D thermal diffusion model at a single surface point.

    The simulation covers *ndays* complete lunar days.  The first (ndays − 1)
    days act as spin-up so the near-surface temperature field reaches a
    periodic steady state; analysis is done on the final day.

    For accurate temperatures below ~0.5 m, initialise T_init with the
    output of compute_equilibrium_profile() instead of a uniform value.

    Parameters
    ----------
    z_grid     : depth grid (m), from create_depth_grid()
    T_init     : initial temperature — scalar (K) for uniform start, or
                 1-D array of length nz for a pre-computed geothermal profile
                 (recommended; use compute_equilibrium_profile())
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
    dt_frac    : timestep as a fraction of the Von Neumann stability limit.
                 Default 0.20 gives a 2.5× safety margin and is ~15× faster
                 than the older default of 0.01.
    albedo     : Bond albedo (fraction of sunlight reflected)
    emissivity : IR emissivity

    Returns
    -------
    T_profile : (n_snapshots, n_depths) float32 — temperature (K) vs time and depth
    t_arr     : (n_snapshots,) float32          — times of each snapshot (seconds)
    """
    nz = len(z_grid)

    # Initialise temperature — accept scalar or profile array
    T = np.empty(nz, dtype=np.float64)
    if hasattr(T_init, '__len__') and len(T_init) == nz:
        for iz in range(nz):
            T[iz] = T_init[iz]
    else:
        for iz in range(nz):
            T[iz] = T_init

    # ── Timestep from stability criterion ─────────────────────────────────────
    # dt < 0.5 · Δz² / α_max   where  α_max = k_max / (ρ_min · c_min)
    #
    # Conservative upper bounds valid for chi in [1.5, 4], T_max = 420 K:
    #   k_max   ≈ k_solid_surface × (1 + 4 × (420/350)³) ≈ 8e-3 W/m/K
    #             (k_solid_deep = 1.2e-2 at cold T gives similar α since ρ·c
    #              is also larger there; surface is the critical zone)
    #   ρ_min   = 1100 kg/m³  (fluffy surface layer)
    #   c_min   = 400 J/kg/K  (cold regolith at ~100 K)
    #   α_max   ≈ 8e-3 / (1100 × 400) = 1.8e-8 m²/s
    # Using α_max = 2e-8 (rounded up for safety) gives a conservative limit.
    dz_min    = np.min(np.diff(z_grid))
    alpha_max = 2.0e-8   # m²/s — conservative upper bound for lunar regolith
    dt_stable = 0.5 * dz_min ** 2 / alpha_max
    dt        = dt_frac * dt_stable

    nt_total      = int(np.ceil(ndays * LUNAR_DAY / dt))
    save_interval = max(1, nt_total // 10_000)
    nt_save       = nt_total // save_interval + 1

    T_profile = np.zeros((nt_save, nz), dtype=np.float32)
    t_arr     = np.zeros(nt_save, dtype=np.float32)

    dz = np.diff(z_grid)

    T_profile[0, :] = T.astype(np.float32)
    save_idx = 1

    # Material properties (allocated once; updated each step)
    rho = np.empty(nz, dtype=np.float64)
    k   = np.empty(nz, dtype=np.float64)
    cp  = np.empty(nz, dtype=np.float64)

    for it in range(nt_total):
        t = it * dt

        # Solar geometry
        zenith, azimuth, _ = solar_geometry(lat_deg, lon_deg, t)

        # Shadowing check
        lit = check_illumination(zenith, azimuth, horizons, az_angles)
        Q_solar = (direct_solar_flux(zenith, azimuth, slope, aspect,
                                     sunscale, albedo)
                   if lit else 0.0)

        # Material properties at current temperatures
        for iz in range(nz):
            rho[iz] = get_density(z_grid[iz], model_id)
            k[iz]   = thermal_conductivity(T[iz], z_grid[iz], chi, model_id)
            cp[iz]  = heat_capacity(T[iz])

        # Surface BC
        T[0] = _surface_bc(T[1], dz[0], Q_solar, k[0], emissivity)

        # Interior — explicit finite difference with harmonic-mean face conductivity
        # Harmonic mean: k_face = 2 k_i k_{i+1} / (k_i + k_{i+1})
        # Physically correct for materials in series; preserves flux continuity
        # across layer boundaries (unlike arithmetic mean, which overestimates
        # the interface conductance at sharp discontinuities).
        T_new = T.copy()
        for iz in range(1, nz - 1):
            k_m  = 2.0 * k[iz - 1] * k[iz]     / (k[iz - 1] + k[iz])
            k_p  = 2.0 * k[iz]     * k[iz + 1] / (k[iz]     + k[iz + 1])
            q_m  = k_m * (T[iz] - T[iz - 1]) / dz[iz - 1]
            q_p  = k_p * (T[iz + 1] - T[iz]) / dz[iz]
            dz_c = 0.5 * (dz[iz - 1] + dz[iz])
            T_new[iz] = T[iz] + dt * (q_p - q_m) / (dz_c * rho[iz] * cp[iz])

        # Bottom BC — constant basal heat flux (first-order Neumann)
        T_new[-1] = T[-2] + Q_basal * dz[-1] / k[-1]

        T = T_new

        # Save snapshot (downcast to float32 for storage efficiency)
        if (it + 1) % save_interval == 0 and save_idx < nt_save:
            T_profile[save_idx, :] = T.astype(np.float32)
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
    dt_frac=0.20,
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

    T_init may be a scalar or a 1-D array (geothermal profile); see
    compute_equilibrium_profile().
    """
    if density_fn is None and k_solid_fn is None:
        return solve_thermal_model(
            z_grid, T_init,
            lat_deg, lon_deg,
            slope, aspect,
            horizons, az_angles,
            chi, model_id, sunscale, ndays,
            dt_frac=dt_frac, albedo=albedo, emissivity=emissivity,
        )

    # Pure-Python fallback for H-sweep
    from lunar.models   import heat_capacity as cp_fn
    from lunar.solar    import solar_geometry as sol_geo
    from lunar.horizon  import check_illumination as chk_ill
    from lunar.solar    import direct_solar_flux as dflux
    from lunar.constants import sigma as _sigma, Q_basal as _Qb, LUNAR_DAY as _LD

    nz  = len(z_grid)
    dz  = np.diff(z_grid)
    dz_min = float(np.min(dz))

    # Initialise T from scalar or profile (float64 throughout)
    T_init_arr = np.asarray(T_init, dtype=np.float64)
    if T_init_arr.ndim == 0 or len(T_init_arr) != nz:
        T = np.full(nz, float(T_init), dtype=np.float64)
    else:
        T = T_init_arr.copy()

    # Stability: same physically motivated criterion as Numba solver
    alpha_max = 2.0e-8   # m²/s — conservative upper bound
    dt_stable = 0.5 * dz_min ** 2 / alpha_max
    dt        = dt_frac * dt_stable
    nt_total  = int(np.ceil(ndays * _LD / dt))
    save_int  = max(1, nt_total // 5_000)
    nt_save   = nt_total // save_int + 1
    T_profile = np.zeros((nt_save, nz), dtype=np.float32)
    t_arr     = np.zeros(nt_save, dtype=np.float32)
    T_profile[0, :] = T.astype(np.float32)
    sidx = 1

    for it in range(nt_total):
        t_now = it * dt
        zen, az, _ = sol_geo(lat_deg, lon_deg, t_now)
        lit = bool(chk_ill(zen, az, horizons, az_angles))
        Q = (dflux(zen, az, slope, aspect, sunscale, albedo)
             if lit else 0.0)

        rho = np.array([density_fn(float(z), H_param) for z in z_grid])
        ks  = np.array([k_solid_fn(float(z), H_param) for z in z_grid])
        cp  = np.array([cp_fn(T_)  for T_ in T])   # note: T_ is temperature, not time
        k   = ks * (1.0 + chi * (T / 350.0) ** 3)

        # Surface BC (Newton-Raphson, float64)
        T0 = max(40.0, float(T[1]))
        for _ in range(10):
            f  = k[0] * (T[1] - T0) / dz[0] + Q - emissivity * _sigma * T0 ** 4
            df = -k[0] / dz[0] - 4.0 * emissivity * _sigma * T0 ** 3
            if abs(df) < 1e-10:
                break
            T0n = T0 - f / df
            if abs(T0n - T0) < 1e-6:
                break
            T0 = max(20.0, min(450.0, T0n))
        T[0] = T0

        # Interior — harmonic-mean face conductivity
        T_new = T.copy()
        for iz in range(1, nz - 1):
            km   = 2.0 * k[iz - 1] * k[iz]     / (k[iz - 1] + k[iz])
            kp   = 2.0 * k[iz]     * k[iz + 1] / (k[iz]     + k[iz + 1])
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
