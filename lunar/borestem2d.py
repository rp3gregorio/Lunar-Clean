"""
borestem2d.py  ─  2-D axisymmetric steady-state borestem thermal correction
                  for the Apollo Heat Flow Experiment.

Governing equation (cylindrical coordinates, steady-state)
-----------------------------------------------------------

    1   ∂       ∂T         ∂       ∂T
   ─── ── [ r k ── ]  +  ── [ k  ── ]  =  0
    r  ∂r      ∂r         ∂z      ∂z

Material zones
--------------
    r < r_i           : k = k_reg(z)      enclosed regolith  (sensor site)
    r_i ≤ r ≤ r_o     : k = K_FIBERGLASS  fiberglass wall
    r > r_o            : k = k_reg(z)      surrounding regolith
    z > BORESTEM_DEPTH : k = k_reg(z)      (casing ends; all regolith below)

Boundary conditions
-------------------
    r = 0    ∂T/∂r = 0          symmetry axis
    r = R∞   T = T_1D(z)        far field = undisturbed 1-D mean profile
    z = 0    T = T_surf_mean    mean surface temperature
    z = L    k ∂T/∂z = Q_basal  geothermal (upward) flux

The borestem warm bias at the sensors (r ≈ 0) is:

    ΔT_bs(z) = T_2d(r=0, z) − T_1D(z)   [K,  positive → sensor reads warmer]

Method
------
The steady-state elliptic PDE is discretised with a 2nd-order finite-volume
(control-volume) scheme on a non-uniform (r, z) grid.  At r = 0 the cylindrical
singularity is handled analytically via l'Hôpital's rule.  Interface
conductivities use harmonic averaging.  The resulting sparse linear system is
solved with scipy's direct solver (SuperLU).

Radial grid:  fine inside the casing and across the fiberglass wall,
              coarser in the far field.

References
----------
Grott M. et al. (2010) JGR 115 E12004 — 2-D FD borestem correction (HP³/InSight)
Carslaw H.S. & Jaeger J.C. (1959) "Conduction of Heat in Solids", §7.9
Langseth M.G. et al. (1976) JGR 81, 3143–3161
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from lunar.constants import (
    BORESTEM_OUTER_RADIUS_M,
    BORESTEM_WALL_M,
    K_FIBERGLASS,
    BORESTEM_DEPTH_M,
    Q_basal,
)

__all__ = [
    'solve_borestem_2d_steady',
    'borestem_2d_correction',
]

# Far-field radius where T is clamped to the undisturbed 1-D profile.
# 40× the borestem outer radius is comfortably in the far field.
_R_FAR_M = 0.50   # m


# ─────────────────────────────────────────────────────────────────────────────
# GRID AND CONDUCTIVITY FIELD
# ─────────────────────────────────────────────────────────────────────────────

def _build_r_grid() -> np.ndarray:
    """
    Non-uniform radial grid that resolves the fiberglass wall and far field.

    Zones (approximate node spacing):
        0       –  r_i   :  1.0 mm  — enclosed regolith / sensor region
        r_i     –  r_o   :  0.5 mm  — fiberglass wall (exactly resolved)
        r_o     –  5 cm  :  3.0 mm  — mechanically disturbed zone
        5 cm    –  50 cm :  50  mm  — undisturbed far field
    """
    r_i = BORESTEM_OUTER_RADIUS_M - BORESTEM_WALL_M   # inner wall face
    r_o = BORESTEM_OUTER_RADIUS_M                      # outer wall face

    segs = [
        np.arange(0.0,  r_i,           1.0e-3),          # interior
        np.arange(r_i,  r_o,           5.0e-4),          # wall
        np.arange(r_o,  5.0e-2,        3.0e-3),          # disturbed zone
        np.arange(5e-2, _R_FAR_M + 1e-9, 5.0e-2),        # far field
    ]
    r = np.unique(np.concatenate(segs)).astype(np.float64)
    # Guarantee that r_i and r_o are exact grid points (avoids floating-point gaps)
    r = np.sort(np.unique(np.concatenate([r, [r_i, r_o, _R_FAR_M]])))
    return r


def _build_k2d(r_grid: np.ndarray,
               z_grid: np.ndarray,
               k_1d:   np.ndarray,
               borestem_depth_m: float = BORESTEM_DEPTH_M) -> np.ndarray:
    """
    Assemble k[n_z, n_r] from the 1-D conductivity profile and borestem geometry.

    Inside the fiberglass wall (r_i <= r <= r_o, z <= borestem_depth_m) the
    conductivity is K_FIBERGLASS; everywhere else it equals k_1d(z).

    Parameters
    ----------
    r_grid           : (n_r,) radial node positions (m)
    z_grid           : (n_z,) depth node positions  (m)
    k_1d             : (n_z,) 1-D conductivity profile from the thermal model (W/m/K)
    borestem_depth_m : actual borehole depth (m); A15=1.62 m, A17=2.36 m

    Returns
    -------
    k2d : (n_z, n_r) conductivity field (W/m/K)
    """
    r_i = BORESTEM_OUTER_RADIUS_M - BORESTEM_WALL_M
    r_o = BORESTEM_OUTER_RADIUS_M

    # Start with the regolith profile broadcast across all radii
    k2d = np.outer(k_1d, np.ones(len(r_grid)))  # (n_z, n_r)

    # Overwrite the fiberglass wall region
    j_wall = np.where((r_grid >= r_i - 1e-9) & (r_grid <= r_o + 1e-9))[0]
    i_bore = np.where(z_grid <= borestem_depth_m + 1e-9)[0]
    k2d[np.ix_(i_bore, j_wall)] = K_FIBERGLASS

    return k2d


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE SYSTEM ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def _hmean(a: float, b: float) -> float:
    """Harmonic mean of two conductivities (safe against a + b = 0)."""
    s = a + b
    return 2.0 * a * b / s if s != 0.0 else 0.0


def _assemble(r_grid:      np.ndarray,
              z_grid:      np.ndarray,
              k2d:         np.ndarray,
              T_surf_mean: float,
              T_1d:        np.ndarray
              ) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Build the sparse linear system  A · T_vec = b.

    Node ordering: T[i, j]  →  linear index  p = i * n_r + j
        i  = depth index   (0 … n_z-1)
        j  = radial index  (0 … n_r-1)

    Boundary conditions applied in this order (first match wins):
        i == 0       : Dirichlet T = T_surf_mean     (surface)
        j == n_r-1   : Dirichlet T = T_1d[i]          (far field)
        i == n_z-1   : Neumann   k ∂T/∂z = Q_basal   (bottom)
        otherwise    : interior finite-volume stencil
    """
    n_z, n_r = len(z_grid), len(r_grid)
    N = n_z * n_r

    dr = np.diff(r_grid)   # (n_r-1,)
    dz = np.diff(z_grid)   # (n_z-1,)

    # Use COO triplets for sparse assembly
    rows_list: list[int] = []
    cols_list: list[int] = []
    vals_list: list[float] = []
    b = np.zeros(N, dtype=np.float64)

    def add(r_idx: int, c_idx: int, v: float) -> None:
        rows_list.append(r_idx)
        cols_list.append(c_idx)
        vals_list.append(v)

    def idx(i: int, j: int) -> int:
        return i * n_r + j

    for i in range(n_z):
        for j in range(n_r):
            p = idx(i, j)

            # ── Dirichlet: surface ────────────────────────────────────────────
            if i == 0:
                add(p, p, 1.0)
                b[p] = T_surf_mean
                continue

            # ── Dirichlet: far field ──────────────────────────────────────────
            if j == n_r - 1:
                add(p, p, 1.0)
                b[p] = T_1d[i]
                continue

            # ── Neumann: bottom  k ∂T/∂z = Q_basal ───────────────────────────
            #   One-sided FD: (T[n_z-1,j] − T[n_z-2,j]) / dz[-1] = Q_basal / k
            if i == n_z - 1:
                add(p, p,            1.0)
                add(p, idx(i-1, j), -1.0)
                b[p] = Q_basal * dz[-1] / k2d[i, j]
                continue

            # ── Interior finite-volume node ───────────────────────────────────
            r_j  = r_grid[j]
            dz_c = 0.5 * (dz[i-1] + dz[i])   # control-volume half-height × 2

            # -- z-direction (identical to 1-D solver; harmonic-mean faces) --
            k_im = _hmean(k2d[i-1, j], k2d[i,   j])
            k_ip = _hmean(k2d[i,   j], k2d[i+1, j])
            c_im = k_im / (dz[i-1] * dz_c)    # coefficient for T[i-1, j]
            c_ip = k_ip / (dz[i]   * dz_c)    # coefficient for T[i+1, j]

            # -- r-direction --
            if j == 0:
                # ── Symmetry axis (r = 0) ─────────────────────────────────────
                # lim_{r→0} (1/r) d/dr[r k dT/dr]  =  4 k½ (T₁ − T₀) / dr₀²
                # Derived from control-volume over disk 0 ≤ r ≤ dr₀/2:
                #   flux through outer face / volume = 4 k½ (T₁-T₀) / dr₀²
                k_rp = _hmean(k2d[i, 0], k2d[i, 1])
                c_rp = 4.0 * k_rp / (dr[0] ** 2)

                add(p, p,           -(c_rp + c_im + c_ip))
                add(p, idx(i,   1),   c_rp)
                add(p, idx(i-1, j),   c_im)
                add(p, idx(i+1, j),   c_ip)

            else:
                # ── General interior radial node ──────────────────────────────
                # Control-volume: r_{j-½} to r_{j+½} (area = 2π r_j dr_c per unit z)
                r_pm = 0.5 * (r_grid[j-1] + r_j)   # left  face radius
                r_pp = 0.5 * (r_j + r_grid[j+1])   # right face radius
                dr_c = 0.5 * (dr[j-1] + dr[j])     # control-volume width in r

                k_rm = _hmean(k2d[i, j-1], k2d[i, j])
                k_rp = _hmean(k2d[i, j],   k2d[i, j+1])
                # Coefficients include the 1/r factor from cylindrical divergence
                c_rm = r_pm * k_rm / (r_j * dr[j-1] * dr_c)
                c_rp = r_pp * k_rp / (r_j * dr[j]   * dr_c)

                add(p, p,            -(c_rm + c_rp + c_im + c_ip))
                add(p, idx(i,   j-1),  c_rm)
                add(p, idx(i,   j+1),  c_rp)
                add(p, idx(i-1, j),    c_im)
                add(p, idx(i+1, j),    c_ip)

    A = sp.csr_matrix(
        (vals_list, (rows_list, cols_list)),
        shape=(N, N),
        dtype=np.float64,
    )
    return A, b


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def solve_borestem_2d_steady(
    z_grid:          np.ndarray,
    T_mean:          np.ndarray,
    k_1d:            np.ndarray,
    T_surf_mean:     float,
    borestem_depth_m: float = BORESTEM_DEPTH_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the 2-D axisymmetric steady-state heat equation for the borestem.

    Parameters
    ----------
    z_grid           : (n_z,) depth array in metres, monotonically increasing from 0
    T_mean           : (n_z,) time-mean temperature profile from the 1-D solver (K)
    k_1d             : (n_z,) thermal conductivity evaluated at T_mean (W/m/K)
    T_surf_mean      : mean surface temperature (K)  [= T_mean[0] in most cases]
    borestem_depth_m : actual borehole depth (m); A15=1.62 m, A17=2.36 m

    Returns
    -------
    T_axis : (n_z,) temperature at the symmetry axis r = 0  (K)
             This is the temperature the sensor would read.
    dT_bs  : (n_z,) warm bias  T_axis − T_mean  (K)
             Positive values mean the sensor reads warmer than undisturbed regolith.
             Zero below borestem_depth_m (no casing there).
    T_2d   : (n_z, n_r) full 2-D temperature field (K)
    r_grid : (n_r,) radial grid used (m)

    Notes
    -----
    The 1-D profile T_mean is used simultaneously as:
      - the far-field Dirichlet BC at r = R∞
      - the reference for computing ΔT_bs

    The linear system has O(n_z × n_r) ≈ 10 000 unknowns and is solved
    exactly by SuperLU via scipy.sparse.linalg.spsolve.
    """
    z_grid  = np.asarray(z_grid,  dtype=np.float64)
    T_mean  = np.asarray(T_mean,  dtype=np.float64)
    k_1d    = np.asarray(k_1d,    dtype=np.float64)

    r_grid = _build_r_grid()
    k2d    = _build_k2d(r_grid, z_grid, k_1d, borestem_depth_m)
    A, b   = _assemble(r_grid, z_grid, k2d, float(T_surf_mean), T_mean)

    T_vec = spsolve(A, b)

    n_r   = len(r_grid)
    T_2d  = T_vec.reshape(len(z_grid), n_r)
    T_axis = T_2d[:, 0].copy()

    dT_bs = T_axis - T_mean
    # Below the borestem there is no casing; zero-out any numerical noise
    dT_bs[z_grid > borestem_depth_m] = 0.0

    return T_axis, dT_bs, T_2d, r_grid


def borestem_2d_correction(
    z_grid:          np.ndarray,
    T_mean:          np.ndarray,
    k_1d:            np.ndarray,
    T_surf_mean:     float,
    borestem_depth_m: float = BORESTEM_DEPTH_M,
) -> np.ndarray:
    """
    Convenience wrapper — returns only ΔT_bs(z) for apply_all_corrections().

    Parameters
    ----------
    z_grid, T_mean, k_1d, T_surf_mean  — same as solve_borestem_2d_steady()
    borestem_depth_m : actual borehole depth (m); A15=1.62 m, A17=2.36 m

    Returns
    -------
    dT_bs : (n_z,) K — positive = sensor reads warmer than undisturbed regolith
    """
    _, dT_bs, _, _ = solve_borestem_2d_steady(z_grid, T_mean, k_1d, T_surf_mean,
                                               borestem_depth_m)
    return dT_bs
