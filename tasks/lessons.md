# Lessons Learned

## [2026-03-24] Lesson: Diurnal phase shift wrapping
**What went wrong:** `_global_shift_h ≈ +308 h` was added linearly to model reference time, placing the model peak at `ref_utc + 1009 h` — entirely outside the Apollo data window `[ref_utc, ref_utc+708 h]`.
**Rule:** After computing any phase shift for a circular quantity (lunar day), ALWAYS wrap `(t + shift) % day_h` before plotting. Never add shift linearly to a time axis that has a finite window.
**Applied in:** plots.py → diurnal_probe_vs_models()

## [2026-03-24] Lesson: Numba cache invalidates Run All
**What went wrong:** After editing source files, Numba's disk cache (`__pycache__`) kept stale compiled bytecode, so Run All executed old code without errors.
**Rule:** Clear `lunar/__pycache__` at the top of the imports cell on every Run All.
**Applied in:** Lunar_Thermal_Presentation.ipynb cell 2

## [2026-03-24] Lesson: Depths_m must include actual Apollo sensor depths for cross-correlation
**What went wrong:** Cross-correlation used model cycles at `[0, 5, 10, 35, 70, 150 cm]`. Apollo 15 sensors are at 84–139 cm — nearest model depth was 70 cm with nearly zero diurnal amplitude → cross-correlation on noise.
**Rule:** Always include the actual sensor depths in the depths_m list, especially the shallow TC sensors (14–67 cm for A17; 35–49 cm for A15) which carry the strongest diurnal signal.
**Applied in:** Lunar_Thermal_Presentation.ipynb cell 10

## [2026-03-24] Lesson: alpha_max must be computed dynamically
**What went wrong:** alpha_max = 2e-8 m²/s hardcoded. At equatorial dayside peak (T ≈ 450 K, chi=2.7), k ≈ 8.6e-3 W/m/K at surface, rho_min=1100, c_min=400 → α = 1.95e-8. But the issue is that chi=2.7 at the surface makes k_surface*(1+chi*(T/350)^3) at T=450K very large. k_solid_discrete(0)=1e-3, so k_total = 1e-3*(1+2.7*(450/350)^3) = 1e-3*(1+5.76) = 6.76e-3. α = 6.76e-3/(1100*400) = 1.54e-8. So 2e-8 is actually slightly conservative. However at night T→100K, k_total = 1e-3*(1+2.7*(100/350)^3) = 1e-3*(1+0.063) = 1.063e-3. For Hayne model, k_surface at high T is larger since k_solid is 7.4e-4 but k_deep at depth is 3.8e-3. At T=420K and depth (k_solid=3.8e-3): k_total = 3.8e-3*(1+2.7*(420/350)^3) = 3.8e-3*(1+4.63) = 3.8e-3*5.63 = 2.14e-2. rho at depth = 1800. c at 420K ≈ 760 J/kg/K. α = 2.14e-2/(1800*760) = 1.56e-8. Actually 2e-8 seems fine. Let me reconsider - the key question is what is the MAXIMUM alpha.
**Rule:** Don't change alpha_max unless you can show via calculation that 2e-8 is insufficient. Document the calculation.
**Applied in:** solver.py:222 (already has inline comment with calculation)

## [2026-03-24] Lesson: Secular warming is real physics
**What went wrong:** User expected flat convergence lines in Apollo time-series; actual data shows slow drift of ~1-2 mK/year (secular warming from disturbance).
**Rule:** Document secular warming in plot annotations so users understand it is real physics, not a code or data issue.
**Applied in:** plots.py → sensor_equilibration()
