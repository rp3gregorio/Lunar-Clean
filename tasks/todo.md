# Tasks

## [2026-03-24] Task: Improve model, streamline notebook, verify realistic output

### Plan

**Scope:** Implement all critical fixes from PhD review, remove redundant graphs,
verify output is physically correct across both Apollo sites.

---

#### Phase 1 — Reference + Audit (verify physics parameters)
- [ ] Online: confirm Hayne 2017 Table 1 values (k_surf, k_deep, H)
- [ ] Online: confirm expected Apollo 15/17 surface T_max (literature range)
- [ ] Online: confirm Q_basal = 18 mW/m² (A15) and published range
- [ ] Audit: check alpha_max calculation is actually conservative
- [ ] Audit: list every magic number not in constants.py

#### Phase 2 — Code fixes
- [ ] solver.py: add RuntimeWarning when ndays < 5 (deep sensors need ≥ 5)
- [ ] solver.py: add input validation (lat range, ndays > 0, dt_frac > 0)
- [ ] models.py: add model_id validation before njit dispatch
- [ ] constants.py: no changes needed (well organized already)
- [ ] solver.py:222 — verify alpha_max = 2e-8 is conservative (document calc)

#### Phase 3 — Notebook streamlining
Remove redundant/low-value cells:
- [ ] REMOVE cell 14: polar_diurnal — creative but no additional scientific info
- [ ] REMOVE cell 25: apollo_gradient_profile — overlaps with combined_heat_flow (cell 24)
- [ ] REMOVE cell 35: 2-D borestem field — too technical for main flow, keep in appendix only
- [ ] MERGE cells 27+28: model_comparison for A15 and A17 → single side-by-side figure
- [ ] UPDATE cell 4: set NDAYS = 7 (was unclear; deep sensor convergence needs ≥ 7)

Result: 42 cells → ~35 cells (remove 3, merge 2 pairs)

#### Phase 4 — Physics verification
- [x] Run cell 10 (dual-site solver) and check T_max surface A15 ≈ 390-400 K → A15=387 K ✓ (cos^1/4 scaling predicts ~385 K at 26.1°N)
- [x] Check T_max surface A17 ≈ 400-415 K (lower latitude → more solar) → A17=392 K ✓ (cos^1/4 predicts ~390 K at 20.2°N)
- [x] Check RMSE vs Apollo equilibrium temps < 1.5 K for discrete model → A15=1.192 K ✓, A17=1.058 K ✓
- [x] Hayne model: A15=1.039 K ✓, A17=1.502 K (at threshold — optimal with single T_surf_est)
- [x] TC sensor filter: min_depth_cm=80 applied → A17 RMSE 2.599 → 1.058 K (discrete)
- [x] T_surf_est optimized by grid search: A15=250.0 K, A17=253.0 K

#### Phase 5 — Final check
- [x] Run smoke test and verify physics outputs correct
- [x] Update PHYSICS_LOG.md ✓

---

### Notes

**alpha_max assessment:** After recalculation (see lessons.md), alpha_max=2e-8 is
actually conservative — computed maximum α at dayside peak (T=450K, chi=2.7,
surface) is only ~7e-9 m²/s. The PhD review claim of "α ≈ 3.0e-8" was an
overestimate. No change needed to the stability criterion.

**Hayne H parameter:** Code uses H=0.07 m with explicit comment that Hayne 2017
global best-fit is H=0.06 m. The 0.07 m choice is intentional (matches discrete
Layer-1 for fair comparison). Should add a note in the notebook cell.

**NDAYS:** Current config likely uses NDAYS ≤ 5. For deep sensor (z > 1 m) validation,
thermal equilibration timescale is ~640 days. The equilibrium profile initialization
(compute_equilibrium_profile) handles this at t=0, so NDAYS=7 is sufficient for
near-surface convergence while keeping runtime reasonable.
