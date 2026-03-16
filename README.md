# 🌑 Apollo Landing Site Thermal Model

A clean, modular 1-D thermal model for the lunar subsurface, validated against
Apollo 15 and Apollo 17 Heat Flow Experiment drill-hole measurements.

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook Lunar_Thermal.ipynb
```

The LOLA DEM files are included in this repository — no separate download needed:

| File | Description |
|------|-------------|
| `LDEM_4.IMG` | Binary elevation data, 720 × 1440 pixels (4 pix/deg, ~7.6 km/pix) |
| `LDEM_4.LBL` | PDS3 label for the IMG file |
| `LDEM_4.JP2` | JPEG2000-compressed version of the same data |
| `LDEM_4_JP2.LBL` | PDS3 label for the JP2 file |

These are LRO LOLA Global Digital Elevation Model (LDEM) V3.0 products
(Dataset: `LRO-L-LOLA-4-GDR-V1.0`).  Elevation values are heights in metres
above a 1737.4 km reference sphere, scaled by 0.5 m/DN.  The model
auto-selects higher-resolution files if present (see `lunar/dem.py`).

## Repository layout

```
Lunar_Thermal.ipynb   ← main notebook (run this)
requirements.txt
lunar/
  __init__.py
  constants.py     ← physical constants, depth-grid settings, Apollo data
  models.py        ← density models (discrete, Hayne 2017, custom)
  dem.py           ← LOLA DEM loading and coordinate extraction
  horizon.py       ← horizon-profile computation and sky-view factor
  solar.py         ← solar geometry and direct solar-flux calculation
  solver.py        ← 1-D finite-difference thermal solver (Numba-compiled)
  analysis.py      ← statistics, sensitivity sweeps, batch processing
  plots.py         ← all matplotlib plotting functions
```

## Notebook sections

| # | Section | Purpose |
|---|---------|---------|
| 0 | Imports | Load all modules |
| 1 | Configuration | **Edit here** — location, model, parameters |
| 2 | Load DEM | Load LOLA elevation map, snap to target pixel |
| 3 | Run Model | Execute the thermal simulation |
| 4 | Diurnal Cycles | Temperature vs time at different depths |
| 5 | Heatmap | 2-D depth × time temperature view |
| 6 | Apollo Comparison | Validate against real drill-hole data |
| 7 | Model Comparison | Discrete layers vs Hayne 2017 |
| 8 | Sensitivity Analysis | Effect of changing one parameter |
| 9 | Single Point Analysis | Two models at custom locations |
| 10 | Batch Processing | Many locations from a list |

## Density models

| Key | Description |
|-----|-------------|
| `'discrete'` | Three layers from Apollo drill-core measurements: fluffy (0–H), transitional (H–20 cm), consolidated (>20 cm). |
| `'hayne_exponential'` | Smooth exponential compaction from Diviner global infrared mapping (Hayne et al. 2017, doi:10.1002/2017JE005387). |
| `'custom'` | Edit `density_custom()` and `k_solid_custom()` in `lunar/models.py`. |

## Tuning guide

| Model too cold | Model too hot |
|---|---|
| Increase `SUNSCALE` (try 1.05–1.15) | Decrease `SUNSCALE` (try 0.95–1.00) |
| Decrease `ALBEDO` (try 0.07–0.09) | Increase `ALBEDO` (try 0.12–0.14) |

## Apollo reference coordinates

| Site | Latitude | Longitude |
|------|----------|-----------|
| Apollo 15 (Hadley-Apennine) | 26.1323 °N | 3.6285 °E |
| Apollo 17 (Taurus-Littrow) | 20.1911 °N | 30.7723 °E |

## Physical model summary

```
Heat conduction:  ρ(z) · c(T) · ∂T/∂t = ∂/∂z [ k(T,z) · ∂T/∂z ]
Conductivity:     k(T,z) = k_solid(z) · [1 + χ · (T / 350)³]
Heat capacity:    c(T)   = polynomial (Hemingway et al. 1973)
Surface BC:       k ∂T/∂z + Q_solar = ε σ T⁴   (Newton-Raphson)
Bottom BC:        constant basal flux Q = 18 mW/m²
```
