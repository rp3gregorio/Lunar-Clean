"""
lunar — Python package for the Apollo Landing Site Thermal Model.

Sub-modules
-----------
constants  : physical constants, depth-grid settings, Apollo HFE data
models     : density and thermal-conductivity models (discrete / Hayne / custom)
dem        : LOLA DEM loading and topographic extraction
horizon    : horizon-profile computation and sky-view factor
solar      : solar geometry and direct solar-flux calculation
solver     : 1-D finite-difference thermal solver
analysis   : post-processing: statistics, sensitivity sweeps, batch runs
plots      : all matplotlib plotting functions

Typical import pattern in the notebook
---------------------------------------
    from lunar import constants, models, dem, horizon, solver, analysis, plots
"""
