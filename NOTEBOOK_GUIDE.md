# Lunar Thermal Notebook — Plain-Language Guide

A guide to every section of `Lunar_Thermal_Presentation.ipynb`, written so that
someone unfamiliar with the code can understand what each cell does and why.

---

## Background: What This Notebook Does

We are simulating the temperature of the Moon's surface and shallow underground
layers across a full lunar day (~29.5 Earth days). Then we compare those simulations
against real measurements from the Apollo 15 and Apollo 17 Heat Flow Experiment (HFE),
which buried temperature probes up to 2.3 m deep in the lunar soil.

The goal is to check whether our physics model correctly predicts the temperatures
that the Apollo astronauts measured, both the **average** temperature at each depth
and how much the temperature **oscillates** over a lunar day.

---

## §1 — Apollo HFE Data Quality (Cells 7–8)

**What it shows:** Raw temperature time-series from every sensor in both Apollo
probes, plotted over the full operational lifetime of each probe.

**Why it matters:** The Apollo probes were not always in a clean, undisturbed state.
Several things contaminated the data:
- **Emplacement transient** — heat from drilling the borehole takes weeks to dissipate
- **Thermal disturbances** — astronaut visits, equipment nearby, or unexplained jumps
- **Data gaps / flat-line artefacts** — sensor failure or transmission loss

The plots shade these bad periods in orange and highlight the "stable window" in green.
Only the green portions are used for scientific comparison.

**Key terms:**
- **TG sensor** (Thermogradient bridge): Measures temperature *differences* between
  two points, used to compute the heat flow. Most reliable.
- **TR sensor** (Reference thermocouple): Measures absolute temperature at depth.
- **TC sensor** (Cable thermocouple): Sits in the cable above the probe; very shallow,
  strongly affected by day/night cycles, not used for equilibrium comparison.

---

## §2 — Thermal Model Properties (Cell 10)

**What it shows:** How the thermal conductivity (k) and density (ρ) of the lunar
soil change with depth in each of our two models.

**Why it matters:** The lunar surface is covered in *regolith* — a powdery layer
of crushed rock created by billions of years of meteorite impacts. The top few
centimetres are extremely fine-grained and poorly conducting (k ≈ 0.001 W/m/K,
roughly the same as aerogel). Deeper down, the grains are larger and pack together
under their own weight, so conductivity increases.

**The two models compared throughout the notebook:**

| Model | Description |
|-------|-------------|
| **Discrete (layered)** | Regolith is divided into discrete layers with different properties. Matches published lab measurements and retains the DEM topographic horizon. |
| **Hayne 2017** | Uses a smooth exponential function for how density and conductivity change with depth, as published by Hayne et al. (2017). Assumes a flat horizon (no topographic shadowing). |

Both models include a **radiative conductivity** term (k increases with T³) because
at very low temperatures, heat radiation between soil grains becomes important.

---

## §3 — Surface Diurnal Temperature Cycles (Cells 12–13)

**What it shows:** How the temperature oscillates over one lunar day at different
depths. The surface can swing from ~100 K at night to ~380 K at noon. By 50 cm
depth, the swing is almost gone.

**Key concept — thermal skin depth:** Heat diffuses slowly into the ground.
The "skin depth" is the depth at which the diurnal oscillation drops to 1/e ≈ 37%
of its surface amplitude. For the Moon, this is about 50 cm. Below that, the soil
barely feels the day/night cycle and sits at a nearly constant temperature set by
the balance between solar heat absorbed at the surface and geothermal heat flowing
up from the interior.

---

## §4 — Model vs Apollo Readings: Mean Temperature Profile (Cells 15–16)

**What it shows:** The time-averaged temperature at each depth from the model
(solid line) versus the stable-window median temperatures from each Apollo sensor
(coloured dots).

**How to read it:** A perfect model would have its line pass exactly through all
the dots. Deviations tell us where the model's physics is off.

**Typical result:** Both models slightly over-predict deep temperatures, and the
borestem effect (explained in §5) accounts for part of the discrepancy.

---

## §5 — Borestem Thermal Correction (Cells 17–19)

### What is the borestem and why does it matter?

When the Apollo astronauts drilled the boreholes, they used a hollow fiberglass
casing called the **borestem** to keep the hole open. This casing stayed in the
ground permanently.

**The problem:** Fiberglass is about 40× more thermally conductive than the
fine-grained surface regolith (k = 0.04 vs 0.001 W/m/K). This means heat from
the warm near-surface layers flows much more easily down through the fiberglass
tube than it would through undisturbed soil — like a thermal short-circuit.
The sensors inside the tube therefore read slightly *warmer* than the true
undisturbed temperature.

**The fix:** We solve a 2-D heat equation (cylindrical geometry: radius × depth)
to compute how much warmer the sensors read because of the borestem. Then we
subtract that offset from the model to predict what the sensors *should* have read
if they had been in the ground without the casing.

**Expected magnitude:**
- Borestem warm bias: +1.2 to +1.8 K
- Probe-top solar absorption (probe head near surface): +0.5 to +2.1 K
- Total: roughly +1.7 to +3.5 K warmer than true undisturbed soil

**What the plot shows (Cells 18–19):** The dashed lines show the borestem-corrected
model profiles. Ideally the dashed lines pass closer through the Apollo data points
than the solid (uncorrected) lines.

**Important surface boundary condition note:** The correction uses a surface
temperature of ~250–253 K (the geothermal mean), *not* the arithmetic mean of
the diurnal cycle (~214 K). Using the wrong value would create a spurious −20 K
cooling artefact in the correction.

---

## §6 — Geothermal Heat Flow (Cell 21)

**What it shows:** How the model's deep temperature gradient compares to the
published Apollo heat flow measurements.

**Why it matters:** The Moon's interior is slowly cooling. Radioactive decay of
uranium, thorium, and potassium releases heat that flows upward and maintains a
temperature gradient in the deep subsurface. This "geothermal heat flow" is:

> Q = k × (dT/dz)

where k is thermal conductivity and dT/dz is the temperature gradient.

**Published values (Langseth et al. 1976):**
- Apollo 15: 21 mW/m² (milliwatts per square metre)
- Apollo 17: 16 mW/m²
- Average: 18 mW/m²

These are some of the most important measurements for understanding the Moon's
bulk composition and its thermal history over 4.5 billion years.

---

## §7 — Animations (Cells 23–24)

**What they show:** Animated GIFs of the temperature profile T(z) evolving
through one lunar day.

- **Animation 1:** Depth profile only — watch the surface temperature peak at noon
  then cool at night while the underground layers respond with a time delay.
- **Animation 2:** Side-by-side view of surface forcing (solar flux) and the
  evolving depth profile.

**Key insight:** Notice that deeper layers don't just oscillate with smaller
amplitude — they also peak *later*. This time delay (called the **thermal phase lag**)
is a fundamental consequence of heat diffusion. It is preserved in the model plots
by aligning the entire model to the Apollo time frame using a *single* global time
shift (see §10 for details).

---

## §8 — Plain-Language Summary (Cell 27 markdown)

A non-technical plain-English description of the entire model chain, suitable for
a general audience.

---

## §9 — Deep Temperature Profile (Cell 26)

The same comparison as §4 but extended to the full 3 m model depth, showing the
transition from the diurnally active zone into the thermally stable deep layer.

---

## §10 — Apollo Model Comparison Graph (Cells 28–29)

### Overview

This is the most comprehensive single figure in the notebook. It combines four
types of information into one multi-panel view for each Apollo site.

**Row 1 — Full time-series:**
The complete temperature record from both Apollo probes (years of data), coloured
by sensor depth. Green shading marks the stable window used for all comparisons.
Orange shading marks known disturbances.

**Row 2 — Amplitude comparison:**
At each sensor depth, how large is the peak-to-peak temperature swing over a
lunar day? The model and Apollo data are shown side by side. This tests whether the
model correctly captures how rapidly the diurnal oscillation decays with depth.

**Row 3 — Phase-matched diurnal cycles:**
Each panel shows one sensor depth. Both the Apollo data (solid) and model (dashed)
are plotted as temperature *anomaly* (i.e. T minus its own mean), so we can compare
shapes and timing without absolute temperature offsets confusing the picture.

**Row 4 — Topographic shadowing effect:**
Shows the discrete model with and without DEM (Digital Elevation Model) horizon
shadows, so you can see how much local terrain features (hills, craters) affect
the temperature prediction.

### Phase alignment — how we synchronise the clocks

A key technical challenge: the model simulation starts at an arbitrary time (t = 0
is just when we started computing), and the Apollo data has its own calendar clock.
Neither zero point has any physical meaning. To compare them, we need to put both
on the same time axis.

**The method:** Both the model and the Apollo data should show their highest surface
temperature at *local solar noon* (when the Sun is directly overhead). We:
1. Find the time of peak surface temperature in the Apollo data.
2. Find the time of peak surface temperature in the model.
3. Compute the difference and shift the model curves by that amount.
4. Apply the **same** shift to all depths.

Step 4 is critical. If we computed a separate shift for each depth, every panel
would look perfectly aligned, but we would have destroyed the physically real
depth-dependent phase lag (the fact that deeper sensors peak later). Using one
global shift preserves those lags.

**Why Hayne and discrete use different shadow settings:**
The Hayne 2017 model was formulated assuming a flat horizon (no hills or craters).
Adding topographic shadows would be physically inconsistent with that model's
assumptions. Therefore:
- **Discrete model** uses real DEM horizon profiles
- **Hayne model** uses a flat horizon (no shadows)

This can be changed by the toggle flags `DISC_USE_SHADOWS` and `HAYNE_USE_SHADOWS`
in Cell 4.

---

## §11 — Borestem-Corrected Diurnal Cycles (Cells 30–31)

### Cell 30 — Borestem solver re-run

This cell re-runs the full thermal simulation for both models at both sites, but
this time using a **modified thermal conductivity** that accounts for the presence
of the fiberglass borestem casing.

The borestem effectively raises the conductivity in the zone where the casing sits.
Higher conductivity means heat diffuses faster, which:
- Shifts the peak of the diurnal wave to arrive slightly earlier at depth
- Changes the apparent phase of the temperature oscillation

**How it works technically:**
1. Compute `k_eff(z)` = weighted average of regolith k and fiberglass k at each depth
2. Pass this custom k profile directly to the solver (`solve_with_ksolid`)
3. Run the full time-stepping simulation with the modified k
4. Extract the diurnal cycles as usual

Both models again respect the shadow toggle: discrete uses DEM horizons, Hayne
uses flat horizon.

### Cell 31 — Plot: borestem-corrected diurnal cycles

Same multi-depth comparison plot as §10 Row 3, but using the borestem-corrected
model runs. Compare this to Cell 32 (same plot without borestem correction) to see
how much the borestem shifts the phase and amplitude of the diurnal signal.

---

## §12 — Plain Diurnal Cycles (No Borestem) (Cell 32)

The same plot as Cell 31, but using the plain model runs from Cell 4. No borestem
correction is applied.

**Why show both?** The borestem is a real physical effect — the fiberglass tube
was physically present in the ground. But it is small (~1–2 K bias on mean T,
smaller effect on phase). Comparing the two versions tells us:
- How much of any model-Apollo phase mismatch is due to the borestem
- Whether the borestem correction improves or worsens the phase match

---

## §13 — Borestem Impact Statistics (Cell 33)

### What this figure shows

A three-panel quantitative comparison of all four model variants:

| Variant | Description |
|---------|-------------|
| Discrete (plain) | Standard discrete model, no borestem correction |
| Discrete (+ borestem) | Discrete model with borestem k_eff applied |
| Hayne (plain) | Hayne 2017 exponential model, no borestem |
| Hayne (+ borestem) | Hayne 2017 with borestem k_eff applied |

**Panel 1 — Mean temperature profile:**
All four model lines plotted against the Apollo sensor data points. Solid lines
are uncorrected; dashed lines are borestem-corrected. Apollo sensors are colour-coded
by sensor type (TG = gradient bridge, TR = reference, TC = cable). Shows whether the
borestem correction moves the model closer to the data.

**Panel 2 — Residuals (model − Apollo):**
The difference between each model variant and the Apollo data at each sensor depth.
A residual of zero means the model exactly matches the measurement. Positive values
mean the model predicts too high a temperature; negative means too low.
This panel makes it easy to see systematic biases across depths.

**Panel 3 — Error metrics bar chart:**
Three grouped bars per variant:
- **RMSE** (Root-Mean-Square Error): Overall mismatch, in Kelvin. Lower is better.
  Penalises large errors more than small ones.
- **|Bias|** (absolute value of mean residual): Systematic offset. If positive,
  the model is consistently too warm; if negative, too cold.
- **MAE** (Mean Absolute Error): Average mismatch, in Kelvin. Similar to RMSE
  but treats all errors equally regardless of size.

A "good" model variant would show all three bars as small as possible.

**What to look for:**
- Does the borestem correction reduce RMSE, or make it worse?
- Do the Hayne and discrete models have similar error levels, or does one
  consistently outperform the other?
- Is the dominant error random (high RMSE, low bias) or systematic
  (RMSE ≈ bias, meaning a constant offset)?

---

## Toggle Flags Reference

The following boolean flags in the notebook control key physics options:

| Cell | Flag | Default | Effect |
|------|------|---------|--------|
| Cell 4 | `DISC_USE_SHADOWS` | `True` | Discrete model uses DEM horizon |
| Cell 4 | `HAYNE_USE_SHADOWS` | `False` | Hayne model uses flat horizon |
| Cell 30 | `_BS_DISC_USE_SHADOWS` | `True` | Borestem discrete: use DEM horizon |
| Cell 30 | `_BS_HAYNE_USE_SHADOWS` | `False` | Borestem Hayne: flat horizon |

Setting `HAYNE_USE_SHADOWS = True` would be physically inconsistent with the Hayne
2017 paper (which assumes flat terrain), but the option is there if you want to
experiment.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Regolith** | Loose, fragmented surface material covering solid rock. On the Moon it is impact-generated "rock flour" up to several metres thick. |
| **Diurnal** | Relating to a day/night cycle. |
| **Skin depth** | Depth at which an oscillating surface signal has decayed to 37% of its surface amplitude. |
| **Thermal phase lag** | The delay between when the surface temperature peaks and when a deeper sensor peaks, caused by slow heat diffusion. |
| **k (conductivity)** | How easily heat flows through a material (W/m/K). Higher = better conductor. |
| **RMSE** | Root-Mean-Square Error. A single number summarising how far predictions are from measurements on average. |
| **HFE** | Heat Flow Experiment. |
| **DEM** | Digital Elevation Model — a 2-D map of surface heights. |
| **LLT** | Local Lunar Time — time measured relative to the local Sun position (0 = midnight, 0.5 = noon). |
| **Borestem** | The fiberglass casing left in the borehole after drilling. Acts as a thermal short-circuit. |
