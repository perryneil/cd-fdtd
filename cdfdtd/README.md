# cdfdtd: the simulation package

This folder holds the Tidy3D implementation of the study *A Plasmonic Probe of Free Ionic
Cadmium in Fish Muscle and the Detection Floor Set by Tissue Microstructure: A Simulation
Study*. This file describes the modules and how to use them. The README at the repository
root lists the main results and the commands that reproduce them.

Everything runs offline by default. Nothing is uploaded to Tidy3D and no credits are
spent unless you pass `--submit`.

```
pip install -r requirements.txt
python run_all.py                # every free stage, nothing uploaded
python run_all.py --estimate     # ask Tidy3D what the simulations would cost
python run_all.py --submit       # run the simulations (spends FlexCredits)
```

---

## What the package computes

A gold bowtie nanoantenna pressed against fish muscle shifts its plasmon resonance when
cadmium binds to the recognition layer on the gold. Muscle is heterogeneous, so its
refractive index varies from point to point and the resonance moves between sites even
when no cadmium is present. The package builds a synthetic muscle medium, simulates the
antenna against many realizations of it, measures the spread of the resonance, and
compares that spread with the shift produced by bound cadmium. The result is a detection
floor in concentration units and the number of independent sites an array needs.

---

## Modules

### Core pipeline

| File | What it does |
|---|---|
| `config.py` | Every adjustable parameter, grouped in dataclasses, with an explanation of each one. Start here. |
| `references.py` | Machine-readable bibliography. Each sourced parameter is paired with the paper it came from, and each modelling assumption is flagged. `python references.py` prints the provenance table. |
| `tissue.py` | Generates a three-dimensional Whittle-Matérn random refractive index field with the texture of muscle and converts it into a Tidy3D medium. |
| `optical_properties.py` | Calibrates the random medium so that it scatters like measured fish muscle, using the analytic Born approximation. |
| `sensor.py` | Builds the gold bowtie, the recognition and fouling layers, and the steric mask that decides where the analyte can reach. |
| `simulation.py` | Assembles the Tidy3D simulation: mesh, boundaries, source and monitors. Also prices a campaign before anything is submitted. |
| `observables.py` | Turns recorded fields into a resonance wavelength, a linewidth, a sensitivity map and a layer response ratio. |
| `ensemble.py` | Runs many tissue realizations and separates tissue speckle from finite-box drift. |
| `noise_floor.py` | Converts a resonance spread into a detection floor in coverage, molar and mg/kg units. |
| `convergence.py` | Checks that the tissue statistics and the resonance do not depend on the grid. |
| `run_all.py` | Runs the stages of the study in order (see below). |

### Measured values and design tables

| File | What it does |
|---|---|
| `redesign.py` | Holds the values measured from the stored simulations (`MEASURED`, `MEASURED_TISSUE_SIGNAL`, `MEASURED_LAYER_SERIES`) and derives the signal to noise ratio, site counts and detection limits from them, on the same basis as the paper (signal on tissue against the effective noise). `sites_for_signal` sizes an array, `design_points` gives the as-built and brush signals, `direct_contact_spec` gives the detection limits, and `redesign.self_check()` recomputes the full-coverage index change from `config.py`. |
| `recognition_layer.py` | Derives the refractive index change of the recognition layer on binding from published dn/dc values, layer contraction and refractive index increment changes, using de Feijter's formula. |
| `conformational_sweep.py` | Sweeps the layer contraction and refractive index increment terms together as a two-dimensional surface, because the two can cancel. |

### Supporting checks

| File | What it does |
|---|---|
| `anisotropy.py` | Anisotropic Born integral for fibre-aligned tissue, and the effect of antenna orientation on the fibre direction. |
| `coating.py` | Measures the thickness of the conformal layers against an exact signed distance function, and reports the effective gap. |
| `photon_budget.py` | Brackets the number of photons reaching the antenna through tissue and checks whether the measurement is photon limited. |

### Simulation runners (these can spend credits)

| File | What it does |
|---|---|
| `paired_run.py` | The paired protocol: unbound, bound and bulk-shifted runs that give the bulk sensitivity and the layer response ratio `f`. |
| `layer_series.py` | Layer thickness series used to fit the sensing decay length and the surface sensitivity. |
| `bulk_step_check.py` | Checks that the bulk sensitivity does not depend on the size of the index step used to measure it. |
| `boundary_test.py` | Boundary depth series comparing PML with the adiabatic absorber. |
| `find_resonance.py` | Wide-band survey that locates the antenna resonance before narrow-band runs. |

### Utilities

| File | What it does |
|---|---|
| `inspect_results.py` | Reads stored `.hdf5` results and reports field decay, resonance position and linewidth. |
| `fetch_task.py` | Downloads the result of a Tidy3D task that finished on the server but did not reach this machine. |

Most modules have a `__main__` block that demonstrates and checks the module offline:

```
python references.py
python recognition_layer.py
python tissue.py
python optical_properties.py
python sensor.py
python observables.py
python ensemble.py
python convergence.py
python noise_floor.py
python redesign.py
```

Scripts that run simulations print a free preflight (geometry, cell count and cost
estimate) when called without `--submit`. `bulk_step_check.py` and `layer_series.py` also
accept `--analyse`, which re-reads stored results without submitting anything.

---

## Stages of `run_all.py`

| Stage | Name | What it does | Cost |
|---|---|---|---|
| 0 | Configuration | Prints every parameter so the run documents itself. | Free |
| 1 | Calibration | Fits the tissue model to measured scattering. | Free |
| 2 | Validity checks | Born approximation and box size against the scattering length. 2b anisotropy, 2c reduced scattering coefficient. | Free |
| 3 | Convergence | Tissue index contrast against voxel size. Stop if it has not converged. | Free |
| 4 | Geometry | Builds the sensor and reports how much of the hot spot the analyte can reach. 4b conformal shell accuracy. | Free |
| 5 | Inventory | Number of cadmium ions available at the regulatory limit and the volume they are gathered from. 5c conformational sweep, 5d photon budget. | Free |
| 6 | Simulations | Builds the water reference and the tissue ensemble. 6b paired run. | Free to build, credits to run |
| 7 | Ensemble statistics | Separates speckle from finite-box drift (shown on synthetic data when no results are present). | Free |
| 8 | Noise floor | Detection floors against the 0.05 mg/kg limit. 8b free cadmium fraction. | Free |

`python run_all.py --stage N` runs one stage and its lettered sub-stages. `--seeds` sets
the number of realizations, `--out` sets the output folder, and `--paired` adds the paired
runs to a `--submit`.

---

## How the code maps onto the manuscript

| Manuscript section | Module |
|---|---|
| Scale separation between FDTD box and scattering length | `optical_properties.check_scale_splitting` |
| Heterogeneous tissue medium | `tissue.generate_tissue` |
| Calibration against measured optical properties | `optical_properties.calibrate` |
| Transducer and surface chemistry | `sensor.build_bowtie`, `sensor.build_surface_stack` |
| Steric accessibility | `sensor.steric_accessibility_mask` |
| FDTD configuration | `simulation.py` |
| Paired protocol and layer response ratio | `paired_run.py`, `layer_series.py`, `observables.py` |
| Tissue ensemble and noise floor | `ensemble.py`, `noise_floor.py`, `data/ensemble_pml/` |
| Signal to noise, array size and detection limit | `redesign.py` |

---

## Modelling choices worth knowing

**Graded mesh.** `simulation.build_grid_spec` uses Tidy3D's `GridSpec.auto` with
`MeshOverrideStructure` regions, so the gap and the gold surfaces are meshed finely while
the surrounding tissue uses a coarser grid.

**Normalised correlation function with an inner cutoff.** In the mass-fractal regime that
describes tissue (`m < 1.5`) the Whittle-Matérn spectrum has unbounded variance unless the
smallest structure size is set explicitly. `TissueConfig` carries a physical inner cutoff
and a variance normalisation so that the index contrast does not change with the voxel
size. `convergence.py` checks this for any choice of cutoff.

**Taper before the boundary.** Absorbing boundaries assume a medium that is uniform along
the absorption direction. `tissue.generate_tissue` tapers the index fluctuations to the
uniform background before the boundary region.

**Boundary type.** All values reported in the paper use a perfectly matched layer (PML).
Pass `--boundary pml` to the scripts that accept it (`paired_run.py`, `layer_series.py`,
`ensemble.py`, `bulk_step_check.py`, `boundary_test.py` and the simulation studies), or
set `FDTDConfig.boundary = "pml"` before using `run_all.py`. The default in `config.py`
is the adiabatic absorber, which reproduces the reference ensemble stored in `results/`.
`boundary_test.py` and `data/boundary_test/` compare the two.

**Dispersive energy density.** For gold the mode energy must use `∂(ωε)/∂ω |E|²` rather
than `ε |E|²`, because ε is negative in the visible and near infrared.
`observables.brillouin_factor` differentiates the material model numerically, so it stays
correct for any gold fit.

**Finite-box drift.** Each finite box has its own mean refractive index, and the
resonance follows it through the bulk sensitivity. `ensemble.py` regresses the resonance
against the local mean index and reports the residual spread as the tissue speckle.

**Detection floor and affinity.** The Langmuir inversion `C = θ / (K (1 − θ))` depends on
the binding constant `K`. The minimum detectable coverage does not, so `noise_floor.py`
reports the coverage floor, the concentration floor as a function of `K`, and the
inventory limited bound separately.

**Counting noise.** With `N` ions bound on average the count fluctuates by `√N`.
`noise_floor.compute_noise_floor` reports this Poisson term alone and combined with the
tissue speckle.

---

## Parameters taken from the literature

`python references.py` prints the full table with page and table numbers.

| Parameter | Value | Source |
|---|---|---|
| Cd(II) aptamer affinity | Kd = 34.5 nM | Wu et al., *Analyst* 139:1550 (2014) |
| Spread of published Cd aptamer affinities | 34.5 nM to 81.4 µM | Gao et al., *Biosensors* 13:612 (2023) |
| dn/dc | DNA 0.17, protein 0.185, chitosan 0.16 to 0.18 mL/g | Malvern, Wyatt TN4002 |
| Layer index relation | de Feijter's formula | de Feijter, Benjamins and Veer, *Biopolymers* 17:1759 (1978) |
| Aptamer site density | 4 × 10¹² cm⁻² = 0.04 nm⁻² | Steel, Herne and Tarlov, *Anal. Chem.* 70:4670 (1998) |
| Layer contraction on binding | 5.2 to 3.8 nm (fitted folding ratio 0.73) | Dejeu et al., *J. Phys. Chem. C* (2018) |
| Refractive index increment on binding | 0.252 to 0.241 cm³/g | Pons et al., *Analyst* 147:4197 (2022) |
| Cd bioaccessibility, raw fish muscle | 73.7 to 93.2% | He, Ke and Wang, *J. Agric. Food Chem.* 58:3517 (2010) |
| Cod muscle scattering coefficient | 4 to 6 mm⁻¹ (500 to 1700 nm) | Vraalstad et al., *Food Bioproc. Technol.* 18:9392 (2025) |
| Cod muscle anisotropy factor g | 0.90 to 0.97 | same |

The background index of raw muscle is not measured directly in any fish species.
`config.py` shows the two-component mixing calculation used to set it, so the value can
be checked or replaced.

---

## Scope notes built into the code

**Tissue anisotropy.** The correlation aspect ratio of muscle has not been measured, so
`anisotropy.py` constrains it from the reduced scattering ratio between probe
orientations and reports how much the antenna orientation matters. It also compares the
measured birefringence of fish muscle with the index fluctuation of the model.

**Coating accuracy and effective gap.** `sensor.build_surface_stack` offsets each
triangle exactly, and `coating.py` measures the result against a signed distance function.
The `gap_nm` parameter positions the sharp apices, so rounding makes the physical gap
larger. Quote `coating.effective_gap_nm`.

**Photon transport.** `photon_budget.py` brackets the delivered photon count between the
Beer-Lambert and diffusion limits. The measurement is photon limited only at depths far
beyond those this sensor works at, so no Monte Carlo transport stage is included.

**Borrowed conformational terms.** The contraction and refractive index increment values
come from other aptamer systems. `conformational_sweep.py` sweeps both and reports the
fraction of the plausible region with each sign.

**Free cadmium fraction.** Bioaccessibility measured after digestion is not the free ionic
fraction seen by a sensor on intact tissue. `RegulatoryConfig.free_fraction` carries this
factor through every conversion, and `RegulatoryConfig.free_fraction_direct_contact`
holds a bound derived from glutathione stability constants.

---

## Before running simulations

Cost scales with the number of cells times the number of time steps. Refining the mesh
raises both, because the Courant condition ties the time step to the smallest cell in the
domain. Relative to the 2 nm ensemble mesh on this geometry:

| gap mesh | cells | time steps | relative cost |
|---------:|------:|-----------:|--------------:|
| 4.0 nm | 13.9 M | 70,969 | 0.33x |
| 2.0 nm | 21.4 M | 138,464 | 1.00x |
| 1.0 nm | 41.5 M | 268,987 | 3.77x |
| 0.5 nm | 112.3 M | 530,820 | 20.11x |

For this reason `FDTDConfig.mesh_study_dl_nm` defaults to 4, 2 and 1 nm, the mesh series
runs in a smaller water box (`mesh_study_domain_um`), and
`simulation.estimate_campaign_cost` prices each run at its own mesh.

`BudgetConfig.flexcredits` is a spending ceiling. `--submit` refuses to start when the
projected total exceeds it and prints the settings that reduce the cost. `--over-budget`
overrides the check.

A suggested order:

1. `python convergence.py` to confirm that the tissue variance has converged.
2. `python optical_properties.py` to confirm the Born approximation holds and the box is
   smaller than a scattering length.
3. `python run_all.py --stage 4` to check that the analyte can reach the hot spot.
4. `python run_all.py --stage 5` to check that the index change gives a detectable shift.
5. `python run_all.py --estimate` to price the whole campaign.
6. `python run_all.py --stage 6 --submit --seeds 1` to run a single simulation. Confirm
   that `check_decay` passes and that the resonance lies inside the monitored band.
7. `python run_all.py --submit` for the full ensemble.

---

## Requirements

Python 3.10 or newer, `tidy3d >= 2.9`, `numpy` and `scipy`. A Tidy3D account and API key
are needed only for `--estimate` and `--submit`.
