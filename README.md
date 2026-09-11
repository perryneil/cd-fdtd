# cdfdtd: tissue noise floor of a contact plasmonic cadmium sensor

Simulation code and data for the paper

> P. N. J. Fernandez, F. Dela Cruz, F. De Guzman and R. F. Pobre, *A Plasmonic Probe of Free Ionic Cadmium in Fish
> Muscle and the Detection Floor Set by Tissue Microstructure: A Simulation Study*
> (submitted to Engineering Research Express).

The study uses finite-difference time-domain (FDTD) simulation in Tidy3D (version 2.12.0)
to place a gold bowtie nanoantenna against a Whittle-Matérn random medium calibrated to
fish muscle. It measures how much the tissue alone moves the plasmon resonance and turns
that into a detection limit and an array specification.

## Main results reproduced here

| Quantity | Value | Source file |
|---|---|---|
| Resonance standard deviation from tissue microstructure, 34 realizations, no analyte | 481.8 pm (95% CI 338 to 580 pm) | `cdfdtd/data/ensemble_pml/seed_results.json` |
| Bulk sensitivity | 194.1 nm/RIU | `cdfdtd/redesign.py` (`MEASURED`) |
| Surface sensitivity | 120 nm/RIU | `cdfdtd/redesign.py` |
| Full-coverage cadmium shift, measured on tissue | 1446 pm | `cdfdtd/redesign.py` (`tissue_signal_pm`), from `cdfdtd/data/tissue_signal/` |
| Independent sites for the quarter-occupancy design target | 17 at 15 µm pitch | `studies/audit_values.py` |

Every value quoted in the paper uses a perfectly matched layer (PML) boundary. The
`cdfdtd/results/` folder holds a reference ensemble computed with the adiabatic absorber
boundary, which the supplementary material reports for comparison only. The default
boundary in `cdfdtd/config.py` is the absorber so that this reference can be reproduced.
Pass `--boundary pml` (or set `FDTDConfig.boundary = "pml"`) for new runs.

## Quick start

Everything below runs offline and costs nothing.

```bash
pip install -r requirements.txt

# Regenerate all ten figures from the PML ensemble
python figs/make_figures.py

# Case-resampling bootstrap for the confidence interval on the noise floor
python studies/bootstrap_refit.py

# Recompute every derived number and check it against the manuscript source
python studies/audit_values.py --tex path/to/manuscript.tex --supp path/to/supplementary.tex
```

Running new simulations requires a Tidy3D account and API key. No script submits a job
unless it is given `--submit`. Each script prints a free preflight first.

## Repository layout

| Path | Contents |
|---|---|
| `cdfdtd/` | The simulation package: tissue generation, sensor geometry, simulation assembly, observables, ensemble and noise floor analysis. `cdfdtd/README.md` describes each module. |
| `cdfdtd/data/ensemble_pml/` | The 34 tissue realizations under PML. This is the central dataset. |
| `cdfdtd/data/tissue_signal/` | Bound versus unbound shift measured on tissue, 8 seeds. |
| `cdfdtd/data/covariance/` | Resonance against lateral offset along x and y, a check on whether neighbouring sites read independent tissue. |
| `cdfdtd/data/boundary_test/` | Boundary depth series showing the absorber does not converge and PML does. |
| `cdfdtd/data/bulkstep0.0013_pml/` | Bulk index step check. |
| `cdfdtd/data/ensemble_bigbox/` | Larger domain ensemble for the finite box comparison. |
| `cdfdtd/results/` | Reference ensemble with the absorber boundary, and the configuration it ran with. |
| `figs/` | Figure scripts and the PDF figures they produce. |
| `studies/` | Offline analysis scripts (bootstrap, value audit) and additional simulation studies (tissue signal, lateral covariance, contact gap, moving boundary, box size). `studies/README.md` describes each one. |

## Data not included

The raw Tidy3D field files (`.hdf5`, 50 to 550 MB each, about 55 GB in total) are not in
this repository. Every quantity the paper uses has been extracted into the JSON files
above. The raw files are available from the corresponding author on request.

## Citation

Please cite the paper above. A Zenodo DOI for this repository will be added on
acceptance.

## License

Code is released under the MIT License (see `LICENSE`). Data and figures are released
under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Contact

Perry Neil J. Fernandez, Department of Physics, De La Salle University, Manila, and
University of the Philippines Visayas, Iloilo, Philippines. pjfernandez@up.edu.ph.
ORCID [0000-0003-0789-8135](https://orcid.org/0000-0003-0789-8135)
