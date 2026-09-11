# studies/

Verification scripts and follow-up simulation studies. Run them from the
repository root. Each script locates the `cdfdtd/` package, adds it to
`sys.path` and works inside it, so results are read from and written to the same
`cdfdtd/data/` folder as the rest of the project.

## Offline scripts (no solver, no cost)

| Script | What it does |
|---|---|
| `audit_values.py` | Recomputes every derived number in the paper from the measured values in `cdfdtd/redesign.py` and, if given the manuscript source, checks that each value appears in it. |
| `bootstrap_refit.py` | Case-resampling bootstrap for the 95% confidence interval on the tissue noise floor. Refits slope and intercept on every resample. |

On `cdfdtd/data/ensemble_pml/seed_results.json` these give sigma = 481.8 pm, a
regression slope of 180.6 nm/RIU, 77.7% of the variance removed by the local
index correction, and a 95% confidence interval of 338 to 580 pm.

## Simulation studies (Tidy3D account required)

Every script prints a free preflight (boundary, cases, number of runs, output
folder) and only runs simulations when given `--submit`. Runs resume: results
already on disk are reused. All scripts default to `--boundary pml`, which is
the boundary used for every value in the paper. Note that the default in
`cdfdtd/config.py` is `absorber`, so pass `--boundary pml` explicitly when
calling other package scripts.

| Script | Runs | Purpose | Results in this repository |
|---|---|---|---|
| `tissue_signal.py` | 2 per seed | Binding shift measured on tissue rather than in water, using the ensemble's own seeds. | `cdfdtd/data/tissue_signal/` (8 seeds) |
| `lateral_covariance.py` | 9 per axis | Autocovariance of the resonance against lateral offset, to check that sites 15 um apart are independent. Memory grows with `--max-offset` (about 3.6 GiB at 8 um). | `cdfdtd/data/covariance/` (x and y) |
| `moving_boundary.py` | 4 | Compares a fixed-thickness index change with a layer that physically contracts, at equal mass per unit area. | not run |
| `box_pml.py` | 3 | Effect of domain depth on the resonance with identical tissue in every run. | not run |
| `contact_gap.py` | 2 per gap per seed | Effect of a fluid gap between sensor and tissue on signal and variability. Use `--seeds 4` or more for the variability trend. | not run |
