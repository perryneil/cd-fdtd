#!/usr/bin/env python3
"""
bootstrap_refit.py -- confidence interval on the tissue noise floor by
case-resampling bootstrap. Runs offline, calls no solver.

    python studies/bootstrap_refit.py                       # PML ensemble
    python studies/bootstrap_refit.py --results <path.json>

The noise floor sigma_tissue is the residual standard deviation of resonance
wavelength after regressing out the local mean refractive index. Its 95%
confidence interval must include the uncertainty in the fitted line as well as
the scatter about it. This script therefore resamples (mean index, resonance)
PAIRS with replacement, refits slope and intercept on every resample, and takes
the residual standard deviation of each refit. This is the interval quoted in
the paper.

For comparison it also prints the narrower interval obtained by resampling the
residuals of a single fixed fit, which ignores uncertainty in the line.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import _bootstrap
_bootstrap.setup()


def load(path):
    with open(path) as fh:
        raw = json.load(fh)
    rows = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
    keep = []
    for r in rows:
        lam = r.get("resonance_um")
        fw = r.get("fwhm_um")
        nb = r.get("mean_index_local")
        ok = (r.get("decay_ok", True) and r.get("fit_ok", True)
              and lam is not None and fw is not None and nb is not None
              and np.isfinite(lam) and np.isfinite(fw) and fw > 0)
        if ok:
            keep.append((float(nb), float(lam) * 1000.0))
    if not keep:
        raise SystemExit(f"no usable rows in {path}")
    nb, lam = map(np.asarray, zip(*keep))
    return nb, lam


def case_bootstrap(nbar, lam, n_boot=5000, seed=0):
    """Resample PAIRS, refit the line on each resample."""
    rng = np.random.default_rng(seed)
    n = len(lam)
    out = np.empty(n_boot)
    kept = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        x, y = nbar[idx], lam[idx]
        if np.std(x) < 1e-14:          # degenerate resample, no slope defined
            out[b] = np.nan
            continue
        sl, ic = np.polyfit(x, y, 1)
        out[b] = np.std(y - (sl * x + ic), ddof=2)
        kept += 1
    return out[np.isfinite(out)], kept


def residual_bootstrap(nbar, lam, n_boot=5000, seed=0):
    """Resample residuals of one fixed fit (line uncertainty ignored)."""
    sl, ic = np.polyfit(nbar, lam, 1)
    resid = lam - (sl * nbar + ic)
    rng = np.random.default_rng(seed)
    n = len(resid)
    out = np.empty(n_boot)
    for b in range(n_boot):
        out[b] = np.std(resid[rng.integers(0, n, n)], ddof=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=os.path.join(
        "data", "ensemble_pml", "seed_results.json"),
                    help="seed_results.json to analyse. Relative paths are "
                         "taken from the cdfdtd/ package directory.")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    nbar, lam = load(args.results)
    n = len(lam)
    slope, intercept = np.polyfit(nbar, lam, 1)
    resid = lam - (slope * nbar + intercept)
    sigma = float(np.std(resid, ddof=2))
    raw = float(np.std(lam, ddof=1))

    case, kept = case_bootstrap(nbar, lam, args.n_boot, args.seed)
    res = residual_bootstrap(nbar, lam, args.n_boot, args.seed)
    ci_case = (np.percentile(case, 2.5), np.percentile(case, 97.5))
    ci_res = (np.percentile(res, 2.5), np.percentile(res, 97.5))

    print()
    print("CASE-RESAMPLING BOOTSTRAP")
    print("=" * 66)
    print(f"  source                      {args.results}")
    print(f"  realizations                {n}   (residual dof {n - 2})")
    print(f"  raw spread                  {raw * 1000:.1f} pm")
    print(f"  fitted slope                {slope:.1f} nm/RIU")
    print(f"  variance removed            "
          f"{100 * (1 - (sigma / raw) ** 2):.1f}%")
    print(f"  sigma_tissue                {sigma * 1000:.1f} pm")
    print()
    print(f"  residual resample (line held fixed)")
    print(f"     95% CI                   {ci_res[0]*1000:.0f} to "
          f"{ci_res[1]*1000:.0f} pm    width {1000*(ci_res[1]-ci_res[0]):.0f} pm")
    print(f"  case resample (slope and intercept refitted)")
    print(f"     95% CI                   {ci_case[0]*1000:.0f} to "
          f"{ci_case[1]*1000:.0f} pm    width {1000*(ci_case[1]-ci_case[0]):.0f} pm")
    print(f"     resamples used           {kept} of {args.n_boot}")
    print()
    widen = ((ci_case[1] - ci_case[0]) / (ci_res[1] - ci_res[0]) - 1) * 100
    print(f"  The case-resampling interval is {widen:+.0f}% wider.")
    print()
    print("  Method: case-resampling bootstrap, "
          f"{args.n_boot} resamples, seed {args.seed}.")
