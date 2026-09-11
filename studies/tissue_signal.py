#!/usr/bin/env python3
"""
tissue_signal.py -- measure the binding signal on tissue rather than in water.
Runs 2 simulations per seed (Tidy3D credits required).

    python studies/tissue_signal.py                     # preflight, free
    python studies/tissue_signal.py --seeds 8 --submit  # 2 simulations per seed
    python studies/tissue_signal.py --analyse           # re-read results, free

The surface sensitivity comes from the paired protocol in water, while the
noise floor comes from random tissue. This script links the two: for each
tissue realization it runs the recognition layer unbound and bound on the SAME
tissue, grid and seed, and takes the difference.

The seeds come from `ensemble.seed_list`, so these are the same realizations on
which the noise floor was measured.

OUTPUT
------
  mean binding shift in tissue      compared with the shift measured in water
  its standard deviation            spread of the signal across placements
  its coefficient of variation      a small value means the signal is stable
  correlation with local index      whether the shift tracks the local tissue
  propagated values                 signal, effective noise and site counts at
                                    full coverage

COST
----
Two simulations per seed. Eight seeds (16 runs, as used in the paper) resolve a
coefficient of variation of order 10%.
"""
from __future__ import annotations

import argparse
import json
import os

import _bootstrap
_bootstrap.setup()

OUT = os.path.join("data", "tissue_signal")


def case_path(seed, state):
    return os.path.join(OUT, f"seed{seed}_{state}.hdf5")


def build(cfg, seed, bound):
    from simulation import build_sensor_simulation
    from tissue import generate_tissue
    real = generate_tissue(cfg.tissue, cfg.fdtd, seed=seed)
    return build_sensor_simulation(cfg, realization=real, cd_bound=bound)


def measure(cfg, seed):
    import tidy3d as td
    from ensemble import local_index_region
    from observables import hotspot_spectrum, shift_stability
    from simulation import hdf5_looks_complete
    from tissue import generate_tissue

    lo, hi = case_path(seed, "unbound"), case_path(seed, "bound")
    if not (hdf5_looks_complete(lo) and hdf5_looks_complete(hi)):
        return None
    wl, y0 = hotspot_spectrum(td.SimulationData.from_file(lo))
    _, y1 = hotspot_spectrum(td.SimulationData.from_file(hi))
    st = shift_stability(wl, y0, y1)
    real = generate_tissue(cfg.tissue, cfg.fdtd, seed=seed)
    xlim, ylim, zlim = local_index_region(cfg)
    return {"seed": seed,
            "shift_pm": float(st["shift_pm"]),
            "resolved": bool(st.get("ok", False)),
            "mean_index_local": real.mean_index_in_region(xlim, ylim, zlim)}


def report(rows):
    import numpy as np
    from redesign import MEASURED, MEASURED_LAYER_SERIES

    water = MEASURED_LAYER_SERIES[3.0]["layer_shift_pm"]
    sh = np.array([r["shift_pm"] for r in rows])
    nb = np.array([r["mean_index_local"] for r in rows])
    print()
    print("BINDING SHIFT MEASURED IN TISSUE")
    print("=" * 70)
    print(f"  realizations                {len(sh)}")
    print(f"  unresolved shifts           "
          f"{sum(1 for r in rows if not r['resolved'])}")
    print(f"  mean shift in tissue        {sh.mean():.2f} pm")
    print(f"  same shift in water         {water:.2f} pm")
    print(f"  difference                  {100 * (sh.mean() / water - 1):+.1f}%")
    if len(sh) > 1:
        cv = sh.std(ddof=1) / sh.mean()
        print(f"  standard deviation          {sh.std(ddof=1):.2f} pm")
        print(f"  coefficient of variation    {100 * cv:.1f}%")
        if np.std(nb) > 0:
            r = float(np.corrcoef(nb, sh)[0, 1])
            print(f"  correlation with local n    r = {r:+.2f}")
    if len(sh) < 2:
        return

    # Two separate questions. Is the MEAN shift different from the water
    # value (t-test on the mean)? And does the shift VARY enough across
    # placements to act as a second noise term (added in quadrature)?
    import math
    sem = sh.std(ddof=1) / math.sqrt(len(sh))
    t = (sh.mean() - water) / sem if sem > 0 else float("inf")
    scale = MEASURED["dn_full_coverage"] / MEASURED["dn_applied"]
    sig_t = sh.mean() * scale
    spread = sh.std(ddof=1) * scale
    eff = math.hypot(MEASURED["speckle_pm"], spread)
    print(f"  offset significance         t = {t:.2f} on {len(sh) - 1} df")
    print()
    print("  PROPAGATED TO FULL COVERAGE")
    print(f"    signal, from tissue       {sig_t:.1f} pm")
    print(f"    its spread                {spread:.1f} pm")
    print(f"    effective noise           {eff:.1f} pm  "
          f"({100 * (eff / MEASURED['speckle_pm'] - 1):+.2f}% on sigma alone)")
    print(f"    signal to noise           {sig_t / eff:.2f}")
    print(f"    sites at full coverage    "
          f"{math.ceil((3 * eff / sig_t) ** 2)}")
    print(f"    sites at theta = 0.25     "
          f"{math.ceil((3 * eff / (0.25 * sig_t)) ** 2)}")
    print()
    if abs(t) > 2.5:
        print(f"  The mean differs from the water value by "
              f"{100 * (sh.mean() / water - 1):+.1f}% at t = {t:.1f}. That is a")
        print("  real offset, not sampling noise, so the tissue value is used as")
        print("  the signal.")
    else:
        print("  The mean is not distinguishable from the water value at this")
        print("  sample size.")
    print()
    if eff / MEASURED["speckle_pm"] < 1.05:
        print("  The shift varies little across placements, so carrying its")
        print("  scatter as a second noise term changes the noise by under 5%.")
    else:
        print("  The shift varies enough across placements that its scatter is")
        print("  a second noise term in its own right. Use the effective noise")
        print("  above in every derived quantity.")


if __name__ == "__main__":
    from simulation import require_tidy3d
    require_tidy3d()
    from config import StudyConfig
    from ensemble import seed_list

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8,
                    help="How many of the ensemble's seeds to use (default 8).")
    ap.add_argument("--boundary", choices=["pml", "absorber"], default="pml",
                    help="Boundary treatment (default pml, used for all "
                         "values in the paper).")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()

    from dataclasses import replace
    cfg = StudyConfig()
    cfg = replace(cfg, fdtd=replace(cfg.fdtd, boundary=args.boundary))
    seeds = seed_list(cfg)[:args.seeds]

    if args.analyse:
        rows = [r for r in (measure(cfg, s) for s in seeds) if r]
        if not rows:
            raise SystemExit(f"nothing on disk in {OUT}")
        report(rows)
        raise SystemExit(0)

    todo = [(s, st) for s in seeds for st in ("unbound", "bound")
            if not os.path.isfile(case_path(s, st))]
    print()
    print("TISSUE SIGNAL PREFLIGHT -- free")
    print(f"  boundary                  {cfg.fdtd.boundary}")
    print(f"  seeds                     {len(seeds)}  (from ensemble.seed_list)")
    print(f"  simulations to run        {len(todo)}  of {2 * len(seeds)}")
    print(f"  output                    {OUT}")
    print()
    print("  Measures the binding shift on tissue rather than in water.")
    if not args.submit:
        print("\n  Run again with --submit to spend.")
        raise SystemExit(0)

    from simulation import run as run_sim
    os.makedirs(OUT, exist_ok=True)
    for seed, state in todo:
        print(f"\n  seed {seed} {state}")
        run_sim(build(cfg, seed, state == "bound"),
                task_name=f"tissue_signal_{seed}_{state}",
                path=case_path(seed, state))

    rows = [r for r in (measure(cfg, s) for s in seeds) if r]
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    report(rows)
