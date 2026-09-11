#!/usr/bin/env python3
"""
layer_series.py -- four-thickness layer series: what does the paired
protocol's `f` measure, and what is the sensing decay length?

    python layer_series.py                 # preflight, free
    python layer_series.py --submit        # 6 simulations (Tidy3D credits)
    python layer_series.py --analyse       # fit the stored runs, free

Run from inside cdfdtd/.

PURPOSE
-------
The paired protocol gives f = layer shift / (S_bulk * dn_applied). For a 7 nm
active layer this ratio is 1.09, so f cannot be a fraction of the mode's
energy. Measuring f at four active-layer thicknesses (2, 3, 5 and 7 nm) lets a
one-parameter decay model be tested against four points rather than fitted to
two, and gives the sensing decay length delta.

GEOMETRY
--------
    3 nm case    recognition 3 nm + fouling 4 nm  = 7 nm stack
    7 nm case    recognition 7 nm + fouling 0 nm  = 7 nm stack
    recognition n = 1.4530,  fouling n = 1.4500

The total stack is held at 7 nm throughout and only the boundary between
active and inactive material moves inside it. Every run therefore shares one
water compartment, which is the one the bulk step changes. A nearly constant
S_bulk across the series is expected for that reason.

THE THREE CANDIDATE READINGS
----------------------------
Write W(r) for the fraction of the mode's dielectric sensing weight within
distance r of the metal, and take W(r) = 1 - exp(-2r/delta).  The layer step
probes 0..d; the bulk step probes r > T with T = 7 nm.

    M1  f = W(d)                          f as a fraction
    M2  f = W(d) / (1 - W(d))             layer against everything else
    M3  f = W(d) / (1 - W(T))             layer against WATER only

M3 is what the simulations construct, because the bulk run raises
`background_n` and the water starts at the outside of the stack.  Each model
has one free parameter, and four thicknesses are enough to separate them.

COST
----
Two new paired runs (2 and 5 nm), six simulations. The 3 and 7 nm runs are
read from `data/paired_pml` and `data/paired_redesign_pml`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings

import numpy as np

STACK_TOTAL_NM = 7.0

#: Thicknesses measured by earlier paired runs, and where they are stored.
#: Both are 7 nm stacks under PML, so they belong to the same series.
EXISTING = {
    3.0: "data/paired_pml",
    7.0: "data/paired_redesign_pml",
}

#: The two new points.  2 nm is the thinnest the 0.933 nm gap mesh can carry
#: at about two cells across the layer; below that the layer is not resolved
#: and the run would measure the mesh.
NEW = [2.0, 5.0]


def series_config(cfg, layer_nm: float, boundary: str | None):
    """A config with `layer_nm` of active layer inside a fixed 7 nm stack."""
    from dataclasses import replace

    fouling = max(0.0, STACK_TOTAL_NM - layer_nm)
    c = replace(cfg, surface=replace(cfg.surface,
                                     recognition_thickness_nm=layer_nm,
                                     fouling_thickness_nm=fouling))
    if boundary is not None:
        c = replace(c, fdtd=replace(c.fdtd, boundary=boundary))
    return c


def out_dir_for(layer_nm: float, boundary: str) -> str:
    return os.path.join("data", f"layer{layer_nm:g}nm_{boundary}")


# ==========================================================================
#  The models
# ==========================================================================
def f_M1(d, delta):
    return 1.0 - np.exp(-2.0 * d / delta)


def f_M2(d, delta):
    w = 1.0 - np.exp(-2.0 * d / delta)
    return w / (1.0 - w)


def f_M3(d, delta, T=STACK_TOTAL_NM):
    w = 1.0 - np.exp(-2.0 * d / delta)
    return w / np.exp(-2.0 * T / delta)


MODELS = [
    ("M1  f = W(d)                 [the paper's reading]", f_M1),
    ("M2  f = W(d)/(1-W(d))        [layer vs everything else]", f_M2),
    ("M3  f = W(d)/(1-W(T))        [layer vs water only]", f_M3),
]


def fit(model, ds, fs):
    """Least squares on log residuals, one parameter. Returns (delta, rms%)."""
    ds, fs = np.asarray(ds, float), np.asarray(fs, float)

    def cost(delta):
        if delta <= 0:
            return 1e9
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            pred = model(ds, delta)
        if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
            return 1e9
        return float(np.sum((np.log(pred) - np.log(fs)) ** 2))

    grid = np.geomspace(0.3, 400.0, 4000)
    best = min(grid, key=cost)
    # local refinement
    for span in (0.3, 0.05, 0.01):
        lo, hi = best * (1 - span), best * (1 + span)
        g = np.linspace(lo, hi, 400)
        best = min(g, key=cost)
    pred = model(ds, best)
    rms = 100.0 * float(np.sqrt(np.mean((pred / fs - 1.0) ** 2)))
    return float(best), rms, pred


# ==========================================================================
#  Measuring one trio
# ==========================================================================
def measure(out_dir: str, cfg):
    """Recompute a stored trio. Costs nothing; returns None if not on disk."""
    import tidy3d as td

    from observables import hotspot_spectrum, fit_resonance, shift_stability
    from paired_run import build_pair

    need = [os.path.join(out_dir, f"{n}.hdf5")
            for n in ("unbound", "bound", "bulk_hi")]
    if not all(os.path.isfile(p) for p in need):
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = build_pair(cfg)

    spectra = {}
    for name, path in zip(("unbound", "bound", "bulk_hi"), need):
        data = td.SimulationData.from_file(path)
        spectra[name] = hotspot_spectrum(data)
        if name == "unbound":
            lam = fit_resonance(*spectra[name]).wavelength_um * 1000
        del data

    wl, y = spectra["unbound"]
    layer_pm = shift_stability(wl, y, spectra["bound"][1])["shift_pm"]
    bulk_pm = shift_stability(wl, y, spectra["bulk_hi"][1])["shift_pm"]
    s_bulk = (bulk_pm / 1000.0) / built["d_bulk"]
    surf = built["config"].surface
    dn = surf.recognition_n_bound - surf.recognition_n_unbound
    return {
        "lam_nm": lam,
        "layer_pm": layer_pm,
        "bulk_pm": bulk_pm,
        "s_bulk": s_bulk,
        "f": layer_pm / (dn * s_bulk * 1000.0),
    }


def collect(cfg, boundary: str):
    """Every thickness with results on disk, from both sources above."""
    from config import StudyConfig

    rows = []
    for d, path in sorted(EXISTING.items()):
        r = measure(path, series_config(cfg, d, boundary))
        if r:
            r.update(layer_nm=d, source=path, paid="already")
            rows.append(r)
    for d in NEW:
        path = out_dir_for(d, boundary)
        r = measure(path, series_config(cfg, d, boundary))
        if r:
            r.update(layer_nm=d, source=path, paid="this series")
            rows.append(r)
    return sorted(rows, key=lambda r: r["layer_nm"])


# ==========================================================================
def analyse(cfg, boundary: str):
    rows = collect(cfg, boundary)
    if not rows:
        raise SystemExit("Nothing on disk yet. Run with --submit first.")

    print()
    print("MEASURED SERIES  (7 nm stack throughout, only the active/dead "
          "boundary moves)")
    print("=" * 78)
    print("  layer(nm)  lam(nm)  layer(pm)  bulk(pm)  S_bulk     f        "
          "source")
    for r in rows:
        print(f"  {r['layer_nm']:>9.1f} {r['lam_nm']:8.2f} "
              f"{r['layer_pm']:10.2f} {r['bulk_pm']:9.2f} "
              f"{r['s_bulk']:7.1f} {r['f']:8.4f}   {r['paid']}")

    sb = [r["s_bulk"] for r in rows]
    print(f"\n  S_bulk across the series: {min(sb):.1f} to {max(sb):.1f} "
          f"nm/RIU, spread {100*(max(sb)/min(sb)-1):.1f}%")
    print("  A flat S_bulk is EXPECTED here and is not evidence of anything")
    print("  being wrong: the water compartment is the same in every run.")

    if len(rows) < 3:
        print(f"\n  Only {len(rows)} thicknesses. Three or more are needed "
              f"before a one-parameter")
        print("  model is tested rather than merely fitted. Run --submit.")
        return

    ds = [r["layer_nm"] for r in rows]
    fs = [r["f"] for r in rows]

    print()
    print(f"FITTING ONE PARAMETER TO {len(rows)} POINTS")
    print("=" * 78)
    results = []
    for name, model in MODELS:
        delta, rms, pred = fit(model, ds, fs)
        results.append((rms, name, delta, pred))
        print(f"\n  {name}")
        print(f"    best delta {delta:8.2f} nm      rms residual "
              f"{rms:6.2f}%")
        print("    layer(nm)   measured f   predicted f   error")
        for d, fm, fp in zip(ds, fs, pred):
            print(f"    {d:>9.1f} {fm:12.4f} {fp:13.4f} "
                  f"{100*(fp/fm-1):+7.1f}%")

    results.sort()
    rms_best, name_best, delta_best, _ = results[0]
    rms_next = results[1][0]
    print()
    print("VERDICT")
    print("=" * 78)
    print(f"  best fit   {name_best.split('[')[0].strip()}")
    print(f"             delta = {delta_best:.2f} nm, rms {rms_best:.2f}%")
    print(f"  runner up  {results[1][1].split('[')[0].strip()}  "
          f"rms {rms_next:.2f}%")
    if rms_best < 5.0 and rms_next > 3 * rms_best:
        print("\n  One model fits and the others do not.")
    elif rms_best < 5.0:
        print("\n  More than one model fits acceptably. The thicknesses are")
        print("  too close together to separate them; add a point further out.")
    else:
        print("\n  NOTHING fits well. The exponential-decay picture itself is")
        print("  the thing to question, not which normalisation goes with it.")

    if name_best.startswith("M1"):
        print("\n  M1 fits best: f behaves as a fraction of the sensing weight.")
    else:
        print("\n  M1 does not fit best: f is a RATIO, not a fraction. The")
        print("  physical overlap W(d) is listed below, and the sensing decay")
        print("  length follows from it rather than from f.")
        print("\n    layer(nm)   quoted f    physical W(d)")
        for d, fm in zip(ds, fs):
            print(f"    {d:>9.1f} {fm:11.4f} {f_M1(d, delta_best):15.4f}")


# ==========================================================================
if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import StudyConfig
    from paired_run import run_pair

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boundary", choices=["absorber", "pml"], default="pml",
                    help="One boundary for the whole series (default pml, "
                         "which converges in the boundary depth test).")
    ap.add_argument("--thicknesses", type=float, nargs="+", default=NEW,
                    help=f"Active-layer thicknesses to RUN (default "
                         f"{NEW}). d = 3 and 7 are read from disk.")
    ap.add_argument("--submit", action="store_true",
                    help="Actually run. Without this nothing is uploaded.")
    ap.add_argument("--analyse", action="store_true",
                    help="Only fit what is already on disk.")
    args = ap.parse_args()

    cfg = StudyConfig()

    if args.analyse:
        analyse(cfg, args.boundary)
        raise SystemExit(0)

    print("LAYER SERIES PREFLIGHT -- free")
    print(f"  boundary                {args.boundary}")
    print(f"  stack total             {STACK_TOTAL_NM:g} nm "
          f"(active + dead, held fixed)")
    print(f"  already on disk         "
          f"{', '.join(f'{d:g} nm' for d in sorted(EXISTING))}")
    print(f"  to run                  "
          f"{', '.join(f'{d:g} nm' for d in args.thicknesses)}")
    print()
    print("  layer(nm)  dead(nm)  cells across layer   output")
    ok = True
    for d in args.thicknesses:
        c = series_config(cfg, d, args.boundary)
        dl = min(c.fdtd.mesh_study_dl_nm)
        cells = d / dl
        flag = "" if cells >= 2.0 else "   <- UNDER-RESOLVED"
        if cells < 2.0:
            ok = False
        print(f"  {d:>9.1f} {STACK_TOTAL_NM - d:>9.1f} {cells:>18.1f}   "
              f"{out_dir_for(d, args.boundary)}{flag}")
    print()
    print(f"  simulations             {3 * len(args.thicknesses)} "
          f"({len(args.thicknesses)} trios of 3)")
    print("  the unbound run doubles as the bulk-sensitivity low point, so")
    print("  each trio is three runs and not four")
    if not ok:
        raise SystemExit("\n  A layer under two cells thick measures the mesh, "
                         "not the mode. Refusing.")
    if not args.submit:
        print("\n  Run again with --submit to spend.")
        raise SystemExit(0)

    for d in args.thicknesses:
        out = out_dir_for(d, args.boundary)
        print(f"\n=== {d:g} nm active layer -> {out} ===")
        try:
            res = run_pair(series_config(cfg, d, args.boundary),
                           submit=True, out_dir=out)
            if hasattr(res, "mode_overlap"):
                print(f"  f = {res.mode_overlap:.4f}")
        except RuntimeError as exc:
            # `run_pair` raises for an overlap outside (0, 1). Here f is
            # expected to exceed one at larger thicknesses, and the three runs
            # are already saved when it raises, so the series continues and
            # `analyse` reads them back.
            print(f"  run_pair declined to return a value: {exc}")
            print(f"  the three simulations are saved in {out}; continuing.")

    analyse(cfg, args.boundary)
