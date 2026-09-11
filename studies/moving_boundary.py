#!/usr/bin/env python3
"""
moving_boundary.py -- is a fixed-geometry index change a valid stand-in for a
physical contraction of the recognition layer? Optional study. Runs 4
simulations (Tidy3D credits required).

    python studies/moving_boundary.py           # preflight, free
    python studies/moving_boundary.py --submit  # run the simulations
    python studies/moving_boundary.py --analyse # re-read results, free

The main study represents binding as an index change in a recognition layer of
FIXED thickness. Physically, the layer contracts on binding: it gets thinner
and denser, and the layer/water boundary moves inward through a near field that
varies steeply over a few nanometres. This script compares the two
representations directly.

THE TEST
--------
Two representations of the same bound state, carrying the same mass per unit
area, against one shared unbound reference:

  A  fixed geometry    3.000 nm at the bound index
  B  moving boundary   2.193 nm at the index that conserves mass over that
                       thickness, with the freed 0.807 nm becoming brush

By de Feijter, (n - n_medium) * d is proportional to mass per unit area, so
conserving mass while the thickness falls from d0 to d1 requires
(n1 - n_med) = (n0 - n_med) * d0 / d1. The total stack thickness is held at
7 nm in both cases, so the water compartment is unchanged and only the internal
boundary moves.

Because B changes the geometry it is also run with the gap mesh halved, so the
comparison is not confounded by discretization. Hence four simulations.
"""
from __future__ import annotations

import argparse
import json
import os

import _bootstrap
_bootstrap.setup()

FOLD = 0.269                  # assumed fractional contraction on binding
STACK_TOTAL_NM = 7.0          # recognition + fouling, held fixed
OUT = os.path.join("data", "moving_boundary")


def bound_thickness_nm(cfg):
    return cfg.surface.recognition_thickness_nm * (1.0 - FOLD)


def bound_index(cfg):
    """The index that conserves mass per unit area over the thinner layer."""
    d0 = cfg.surface.recognition_thickness_nm
    d1 = bound_thickness_nm(cfg)
    n_med = cfg.tissue.n0
    return n_med + (cfg.surface.recognition_n_bound - n_med) * d0 / d1


def case_path(tag):
    return os.path.join(OUT, f"{tag}.hdf5")


def config_for(cfg, tag):
    """Return (config, cd_bound) for each case."""
    from dataclasses import replace
    if tag in ("unbound", "fixed"):
        return cfg, (tag == "fixed")
    d1 = bound_thickness_nm(cfg)
    c = replace(cfg, surface=replace(
        cfg.surface,
        recognition_thickness_nm=d1,
        recognition_n_bound=bound_index(cfg),
        fouling_thickness_nm=max(0.0, STACK_TOTAL_NM - d1)))
    if tag == "moving_fine":
        # Halve dl_metal_nm together with dl_gap_nm. If the metal step were
        # left finer than the gap step it would override it, and the refined
        # run would silently use the same mesh as the coarse one.
        c = replace(c, fdtd=replace(c.fdtd,
                                    dl_gap_nm=c.fdtd.dl_gap_nm / 2.0,
                                    dl_metal_nm=c.fdtd.dl_metal_nm / 2.0))
    return c, True


def build(cfg, tag):
    from simulation import build_sensor_simulation
    c, bound = config_for(cfg, tag)
    return build_sensor_simulation(c, realization=None, cd_bound=bound)


def measure():
    import tidy3d as td
    from observables import hotspot_spectrum, shift_stability
    from simulation import hdf5_looks_complete
    ref = case_path("unbound")
    if not hdf5_looks_complete(ref):
        return None
    wl, y0 = hotspot_spectrum(td.SimulationData.from_file(ref))
    out = {}
    for tag in ("fixed", "moving", "moving_fine"):
        p = case_path(tag)
        if not hdf5_looks_complete(p):
            continue
        _, y = hotspot_spectrum(td.SimulationData.from_file(p))
        out[tag] = float(shift_stability(wl, y0, y)["shift_pm"])
    return out or None


def report(cfg, sh):
    print()
    print("FIXED GEOMETRY AGAINST A MOVING BOUNDARY")
    print("=" * 72)
    print(f"  unbound layer               "
          f"{cfg.surface.recognition_thickness_nm:.3f} nm at "
          f"n = {cfg.surface.recognition_n_unbound:.4f}")
    print(f"  contracted layer            {bound_thickness_nm(cfg):.3f} nm at "
          f"n = {bound_index(cfg):.4f}   ({100 * FOLD:.1f}% fold)")
    print(f"  stack total held at         {STACK_TOTAL_NM:.1f} nm")
    print()
    for tag, label in (("fixed", "index change at fixed d"),
                       ("moving", "boundary actually moves"),
                       ("moving_fine", "moving, gap mesh halved")):
        if tag in sh:
            print(f"  {label:34s} {sh[tag]:8.2f} pm")
    if "fixed" in sh and "moving" in sh:
        d = 100 * (sh["moving"] / sh["fixed"] - 1)
        print()
        print(f"  difference                         {d:+8.1f}%")
        if "moving_fine" in sh:
            m = 100 * (sh["moving_fine"] / sh["moving"] - 1)
            print(f"  of which mesh                      {m:+8.1f}%")
        print()
        if abs(d) < 10:
            print("  The fixed-geometry protocol is an adequate stand-in for a")
            print("  physical contraction (difference under 10%).")
        else:
            print("  The approximation differs by more than 10%. Use the")
            print("  moving-boundary shift for the full-coverage signal and")
            print("  propagate it to the detection limit and site count.")


if __name__ == "__main__":
    from dataclasses import replace

    from simulation import require_tidy3d
    require_tidy3d()
    from config import StudyConfig

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boundary", choices=["pml", "absorber"], default="pml")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()

    cfg = StudyConfig()
    cfg = replace(cfg, fdtd=replace(cfg.fdtd, boundary=args.boundary))
    tags = ["unbound", "fixed", "moving", "moving_fine"]

    if args.analyse:
        sh = measure()
        if not sh:
            raise SystemExit(f"nothing on disk in {OUT}")
        report(cfg, sh)
        raise SystemExit(0)

    todo = [t for t in tags if not os.path.isfile(case_path(t))]
    print()
    print("MOVING BOUNDARY PREFLIGHT -- free")
    print(f"  boundary                  {cfg.fdtd.boundary}")
    print(f"  unbound thickness         {cfg.surface.recognition_thickness_nm:.3f} nm")
    print(f"  contracted thickness      {bound_thickness_nm(cfg):.3f} nm")
    print(f"  contracted index          {bound_index(cfg):.4f}  "
          f"(mass per unit area conserved)")
    print(f"  simulations to run        {len(todo)}  ({', '.join(todo)})")
    print(f"  output                    {OUT}")
    print()
    print("  Both representations carry the same mass per unit area, so any")
    print("  difference in shift is the optical cost of moving the boundary")
    print("  rather than a difference in how much material is present.")
    if not args.submit:
        print("\n  Run again with --submit to spend.")
        raise SystemExit(0)

    from simulation import run as run_sim
    os.makedirs(OUT, exist_ok=True)
    for tag in todo:
        print(f"\n  {tag}")
        run_sim(build(cfg, tag), task_name=f"movebound_{tag}",
                path=case_path(tag))

    sh = measure()
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(sh, fh, indent=2)
    report(cfg, sh)
