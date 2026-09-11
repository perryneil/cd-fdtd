#!/usr/bin/env python3
"""
bulk_step_check.py -- does the bulk sensitivity depend on the size of the
index step used to measure it? Runs 1 simulation (Tidy3D credits required).

    python bulk_step_check.py                # preflight, free
    python bulk_step_check.py --submit       # run the one simulation
    python bulk_step_check.py --analyse      # re-read results on disk, free

Run from inside cdfdtd/.

BACKGROUND
----------
The layer response and the bulk response are measured with different index
steps:

    layer step   dn = 1.3e-3   ->  shift  156 pm
    bulk  step   dn = 5.0e-3   ->  shift  970 pm

A plasmon's sensitivity need not be constant in the ambient index, so S_bulk
measured with a large step might differ from the local sensitivity the layer
experiences. S_bulk is the denominator of the layer to water response ratio f,
so this matters for f (but not for the surface sensitivity or the signal,
which do not divide by S_bulk).

THE TEST
--------
The bulk-sensitivity simulation is repeated with the background step matched
to the layer step (1.3e-3). If S_bulk agrees with the 5.0e-3 value, step size
does not affect f.

The reference spectrum is the unbound run in `data/paired_pml`, so only the
new stepped run (bulk_hi) is simulated. The result reported in the paper is in
`data/bulkstep0.0013_pml/result.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

WATER_N = 1.333

#: Paired run whose unbound spectrum is the reference, and the S_bulk it gave
#: with the standard 5.0e-3 background step.
REFERENCE_DIR = os.path.join("data", "paired_pml")
REFERENCE_S_BULK = 194.1
REFERENCE_STEP = 5.0e-3


def out_dir_for(step: float, boundary: str) -> str:
    return os.path.join("data", f"bulkstep{step:g}_{boundary}")


def build(cfg, step: float):
    """The one simulation: same geometry and mesh, smaller background step."""
    from paired_run import build_pair
    from simulation import build_sensor_simulation

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = build_pair(cfg)          # reuse the paired run's fine mesh
    fine = built["config"]
    sim = build_sensor_simulation(fine, realization=None, cd_bound=False,
                                  background_n=WATER_N + step)
    return fine, sim


def measure(out_dir: str, step: float):
    """Shift of the stepped run against the stored unbound reference."""
    import tidy3d as td

    from observables import hotspot_spectrum, shift_stability
    from simulation import hdf5_looks_complete

    ref_path = os.path.join(REFERENCE_DIR, "unbound.hdf5")
    hi_path = os.path.join(out_dir, "bulk_hi.hdf5")
    if not hdf5_looks_complete(ref_path):
        raise SystemExit(f"reference {ref_path} is missing or incomplete")
    if not hdf5_looks_complete(hi_path):
        return None

    wl, y_ref = hotspot_spectrum(td.SimulationData.from_file(ref_path))
    _, y_hi = hotspot_spectrum(td.SimulationData.from_file(hi_path))
    stab = shift_stability(wl, y_ref, y_hi)
    shift_pm = stab["shift_pm"]
    return {
        "step": step,
        "shift_pm": shift_pm,
        "s_bulk": (shift_pm / 1000.0) / step,
        "resolved": bool(stab.get("ok", False)),
        "verdict": stab.get("verdict", ""),
    }


def report(r, cfg):
    from redesign import MEASURED_LAYER_SERIES

    s_new = r["s_bulk"]
    print()
    print("MATCHED-STEP BULK SENSITIVITY")
    print("=" * 74)
    print(f"  step applied              {r['step']:.4e}")
    print(f"  resonance shift           {r['shift_pm']:.2f} pm"
          + ("" if r["resolved"] else "   <- NOT RESOLVED"))
    print(f"  S_bulk at this step       {s_new:.1f} nm/RIU")
    print(f"  S_bulk at {REFERENCE_STEP:.1e}          "
          f"{REFERENCE_S_BULK:.1f} nm/RIU")
    print(f"  difference                {100 * (s_new / REFERENCE_S_BULK - 1):+.1f}%")
    if not r["resolved"]:
        print()
        print("  The shift is not resolved against the frequency grid. A")
        print("  smaller step moves the line less, and this one may be too")
        print("  small to measure at the current freq_points. Raise it and")
        print("  repeat before reading anything into the number.")
        return

    print()
    print("  WHAT IT MEANS FOR THE PARTITION")
    d = 7.0
    L7 = MEASURED_LAYER_SERIES[d]["layer_shift_pm"]
    dn_applied = 1.3e-3
    for label, sb in (("5.0e-3 step", REFERENCE_S_BULK),
                      ("matched step", s_new)):
        f7 = L7 / (dn_applied * sb * 1000.0)
        print(f"    {label:<14} S_bulk {sb:6.1f}  ->  f(7 nm) = {f7:.3f}"
              f"   water share of a partition = {1 / (1 + f7):.3f}")
    print()
    if abs(s_new / REFERENCE_S_BULK - 1) < 0.05:
        print("  S_bulk changes by under 5% with step size, so the step size")
        print("  does not affect f. f is reported as a layer to water response")
        print("  ratio.")
    else:
        print("  S_bulk depends on the step size, so f should be computed with")
        print("  the matched-step value. The surface sensitivity and the signal")
        print("  are unaffected: neither divides by S_bulk.")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataclasses import replace

    from config import StudyConfig

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", type=float, default=1.3e-3,
                    help="Background index step (default 1.3e-3, matched to "
                         "the layer step).")
    ap.add_argument("--boundary", choices=["absorber", "pml"], default="pml")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()

    cfg = StudyConfig()
    cfg = replace(cfg, fdtd=replace(cfg.fdtd, boundary=args.boundary))
    out = out_dir_for(args.step, args.boundary)

    if args.analyse:
        r = measure(out, args.step)
        if r is None:
            raise SystemExit(f"nothing on disk in {out}")
        report(r, cfg)
        raise SystemExit(0)

    # Check for the output file before calling `measure`, which opens two
    # 300 MB result files and needs tidy3d. This keeps the preflight fast.
    if os.path.isfile(os.path.join(out, "bulk_hi.hdf5")):
        existing = measure(out, args.step)
        if existing is not None:
            print(f"  {out} already holds this run; nothing to submit.")
            report(existing, cfg)
            raise SystemExit(0)

    print("BULK STEP CHECK PREFLIGHT -- free")
    print(f"  boundary                  {args.boundary}")
    print(f"  background step           {args.step:.4e}"
          f"   (layer step is 1.3000e-03)")
    print(f"  reference unbound run     {REFERENCE_DIR}/unbound.hdf5"
          f"   (reused)")
    print(f"  output                    {out}")
    print(f"  simulations               1")
    print()
    print("  The standard S_bulk of 194.1 nm/RIU uses a 5.0e-3 step that moves")
    print("  the resonance 970 pm. This repeats the measurement with the step")
    print("  matched to the layer measurement.")
    if not args.submit:
        print("\n  Run again with --submit to spend.")
        raise SystemExit(0)

    from simulation import run as run_sim

    fine, sim = build(cfg, args.step)
    os.makedirs(out, exist_ok=True)
    print(f"\nRunning 1 simulation into {out}. This spends credits.")
    run_sim(sim, task_name=f"bulkstep_{args.step:g}_{args.boundary}",
            path=os.path.join(out, "bulk_hi.hdf5"))

    r = measure(out, args.step)
    with open(os.path.join(out, "result.json"), "w") as fh:
        json.dump(r, fh, indent=2)
    report(r, cfg)
