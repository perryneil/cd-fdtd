#!/usr/bin/env python3
"""
contact_gap.py -- effect of an imperfect contact between sensor and tissue.
Optional study. Runs 2 simulations per gap per seed (Tidy3D credits required).

    python studies/contact_gap.py                     # preflight, free
    python studies/contact_gap.py --submit            # 2 sims per gap per seed
    python studies/contact_gap.py --seeds 4 --submit  # adds the variability trend
    python studies/contact_gap.py --analyse           # re-read results, free

The main study assumes tissue touches the 7 nm coating directly, while the
sensing decay length is only 12 to 21 nm. A real cut fish surface has
roughness and extracellular fluid, so there may be a thin fluid layer between
the chip and the tissue. That layer pushes tissue out of the sensing volume and
reduces both the binding signal and the microstructure variability. The ratio
of the two decides whether imperfect contact is harmless.

METHOD
------
The metal film is centred on z = 0 with slab bounds (-t/2, +t/2) and the
coating extends a further 7 nm, so the outer face of the stack sits at
z = t/2 + 7 nm = 27 nm. A uniform fluid standoff of g nanometres is imposed by
replacing the random medium with extracellular fluid everywhere within
|z| < 27 nm + g, leaving structured tissue only beyond it. That is the
physical picture of a rigid chip pressed onto a wet surface with fluid pooled
around it, and it needs no new geometry code.

The binding shift is measured unbound against bound at each standoff, and the
resonance itself is recorded so that with several seeds the variability trend
comes out alongside the signal trend.
"""
from __future__ import annotations

import argparse
import json
import os

import _bootstrap
_bootstrap.setup()

GAPS_NM = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]
FLUID_N = 1.34
COATING_NM = 7.0
OUT = os.path.join("data", "contact_gap")


def case_path(gap, seed, state):
    return os.path.join(OUT, f"gap{gap:05.1f}_seed{seed}_{state}.hdf5")


def stack_outer_face_um(cfg) -> float:
    """Outer face of the coated metal, in microns, measured from z = 0."""
    return (cfg.bowtie.thickness_nm / 2.0 + COATING_NM) * 1e-3


def with_standoff(cfg, real, gap_nm: float):
    """Replace the random medium with fluid within the standoff slab."""
    import numpy as np
    from tissue import TissueRealization
    if gap_nm <= 0:
        return real
    half = stack_outer_face_um(cfg) + gap_nm * 1e-3
    n = real.n.copy()
    inside = np.abs(real.z) < half
    n[:, :, inside] = FLUID_N
    diag = dict(real.diagnostics)
    diag.update({"standoff_nm": gap_nm, "standoff_index": FLUID_N})
    return TissueRealization(n=n, x=real.x, y=real.y, z=real.z,
                             seed=real.seed, diagnostics=diag)


def build(cfg, gap_nm, seed, bound):
    from simulation import build_sensor_simulation
    from tissue import generate_tissue
    real = with_standoff(cfg, generate_tissue(cfg.tissue, cfg.fdtd, seed=seed),
                         gap_nm)
    return build_sensor_simulation(cfg, realization=real, cd_bound=bound)


def measure(gap, seed):
    import tidy3d as td
    from observables import fit_resonance, hotspot_spectrum, shift_stability
    from simulation import hdf5_looks_complete
    lo, hi = case_path(gap, seed, "unbound"), case_path(gap, seed, "bound")
    if not (hdf5_looks_complete(lo) and hdf5_looks_complete(hi)):
        return None
    wl, y0 = hotspot_spectrum(td.SimulationData.from_file(lo))
    _, y1 = hotspot_spectrum(td.SimulationData.from_file(hi))
    st = shift_stability(wl, y0, y1)
    fit = fit_resonance(wl, y0)
    return {"gap_nm": gap, "seed": seed,
            "shift_pm": float(st["shift_pm"]),
            "resonance_nm": float(fit.wavelength_um * 1000.0)}


def report(rows):
    import numpy as np
    from redesign import MEASURED
    sig0 = MEASURED["speckle_pm"]
    print()
    print("CONTACT GAP SWEEP")
    print("=" * 74)
    print(f"  {'gap (nm)':>9}  {'binding shift (pm)':>19}  "
          f"{'resonance spread (pm)':>22}  {'seeds':>6}")
    base_sig = base_var = None
    for gap in sorted({r["gap_nm"] for r in rows}):
        sel = [r for r in rows if r["gap_nm"] == gap]
        sh = np.array([r["shift_pm"] for r in sel])
        lam = np.array([r["resonance_nm"] for r in sel])
        spread = 1000 * lam.std(ddof=1) if len(lam) > 1 else float("nan")
        if base_sig is None:
            base_sig, base_var = sh.mean(), spread
        print(f"  {gap:9.1f}  {sh.mean():19.2f}  {spread:22.1f}  {len(sel):6d}")
    print()
    print("  The ratio of signal to variability decides whether imperfect")
    print("  contact is harmless. Fluid pushes tissue out of the sensing volume,")
    print("  so both fall. If the variability falls faster, poor contact is")
    print("  harmless. If the signal falls faster, contact quality becomes a")
    print("  requirement on the instrument.")
    print()
    print(f"  Ideal-contact floor for comparison: {sig0:.0f} pm.")
    if base_var is None or not np.isfinite(base_var):
        print("  With one seed the spread column is undefined. Rerun with")
        print("  --seeds 4 or more to get the variability trend that matters.")


if __name__ == "__main__":
    from dataclasses import replace

    from simulation import require_tidy3d
    require_tidy3d()
    from config import StudyConfig
    from ensemble import seed_list

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--boundary", choices=["pml", "absorber"], default="pml")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()

    cfg = StudyConfig()
    cfg = replace(cfg, fdtd=replace(cfg.fdtd, boundary=args.boundary))
    seeds = seed_list(cfg)[:args.seeds]

    if args.analyse:
        rows = [r for r in (measure(g, s) for g in GAPS_NM for s in seeds) if r]
        if not rows:
            raise SystemExit(f"nothing on disk in {OUT}")
        report(rows)
        raise SystemExit(0)

    todo = [(g, s, st) for g in GAPS_NM for s in seeds
            for st in ("unbound", "bound")
            if not os.path.isfile(case_path(g, s, st))]
    print()
    print("CONTACT GAP PREFLIGHT -- free")
    print(f"  boundary                  {cfg.fdtd.boundary}")
    print(f"  gaps (nm)                 {GAPS_NM}")
    print(f"  fluid index               {FLUID_N}")
    print(f"  stack outer face          {1000 * stack_outer_face_um(cfg):.1f} nm from z = 0")
    print(f"  seeds                     {len(seeds)}")
    print(f"  simulations to run        {len(todo)}")
    print(f"  output                    {OUT}")
    if not args.submit:
        print("\n  Run again with --submit to spend.")
        raise SystemExit(0)

    from simulation import run as run_sim
    os.makedirs(OUT, exist_ok=True)
    for gap, seed, state in todo:
        print(f"\n  gap {gap} nm, seed {seed}, {state}")
        run_sim(build(cfg, gap, seed, state == "bound"),
                task_name=f"gap_{gap:g}_{seed}_{state}",
                path=case_path(gap, seed, state))

    rows = [r for r in (measure(g, s) for g in GAPS_NM for s in seeds) if r]
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    report(rows)
