#!/usr/bin/env python3
"""
box_pml.py -- does the size of the simulation domain move the resonance?
Optional study. Runs 3 simulations (Tidy3D credits required with --submit).

    python studies/box_pml.py            # preflight, free
    python studies/box_pml.py --submit   # run the simulations
    python studies/box_pml.py --analyse  # re-read results on disk, free

Changing the domain depth normally changes the grid on which the random tissue
field is generated, so the same seed would not reproduce the same tissue. To
isolate the effect of the domain alone, this script generates one untapered
master field on the deepest grid (using `boundary_test.master_field`) and crops
it symmetrically for each shallower depth. The tissue inside the interrogation
volume is then identical in every run, all runs use a PML boundary, and any
difference in resonance is the finite-domain contribution.
"""
from __future__ import annotations

import argparse
import json
import os

import _bootstrap
_bootstrap.setup()

DEPTHS_UM = [2.50, 3.76, 5.00]
OUT = os.path.join("data", "box_pml")


def case_path(depth):
    return os.path.join(OUT, f"depth{depth:05.2f}um.hdf5")


def measure():
    import tidy3d as td
    from observables import fit_resonance, hotspot_spectrum
    from simulation import hdf5_looks_complete
    out = []
    for d in DEPTHS_UM:
        p = case_path(d)
        if not hdf5_looks_complete(p):
            continue
        wl, y = hotspot_spectrum(td.SimulationData.from_file(p))
        r = fit_resonance(wl, y)
        out.append({"depth_um": d,
                    "resonance_nm": float(r.wavelength_um * 1000.0),
                    "fwhm_nm": float(r.fwhm_um * 1000.0)})
    return out


def report(rows):
    import numpy as np
    print()
    print("DOMAIN SIZE UNDER PML, TISSUE HELD BIT-IDENTICAL")
    print("=" * 72)
    print(f"  {'depth (um)':>11}  {'resonance (nm)':>15}  {'FWHM (nm)':>11}")
    for r in sorted(rows, key=lambda r: r["depth_um"]):
        print(f"  {r['depth_um']:11.2f}  {r['resonance_nm']:15.4f}  "
              f"{r['fwhm_nm']:11.3f}")
    if len(rows) < 2:
        return
    lam = np.array([r["resonance_nm"] for r in rows])
    fw = np.array([r["fwhm_nm"] for r in rows])
    swing = 1000 * (lam.max() - lam.min())
    print()
    print(f"  resonance swing across a doubling of depth   {swing:.1f} pm")
    print(f"  linewidth swing                              "
          f"{1000 * (fw.max() - fw.min()):.1f} pm")
    print()
    print("  Because the tissue in the interrogation volume is identical,")
    print("  this swing is the finite-domain contribution to the resonance.")
    print("  Compare it with the ensemble-based bound of about 18 pm.")
    print()
    if swing < 50:
        print("  A swing this small is consistent with the ensemble bound: the")
        print("  production domain depth is adequate.")
    else:
        print("  A swing this large means the domain size matters, and the noise")
        print("  floor should be recomputed with a deeper domain.")


if __name__ == "__main__":
    from dataclasses import replace

    from simulation import require_tidy3d
    require_tidy3d()
    from config import StudyConfig
    from ensemble import seed_list

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()

    cfg = StudyConfig()
    cfg = replace(cfg, fdtd=replace(cfg.fdtd, boundary="pml"))

    if args.analyse:
        rows = measure()
        if not rows:
            raise SystemExit(f"nothing on disk in {OUT}")
        report(rows)
        raise SystemExit(0)

    todo = [d for d in DEPTHS_UM if not os.path.isfile(case_path(d))]
    print()
    print("BOX-UNDER-PML PREFLIGHT -- free")
    print(f"  depths (um)               {DEPTHS_UM}")
    print(f"  boundary                  pml")
    print(f"  simulations to run        {len(todo)} of {len(DEPTHS_UM)}")
    print(f"  output                    {OUT}")
    print()
    print("  One master field, cropped symmetrically, so every depth sees")
    print("  bit-identical tissue at the hot spot and the only variable is")
    print("  the distance to the boundary.")
    if not args.submit:
        print("\n  Run again with --submit to spend.")
        raise SystemExit(0)

    from boundary_test import crop_to_depth, master_field
    from simulation import build_sensor_simulation
    from simulation import run as run_sim

    os.makedirs(OUT, exist_ok=True)
    seed = seed_list(cfg)[0]
    master = master_field(cfg, seed, max(DEPTHS_UM))
    for d in todo:
        real, achieved = crop_to_depth(master, cfg, d)
        c = replace(cfg, fdtd=replace(cfg.fdtd,
                                      domain_um=(cfg.fdtd.domain_um[0],
                                                 cfg.fdtd.domain_um[1],
                                                 achieved)))
        print(f"\n  depth {d} um (achieved {achieved:.3f})")
        run_sim(build_sensor_simulation(c, realization=real, cd_bound=False),
                task_name=f"boxpml_{d:g}um", path=case_path(d))

    rows = measure()
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    report(rows)
