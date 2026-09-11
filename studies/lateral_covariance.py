#!/usr/bin/env python3
"""
lateral_covariance.py -- how far apart must two sensing sites be to read
independent tissue? Runs 9 simulations per axis (Tidy3D credits required).

    python studies/lateral_covariance.py                     # preflight, free
    python studies/lateral_covariance.py --submit
    python studies/lateral_covariance.py --axis both --submit
    python studies/lateral_covariance.py --max-offset 4 --submit   # cheaper
    python studies/lateral_covariance.py --analyse           # re-read results, free

The array design averages N sites and assumes their readings are independent at
a 15 um pitch. The relevant quantity is the autocovariance of the SENSOR OUTPUT,
which differs from the correlation of the refractive index field itself: the
antenna integrates the index over its mode volume with its own weighting. This
script measures that output autocovariance C(r) directly.

METHOD
------
One untapered master field is generated and windowed at a ladder of lateral
offsets, so every case sees the same medium sampled at a different position and
the series carries no seed-to-seed noise (the same construction as
`boundary_test.master_field`).

Memory is the practical limit. Lateral offsets are micrometres, so the master
field is padded only along the swept axis, and only by the maximum offset. At
the default 8 um reach this is 13 x 5 x 2.5 um, about 3.6 GiB peak. The
preflight prints the estimate and stops if it exceeds --mem-limit.

WHY 8 um IS ENOUGH
------------------
The tissue correlation length is 1.645 um and the literature outer scale is 4
to 10 um. If C(r) has fallen below 1/e by 8 um, a 15 um pitch is comfortably
justified. If not, rerun with a larger --max-offset.

ANISOTROPY
----------
The tissue model is isotropic, while real fish muscle is fibre oriented.
`--axis both` gives the direction dependence of the isotropic model and can be
reused unchanged with an anisotropic medium. Each axis builds and frees its own
master field in turn.
"""
from __future__ import annotations

import argparse
import json
import os

import _bootstrap
_bootstrap.setup()

#: Geometric ladder in microns, trimmed to --max-offset. Spans well below the
#: correlation length to comfortably beyond the literature outer scale.
LADDER_UM = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
DEFAULT_MAX_OFFSET_UM = 8.0
MEM_LIMIT_GIB = 8.0
OUT = os.path.join("data", "covariance")


def offsets_for(max_offset_um):
    return [o for o in LADDER_UM if o <= max_offset_um + 1e-9]


def master_domain(cfg, axis, max_offset_um):
    lx, ly, lz = cfg.fdtd.domain_um
    if axis == "x":
        return (lx + max_offset_um, ly, lz)
    return (lx, ly + max_offset_um, lz)


def peak_gib(domain_um, voxel_nm):
    dv = voxel_nm * 1e-3
    n = 1.0
    for L in domain_um:
        n *= max(8, round(L / dv))
    return 24.0 * n / 2 ** 30          # ~3 working arrays at 8 B/voxel


def case_path(axis, off):
    return os.path.join(OUT, f"{axis}_{off:06.2f}um.hdf5")


def wide_master(cfg, seed, axis, max_offset_um):
    """One untapered field, padded along the swept axis only.

    Untapered for the same reason `boundary_test.master_field` is: a taper is
    measured from the domain faces, so a field tapered for the wide box and
    then windowed would be untapered at its new faces. We window first and
    taper per case. Clipping is disabled so that subtracting n0 recovers the
    fluctuation exactly.
    """
    from dataclasses import replace
    from tissue import generate_tissue
    fd = replace(cfg.fdtd, domain_um=master_domain(cfg, axis, max_offset_um))
    wide = replace(cfg.tissue,
                   n_clip_min=cfg.tissue.n0 - 10.0,
                   n_clip_max=cfg.tissue.n0 + 10.0)
    real = generate_tissue(wide, fd, seed=seed, apply_taper=False)
    real.n -= cfg.tissue.n0                    # real.n is now the FLUCTUATION
    return real


def window(cfg, master, axis, off_um):
    """Cut the production-sized window whose near edge sits at `off_um`.

    The window slides along the padded axis from one end; the antenna stays at
    the origin of its own box, so the tissue moves past a fixed antenna.
    """
    import numpy as np
    from tissue import TissueRealization, _edge_taper

    dv = cfg.fdtd.tissue_voxel_nm * 1e-3
    domain = cfg.fdtd.domain_um
    want = [max(8, int(round(L / dv))) for L in domain]
    have = master.n.shape
    step = int(round(off_um / dv))

    def span(i, shifted):
        lo = step if shifted else (have[i] - want[i]) // 2
        if lo < 0 or lo + want[i] > have[i]:
            raise SystemExit(
                f"offset {off_um} um does not fit the master field along "
                f"{axis}: need {lo + want[i]} voxels, have {have[i]}")
        return slice(lo, lo + want[i])

    sl = [span(0, axis == "x"), span(1, axis == "y"), span(2, False)]
    f = np.ascontiguousarray(master.n[sl[0], sl[1], sl[2]])

    x = (np.arange(want[0]) - (want[0] - 1) / 2) * dv
    y = (np.arange(want[1]) - (want[1] - 1) / 2) * dv
    z = (np.arange(want[2]) - (want[2] - 1) / 2) * dv
    if cfg.fdtd.taper_width_um > 0:
        f *= _edge_taper(x, y, z, cfg.fdtd.taper_width_um, domain)

    raw = cfg.tissue.n0 + f
    n = np.clip(raw, cfg.tissue.n_clip_min, cfg.tissue.n_clip_max)
    diag = dict(master.diagnostics)
    diag.update({"domain_um": domain, "tapered": True,
                 "lateral_offset_um": off_um, "offset_axis": axis})
    return TissueRealization(n=n, x=x, y=y, z=z, seed=master.seed,
                             diagnostics=diag)


def measure(axis, off):
    import tidy3d as td
    from observables import fit_resonance, hotspot_spectrum
    from simulation import hdf5_looks_complete
    p = case_path(axis, off)
    if not hdf5_looks_complete(p):
        return None
    wl, y = hotspot_spectrum(td.SimulationData.from_file(p))
    r = fit_resonance(wl, y)
    return {"axis": axis, "offset_um": off,
            "resonance_nm": float(r.wavelength_um * 1000.0)}


def report(rows, pitch_um=15.0):
    import numpy as np
    print()
    print("LATERAL AUTOCOVARIANCE OF THE RESONANCE")
    print("=" * 72)
    for axis in sorted({r["axis"] for r in rows}):
        sel = sorted([r for r in rows if r["axis"] == axis],
                     key=lambda r: r["offset_um"])
        if len(sel) < 4:
            print(f"\n  axis {axis}: only {len(sel)} points, need at least 4")
            continue
        off = np.array([r["offset_um"] for r in sel])
        lam = np.array([r["resonance_nm"] for r in sel])
        d = lam - lam.mean()
        c0 = float(np.mean(d * d))
        print(f"\n  axis {axis}")
        print(f"    {'offset (um)':>12}  {'resonance (nm)':>15}  {'C(r)/C(0)':>11}")
        rho = []
        for i, o in enumerate(off):
            rr = float(d[i] * d[0] / c0) if c0 > 0 else float("nan")
            rho.append(rr)
            print(f"    {o:12.2f}  {lam[i]:15.4f}  {rr:11.3f}")
        rho = np.array(rho)
        below = off[np.abs(rho) < 1.0 / np.e]
        r_e = float(below[0]) if len(below) else float("nan")
        print(f"\n    resonance spread over the sweep  "
              f"{1000 * lam.std(ddof=1):.1f} pm")
        if np.isfinite(r_e):
            print(f"    effective correlation length     {r_e:.2f} um")
            if r_e > pitch_um:
                print(f"    WARNING: slower than the assumed {pitch_um:.0f} um")
                print("    pitch. Space the array further apart, or replace N")
                print("    with N_eff in the averaging argument.")
            else:
                print(f"    Comfortably inside the assumed {pitch_um:.0f} um")
                print("    pitch, so sites at that separation are effectively")
                print("    independent and the array argument stands.")
        else:
            print(f"    C(r) has not fallen below 1/e by {off.max():.0f} um.")
            print("    Re-run with a larger --max-offset before relying on the")
            print("    pitch.")


if __name__ == "__main__":
    from dataclasses import replace

    from simulation import require_tidy3d
    require_tidy3d()
    from config import StudyConfig
    from ensemble import seed_list

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", choices=["x", "y", "both"], default="x")
    ap.add_argument("--max-offset", type=float,
                    default=DEFAULT_MAX_OFFSET_UM,
                    help="Largest lateral offset in um (default 8). Memory "
                         "grows linearly with this; see the preflight.")
    ap.add_argument("--mem-limit", type=float, default=MEM_LIMIT_GIB,
                    help="Refuse to build a master field above this many GiB.")
    ap.add_argument("--boundary", choices=["pml", "absorber"], default="pml")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()

    cfg = StudyConfig()
    cfg = replace(cfg, fdtd=replace(cfg.fdtd, boundary=args.boundary))
    axes = ["x", "y"] if args.axis == "both" else [args.axis]
    offs = offsets_for(args.max_offset)
    cases = [(a, o) for a in axes for o in offs]
    if args.axis == "both":                    # offset 0 is the same run twice
        cases = [c for c in cases if not (c[0] == "y" and c[1] == 0.0)]

    if args.analyse:
        rows = [r for r in (measure(a, o) for a, o in cases) if r]
        if args.axis == "both":
            z = measure("x", 0.0)
            if z:
                rows.append({**z, "axis": "y"})
        if not rows:
            raise SystemExit(f"nothing on disk in {OUT}")
        report(rows)
        raise SystemExit(0)

    todo = [c for c in cases if not os.path.isfile(case_path(*c))]
    print()
    print("LATERAL COVARIANCE PREFLIGHT -- free")
    print(f"  boundary                  {cfg.fdtd.boundary}")
    print(f"  axes                      {', '.join(axes)}")
    print(f"  offsets (um)              {offs}")
    print(f"  simulations to run        {len(todo)} of {len(cases)}")
    print(f"  output                    {OUT}")
    print()
    worst = 0.0
    for a in axes:
        dom = master_domain(cfg, a, args.max_offset)
        g = peak_gib(dom, cfg.fdtd.tissue_voxel_nm)
        worst = max(worst, g)
        print(f"  master field, axis {a}      "
              f"{dom[0]:.1f} x {dom[1]:.1f} x {dom[2]:.1f} um, untapered"
              f"   ~{g:.1f} GiB peak")
    print()
    print("  The master field is padded along the swept axis only, which keeps")
    print("  memory within reach of a normal workstation.")
    if worst > args.mem_limit:
        raise SystemExit(
            f"\n  REFUSING: the master field needs about {worst:.1f} GiB, over "
            f"the {args.mem_limit:.1f} GiB limit.\n"
            f"  Lower it with --max-offset 4 (about "
            f"{peak_gib(master_domain(cfg, axes[0], 4.0), cfg.fdtd.tissue_voxel_nm):.1f}"
            f" GiB), or raise --mem-limit if you have the memory.\n")
    print()
    print("  Every offset is a window on ONE master field, so the series")
    print("  carries no seed-to-seed noise and is a direct sample of C(r).")
    if not args.submit:
        print("\n  Run again with --submit to spend.")
        raise SystemExit(0)

    from simulation import build_sensor_simulation
    from simulation import run as run_sim

    os.makedirs(OUT, exist_ok=True)
    seed = seed_list(cfg)[0]
    for axis in axes:
        pending = [o for a, o in todo if a == axis]
        if not pending:
            continue
        print(f"\n  building master field for axis {axis} ...")
        master = wide_master(cfg, seed, axis, args.max_offset)
        for off in pending:
            print(f"\n  axis {axis}, offset {off} um")
            sim = build_sensor_simulation(
                cfg, realization=window(cfg, master, axis, off),
                cd_bound=False)
            run_sim(sim, task_name=f"cov_{axis}_{off:g}um",
                    path=case_path(axis, off))
        del master                      # free before the next axis

    rows = [r for r in (measure(a, o) for a, o in cases) if r]
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    report(rows)
