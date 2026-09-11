#!/usr/bin/env python3
"""
boundary_test.py -- is the linewidth a property of the antenna or of the box?

PURPOSE
-------
The linewidth (and, through it, the resonance fit) should not depend on how
far the domain boundary is from the antenna. If it does, the boundary is
contaminating the result. Three box effects can do this:

  1. the absorbing boundary is close enough to the antenna to add parasitic
     loss, which broadens the line;
  2. the boundary is reflecting, and the return interferes with the hot-spot
     spectrum;
  3. the field rings longer in a shallow box than `run_time_ps` allows, so
     the spectrum is truncation-broadened.

Those three have different signatures, and this script separates them by
changing ONE thing at a time (domain depth, absorber thickness, or boundary
type). With a PML boundary the linewidth is flat across depth; with the
adiabatic absorber it is not, which is why all results in the paper use PML.

METHOD
------
Running independent tissue ensembles at two depths cannot isolate the boundary:
`generate_tissue` builds the random field on a grid derived from `domain_um`,
so changing the depth changes the tissue produced by the same seed, and the
comparison carries the sampling error of two independent ensembles.

Instead, the index field is generated ONCE, on the deepest grid, with no edge
taper. Each shallower case is a symmetric crop about z = 0 with the taper that
case needs. Cropping about the centre leaves the field near the antenna
bit-for-bit identical (the preflight checks this, offline, before anything is
submitted). The only thing that varies between runs is where the boundary sits
and what kind it is, so any difference in linewidth is a boundary effect, from
a handful of simulations rather than an ensemble.

READING THE RESULT
------------------
  FWHM falls as depth grows, and ALSO falls when absorber layers are added at
  fixed depth
        -> the boundary was too close and too thin.  Parasitic loss.  Use
           the converged linewidth.

  FWHM falls with depth but absorber thickness does nothing
        -> the absorber is fine; what mattered was the distance.  Either
           evanescent overlap with the boundary or a reflected return.  Look
           at whether the resonance position also moves monotonically.

  FWHM barely moves at all
        -> the box is not the cause.  Check `tail_over_peak` below first:
           if the shallow cases
           are near or above 0.01 the lines are truncation-broadened and
           `--run-time-ps` is the knob, not the boundary.

USAGE
-----
    python boundary_test.py                       # preflight only, free
    python boundary_test.py --submit              # 4 simulations
    python boundary_test.py --depths 2.5 5.0 --absorbers 20 60 --submit
    python boundary_test.py --analyse-only        # re-read what is on disk

Nothing is uploaded without --submit.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

import numpy as np

from config import StudyConfig
from tissue import generate_tissue, _edge_taper, TissueRealization
from observables import fit_resonance, hotspot_spectrum
from simulation import build_sensor_simulation, check_decay, bulk_cell_um
from ensemble import local_index_region


# ==========================================================================
#  Building the cases
# ==========================================================================
def master_field(cfg: StudyConfig, seed: int, max_depth_um: float):
    """
    One untapered realisation on the deepest grid.  Every case is a crop of
    this, so every case shares the same tissue.

    The taper is deliberately OFF here.  A taper is a window whose width is
    measured from the domain faces, so a field tapered for a 5.0 um box and
    then cropped to 2.5 um would be untapered at its new faces -- the absorber
    would see a fluctuating medium, which is the one thing the taper exists to
    prevent.  We crop first and taper per case instead.
    """
    fd = replace(cfg.fdtd,
                 domain_um=(cfg.fdtd.domain_um[0],
                            cfg.fdtd.domain_um[1],
                            float(max_depth_um)))

    # CLIPPING IS DISABLED FOR THE MASTER, AND THAT IS NOT A SHORTCUT.
    #
    # `generate_tissue` clips as its LAST step, after the taper.  We need the
    # unclipped fluctuation here so that subtracting n0 recovers it exactly;
    # otherwise a voxel that saturated in the master would carry a different
    # value into every crop and the fields would not be identical.  Each crop
    # applies the real clip bounds after its own taper, which is precisely the
    # order `generate_tissue` uses, so no case sees an out-of-range index.
    wide = replace(cfg.tissue,
                   n_clip_min=cfg.tissue.n0 - 10.0,
                   n_clip_max=cfg.tissue.n0 + 10.0)
    real = generate_tissue(wide, fd, seed=seed, apply_taper=False)

    # IN PLACE.  At 5.0 um and a 10 nm voxel this array is 125 million
    # doubles, one gigabyte.  A copy here doubles the peak for no reason.
    real.n -= cfg.tissue.n0
    return real                            # real.n is now the FLUCTUATION


def crop_to_depth(master: TissueRealization, cfg: StudyConfig,
                  depth_um: float) -> tuple[TissueRealization, float]:
    """
    Crop the master fluctuation symmetrically about z = 0, taper it for its new
    domain, and rebuild an index field.

    Returns the realisation and the depth actually achieved, which can differ
    from the request by one voxel -- see the parity note below.
    """
    dv = cfg.fdtd.tissue_voxel_nm * 1e-3
    nz_m = master.n.shape[2]
    nz_t = max(8, int(round(depth_um / dv)))

    # PARITY IS LOAD-BEARING.  The coordinate axes are centred on zero, so a
    # crop is only exactly centred when the master and target cell counts have
    # the same parity.  Off by one and the whole tissue shifts half a voxel
    # -- 5 nm -- relative to the antenna, which is most of a sensing decay
    # length.  Nudging nz_t by one cell moves the depth by 10 nm instead.
    if (nz_m - nz_t) % 2:
        nz_t += 1
    if nz_t > nz_m:
        raise SystemExit(f"depth {depth_um} um exceeds the master field")

    k0 = (nz_m - nz_t) // 2
    z = master.z[k0:k0 + nz_t]
    achieved = nz_t * dv

    domain = (cfg.fdtd.domain_um[0], cfg.fdtd.domain_um[1], achieved)
    f = master.n[:, :, k0:k0 + nz_t].copy()
    if cfg.fdtd.taper_width_um > 0:
        f *= _edge_taper(master.x, master.y, z,
                         cfg.fdtd.taper_width_um, domain)

    raw = cfg.tissue.n0 + f
    n = np.clip(raw, cfg.tissue.n_clip_min, cfg.tissue.n_clip_max)
    clipped = float((n != raw).mean())
    del f, raw

    diag = dict(master.diagnostics)
    diag.update({"domain_um": domain, "tapered": True,
                 "clipped_fraction": clipped,
                 "cropped_from_um": master.diagnostics["domain_um"]})
    return TissueRealization(n=n, x=master.x, y=master.y, z=z,
                             seed=master.seed, diagnostics=diag), achieved


def _region_patch(real: TissueRealization, region) -> np.ndarray:
    """The index values inside the interrogation volume, as an array.

    Comparing means across cases would be far too weak a test -- two different
    fields share a mean easily.  This returns the actual voxels so the
    preflight can compare them element by element.
    """
    (xl, xh), (yl, yh), (zl, zh) = region
    ix = (real.x >= xl) & (real.x <= xh)
    iy = (real.y >= yl) & (real.y <= yh)
    iz = (real.z >= zl) & (real.z <= zh)
    return real.n[np.ix_(ix, iy, iz)].copy()


def case_config(cfg: StudyConfig, depth_um: float, absorber_layers: int,
                run_time_ps: float | None,
                boundary: str | None = None) -> StudyConfig:
    """The study config for one case.  Only boundary properties change."""
    fd = replace(cfg.fdtd,
                 domain_um=(cfg.fdtd.domain_um[0],
                            cfg.fdtd.domain_um[1], float(depth_um)),
                 absorber_layers=int(absorber_layers))
    if run_time_ps is not None:
        fd = replace(fd, run_time_ps=float(run_time_ps))
    if boundary is not None:
        fd = replace(fd, boundary=boundary)
    return replace(cfg, fdtd=fd)


def boundary_of(rec: dict) -> str:
    """The boundary a stored record used.

    Records without a boundary field were produced under the config default,
    so that is what they are labelled -- never guessed from the current
    config, which may have been overridden on the command line.
    """
    return rec.get("boundary", "absorber")


def case_key(rec: dict):
    """Identity of a case, for resume.  Every knob that changes the answer."""
    return (round(rec["depth_um"], 4), int(rec["absorber_layers"]),
            boundary_of(rec), round(float(rec.get("run_time_ps", 0.0)), 6))


def absorber_geometry(cfg: StudyConfig, depth_um: float, layers: int):
    """
    How thick the absorbing layer is, and how far it starts from the antenna.

    Tidy3D appends PML and absorber layers OUTSIDE the simulation domain
    (`GridSpec._add_pml_to_bounds`), so the absorber starts at the domain edge
    and adding layers enlarges the grid without moving the absorber closer to
    the antenna.  Depth and layer count are therefore independent: depth sets
    the clearance, and layer count sets the absorber thickness and the extra
    cells it costs.

    Returns (thickness_um, clearance_um, min_safe_um), where clearance is the
    gap between the antenna plane at z = 0 and the start of the absorber, and
    min_safe is the usual half-wavelength-in-medium rule of thumb.
    """
    cell = bulk_cell_um(cfg)
    thickness = layers * cell
    clearance = depth_um / 2.0
    lam = float(np.mean(cfg.calibration.band_um))
    return thickness, clearance, lam / (2.0 * cfg.tissue.n0)


def build_cases(depths, absorbers):
    """
    Vary one thing at a time, from a common corner.

    The first depth and the first absorber count are the reference, which
    should be the production geometry.  Every other case differs from it in
    exactly one respect, so each contrast has a single cause.
    """
    d0, a0 = float(depths[0]), int(absorbers[0])
    cases = [(float(d), a0) for d in depths]
    cases += [(d0, int(a)) for a in absorbers[1:]]
    return cases


# ==========================================================================
#  Preflight -- free, and it is where the design is proved
# ==========================================================================
def preflight(cfg, master, cases, args):
    dv = cfg.fdtd.tissue_voxel_nm * 1e-3
    region = local_index_region(cfg)
    lx, ly = cfg.fdtd.domain_um[0], cfg.fdtd.domain_um[1]

    print("BOUNDARY TEST PREFLIGHT -- all of this is free")
    print(f"    tissue seed                 {master.seed}")
    print(f"    master field                {lx:g} x {ly:g} x "
          f"{master.diagnostics['domain_um'][2]:g} um at "
          f"{cfg.fdtd.tissue_voxel_nm:g} nm  "
          f"({np.prod(master.n.shape) / 1e6:.0f} M voxels)")
    print(f"    correlation length          "
          f"{cfg.tissue.l_c_um * 1000:.0f} nm")
    print(f"    run time                    {cfg.fdtd.run_time_ps:g} ps")
    print()
    print("    case  depth(um)  absorber  L/l_c   voxels(M)  payload(GB)"
          "  absorber(um)  clearance(um)")

    # ONE AT A TIME.  Holding every crop at once would cost several gigabytes
    # for no reason -- each is regenerated from the master when its turn to
    # run comes.  The master is the only large array that persists.
    achieved_all, local_idx, payload_gb, unsafe = [], [], [], []
    for i, (d, a) in enumerate(cases):
        real, achieved = crop_to_depth(master, cfg, d)
        vox = int(np.prod(real.n.shape))
        thick, clear, min_safe = absorber_geometry(cfg, achieved, a)
        mark = "" if clear >= min_safe else "   <- TOO THIN A DOMAIN"
        if clear < min_safe:
            unsafe.append((i, achieved, a, clear))
        print(f"    {i:>4}  {achieved:>9.2f}  {a:>8}  "
              f"{achieved / cfg.tissue.l_c_um:>5.2f}   "
              f"{vox / 1e6:>9.1f}  {4 * vox / 1e9:>10.2f}  "
              f"{thick:>12.3f}  {clear:>13.3f}{mark}")
        local_idx.append((real.mean_index_in_region(*region),
                          _region_patch(real, region)))
        achieved_all.append(achieved)
        payload_gb.append(4 * vox / 1e9)
        del real

    if unsafe:
        _, _, _, _ = unsafe[0]
        cell = bulk_cell_um(cfg)
        _, _, min_safe = absorber_geometry(cfg, achieved_all[0], cases[0][1])
        print()
        print("    FAIL: at least one case leaves the antenna less than "
              f"{min_safe * 1000:.0f} nm clear of")
        print("          the absorbing layer, which is half a wavelength in "
              "the medium.")
        for i, d, a, c in unsafe:
            print(f"            case {i}: depth {d:.2f} um, {a} layers "
                  f"({a * cell:.2f} um) leaves {c * 1000:+.0f} nm")
        print("          Boundary layers are appended outside the domain, so "
              "only the depth")
        print("          sets this clearance.  Raise the shallowest depth.")
        return None, False

    # The taper must not reach the interrogation volume, or the cases would
    # differ there for a reason that has nothing to do with the boundary.
    shallowest = min(achieved_all)
    core = shallowest / 2.0 - cfg.fdtd.taper_width_um
    half_span = max(abs(region[2][0]), abs(region[2][1]))
    print()
    print(f"    untapered core, shallowest   +/-{core:.3f} um")
    print(f"    interrogation volume         +/-{half_span:.3f} um")
    if core <= half_span:
        print("    FAIL: the edge taper reaches into the interrogation "
              "volume at the")
        print("          shallowest depth, so the cases cannot share a "
              "tissue there.")
        print("          Raise the shallowest depth or lower "
              "fdtd.taper_width_um.")
        return None, False

    print()
    print("    THE INVARIANCE CHECK -- this is the point of the script")
    print("    Index inside the interrogation volume, per case:")
    ref_mean, ref_patch = local_idx[0]
    identical = True
    for i, (mean_i, patch_i) in enumerate(local_idx):
        same = (patch_i.shape == ref_patch.shape
                and np.array_equal(patch_i, ref_patch))
        identical &= same
        print(f"      case {i}  mean {mean_i:.12f}  "
              f"{patch_i.size:>6} voxels  "
              f"{'identical to case 0' if same else 'DIFFERS FROM CASE 0'}")
    if identical:
        print("    pass: every case sees a bit-identical tissue at the hot "
              "spot.")
        print("          Any difference in the results is the boundary, not "
              "the tissue,")
        print("          and it carries no sampling error.")
    else:
        print("    FAIL: the crops do not agree voxel for voxel.  Something "
              "shifted the")
        print("          field relative to the antenna -- check the parity "
              "handling in")
        print("          crop_to_depth.")
        return None, False

    print()
    print(f"    simulations                 {len(cases)}")
    print(f"    total upload                ~{sum(payload_gb):.1f} GB")
    print(f"    peak memory                 "
          f"~{(master.n.nbytes + 3 * max(payload_gb) * 1e9) / 1e9:.1f} GB")
    print()
    print("    The solver prints its own FlexCredit estimate before each run.")
    print("    Watch the first one; the run resumes from whatever completed.")
    return list(zip(cases, achieved_all)), True


# ==========================================================================
#  Running
# ==========================================================================
def run_cases(cfg, master, prepared, args):
    from simulation import run as run_sim

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "boundary_results.json")
    results = []
    if args.resume and os.path.exists(results_path):
        with open(results_path) as fh:
            results = json.load(fh)
    # THE RESUME KEY MUST NAME EVERY KNOB THIS SCRIPT TURNS (depth, layer
    # count AND boundary type). Otherwise a run with a different boundary
    # would match records already on disk and skip cases it has not run.
    done = {case_key(r) for r in results}

    for i, ((d_req, a), achieved) in enumerate(prepared):
        ccfg = case_config(cfg, achieved, a, args.run_time_ps, args.boundary)
        bnd = ccfg.fdtd.boundary
        key = case_key({"depth_um": achieved, "absorber_layers": a,
                        "boundary": bnd,
                        "run_time_ps": ccfg.fdtd.run_time_ps})
        if key in done:
            print(f"  [{i + 1}/{len(prepared)}] depth {achieved:.2f} um, "
                  f"absorber {a}, {bnd}: already done, skipping")
            continue

        real, _ = crop_to_depth(master, cfg, achieved)
        sim = build_sensor_simulation(ccfg, realization=real,
                                      light_monitors=True)
        raw = os.path.join(args.out,
                           f"depth{achieved:.2f}_abs{a}_{bnd}.hdf5")
        print(f"  [{i + 1}/{len(prepared)}] depth {achieved:.2f} um, "
              f"absorber {a}, {bnd} ...", flush=True)

        rec = {"depth_um": achieved, "absorber_layers": a,
               "boundary": ccfg.fdtd.boundary,
               "run_time_ps": ccfg.fdtd.run_time_ps, "seed": real.seed,
               "mean_index_local": real.mean_index_in_region(
                   *local_index_region(cfg)),
               "mean_index_box": real.mean_index}
        try:
            data = run_sim(
                sim, task_name=f"boundary_d{achieved:.2f}_a{a}_{bnd}",
                path=raw, verbose=False)
            wl, intensity = hotspot_spectrum(data)
            fit = fit_resonance(wl, intensity)
            decay = check_decay(data)
            rec.update({
                "resonance_nm": 1000 * fit.wavelength_um,
                "fwhm_nm": 1000 * fit.fwhm_um,
                "fit_ok": bool(np.isfinite(fit.wavelength_um)
                               and np.isfinite(fit.fwhm_um)
                               and fit.fwhm_um > 0 and fit.converged),
                "tail_over_peak": decay.get("tail_over_peak", float("nan")),
                "decay_ok": bool(decay.get("ok", False)),
                "note": fit.warning or "",
            })
            del data
        except Exception as exc:                      # keep completed cases
            rec.update({"resonance_nm": float("nan"), "fwhm_nm": float("nan"),
                        "fit_ok": False, "decay_ok": False,
                        "tail_over_peak": float("nan"), "note": str(exc)})

        results.append(rec)
        with open(results_path, "w") as fh:           # save after EVERY case
            json.dump(results, fh, indent=2)

        if not args.keep_raw:
            try:
                os.remove(raw)
            except OSError:
                pass
        # MEMORY.  The realisation and the Simulation each hold a copy of the
        # index field.  Drop both before the next crop is built, or the peak
        # climbs case by case until the process is killed.
        del sim, real

    return results


# ==========================================================================
#  Reading the answer
# ==========================================================================
def report(results, cfg):
    good = [r for r in results if r.get("fit_ok")]
    if len(good) < 2:
        print("\nFewer than two usable cases; nothing to compare.")
        return

    good.sort(key=lambda r: (boundary_of(r), r["absorber_layers"],
                             r["depth_um"]))
    print("\nBOUNDARY TEST RESULT")
    print("=" * 74)
    print("  depth(um)  absorber  bnd  resonance(nm)   FWHM(nm)      Q   "
          "tail/peak")
    for r in good:
        q = r["resonance_nm"] / r["fwhm_nm"]
        flag = "" if r["decay_ok"] else "  <- STILL RINGING"
        print(f"  {r['depth_um']:>9.2f}  {r['absorber_layers']:>8}  "
              f"{boundary_of(r)[:3]:>3}  "
              f"{r['resonance_nm']:>13.3f}  {r['fwhm_nm']:>9.3f}  "
              f"{q:>5.2f}  {r['tail_over_peak']:>9.4f}{flag}")

    ref = good[0]
    a0 = ref["absorber_layers"]
    b0 = boundary_of(ref)

    # THE ARM MUST BE ONE BOUNDARY.  Mixing an absorber depth series with a
    # PML one and calling the result a trend in depth would manufacture an
    # oscillation out of two perfectly clean curves.
    depth_arm = sorted((r for r in good
                        if r["absorber_layers"] == a0
                        and boundary_of(r) == b0),
                       key=lambda r: r["depth_um"])

    # Head to head, where the same depth was run under both boundaries.
    # Restricted to the reference layer count: keying on depth alone would let
    # a different-thickness row overwrite the one being compared, and quietly
    # put two unlike cases side by side.
    by_b = {}
    for r in good:
        if r["absorber_layers"] != a0:
            continue
        by_b.setdefault(boundary_of(r), {})[round(r["depth_um"], 2)] = r
    if len(by_b) > 1:
        kinds = sorted(by_b)
        shared = sorted(set.intersection(*(set(v) for v in by_b.values())))
        if shared:
            print(f"\n  HEAD TO HEAD, {' vs '.join(kinds)}, same tissue")
            print("    depth(um)   " + "".join(f"{k+' FWHM':>16}"
                                               for k in kinds) + "   ratio")
            for d in shared:
                fs = [by_b[k][d]["fwhm_nm"] for k in kinds]
                print(f"    {d:>9.2f}   "
                      + "".join(f"{v:>16.3f}" for v in fs)
                      + f"   {max(fs) / min(fs):>5.2f}x")
            swings = {k: (max(x["fwhm_nm"] for x in by_b[k].values())
                          - min(x["fwhm_nm"] for x in by_b[k].values()))
                      for k in kinds if len(by_b[k]) > 1}
            if len(swings) > 1:
                print("    FWHM swing across depths:  "
                      + ",  ".join(f"{k} {v:.1f} nm"
                                   for k, v in swings.items()))
                best = min(swings, key=swings.get)
                worst = max(swings, key=swings.get)
                if swings[worst] > 3 * max(swings[best], 1e-9):
                    print(f"    The swing collapses under '{best}'.  The "
                          f"'{worst}' boundary was")
                    print("    the cause, not the domain size.  Re-run the "
                          "ensemble under")
                    print(f"    '{best}' and quote the linewidth it "
                          f"converges to.")
                else:
                    print("    Both boundaries swing comparably, so the "
                          "boundary KIND is not")
                    print("    the cause.  Look at the source and monitor "
                          "geometry next.")

    print("\n  FOR REFERENCE, ABSORBER-BOUNDARY ENSEMBLES")
    print("    ensemble at depth 2.50 um:           FWHM 150.168 nm, Q 5.64")
    print("    ensemble at depth 5.00 um:           FWHM 117.174 nm, Q 7.21")
    print("    difference:                          32.994 nm")

    print("\n  VERDICT")
    if [r for r in good if not r["decay_ok"]]:
        print("    Some cases were still ringing when the run ended, so their")
        print("    linewidths are truncation-broadened and nothing below is")
        print("    clean.  Raise --run-time-ps and repeat.")
        return
    if len(depth_arm) < 3:
        print(f"    Fewer than three depths on the '{b0}' boundary at "
              f"{a0} layers;")
        print("    a trend cannot be distinguished from an oscillation.  "
              "Add depths.")
        return
    print(f"    boundary                 {b0}, {a0} layers")

    f = [r["fwhm_nm"] for r in depth_arm]
    lam = [r["resonance_nm"] for r in depth_arm]
    d = [r["depth_um"] for r in depth_arm]
    swing_f, swing_l = max(f) - min(f), max(lam) - min(lam)
    rising = all(b > a for a, b in zip(f, f[1:]))
    falling = all(b < a for a, b in zip(f, f[1:]))

    print(f"    FWHM across depths       "
          f"{min(f):.1f} to {max(f):.1f} nm, swing {swing_f:.1f} nm "
          f"({max(f) / min(f):.2f}x)")
    print(f"    resonance across depths  "
          f"{min(lam):.1f} to {max(lam):.1f} nm, swing "
          f"{1000 * swing_l:.0f} pm")
    print(f"    monotonic in depth?      "
          f"{'yes' if (rising or falling) else 'NO'}")
    print()

    if rising or falling:
        print("    The linewidth moves monotonically with distance to the")
        print("    boundary.  That is the signature of parasitic loss: the")
        print("    antenna is close enough to the absorber to leak into it.")
        print("    Push the depth out until the linewidth stops changing, and")
        print("    quote the converged value.")
    else:
        worst = d[int(np.argmax(f))]
        print("    The linewidth is NOT monotonic in depth -- it is worst at")
        print(f"    {worst:.2f} um, between two better cases.  Parasitic loss")
        print("    cannot do that, because it falls off with distance.  What")
        print("    does is a REFLECTING boundary: the return interferes with")
        print("    the antenna spectrum, and the phase depends on the")
        print("    round-trip path, so the damage rises and falls with depth.")
        print()
        print("    Consequences:")
        print("      * No depth in this series has a converged linewidth. Do")
        print("        not quote any of them, including the narrowest, as a")
        print("        property of the antenna.")
        print("      * The absorber is the suspect, not the domain size. Try")
        print("        --boundary pml, which the index taper already makes")
        print("        safe, and check the swing collapses.")

    # How much of this leaks into the seed-to-seed scatter?  Within one
    # ensemble the depth is fixed, so the interference phase only moves
    # through the realisation's own mean index -- a far smaller lever than
    # changing the box.  Bounding it says whether the boundary affects
    # sigma_tissue.
    n_box_spread = 1.46e-4                     # measured spread of mean index
    dmin, dmax = min(d), max(d)
    n0 = 1.35
    path_range = 2 * (dmax - dmin) * n0 * 1000            # nm
    path_per_seed = 2 * dmin * n_box_spread * 1000        # nm
    if path_range > 0:
        leak = 1000 * swing_l * path_per_seed / path_range   # pm
        print()
        print("  DOES THIS CONTAMINATE THE NOISE FLOOR?")
        print(f"    resonance swing over the depth series   "
              f"{1000 * swing_l:>8.0f} pm")
        print(f"    optical path change that produced it    "
              f"{path_range:>8.0f} nm")
        print(f"    path change from seed to seed at fixed depth "
              f"{path_per_seed:>3.2f} nm")
        print(f"    implied boundary contribution to sigma  "
              f"{leak:>8.2f} pm")
        print("    against sigma_tissue (absorber ensemble)  533.30 pm")
        if leak < 53.3:
            print("    The boundary systematic is common mode within an")
            print("    ensemble: it moves every realisation together and adds")
            print("    negligibly to the scatter between them.  The noise")
            print("    floor survives this; the linewidth does not.")
        else:
            print("    This is a material fraction of the noise floor. Fix the")
            print("    boundary before using sigma_tissue.")


# ==========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Separate the boundary artefact from tissue statistics.")
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[2.5, 3.75, 5.0],
                    help="Domain depths in um.  The FIRST is the reference "
                         "and should be the production depth (default "
                         "2.5 3.75 5.0).")
    ap.add_argument("--absorbers", type=int, nargs="+", default=[20],
                    help="Absorber layer counts.  The FIRST is the reference; "
                         "the rest are run at the reference depth only.  "
                         "Boundary layers are appended outside the domain, "
                         "so more layers enlarge the grid but do not move "
                         "the absorber toward the antenna.  To compare "
                         "boundary kinds, use --boundary pml (default 20).")
    ap.add_argument("--boundary", choices=["absorber", "pml"], default=None,
                    help="Override fdtd.boundary for every case.  'pml' is "
                         "the sharp tool and is safe here because the index "
                         "taper already presents a uniform medium to the "
                         "boundary; use it to test whether a reflecting "
                         "absorber is what moves the linewidth.")
    ap.add_argument("--seed", type=int, default=2104521678,
                    help="Tissue seed.  The default is the first seed of the "
                         "ensembles, so the results link to them.")
    ap.add_argument("--run-time-ps", type=float, default=None,
                    help="Override fdtd.run_time_ps for every case.  Use this "
                         "if the preflight or the results show ringing.")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="Skip the correlation-length calibration.  Do not "
                         "use this; the starting guess in config.py is "
                         "several times too small.")
    ap.add_argument("--submit", action="store_true",
                    help="Actually upload and spend.  Without this nothing "
                         "leaves the machine.")
    ap.add_argument("--analyse-only", action="store_true",
                    help="Re-read boundary_results.json and reprint.")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    default=True,
                    help="Re-run cases already recorded in "
                         "boundary_results.json instead of skipping them.")
    ap.add_argument("--discard-raw", dest="keep_raw", action="store_false",
                    default=True,
                    help="Delete each .hdf5 after fitting.  The default keeps "
                         "them (0.25-0.5 GB each) so the lineshape can be "
                         "inspected later.")
    ap.add_argument("--out", default="data/boundary_test")
    args = ap.parse_args()

    if args.submit:
        from simulation import require_tidy3d
        require_tidy3d()

    cfg = StudyConfig()

    if args.analyse_only:
        p = os.path.join(args.out, "boundary_results.json")
        if not os.path.exists(p):
            raise SystemExit(f"nothing at {p}")
        with open(p) as fh:
            report(json.load(fh), cfg)
        raise SystemExit(0)

    # CALIBRATE FIRST, for the same reason ensemble.py does: every L/l_c the
    # preflight prints is wrong in the optimistic direction otherwise.
    if not args.no_calibrate:
        import contextlib as _ctx, io as _io, warnings as _warn
        from optical_properties import calibrate

        with _warn.catch_warnings():
            _warn.simplefilter("ignore")
            with _ctx.redirect_stdout(_io.StringIO()):
                fit = calibrate(cfg.tissue, cfg.calibration)
        start = cfg.tissue.l_c_um
        cfg = replace(cfg, tissue=fit.as_tissue_config(cfg.tissue))
        print(f"  correlation length: starting guess {start * 1000:.0f} nm, "
              f"calibrated {cfg.tissue.l_c_um * 1000:.0f} nm")
        print()

    cases = build_cases(args.depths, args.absorbers)
    max_depth = max(args.depths)

    print(f"Generating the master tissue field at {max_depth:g} um depth.  "
          f"This is the slow part and it happens once.")
    master = master_field(cfg, args.seed, max_depth)
    print()

    prepared, ok = preflight(cfg, master, cases, args)
    if not ok:
        raise SystemExit("Preflight failed; refusing to spend.")
    if not args.submit:
        print("\nRun again with --submit to spend.")
        raise SystemExit(0)

    print(f"\nRunning {len(prepared)} simulations into {args.out}. "
          f"This spends credits.")
    results = run_cases(cfg, master, prepared, args)
    report(results, cfg)
