"""
convergence.py — proving the answer is physics, not an artefact of the grid.
============================================================================

WHY THIS MODULE IS NOT OPTIONAL
-------------------------------
Every number this project produces comes out of a computer that had to chop
space into cells and pick a box size.  Neither of those is part of the physics.
If the answer changes when you change them, the answer is about your computer
settings, not about fish.

Three things must be shown to be stable before any result can be quoted:

  1. THE TISSUE VOXEL SIZE.  The random medium is generated on a grid.  In the
     mass-fractal regime that describes real tissue, finer grids capture more
     of the index variance -- unless an explicit physical inner cutoff stops
     them.  This is the single most likely source of a spurious result in the
     whole project, and `tissue_voxel_study` is the check.

  2. THE FDTD CELL SIZE IN THE GAP.  Plasmonic hot spots concentrate the field
     into a few nanometres.  Too coarse a mesh and the peak field, the
     resonance position and the linewidth are all wrong.

  3. THE SUPERCELL SIZE.  A finite box has its own average index, which drifts
     from realisation to realisation and inflates the apparent spread.  The
     speckle residual (see `ensemble.py`) should be stable as the box grows
     even though the raw spread shrinks.

THE FIRST STUDY IS FREE
-----------------------
`tissue_voxel_study` requires no FDTD at all -- it is pure random-field
generation and arithmetic.  Run it first, always.  It costs seconds and it
catches the most dangerous error.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from config import StudyConfig
from tissue import generate_tissue, band_limited_variance


# ==========================================================================
#  1. Tissue voxel size  (free -- no FDTD)
# ==========================================================================
@dataclass
class VoxelStudyRow:
    voxel_nm: float
    realised_variance: float
    capture_fraction: float
    mean_index: float
    clipped_fraction: float
    lost_low_k: float = 0.0
    """Fraction of the nominal variance that falls BELOW the lowest spatial
    frequency the box can hold, i.e. blobs bigger than the simulation box.
    A property of the BOX, not the mesh -- refining the voxel cannot recover
    it, so it must not be read as a convergence failure."""

    lost_high_k: float = 0.0
    """Fraction that falls ABOVE the voxel Nyquist frequency, i.e. structure
    finer than the mesh can represent.  A property of the MESH.  THIS is the
    one that should go to zero as the voxel shrinks below `l_min`."""


def tissue_voxel_study(
    cfg: StudyConfig,
    voxels_nm: tuple[float, ...] = (20, 10, 5, 2.5),
    seed: int = 1,
    domain_um: tuple[float, float, float] | None = None,
    max_voxels: int = 30_000_000,
) -> list[VoxelStudyRow]:
    """
    Does the tissue's index contrast depend on how finely we voxelise it?

    WHAT YOU ARE LOOKING FOR
    ------------------------
    The realised variance should RISE as the voxel shrinks, and then LEVEL OFF
    once the voxel is comfortably smaller than the physical inner cutoff
    `l_min_um`.  The plateau is the physical answer.

    If it never levels off, the inner cutoff is not doing its job -- either it
    is set smaller than your finest voxel (so the grid is still the cutoff), or
    it was left out.  In that case every downstream number, including the noise
    floor, is a function of your mesh and cannot be quoted.

    Run this before anything else.  It takes seconds and costs nothing.
    """
    # A deliberately small box.  The point of this study is to vary the
    # VOXEL while holding the box fixed, so the low-frequency end of the
    # represented band does not move.  A small box keeps the finest voxel
    # affordable; the absolute variance values are therefore lower than a
    # production run would give, but the CONVERGENCE BEHAVIOUR - which is
    # what this study is about - is the same.
    domain = domain_um or (0.6, 0.6, 0.6)
    rows = []
    for v in voxels_nm:
        n_vox = int(np.prod([max(8, round(L / (v * 1e-3))) for L in domain]))
        if n_vox > max_voxels:
            print(f"  (skipping {v} nm: {n_vox:,} voxels exceeds the "
                  f"{max_voxels:,} memory guard)")
            continue
        real = generate_tissue(
            cfg.tissue, cfg.fdtd, seed=seed,
            domain_um=domain, voxel_nm=v, apply_taper=False,
        )
        d = real.diagnostics
        t = cfg.tissue

        # Split the missing variance into its two quite different causes.
        k_box = 2 * np.pi / max(domain)
        k_nyq = np.pi / (v * 1e-3)
        low_k = band_limited_variance(
            t.delta_n_sq, t.l_c_um, t.m, t.l_min_um, 1e-9, k_box
        ) / t.delta_n_sq
        high_k = band_limited_variance(
            t.delta_n_sq, t.l_c_um, t.m, t.l_min_um, k_nyq, 1e9
        ) / t.delta_n_sq

        rows.append(VoxelStudyRow(
            voxel_nm=v,
            realised_variance=d["realised_variance"],
            capture_fraction=d["variance_capture_fraction"],
            mean_index=real.mean_index,
            clipped_fraction=d["clipped_fraction"],
            lost_low_k=float(low_k),
            lost_high_k=float(high_k),
        ))
    return rows


def print_voxel_study(rows: list[VoxelStudyRow], l_min_nm: float) -> None:
    """Print the table, and say plainly whether it converged.

    THE DISTINCTION THIS TABLE EXISTS TO MAKE
    -----------------------------------------
    Variance goes missing for two completely different reasons, and only one of
    them is a convergence problem:

      LOST LOW-k   blobs larger than the simulation box.  A property of the
                   BOX.  Refining the mesh cannot recover it, and it does not
                   invalidate anything -- it just means your box is smaller
                   than the tissue's outer scale.  Fix it by enlarging the box
                   if you care, or state it.

      LOST HIGH-k  structure finer than the voxel.  A property of the MESH.
                   THIS is the convergence question.  It must fall towards
                   zero as the voxel drops below `l_min`, and the verdict below
                   is based on it alone.

    Judging convergence on the total capture fraction would conflate the two
    and could report "not converged" for a study that has converged perfectly
    well on a box that is simply small.
    """
    print(f"  {'voxel':>8}  {'realised (dn)^2':>16}  {'RMS dn':>9}  "
          f"{'captured':>9}  {'lost low-k':>11}  {'lost high-k':>12}")
    for r in rows:
        print(f"  {r.voxel_nm:6.1f}nm  {r.realised_variance:16.4e}  "
              f"{r.realised_variance ** 0.5:9.5f}  "
              f"{100 * r.capture_fraction:8.1f}%  "
              f"{100 * r.lost_low_k:10.1f}%  {100 * r.lost_high_k:11.1f}%")

    if len(rows) < 3:
        return

    print()
    print(f"  physical inner cutoff l_min = {l_min_nm:.1f} nm")

    # The box term is the same for every row; report it once, and make clear
    # it is not a mesh issue.
    box_loss = rows[0].lost_low_k
    if box_loss > 0.02:
        print(f"  {100 * box_loss:.1f}% of the variance lives in blobs LARGER "
              f"than the box.")
        print("  That is a box-size statement, not a mesh one, and it is the")
        print("  same in every row.  It caps 'captured' below 100% however")
        print("  fine the voxel gets.  Enlarge the box if you want it, but it")
        print("  does not affect convergence.")
        print()

    # Convergence is judged on the MESH term alone.
    finest = rows[-1]
    mesh_loss = finest.lost_high_k
    last = [r.realised_variance for r in rows[-3:]]
    spread = (max(last) - min(last)) / max(np.mean(last), 1e-30)

    if mesh_loss < 0.02 and spread < 0.05:
        print(f"  CONVERGED - at the finest voxel only "
              f"{100 * mesh_loss:.2f}% of the variance lies above Nyquist,")
        print(f"  and the finest three voxels agree to {100 * spread:.1f}%.")
        print("  The index contrast is set by the physical cutoff, not the")
        print("  mesh, so downstream results are safe to quote.")
    else:
        print(f"  NOT CONVERGED - {100 * mesh_loss:.1f}% of the variance is "
              f"still above the Nyquist")
        print(f"  frequency at the finest voxel, and the finest three differ "
              f"by {100 * spread:.0f}%.")
        print("  The mesh is still acting as the cutoff.  Two ways out:")
        print(f"   (a) refine until the voxel is well below l_min "
              f"({l_min_nm:.1f} nm) -- expensive,")
        print("       since cost scales as the cube of the refinement; or")
        print("   (b) raise l_min_um to a larger, defensible physical scale.")
        print("       Below the myofilament lattice (tens of nm) it is")
        print("       arguable that muscle has no optically relevant index")
        print("       STRUCTURE at all, only composition -- which makes a")
        print("       cutoff of 20-40 nm defensible and affordable.")
        print("  Until this converges, the noise floor is a mesh setting and")
        print("  must not be quoted in mg/kg.")


# ==========================================================================
#  2. FDTD cell size in the gap
# ==========================================================================
def gap_mesh_study(
    cfg: StudyConfig,
    dl_nm: tuple[float, ...] | None = None,
    submit: bool = False,
    out_dir: str = "data/convergence",
    scale_metal: bool = True,
    domain_um=None,
) -> list[dict]:
    """
    Does the resonance move as the gap mesh is refined?

    A MESHING TRAP
    --------------
    Asking for a 4 nm cell in the gap does NOT necessarily give you one.  Two
    other things also set the local cell size: the separate mesh override on
    the metal (`dl_metal_nm`, 2 nm by default) and Tidy3D's automatic grid,
    which refines near high-index material on its own.  Whichever is FINEST
    wins.

    So a naive series of 4, 2, 1, 0.5 nm silently delivers 1.75, 1.75, 0.93,
    0.48 nm.  The first two rows are the same simulation with different labels,
    and plotting them as a convergence series is misleading, because two
    identical points look like perfect convergence.

    Two safeguards are built in:

      * `scale_metal=True` (default) scales `dl_metal_nm` along with the gap
        cell, so the whole mesh actually refines together and the series is a
        real refinement series.

      * every row reports the ACHIEVED minimum cell alongside the requested
        one, and `print_gap_mesh_study` flags any row where they disagree by
        more than 20%.

    OTHER NOTES
    -----------
    Run this in WATER, not tissue.  You are testing the mesh, and a random
    medium adds realisation-to-realisation scatter that would mask the trend.

    The peak field will keep creeping up if your tip radius is too small --
    that is the field singularity warned about in `sensor.py`, and it is a
    geometry problem, not a mesh problem.

    Offline (`submit=False`) this returns cell counts and achieved resolutions
    so you can see what each refinement would cost before committing.

    THE BOX
    -------
    `domain_um` (default `cfg.fdtd.mesh_study_domain_um`) lets this series run
    in a smaller box than production.  It is a water reference: it needs room
    for the antenna, its near field and the absorber, but NOT for the many
    independent correlation volumes the tissue ensemble needs.  Every row uses
    the same box, so the mesh-to-mesh differences this study exists to measure
    are unaffected — and the box is printed, so it is never a silent change.

    THE COST, WHICH IS THE REASON `dl_nm` DEFAULTS TO THREE POINTS
    --------------------------------------------------------------
    Halving the cell multiplies the work by roughly eight: more cells, and
    more time steps, since the Courant condition ties the step to the smallest
    cell.  Measured here, a 0.5 nm run is about 20x a 2 nm run -- on its own
    more than the other three points combined.  Run 4/2/1 first; add 0.5 to
    `cfg.fdtd.mesh_study_dl_nm` only if the peak has not stopped moving.
    """
    from dataclasses import replace as _replace
    from simulation import (
        build_sensor_simulation, report, run, check_decay
    )
    from observables import hotspot_spectrum, fit_resonance

    if dl_nm is None:
        dl_nm = tuple(cfg.fdtd.mesh_study_dl_nm)
    if domain_um is None:
        domain_um = cfg.fdtd.mesh_study_domain_um
    box = tuple(domain_um) if domain_um is not None else tuple(cfg.fdtd.domain_um)

    rows = []
    for dl in dl_nm:
        fdtd_kwargs = {"dl_gap_nm": dl, "domain_um": box}
        if scale_metal:
            # Keep the metal mesh a fixed factor coarser than the gap mesh, so
            # refining the gap refines the whole neighbourhood rather than
            # being overridden by a finer metal setting.
            fdtd_kwargs["dl_metal_nm"] = dl * 2.0
        c = _replace(cfg, fdtd=_replace(cfg.fdtd, **fdtd_kwargs))
        sim = build_sensor_simulation(c, realization=None)   # water reference
        rep_ = report(sim)
        row = {
            "dl_requested_nm": dl,
            "dl_metal_nm": c.fdtd.dl_metal_nm,
            "dl_achieved_nm": rep_.min_step_nm,
            "binding": abs(rep_.min_step_nm - dl) / dl < 0.20,
            "cells": rep_.n_cells,
            "time_steps": int(sim.num_time_steps),
            "domain_um": box,
            "resonance_nm": float("nan"),
            "fwhm_nm": float("nan"),
        }
        if submit:
            import json as _json
            import os as _os

            _os.makedirs(out_dir, exist_ok=True)
            path = f"{out_dir}/mesh_{dl}nm.hdf5"
            if _os.path.isfile(path):
                # RESUME: this row is already on disk.  Load it instead of
                # running it again -- the finest row alone is most of the
                # series cost.
                import tidy3d as _td
                data = _td.SimulationData.from_file(path)
            else:
                data = run(sim, task_name=f"mesh_{dl}nm", path=path)
            wl, inten = hotspot_spectrum(data)
            fit = fit_resonance(wl, inten)
            row["resonance_nm"] = fit.wavelength_um * 1000
            row["fwhm_nm"] = fit.fwhm_um * 1000
            row["decay"] = check_decay(data)
            # Save after EVERY row, so a crash mid-series loses no completed
            # results.
            rows_so_far = rows + [row]
            with open(f"{out_dir}/mesh_results.json", "w") as fh:
                _json.dump(
                    [{k: v for k, v in r.items() if k != "decay"}
                     for r in rows_so_far], fh, indent=2, default=str)
        rows.append(row)
    return rows


def print_gap_mesh_study(rows: list[dict]) -> None:
    """Print the mesh series, flagging any row whose override did not bind."""
    if rows and rows[0].get("domain_um"):
        b = rows[0]["domain_um"]
        print(f"  Box for this series: {b[0]:g} x {b[1]:g} x {b[2]:g} um of "
              f"water (production box is used for the ensemble).")
    print(f"  {'requested':>10}  {'metal':>7}  {'achieved':>9}  "
          f"{'cells':>13}  {'steps':>9}  {'rel cost':>8}  "
          f"{'resonance':>10}  {'FWHM':>8}")
    base_work = None
    for r in rows:
        work = float(r["cells"]) * float(r.get("time_steps") or 0)
        if base_work is None:
            base_work = work or 1.0
        flag = "" if r["binding"] else "   <-- OVERRIDE NOT BINDING"
        res = ("%10.3f" % r["resonance_nm"]) if r["resonance_nm"] == r["resonance_nm"] else "         -"
        fw = ("%8.3f" % r["fwhm_nm"]) if r["fwhm_nm"] == r["fwhm_nm"] else "       -"
        print(f"  {r['dl_requested_nm']:8.2f}nm  {r['dl_metal_nm']:5.1f}nm  "
              f"{r['dl_achieved_nm']:7.3f}nm  {r['cells']:13,}  "
              f"{r.get('time_steps', 0):9,}  {work / base_work:7.2f}x  "
              f"{res}  {fw}{flag}")
    print()
    print("  'rel cost' is (cells x time steps) relative to the first row --")
    print("  what you are actually billed for, and why the finest point can")
    print("  cost more than every other run in the campaign put together.")

    # A subtler failure than a non-binding override: two rows that happen to
    # land on the SAME achieved resolution.  Each may be within tolerance of
    # its own request, yet they are one simulation wearing two labels, and
    # plotting them looks like flawless convergence.
    seen: dict[float, float] = {}
    duplicates = []
    for r in rows:
        key = round(r["dl_achieved_nm"], 3)
        if key in seen:
            duplicates.append((seen[key], r["dl_requested_nm"], key))
        seen[key] = r["dl_requested_nm"]

    bad = [r for r in rows if not r["binding"]]
    print()
    if duplicates:
        for first, second, achieved in duplicates:
            print(f"  DUPLICATE MESH: the {first:.2f} nm and {second:.2f} nm "
                  f"requests both delivered {achieved:.3f} nm.")
        print("  Those are the same simulation twice.  Two identical points")
        print("  look like perfect convergence and prove nothing.  Drop one,")
        print("  or spread the requested values further apart.")
        print()
    if bad:
        print(f"  {len(bad)} row(s) did not get the cell size they asked for.")
        print("  Something finer -- the metal override or Tidy3D's automatic")
        print("  grid -- won instead.  Those rows are NOT independent points")
        print("  in a refinement series; drop them or coarsen the competing")
        print("  setting.  Two rows that silently share a mesh look like")
        print("  perfect convergence and are not.")
    else:
        print("  Every row got the resolution it asked for, so this is a")
        print("  genuine refinement series.")

    if any(r["resonance_nm"] == r["resonance_nm"] for r in rows):
        got = [r for r in rows if r["resonance_nm"] == r["resonance_nm"]]
        if len(got) >= 2:
            drift = abs(got[-1]["resonance_nm"] - got[-2]["resonance_nm"])
            print()
            print(f"  Resonance moved {drift * 1000:.1f} pm between the two "
                  f"finest meshes.")
            print("  Compare that against the tissue speckle spread: if the")
            print("  mesh drift is comparable, refine further before quoting")
            print("  a noise floor.")
    else:
        print()
        print("  Pass submit=True to run these and see whether the resonance")
        print("  and linewidth have stopped moving.  One series in water is")
        print("  enough.")


# ==========================================================================
#  3. Supercell size
# ==========================================================================
def supercell_study(
    cfg: StudyConfig,
    sizes_um: tuple[float, ...] = (1.5, 2.0, 3.0, 4.0),
    n_seeds: int = 32,
    voxel_nm: float = 25.0,
    max_voxels: int = 70_000_000,
    bootstrap: int = 2000,
    rng_seed: int = 0,
) -> list[dict]:
    """
    How much of the resonance spread is just the finite box?

    Runs OFFLINE and needs no FDTD: for each box size it computes the
    realisation-to-realisation spread of the box's own mean refractive index.
    That spread, multiplied by the bulk index sensitivity of the resonance, is
    the finite-box contribution to the apparent noise floor.

    WHY THE SEED COUNT MATTERS HERE MORE THAN ANYWHERE ELSE
    -------------------------------------------------------
    This study measures a SPREAD, and spreads are noisy: the relative error on
    an estimated standard deviation is about 1/sqrt(2(N-1)) -- 21% at N=12.

    That is enough to hide the trend the study exists to show.  With N=12 the
    spread can appear flat between 3 um (4.57e-4) and 4 um (4.49e-4) while the
    underlying spread, recovered at N=48, falls smoothly.

    The default is therefore 32 seeds, and every row carries a bootstrap
    confidence interval plus a comparison against the expected scaling, so a
    flat segment can be recognised as noise rather than physics.

    WHAT TO LOOK FOR
    ----------------
    Averaging over a larger box should reduce the mean-index spread as the
    inverse square root of the volume, i.e. as L^(-3/2).  The `expected`
    column anchors that power law to the first row.  Real values should track
    it within the confidence interval.

    Then compare against your ensemble: if the raw resonance spread falls the
    same way with box size, it was dominated by this artefact, and the
    regression in `ensemble.py` is doing essential work.  The genuine speckle
    residual should NOT fall with box size -- that contrast is the key result
    of this study.

    `oversample=2` below is ESSENTIAL and not a tuning knob.  Spectral
    synthesis on a periodic box gives every realisation EXACTLY the same mean
    index, because the box mean simply IS the zero-frequency component, which
    the generator sets to zero.  Without oversampling this study measures a
    spread of 1e-16 -- i.e. nothing.  Cropping a window out of a larger
    generated field restores the genuine finite-window drift.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    reference = None

    for L in sizes_um:
        n_vox = int(round(2 * L / (voxel_nm * 1e-3))) ** 3   # oversampled
        if n_vox > max_voxels:
            print(f"  (skipping {L} um: {n_vox:,} voxels exceeds the "
                  f"{max_voxels:,} memory guard)")
            continue

        means = []
        for k in range(n_seeds):
            real = generate_tissue(
                cfg.tissue, cfg.fdtd, seed=1000 + k,
                domain_um=(L, L, L),
                voxel_nm=voxel_nm,
                apply_taper=False,
                oversample=2.0,
            )
            means.append(real.mean_index)
        means = np.array(means)
        spread = float(means.std(ddof=1))

        # Bootstrap the spread, so a flat segment can be told from real
        # saturation.
        boots = np.array([
            np.std(means[rng.integers(0, n_seeds, n_seeds)], ddof=1)
            for _ in range(bootstrap)
        ])
        ci = (float(np.percentile(boots, 2.5)),
              float(np.percentile(boots, 97.5)))

        if reference is None:
            reference = (L, spread)
        L0, s0 = reference

        # The L^(-3/2) averaging law assumes the box contains MANY independent
        # correlation volumes.  When the box is comparable to, or smaller than,
        # one correlation length there is nothing to average over and the law
        # simply does not apply -- quoting it would invite the reader to
        # diagnose a departure that is not there.
        blobs = L / cfg.tissue.l_c_um
        law_applies = blobs >= 3.0
        expected = s0 * (L0 / L) ** 1.5 if law_applies else float("nan")

        rows.append({
            "box_um": L,
            "n_seeds": n_seeds,
            "mean_index": float(means.mean()),
            "index_spread": spread,
            "ci95": ci,
            "expected_L15": float(expected),
            "law_applies": bool(law_applies),
            "blobs_across": blobs,
            "independent_volumes": blobs ** 3,
            "l_c_um": cfg.tissue.l_c_um,
        })
    return rows


def print_supercell_study(
    rows: list[dict], bulk_sensitivity_nm_per_riu: float | None = None
) -> None:
    """Print the table, converting index spread into an equivalent resonance
    wobble so the size of the artefact is immediately visible.

    `bulk_sensitivity_nm_per_riu` defaults to
    `TransductionConfig.bulk_sensitivity_nm_per_riu`, so this conversion and
    the one in `run_all.py` stage 5b always use the same value.
    """
    if bulk_sensitivity_nm_per_riu is None:
        from config import TransductionConfig
        bulk_sensitivity_nm_per_riu = (
            TransductionConfig().bulk_sensitivity_nm_per_riu)
    if not rows:
        print("  (no box sizes fit within the memory guard)")
        return

    n = rows[0]["n_seeds"]
    rel_err = 100.0 / (2 * (n - 1)) ** 0.5
    print(f"  {n} seeds per box; expected relative error on each spread "
          f"~{rel_err:.0f}%")
    print()
    l_c_nm = rows[0].get("l_c_um", float("nan")) * 1000
    print(f"  correlation length in use: {l_c_nm:.0f} nm")
    print()
    print(f"  {'box':>7}  {'L/l_c':>6}  {'indep. vols':>11}  "
          f"{'index spread':>13}  {'95% CI':>21}  {'expected':>10}")
    for r in rows:
        lo, hi = r["ci95"]
        exp = (f"{r['expected_L15']:10.3e}" if r.get("law_applies")
               else "       n/a")
        flag = "" if r.get("law_applies") else "  <-- law N/A"
        print(f"  {r['box_um']:5.1f}um  {r['blobs_across']:6.2f}  "
              f"{r['independent_volumes']:11.2f}  "
              f"{r['index_spread']:13.3e}  "
              f"[{lo:.2e}, {hi:.2e}]  {exp}{flag}")

    print()
    n_bad = sum(1 for r in rows if not r.get("law_applies"))
    if n_bad:
        print(f"  {n_bad} of {len(rows)} boxes hold fewer than three")
        print("  correlation lengths, so no L^(-3/2) expectation is quoted for")
        print("  them.  Averaging only shrinks a spread when there is more than")
        print("  one independent thing to average, and the 'indep. vols' column")
        print("  -- (L/l_c)^3 -- is that count.  Below about one, enlarging the")
        print("  box does essentially nothing, which is exactly what a flat")
        print("  column here means.  This is expected: the averaging law only")
        print("  applies once the box holds several independent volumes.")
    else:
        print("  'expected' is the L^(-3/2) averaging law anchored to the first")
        print("  row.  Values should track it within the confidence interval; a")
        print("  row far off is either a real departure or too few seeds.")
    print()
    print(f"  {'box':>7}  {'-> resonance wobble':>21}")
    for r in rows:
        wobble_pm = r["index_spread"] * bulk_sensitivity_nm_per_riu * 1000
        print(f"  {r['box_um']:5.1f}um  {wobble_pm:18.1f} pm")
    print()
    print(f"  (converted using an assumed bulk sensitivity of "
          f"{bulk_sensitivity_nm_per_riu:.0f} nm/RIU;")
    print("   replace with the slope your own ensemble regression measured.)")
    print()
    print("  Every picometre above is finite-box artefact, not tissue")
    print("  heterogeneity.  If it is comparable to your raw ensemble spread,")
    print("  the regression in ensemble.py is carrying the result.")


# ==========================================================================
#  Self-check
# ==========================================================================
if __name__ == "__main__":
    import contextlib as _ctx
    import io as _io
    import warnings as _warn
    from dataclasses import replace as _replace

    import argparse as _argparse

    from optical_properties import calibrate as _calibrate

    _ap = _argparse.ArgumentParser(
        description="Convergence studies for the tissue model and the FDTD "
                    "mesh.  Everything is offline and free unless --submit "
                    "is given.")
    _ap.add_argument(
        "--submit", action="store_true",
        help="RUN the gap mesh series on the cloud solver.  THIS SPENDS "
             "FlexCredits.  Without it, Study 2 only reports the cell counts "
             "and relative costs.")
    _ap.add_argument(
        "--boxes", type=float, nargs="+", default=None,
        help="Supercell sizes in microns for Study 3.  Defaults include the "
             "production domain depth.")
    _args = _ap.parse_args()

    cfg = StudyConfig()

    # The whole study uses the FITTED correlation length.  Running any of the
    # sections below on an uncalibrated config silently uses the starting
    # guess in config.py, which is several times smaller, and every L/l_c
    # ratio downstream is then wrong in the optimistic direction.
    with _warn.catch_warnings():
        _warn.simplefilter("ignore")
        with _ctx.redirect_stdout(_io.StringIO()):
            _fit = _calibrate(cfg.tissue, cfg.calibration)
    _l_c_start = cfg.tissue.l_c_um
    cfg = _replace(cfg, tissue=_fit.as_tissue_config(cfg.tissue))
    print(f"  correlation length: starting guess {_l_c_start*1000:.0f} nm, "
          f"calibrated {cfg.tissue.l_c_um*1000:.0f} nm")
    print("  All studies below use the calibrated value.")
    print()

    print("=" * 74)
    print("STUDY 1a - TISSUE VOXEL SIZE, default inner cutoff (5 nm)")
    print("=" * 74)
    print_voxel_study(tissue_voxel_study(cfg), cfg.tissue.l_min_um * 1000)

    print()
    print("=" * 74)
    print("STUDY 1b - the same, with a COARSER, still-defensible cutoff")
    print("=" * 74)
    print("  l_min raised to 30 nm.  Below the myofilament lattice scale it")
    print("  is arguable that muscle has composition but no optically")
    print("  relevant index STRUCTURE, so this is a physical choice, not a")
    print("  numerical convenience.  Watch what it does to convergence:")
    print()
    good = replace(cfg, tissue=replace(cfg.tissue, l_min_um=0.030))
    print_voxel_study(tissue_voxel_study(good), 30.0)

    print()
    print("=" * 74)
    print("STUDY 1c - and with the cutoff DELIBERATELY REMOVED")
    print("=" * 74)
    print("  (l_min set to 0.1 nm, far below every voxel, so the MESH becomes")
    print("   the effective cutoff -- this is the failure mode to avoid)")
    print()
    bad = replace(cfg, tissue=replace(cfg.tissue, l_min_um=0.0001))
    print_voxel_study(tissue_voxel_study(bad), 0.1)

    print()
    print("=" * 74)
    print("STUDY 2 - FDTD GAP MESH   (offline: cell counts and resolutions)")
    print("=" * 74)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        if _args.submit:
            print("  SUBMITTING.  This spends FlexCredits.")
        print_gap_mesh_study(gap_mesh_study(cfg, submit=_args.submit))

    print()
    print("=" * 74)
    print("STUDY 2b - the same WITHOUT scaling the metal mesh")
    print("=" * 74)
    print("  This is the trap: asking for a coarse gap cell while a finer")
    print("  metal override is still in force.  Watch the flag fire.")
    print()
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        # Always offline: this series only demonstrates the meshing trap
        # described in `gap_mesh_study`.
        print_gap_mesh_study(
            gap_mesh_study(cfg, submit=False, scale_metal=False)
        )

    print()
    print("=" * 74)
    print("STUDY 3 - SUPERCELL SIZE   (free: no FDTD)")
    print("=" * 74)
    print("  Evaluated at the CALIBRATED correlation length, not the starting")
    print("  guess in config.py, which would give optimistic L/l_c ratios.")
    print()
    _boxes = tuple(_args.boxes) if _args.boxes else (
        1.5, 2.0, min(cfg.fdtd.domain_um), 3.0, 4.0)
    print(f"  Box sizes: {', '.join(f'{b:g}' for b in _boxes)} um.  The "
          f"production domain depth is {min(cfg.fdtd.domain_um):g} um.")
    print()
    print_supercell_study(supercell_study(cfg, sizes_um=_boxes))
