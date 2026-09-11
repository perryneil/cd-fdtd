"""
inspect_results.py — read back saved simulation results and report on them.
=============================================================================

Re-opens stored Tidy3D `.hdf5` result files and prints, for each run, whether
the fields decayed, where the resonance landed and how wide it is. When a
linewidth cannot be measured it explains why (see `describe_spectrum`). It
works offline and submits nothing.

USAGE
-----
    python inspect_results.py                       # results/ensemble/
    python inspect_results.py path/to/dir           # somewhere else
    python inspect_results.py path/to/one.hdf5      # a single run

WHAT TO LOOK FOR, IN ORDER OF HOW MUCH IT SHOULD STOP YOU
----------------------------------------------------------
  decay      NO means the field had not died away when the run ended.  The
             linewidth is then set by the run time, not by the physics, and
             every Q, every figure of merit and every sensitivity downstream
             is wrong.  Fix `FDTDConfig.run_time_ps` / `expected_q` and
             re-run before buying anything else.

  in band    NO means the fitted peak is at the edge of the recorded
             wavelength window.  A fitter handed a monotonic spectrum will
             report the edge and call it a resonance.  Retune the antenna or
             widen `CalibrationConfig.band_um`.

  Q          Compare against what the run time supports (about 1.45 x
             `expected_q`).  Above that, the linewidth is truncated even if
             `decay` passed marginally.
"""

from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np


def load_seed_results(path: str) -> list:
    """Re-read saved simulation data and rebuild the per-seed record."""
    import tidy3d as td

    from ensemble import SeedResult
    from observables import hotspot_spectrum, fit_resonance
    from simulation import check_decay

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.hdf5")))
    else:
        files = [path]

    if not files:
        return []

    print(f"  {len(files)} file(s) to read:", flush=True)
    for f in files:
        size_mb = os.path.getsize(f) / 1e6
        print(f"    {os.path.basename(f)}  ({size_mb:.1f} MB)", flush=True)
    print(flush=True)

    out = []
    for f in files:
        print(f"  reading {os.path.basename(f)} ...", end=" ", flush=True)
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            seed = int(name.split("_")[-1])
        except ValueError:
            seed = 0
        try:
            data = td.SimulationData.from_file(f)
            wl, intensity = hotspot_spectrum(data)
            fit = fit_resonance(wl, intensity)
            decay = check_decay(data)
            out.append(SeedResult(
                seed=seed,
                resonance_um=fit.wavelength_um,
                fwhm_um=fit.fwhm_um,
                mean_index_local=float("nan"),
                mean_index_box=float("nan"),
                decay_ok=bool(decay.get("ok", False)),
                fit_ok=bool(np.isfinite(fit.wavelength_um)
                            and np.isfinite(fit.fwhm_um)
                            and fit.fwhm_um > 0
                            and fit.converged),
                note=fit.warning or "",
            ))
            print("ok", flush=True)
        except Exception as exc:
            out.append(SeedResult(
                seed=seed, resonance_um=float("nan"), fwhm_um=float("nan"),
                mean_index_local=float("nan"), mean_index_box=float("nan"),
                decay_ok=False, fit_ok=False, note=f"could not read: {exc}",
            ))
            print(f"FAILED -- {type(exc).__name__}: {exc}", flush=True)
    print(flush=True)
    return out


def describe_spectrum(path: str, cfg=None) -> dict:
    """
    What does the recorded spectrum actually look like?

    WHEN THE LINEWIDTH COMES BACK MISSING, THIS SAYS WHY
    ----------------------------------------------------
    `fit_resonance` needs the response to fall to half its peak on BOTH sides
    of the maximum inside the recorded band.  When it cannot, it returns a NaN
    width and says so -- but "half-maximum not reached on both sides" does not
    tell you which side, or by how much, or whether the fix is a wider band or
    a different antenna.  Those are different problems with different prices.

        RED SIDE fails, peak well inside the band
            The line is broader than the room left above it.  Widening
            `CalibrationConfig.band_um` upward fixes it and costs nothing
            extra in solver time -- a frequency monitor records more points,
            it does not run longer.

        BLUE SIDE fails
            Same, downward.  Watch the mesh: the bulk cell is derived from
            the band MEAN, so extending the blue end without raising
            `min_steps_per_wavelength` under-resolves the new short
            wavelengths.

        PEAK NEAR AN EDGE
            The antenna is mistuned for the band, not the band for the
            antenna.  Widening will chase it; resizing the bowtie is the
            honest fix.

    Everything here is read from a stored result file.
    """
    import tidy3d as td

    from observables import hotspot_spectrum, fit_resonance

    data = td.SimulationData.from_file(path)
    wl, y = hotspot_spectrum(data)
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(wl)
    wl, y = wl[order], y[order]

    fit = fit_resonance(wl, y)
    i = int(np.argmax(y))
    peak = float(y[i])
    half = 0.5 * peak

    left_ok = bool(np.any(y[:i + 1] <= half))
    right_ok = bool(np.any(y[i:] <= half))

    # How far down each side actually gets, as a fraction of the peak.
    blue_floor = float(y[:i + 1].min() / peak) if i > 0 else 1.0
    red_floor = float(y[i:].min() / peak) if i < len(y) - 1 else 1.0

    out = {
        "peak_um": float(wl[i]),
        "fit_um": float(fit.wavelength_um),
        "fwhm_um": float(fit.fwhm_um),
        "band_um": (float(wl[0]), float(wl[-1])),
        "half_max_blue": left_ok,
        "half_max_red": right_ok,
        "blue_floor_fraction": blue_floor,
        "red_floor_fraction": red_floor,
        "headroom_blue_nm": (float(wl[i]) - float(wl[0])) * 1000,
        "headroom_red_nm": (float(wl[-1]) - float(wl[i])) * 1000,
        "warning": fit.warning,
    }

    print("  SPECTRUM DIAGNOSTIC")
    print(f"    recorded band            {out['band_um'][0] * 1000:.0f} - "
          f"{out['band_um'][1] * 1000:.0f} nm")
    print(f"    peak at                  {out['peak_um'] * 1000:.1f} nm")
    print(f"    room below the peak      {out['headroom_blue_nm']:.0f} nm")
    print(f"    room above the peak      {out['headroom_red_nm']:.0f} nm")
    print(f"    falls to half-max blue?  "
          f"{'yes' if left_ok else 'NO'}   "
          f"(gets down to {100 * blue_floor:.0f}% of peak)")
    print(f"    falls to half-max red?   "
          f"{'yes' if right_ok else 'NO'}   "
          f"(gets down to {100 * red_floor:.0f}% of peak)")
    print()

    # A coarse profile, so the shape is visible without plotting.
    print("    profile (response normalised to the peak):")
    step = max(1, len(wl) // 24)
    for j in range(0, len(wl), step):
        frac = y[j] / peak
        bar = "#" * int(round(frac * 40))
        mark = "  <-- peak" if abs(j - i) < step / 2 else ""
        print(f"      {wl[j] * 1000:6.0f} nm {frac:5.2f} {bar}{mark}")
    print("      (the half-max level is 0.50; the line is measurable only if")
    print("       the profile crosses it on BOTH sides inside the band)")
    print()

    # ---- classify the spectrum before blaming the band -----------------
    # A missing linewidth does not always mean the band is too narrow. The
    # band may contain no resonance at all, in which case widening it only
    # records more of the same background, so check for that first.
    peaks, dips = [], []
    for j in range(1, len(y) - 1):
        if y[j] > y[j - 1] and y[j] >= y[j + 1]:
            peaks.append(j)
        if y[j] < y[j - 1] and y[j] <= y[j + 1]:
            dips.append(j)
    # Keep only features with real depth, so sampling noise is not counted.
    span = float(y.max() - y.min())
    peaks = [j for j in peaks if y[j] > y.min() + 0.25 * span]
    dips = [j for j in dips if y[j] < y.max() - 0.25 * span]

    modulation = float(y.min() / y.max())
    out["n_maxima"] = len(peaks)
    out["n_minima"] = len(dips)
    out["modulation_min_over_max"] = modulation

    # Fabry-Perot fringes are equally spaced in 1/lambda, not in lambda.
    fabry_perot = False
    if len(peaks) >= 3:
        k = 1.0 / wl[peaks]
        dk = np.abs(np.diff(k))
        spread = float(dk.std() / dk.mean()) if dk.mean() > 0 else 1.0
        if spread < 0.25:
            fabry_perot = True
            nl = 1.0 / (2.0 * float(dk.mean()))
            out["fabry_perot"] = True
            out["optical_path_nL_um"] = nl
            out["fringe_spacing_inv_um"] = float(dk.mean())
    out.setdefault("fabry_perot", False)

    # Amplitude reflection implied by the modulation depth, from
    # min/max = ((1-r)/(1+r))^2.
    root = np.sqrt(modulation)
    r_amp = float((1.0 - root) / (1.0 + root)) if root < 1 else 0.0
    out["implied_reflection_amplitude"] = r_amp

    if fabry_perot or (len(peaks) >= 3 and modulation > 0.4):
        print("    *** THIS IS NOT A RESONANCE. ***")
        print()
        print(f"    {len(peaks)} maxima and {len(dips)} minima, and the "
              f"response never falls")
        print(f"    below {100 * modulation:.0f}% of its peak. A plasmonic hot "
              f"spot on resonance")
        print("    gives ONE isolated peak rising far above its background,")
        print("    not a shallow ripple.")
        if out.get("fabry_perot"):
            print()
            print(f"    The maxima are equally spaced in 1/lambda, which is "
                  f"the")
            print(f"    signature of FABRY-PEROT INTERFERENCE, not of a mode.")
            print(f"    Round-trip optical path 2nL = "
                  f"{2 * out['optical_path_nL_um']:.2f} um, so nL = "
                  f"{out['optical_path_nL_um']:.2f} um --")
            print(f"    a cavity of about "
                  f"{out['optical_path_nL_um'] / 1.35 * 1000:.0f} nm at the "
                  f"tissue index.")
        print()
        print(f"    The modulation depth implies an amplitude reflection of "
              f"about")
        print(f"    r = {r_amp:.3f} ({100 * r_amp ** 2:.1f}% in power) "
              f"somewhere in the box.")
        print()
        print("    WHAT THIS MEANS FOR THE STUDY")
        print("    The peak the fitter reported is a FRINGE MAXIMUM. Its")
        print("    position is set by the box, not by the antenna, so it would")
        print("    move if you changed the domain size and not if you changed")
        print("    the bowtie. Widening the band records more fringes; it does")
        print("    not find a resonance that is not there.")
        print()
        print("    Two things to do, in this order:")
        print("      1. Find the reflector. Check absorber_layers (a thin")
        print("         absorber reflects) and the substrate interface.")
        print("      2. Then look for the resonance OUTSIDE this band. A gold")
        print("         bowtie of this size in n ~ 1.35 is expected in the")
        print("         near infrared, above 900 nm.")
        out["verdict"] = "fabry_perot_not_resonance"
        return out

    if not (left_ok and right_ok):
        side = ("red (long-wavelength)" if not right_ok
                else "blue (short-wavelength)")
        floor = red_floor if not right_ok else blue_floor
        edge = out["band_um"][1] if not right_ok else out["band_um"][0]
        print(f"    THE {side.upper()} SIDE IS THE PROBLEM. The response only "
              f"falls to")
        print(f"    {100 * floor:.0f}% of the peak by {edge * 1000:.0f} nm, "
              f"so no half-maximum crossing")
        print("    exists inside the band and the linewidth is unmeasurable.")
        print()
        print("    The profile above has a single maximum, so this really is a")
        print("    band problem: extend CalibrationConfig.band_um on that")
        print("    side. A frequency monitor records more points; it does not")
        print("    run longer, so it costs no extra solver time.")
        out["verdict"] = "band_too_narrow"
    else:
        out["verdict"] = "ok"
    return out


def main(argv: list) -> int:
    from config import StudyConfig
    from optical_properties import calibrate
    from dataclasses import replace
    from ensemble import print_seed_results

    target = argv[1] if len(argv) > 1 else os.path.join("results", "ensemble")

    print("=" * 76)
    print("INSPECTING SAVED RESULTS -- offline, nothing is submitted")
    print("=" * 76)
    print(f"  reading {target}")
    print()

    if not os.path.exists(target):
        print(f"  Nothing at {target}.")
        print("  Saved runs live in results/ensemble/ by default; pass a path")
        print("  if you used --out.")
        return 1

    # Calibrate quietly. `calibrate` prints its own report, which is not
    # relevant when inspecting stored runs.
    import contextlib
    import io

    cfg = StudyConfig()
    print("  calibrating the tissue (quietly) ...", end=" ", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = replace(cfg, tissue=calibrate(
                cfg.tissue, cfg.calibration).as_tissue_config(cfg.tissue))
    print("done", flush=True)
    print(flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = load_seed_results(target)

    if not results:
        print("  No .hdf5 files found there.")
        print("  Saved runs land in results/ensemble/ -- check that the")
        print("  submit actually wrote one, and that you are in the cdfdtd")
        print("  folder.")
        return 1

    print(f"  {len(results)} run(s) read.")
    print()
    verdict = print_seed_results(results, cfg)

    # When a linewidth is missing, say WHY -- that decides whether the fix is
    # a wider band (free) or a different antenna (a redesign).
    if not verdict.get("ok"):
        need = [r for r in results
                if not (np.isfinite(r.fwhm_um) and r.fwhm_um > 0)]
        if need:
            files = (sorted(glob.glob(os.path.join(target, "*.hdf5")))
                     if os.path.isdir(target) else [target])
            if files:
                print()
                print("=" * 76)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    describe_spectrum(files[0], cfg)

    print()
    print("  The spectra themselves are in the .hdf5 files; open one with")
    print("  tidy3d.SimulationData.from_file(path) if you want to plot it.")
    return 0 if verdict.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
