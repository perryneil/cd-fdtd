"""
find_resonance.py — wide-band survey to locate the antenna resonance.
======================================================================

    python find_resonance.py            # build offline and report the cost
    python find_resonance.py --submit   # run the survey (about 1 credit)

WHY A SEPARATE SURVEY
---------------------
A gold bowtie with 120 nm sides, a 30 nm physical gap and a 40 nm thickness in
a medium of index about 1.35 is not obviously a 500-900 nm antenna: gap coupling
redshifts a bowtie strongly. Before choosing the production band, the mode has
to be located over a wide range.

A narrow band can also mislead. If the antenna does not resonate inside it, a
point probe records only the incident field modulated by weak reflections in
the box, which appears as Fabry-Perot fringes (maxima equally spaced in
1/lambda). A fitter will report one of those maxima as a "peak" even though its
position is set by the box, not the bowtie. `inspect_results.describe_spectrum`
recognises this case.

SURVEY SETTINGS
---------------
  wide band          500-1700 nm, so the mode cannot fall outside it. The upper
                     limit matches Vraalstad's cod measurements, beyond which
                     the tissue has no published optical properties.
  water, no tissue   the resonance shifts by only about 4 nm between water
                     (n = 1.333) and tissue (n = 1.35), which is negligible on a
                     1200 nm search range. Without tissue the `l_min` mesh
                     constraint no longer applies, so the band can be wide.
  absorber           40 layers, so the plane wave is not reflected back.
  source clearance   a taller box, so the source is at least a fixed fraction
                     of a wavelength from the boundary everywhere in the band.
  coarse gap mesh    4 nm. The survey locates a peak; the linewidth is
                     measured later at production resolution.
  two observables    the hot-spot |E|^2 AND the flux through a box around the
                     antenna. The flux is the more reliable resonance
                     indicator, because a point probe also sees the incident
                     field, which dominates when the antenna is off resonance.

The survey located the resonance near 850 nm used in the production runs.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import replace

import numpy as np

from config import StudyConfig


#: Vraalstad et al. measured cod muscle over 500-1700 nm.  Searching beyond
#: that would find a mode the tissue model cannot be calibrated at.
SURVEY_BAND_UM = (0.50, 1.70)


def survey_config(cfg: StudyConfig | None = None) -> StudyConfig:
    """The configuration this survey runs at, with every change justified."""
    cfg = cfg or StudyConfig()

    # A taller box.  The source sits at -dz/2 + taper, so growing dz moves it
    # away from the boundary in absolute terms as well as in wavelengths.
    dx, dy, _dz = cfg.fdtd.domain_um

    return replace(
        cfg,
        calibration=replace(cfg.calibration, band_um=SURVEY_BAND_UM),
        fdtd=replace(
            cfg.fdtd,
            freq_points=801,
            domain_um=(dx, dy, 4.0),
            absorber_layers=40,
            dl_gap_nm=4.0,
            dl_metal_nm=8.0,
            # The bulk cell is derived from the band mean, which this wide
            # band pushes up.  There is no tissue here so `l_min` does not
            # bind, but keeping the bulk resolved at the blue end still
            # matters for the substrate interface.
            min_steps_per_wavelength=24,
            # The run time must not truncate the mode being searched for.
            # 0.5 ps supports a Q of about 25. Gold's losses fall in the near
            # infrared, so a mode found there could have a higher Q, and a
            # truncated mode is broadened until it may not look like a peak.
            # Covering Q up to about 50 doubles the run time (cost is linear
            # in run time), to about one credit.
            expected_q=50.0,
            run_time_ps=1.1,
        ),
    )


def build(cfg: StudyConfig | None = None) -> dict:
    """Build the survey simulation offline and report what it would cost."""
    from simulation import build_sensor_simulation, report, simulation_work

    base = cfg or StudyConfig()
    c = survey_config(base)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = build_sensor_simulation(c, realization=None)
        w = simulation_work(c)
        w0 = simulation_work(base)["work"]

    from simulation import source_clearance_check
    clr = source_clearance_check(c)
    clr_base = source_clearance_check(base)

    return {
        "clearance": clr,
        "clearance_baseline": clr_base,
        "config": c,
        "sim": sim,
        "report": report(sim),
        "cells": w["cells"],
        "steps": w["time_steps"],
        "relative_cost": w["work"] / w0,
        "source_clearance_nm": clr["clearance_um"] * 1000,
        "clearance_in_wavelengths": clr["wavelengths"],
    }


def run(cfg: StudyConfig | None = None,
        out_dir: str = "data/survey") -> dict:
    """Submit the survey and report where the mode is."""
    from simulation import run as run_sim, check_decay
    from observables import hotspot_spectrum, fit_resonance

    built = build(cfg)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "resonance_survey.hdf5")

    data = run_sim(built["sim"], task_name="cd_fdtd_resonance_survey",
                   path=path)

    wl, inten = hotspot_spectrum(data)
    fit = fit_resonance(wl, inten)
    decay = check_decay(data)

    flux_wl, flux = _flux_spectrum(data)

    return {"path": path, "hotspot": (wl, inten), "fit": fit,
            "decay": decay, "flux": (flux_wl, flux), "built": built}


def _flux_spectrum(sim_data, monitor_name: str = "flux_box"):
    """Net power through the box around the antenna, against wavelength.

    A resonance shows here far more cleanly than in a point probe, because
    the flux responds to what the ANTENNA does rather than to the incident
    field passing through the probe point.
    """
    from config import C_LIGHT_UM_S

    try:
        mon = sim_data[monitor_name]
    except Exception:
        from simulation import FLUX_MONITOR
        try:
            mon = sim_data[FLUX_MONITOR]
        except Exception:
            return None, None
    f = np.array(mon.flux.f)
    y = np.abs(np.array(mon.flux))
    wl = C_LIGHT_UM_S / f
    order = np.argsort(wl)
    return wl[order], y[order]


def print_plan(cfg: StudyConfig | None = None) -> dict:
    """What the survey would do, and what it would cost."""
    b = build(cfg)
    c = b["config"]

    print("  THE QUESTION: where does this bowtie resonate?")
    print()
    print("  SURVEY SETTINGS:")
    print(f"    band                 {c.calibration.band_um[0] * 1000:.0f} - "
          f"{c.calibration.band_um[1] * 1000:.0f} nm")
    print(f"    frequency points     {c.fdtd.freq_points}"
          f"              (free -- a monitor records more, it does not "
          f"run longer)")
    print(f"    medium               water            (no tissue: the mode "
          f"moves ~4 nm, and")
    print(f"                                          the l_min mesh limit "
          f"stops applying)")
    print(f"    absorber layers      {c.fdtd.absorber_layers}")
    print(f"    z domain             {c.fdtd.domain_um[2]:.1f} um"
          f"          (room for source clearance)")
    print(f"    gap mesh             {c.fdtd.dl_gap_nm:.1f} nm"
          f"          (locates a peak, not a precise width)")
    print(f"    run time             {c.fdtd.run_time_ps:.1f} ps"
          f"          (covers Q up to about 50)")
    print()
    print(f"    source clearance     {b['source_clearance_nm']:.0f} nm = "
          f"{b['clearance_in_wavelengths']:.2f} wavelengths in silica at the")
    print(f"                         reddest point in the band (derived from "
          f"the band).")
    print()
    print(f"    grid                 {b['cells']:,} cells x {b['steps']:,} "
          f"steps")
    print(f"    cost                 {b['relative_cost']:.2f}x the production "
          f"simulation")
    print(f"                         about {1.13 * b['relative_cost']:.2f} "
          f"FlexCredits")
    print()
    print("  WHAT COMES BACK")
    print("  Two spectra: the hot-spot |E|^2 and the flux through a box round")
    print("  the antenna. If they agree on a peak, that is the mode. If only")
    print("  the flux shows one, the probe point is being swamped by the")
    print("  incident field. If neither does, the antenna is wrong for any")
    print("  band the tissue literature covers, and the geometry has to")
    print("  change instead.")
    return b


def print_result(res: dict) -> None:
    """Report the survey."""
    from inspect_results import describe_spectrum

    fit = res["fit"]
    print("  HOT-SPOT PROBE")
    print(f"    peak            {fit.wavelength_um * 1000:.1f} nm")
    print(f"    FWHM            "
          + (f"{fit.fwhm_um * 1000:.1f} nm" if np.isfinite(fit.fwhm_um)
             else "not measurable"))
    print(f"    fields decayed  "
          f"{'yes' if res['decay'].get('ok') else 'NO'}")
    if fit.warning:
        print(f"    note            {fit.warning}")
    print()

    fwl, flux = res["flux"]
    if fwl is not None:
        i = int(np.argmax(flux))
        print("  FLUX THROUGH THE ANTENNA BOX")
        print(f"    peak            {fwl[i] * 1000:.1f} nm")
        print(f"    contrast        {flux.max() / max(flux.min(), 1e-30):.1f}x "
              f"between max and min")
        print()
        if abs(fwl[i] - fit.wavelength_um) * 1000 < 60:
            print("    The two observables AGREE. That is the mode.")
        else:
            print("    The two observables DISAGREE by "
                  f"{abs(fwl[i] - fit.wavelength_um) * 1000:.0f} nm. Trust the")
            print("    flux: a point probe sees the incident field too.")
        print()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        describe_spectrum(res["path"])


def main(argv: list) -> int:
    submit = "--submit" in argv
    print("=" * 76)
    print("RESONANCE SURVEY" + ("" if submit else " -- PLAN ONLY, nothing spent"))
    print("=" * 76)

    if not submit:
        print_plan()
        print()
        print("  Run it:  python find_resonance.py --submit")
        return 0

    res = run()
    print()
    print_result(res)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
