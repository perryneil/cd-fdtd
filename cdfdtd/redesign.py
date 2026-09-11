"""
redesign.py — the measured values, and what they imply for the instrument.
==========================================================================

    python redesign.py        # offline design tables

This module holds the measured values from the simulations (`MEASURED`,
`MEASURED_LAYER_SERIES`, `MEASURED_TISSUE_SIGNAL`, `MEASURED_7NM`) and design
calculations built on them: signal against the assumed layer contraction,
layer thickness trade-off, array sizing, the free-cadmium gate, binder
affinity and a direct-contact probe specification.  Figures and
`studies/audit_values.py` read their numbers from here.

MEASURED VALUES (PML boundary)
------------------------------
    resonance                  849.4 nm      (34 tissue realisations)
    linewidth                  162.8 nm      Q = 5.22
    bulk sensitivity           194.1 nm/RIU  (paired background-index step)
    layer/water response f     0.62          (a RATIO, not a fraction)
    surface sensitivity        120 nm/RIU    (layer shift / applied step)
    sensing decay length       12-21 nm      (four-thickness layer series)
    tissue speckle sigma       481.8 pm      (95% CI 338-580, case bootstrap)
    binding shift, on tissue   +6.5% relative to water (8 realisations)

DERIVED FOR THE PAPER (see studies/audit_values.py)
---------------------------------------------------
    full-coverage signal, on tissue   1446 pm  (accessible sites, -26.9% fold)
    effective noise                   485 pm   (speckle + signal scatter)
    signal / noise, one site          2.98
    sites to detect full coverage     2
    sites for theta <= 0.25           17       (15 um pitch)

Every site count, coverage floor and detection limit this module prints uses
the same basis as the paper and the figures: the full-coverage signal measured
on tissue (`tissue_signal_pm`, or a water value times `tissue_gain`) against the
effective noise (`effective_noise_pm`).  `sites_for_signal` does that sizing.
The water values (for example `MEASURED["signal_pm"]`, 1357 pm) are kept because
the paper builds the tissue value from them term by term.

THE LARGEST RISK IS THE CHEMISTRY ASSUMPTION
--------------------------------------------
About 98% of the full-coverage index change is the ASSUMED 26.9% layer
contraction on binding; under 2% is the mass of the cadmium.  That contraction
is a folding ratio fitted to a tyrosinamide aptamer, not a measurement on this
chemistry, and the index change PASSES THROUGH ZERO within the published range
because the folding term and the refractive-index-increment deviation have
opposite sign.  Near that crossing the sensor is blind at any concentration.

Xue et al. (Anal. Chem. 92:10007, 2020) is the one direct observation of a
CADMIUM aptamer layer, by dual polarization interferometry, which resolves
thickness and index simultaneously.  They report that at LOW Cd(II) the ion
binds the phosphate backbone and the ssDNA is STRETCHED rather than folded.
Low concentration is this sensor's operating regime.  Elongation reverses the
sign of the response but its magnitude can still clear the noise.  The
numerical thickness values are not given in the abstract and are not in this
file.

`signal_vs_contraction()` and `print_contraction_sensitivity()` report the
whole curve.

WHY PML
-------
All values above use a perfectly matched layer.  With the adiabatic absorber,
holding the tissue identical and varying only the domain depth, the linewidth
swung from 117 to 267 nm and was worst at the middle depth, the signature of a
reflecting boundary; under PML the same sweep is flat to 0.7%.  The bulk
sensitivity gives an independent check: the ensemble regression and the paired
index-step protocol agree to 0.7% under PML against 30% under the absorber.

OPEN ITEM
---------
The 7 nm trio in `MEASURED_7NM` was run with the absorber boundary, so its
comparisons are provisional.  To repeat it under PML:

    python paired_run.py --redesign --boundary pml --submit
"""

from __future__ import annotations

import math

import numpy as np

from config import StudyConfig


# --- measured baseline (PML boundary) -------------------------------------
MEASURED = {
    # Ensemble of 34 tissue realizations under PML
    # (cdfdtd/data/ensemble_pml/seed_results.json).
    "resonance_nm": 849.4,
    "fwhm_nm": 162.8,                 # Q = 5.22
    # Paired protocol under PML (data/paired_pml/): bound, unbound and a
    # 5e-3 background-index step on the same mesh.  The layer index step the
    # solver applied is 0.0013 (1.4530 -> 1.4543), and that applied step,
    # not the de Feijter full-coverage estimate, is the denominator of f.
    "s_bulk_nm_per_riu": 194.13,
    "overlap": 0.6182,                # layer-to-water response ratio f
    "layer_shift_pm": 156.02,         # measured in water
    "dn_applied": 0.0013,
    # Full-coverage index change from `recognition_layer.estimate_index_change`
    # at the default chemistry.  Only the ACCESSIBLE sites (49%) bind, so the
    # bound cadmium MASS is scaled by that fraction; the layer's thickness
    # change is a collective property and is applied whole (Dejeu et al.
    # 2018, eq. 13; Pons et al. 2022 found the index-increment deviation
    # "around -4.2% ... whatever the surface density is").  The value is
    # recomputed from config by `recompute_full_coverage()`, and
    # `self_check()` asserts the two agree.
    "dn_full_coverage": 1.1309e-2,
    "signal_pm": 1357.3,        # layer_shift * dn_full / dn_applied (water)
    # Tissue speckle: residual standard deviation after regressing out the
    # local mean index.  The CI stored here is the residual-resample interval
    # (344-581 pm); the case-resampling interval quoted in the paper,
    # 338-580 pm, comes from studies/bootstrap_refit.py.
    "speckle_pm": 481.8,
    "speckle_ci_pm": (344.2, 581.0),
    "n_seeds": 34,
}


def recompute_full_coverage(cfg: StudyConfig | None = None,
                            accessible_fraction: float | None = None) -> dict:
    """Rebuild the full-coverage numbers from config, not from literals."""
    import warnings
    from recognition_layer import estimate_index_change

    cfg = cfg or StudyConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = estimate_index_change(cfg.surface, n_solvent=cfg.tissue.n0,
                                    accessible_fraction=accessible_fraction)
    dn = est.delta_n_with_conformation
    signal = MEASURED["layer_shift_pm"] * (dn / MEASURED["dn_applied"])
    return {"delta_n": dn, "signal_pm": signal,
            "mass_only_delta_n": est.delta_n_mass_only,
            "mass_share_percent": 100.0 * est.delta_n_mass_only / dn,
            "accessible_fraction": est.accessible_fraction}


def self_check(cfg: StudyConfig | None = None, tol_percent: float = 0.5) -> None:
    """Fail loudly if the stored constants drift from the config chemistry.

    Every number this module reports downstream is built on MEASURED, so the
    stored constants are checked against the configuration rather than
    trusted.
    """
    live = recompute_full_coverage(cfg)
    for key, got in (("dn_full_coverage", live["delta_n"]),
                     ("signal_pm", live["signal_pm"])):
        want = MEASURED[key]
        drift = abs(got - want) / want * 100.0
        if drift > tol_percent:
            raise AssertionError(
                f"MEASURED[{key!r}] is {want:g} but the configuration gives "
                f"{got:g}, a drift of {drift:.2f}%. Update the constant or "
                "find out which one is wrong before quoting anything.")

#: Paired trio on the UNCHANGED geometry with a 7 nm active layer
#: (data/paired_redesign), run with the ADIABATIC ABSORBER boundary.  Under
#: PML the 3 nm trio differed by 9% in f and 6.7% in S_bulk, so treat every
#: 7 nm figure derived from this dict as provisional.  To repeat under PML:
#:     python paired_run.py --redesign --boundary pml --submit
MEASURED_7NM = {
    "layer_nm": 7.0,
    "resonance_unbound_nm": 833.7660,
    "resonance_bound_nm": 833.9849,
    "layer_shift_pm": 218.95,
    "dn_applied": 0.0013,
    "s_bulk_nm_per_riu": 195.14,
    "overlap": 0.8631,
    "fwhm_nm": 141.5,
    # Predicted from the 3 nm absorber point with a single-exponential decay.
    "overlap_predicted": 0.8585,
    "prediction_error_percent": 0.54,
}

#: Sensing decay length.
#:
#: It cannot be obtained by inverting f = 1 - exp(-2d/delta), because that
#: assumes f is the FRACTION of the mode's sensing weight in the layer.  The
#: bulk run raises the aqueous background only, so f is the layer's response
#: divided by the WATER's response and is not bounded by one (f = 1.093 at
#: 7 nm).
#:
#: layer_series.py measured the decay directly at four thicknesses in a fixed
#: 7 nm stack (data/layer{2,5}nm_pml plus the 3 and 7 nm trios):
#:     unconstrained fit to the layer shifts alone   11.8 +/- 1.1 nm
#:     partition-constrained fit                     18.1 nm
#:     marginal sensitivity, model-free              18-21 nm
#: Every route lands in 12-21 nm.  The midpoint is used here; the spread is
#: real and should be reported as a range.
#:
#: NOTE what this does NOT touch.  The surface sensitivity (layer shift /
#: applied step = 120 nm/RIU) and the full-coverage signal (that shift scaled
#: by the chemistry = 1357 pm) are direct measurements that never use delta or
#: f, and everything downstream -- S/N, coverage floor, array size -- follows
#: from those and from sigma_tissue.
DELTA_SENSING_NM = 15.0

N_SIGMA = 3.0


def gap_floor_nm(cfg: StudyConfig | None = None,
                 tip_radius_nm: float | None = None,
                 apex_deg: float | None = None) -> dict:
    """
    The smallest PHYSICAL gap this bowtie can have, whatever `gap_nm` says.

    `gap_nm` positions the sharp apices; rounding pulls each tip back by
    r*(1/sin(a) - 1).  At the current 5 nm tip radius and 60 degree apex that
    is 5 nm per tip, so the physical gap cannot go below 10 nm even at
    gap_nm = 0.  Any redesign that calls for a 6-8 nm gap must therefore
    change the tip radius or the apex angle first -- setting `gap_nm` smaller
    accomplishes nothing.
    """
    cfg = cfg or StudyConfig()
    r = tip_radius_nm if tip_radius_nm is not None else cfg.bowtie.tip_radius_nm
    a = apex_deg if apex_deg is not None else cfg.bowtie.apex_angle_deg
    half = np.radians(a) / 2.0
    setback = r * (1.0 / np.sin(half) - 1.0)
    return {"tip_radius_nm": r, "apex_deg": a,
            "setback_per_tip_nm": setback, "floor_nm": 2 * setback}


#: No gap mode keeps ALL of its energy in the gap: some always sits in the
#: metal's skin depth and in the medium beyond the gap mouth.  Measured
#: nanoparticle-on-mirror work puts the in-gap share high but not unity
#: (Baumberg et al., Nature Materials 18:668 (2019)).  0.95 is an assumed
#: ceiling, set above the 0.8631 measured by the 7 nm absorber trio.
IN_GAP_CEILING = 0.95


def overlap_from_layer(layer_nm: float,
                       ceiling: float = IN_GAP_CEILING) -> float:
    """
    Predicted mode overlap for an active layer of thickness `layer_nm` on the
    UNCHANGED antenna, from the measured sensing decay length.

    Evaluates f = 1 - exp(-2d/delta) at `DELTA_SENSING_NM`, capped at the
    confinement ceiling.  It assumes a single-exponential profile, which the
    four-thickness series fits only approximately (see `DELTA_SENSING_NM`),
    so treat the result as indicative.  `measured_layer_tradeoff` needs no
    decay model and is preferred within the measured range.

    The value is a fraction of a single-exponential profile, not the measured
    layer-to-water ratio f (which can exceed 1).  Use it through ratios such
    as overlap_from_layer(7) / overlap_from_layer(3), as `signal_from_layer`
    does.
    """
    return float(min(ceiling,
                     1.0 - np.exp(-2.0 * layer_nm / DELTA_SENSING_NM)))


def signal_from_layer(layer_nm: float, sites_scale_with_thickness: bool = False,
                      base_layer_nm: float = 3.0) -> dict:
    """
    Full-coverage signal for an active layer of thickness `layer_nm`.

    THE RESULT THAT MATTERS
    -----------------------
    Thickening the layer does two things at once, and they fight.

      1.  The mode overlap RISES, as 1 - exp(-2d/delta).  It saturates at 1.
      2.  The index change FALLS.  de Feijter gives dn = (dn/dc) * Gamma / d,
          so if the binding sites are a fixed number per unit AREA, spreading
          them through a thicker layer dilutes them exactly as 1/d.  This has
          no floor.

    The signal is the product, so it goes as f(d)/d, and (1 - exp(-u))/u is
    monotonically decreasing for all u > 0.  AT FIXED AREAL SITE DENSITY,
    THINNER IS ALWAYS BETTER.  There is no optimum thickness to find; the
    optimum is the thinnest layer the chemistry allows.

    The measured layer series shows this directly: between 3 and 7 nm the
    layer response rises by 1.77x (156.0 to 276.1 pm) while the dilution costs
    2.33x, so the full-coverage signal on tissue falls from 1446 to 1097 pm
    (see `measured_layer_tradeoff`).

    Set `sites_scale_with_thickness` to True for the other chemistry, where a
    grafted brush carries binding sites along its whole length so that the
    site density per unit VOLUME is preserved and Gamma rises with d.  Then
    the dilution cancels, the overlap gain is kept, and thicker wins.  Which
    of the two applies is a question about the surface chemistry, not about
    the electromagnetics, and no further FDTD run can settle it.

    The model holds the bulk sensitivity fixed at the value measured with the
    3 nm layer.  The absorber-boundary 7 nm trio measured it about 7% higher
    (195.1 against 181.9 nm/RIU), because the thicker high-index layer
    displaces more water from the mode volume, so this function is
    conservative by roughly that much at 7 nm.  `measured_layer_tradeoff`
    gives the value straight from the PML layer series.

    The returned `signal_pm` is on tissue (water model times `tissue_gain`),
    and the coverage floor and site count use `effective_noise_pm`.
    """
    f = overlap_from_layer(layer_nm)
    # Scale the measured 3 nm layer shift by the modelled profile, so the
    # model passes through the measured point at `base_layer_nm`.
    shift_pm = MEASURED["layer_shift_pm"] * (f / overlap_from_layer(base_layer_nm))
    dn_full = MEASURED["dn_full_coverage"]
    if not sites_scale_with_thickness:
        dn_full = dn_full * (base_layer_nm / layer_nm)
    signal_pm = shift_pm * (dn_full / MEASURED["dn_applied"]) * tissue_gain()
    theta = N_SIGMA * effective_noise_pm() / signal_pm
    return {"layer_nm": layer_nm, "overlap": f, "dn_full_coverage": dn_full,
            "signal_pm": signal_pm, "theta_single_site": theta,
            "sites_for_theta_25": sites_for_signal(signal_pm),
            "sites_scale_with_thickness": sites_scale_with_thickness}


def signal_vs_contraction(cfg: StudyConfig | None = None,
                          percents: tuple[float, ...] = (
                              -30.0, -26.9, -20.0, -15.0, -11.0,
                              -5.0, 0.0, 10.0, 29.0),
                          accessible_fraction: float | None = None,
                          n_sites: int = 1) -> list[dict]:
    """
    Detection floor as a function of the ASSUMED conformational contraction.

    WHY THIS EXISTS
    ---------------
    The full-coverage index change is dominated by the conformational term,
    not by the mass of the bound cadmium.  At the default -26.9% contraction
    the bound cadmium mass supplies only about 4% of the index change; the
    rest is the layer folding.  That contraction is not a measurement for
    this chemistry.  It is a folding ratio fitted to a tyrosinamide aptamer
    and carried through two papers, and `THICKNESS_CHANGE_LITERATURE` records
    a serotonin aptamer that ELONGATES by 29% on binding instead.

    So the instrument's performance is not a single detection floor.  It is
    a curve against this assumption.  At zero
    contraction the signal is the cadmium mass alone and falls an order of
    magnitude below the tissue noise.

    The signal is computed from the MEASURED layer shift, scaled by the ratio
    of the physical index change to the one the solver applied.  No model of
    the mode enters, so the only thing varying down the rows is chemistry.

    `signal_pm` is the water value (the contraction figure multiplies it by
    `tissue_gain`).  `signal_tissue_pm`, the coverage floor and the site count
    are on tissue against `effective_noise_pm`, as in the paper.
    """
    import warnings
    from recognition_layer import estimate_index_change

    cfg = cfg or StudyConfig()
    rows = []
    for pct in percents:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = estimate_index_change(
                cfg.surface, n_solvent=cfg.tissue.n0,
                conformational_thickness_change_percent=pct,
                accessible_fraction=accessible_fraction)
        dn = est.delta_n_with_conformation
        signal_pm = MEASURED["layer_shift_pm"] * (dn / MEASURED["dn_applied"])
        # A negative shift is as detectable as a positive one; the magnitude
        # is what has to clear the noise.  What is NOT detectable is a signal
        # near zero, which is where the contraction term and the refractive
        # index increment deviation cancel.  That crossing is the real hazard
        # and it sits inside the published range of layer responses.
        signal_tissue = signal_pm * tissue_gain()
        mag = abs(signal_tissue)
        noise = effective_noise_pm() / math.sqrt(n_sites)
        theta = N_SIGMA * noise / mag if mag > 0 else float("inf")
        rows.append({
            "contraction_percent": pct,
            "delta_n": dn,
            "mass_share_percent": 100.0 * est.delta_n_mass_only / dn
            if dn else float("nan"),
            "signal_pm": signal_pm,
            "signal_tissue_pm": signal_tissue,
            "theta_single_site": theta,
            "sites_for_theta_25": sites_for_signal(mag) if mag > 0 else None,
        })
    return rows


def print_contraction_sensitivity(cfg: StudyConfig | None = None) -> None:
    """Print the full-coverage signal against the assumed layer contraction."""
    cfg = cfg or StudyConfig()
    rows = signal_vs_contraction(cfg)
    print()
    print("  THE SIGNAL IS A CHEMISTRY ASSUMPTION, NOT A MEASUREMENT")
    print("  Full-coverage signal on tissue against the assumed layer contraction.")
    print(f"  {'contraction':>12}{'delta_n':>12}{'mass share':>12}"
          f"{'signal':>10}{'theta':>9}{'sites':>9}")
    for r in rows:
        sites = r["sites_for_theta_25"]
        sites_s = f"{sites:d}" if sites is not None else "n/a"
        print(f"  {r['contraction_percent']:>11.1f}%{r['delta_n']:>12.3e}"
              f"{r['mass_share_percent']:>11.1f}%{r['signal_tissue_pm']:>9.1f}"
              f"{100 * r['theta_single_site']:>8.0f}%{sites_s:>9}")
    print()
    print("  The index change CHANGES SIGN inside the published range of")
    print("  aptamer layer responses, because the contraction term and the")
    print("  refractive index increment deviation pull opposite ways. Near")
    print("  the crossing the sensor is blind at any concentration. That")
    print("  crossing, not the noise floor, is the dominant risk here.")
    print("  Both ENDS of the range can work, because only the magnitude of")
    print("  the signal has to clear the noise (see the table above).")
    print("  Xue et al. 2020 (Anal. Chem. 92:10007) observed a CADMIUM")
    print("  aptamer by dual polarization interferometry and report the")
    print("  ssDNA STRETCHED at low Cd, which is this sensor's regime.")
    print("  That is the far side of the blind window from the assumption")
    print("  used here. Report the curve, not one row of it.")
    signs = {r["delta_n"] > 0 for r in rows}
    if len(signs) > 1:
        lo = max(r["contraction_percent"] for r in rows if r["delta_n"] > 0)
        hi = min(r["contraction_percent"] for r in rows if r["delta_n"] <= 0)
        print(f"\n  Sign change lies between {lo:.1f}% and {hi:.1f}% "
              "contraction.")


def measured_signal_7nm(sites_scale_with_thickness: bool = False) -> dict:
    """Full-coverage signal at 7 nm straight from the stored trio, no model.

    Uses the measured 218.95 pm layer shift and the identity
    signal = shift * (dn_full / dn_applied), in which the bulk sensitivity
    cancels, so the 7% change in S_bulk between the two layer thicknesses
    never enters.  The trio used the absorber boundary; the PML value is in
    `MEASURED_LAYER_SERIES`.  The signal is put on tissue with `tissue_gain`.
    """
    dn_full = MEASURED["dn_full_coverage"]
    if not sites_scale_with_thickness:
        dn_full = dn_full * (3.0 / MEASURED_7NM["layer_nm"])
    signal_pm = (MEASURED_7NM["layer_shift_pm"]
                 * (dn_full / MEASURED_7NM["dn_applied"]) * tissue_gain())
    return {"signal_pm": signal_pm,
            "theta_single_site": N_SIGMA * effective_noise_pm() / signal_pm,
            "sites_for_theta_25": sites_for_signal(signal_pm)}


#: THE LAYER SERIES, MEASURED.  Four active-layer thicknesses inside a fixed
#: 7 nm stack under PML (layer_series.py; data/layer{2,5}nm_pml plus the 3 and
#: 7 nm trios).  `layer_shift_pm` is the bound-minus-unbound shift for the
#: applied step of 1.3e-3; `dn_full` is the de Feijter full-coverage index
#: change at that thickness AT FIXED AREAL SITE DENSITY, which dilutes as the
#: layer grows.
#:
#: These four numbers replace the decay model for the thickness trade-off.
#: Both branches below are arithmetic on measurements and use neither `f` nor
#: DELTA_SENSING_NM, which matters because the decay length is only known to
#: 12-21 nm and the exponential form did not fit the series well.
#: (The dict itself follows `effective_noise_pm` below.)

#: THE BINDING SHIFT MEASURED ON TISSUE, from studies/tissue_signal.py
#: (16 simulations under PML, data/tissue_signal/).
#:
#: The layer shifts above are measured in WATER, while sigma_tissue is
#: measured in TISSUE.  Running the recognition layer unbound and bound on the
#: same eight tissue realizations gives a shift 6.5% LARGER than in water, an
#: offset of 4.3 standard errors.  Across placements the shift itself scatters
#: by only 4.1%, so the second noise term this introduces raises the effective
#: noise by 0.74%, which is why sigma_tissue dominates the noise.
MEASURED_TISSUE_SIGNAL = {
    "n_realizations": 8,
    "layer_shift_pm": 166.22,      # against 156.02 in water
    "layer_shift_sd_pm": 6.75,     # across placements
    "layer_shift_sem_pm": 2.39,
    "offset_vs_water": 0.0654,     # +6.5%, t = 4.27 on 7 df
    "cv": 0.041,
    "corr_with_local_index": -0.30,   # NOT resolved at n = 8 (crit. 0.707)
}


def tissue_signal_pm(dn_full: float | None = None) -> float:
    """Full-coverage shift, measured on tissue. The detection-limit signal."""
    dn = dn_full if dn_full is not None else MEASURED["dn_full_coverage"]
    return MEASURED_TISSUE_SIGNAL["layer_shift_pm"] * (dn / MEASURED["dn_applied"])


def effective_noise_pm(sigma_pm: float | None = None) -> float:
    """sigma_tissue with the binding shift's own scatter added in quadrature.

    Combines the spread of binding-induced shifts across placements with
    the tissue speckle, rather than using the mean shift alone.
    """
    import math
    sig = MEASURED["speckle_pm"] if sigma_pm is None else sigma_pm
    spread = (MEASURED_TISSUE_SIGNAL["layer_shift_sd_pm"]
              * (MEASURED["dn_full_coverage"] / MEASURED["dn_applied"]))
    return math.hypot(sig, spread)


def tissue_gain() -> float:
    """Layer shift on tissue divided by the layer shift in water (1.065).

    Multiply a full-coverage signal computed from a water run by this factor
    to put it on the tissue basis used for detection limits.
    """
    return 1.0 + MEASURED_TISSUE_SIGNAL["offset_vs_water"]


def sites_for_signal(signal_pm: float, theta: float | None = None,
                     noise_pm: float | None = None) -> int:
    """Independent sites for a 3-sigma response at coverage `theta`.

    N = ceil((N_SIGMA * noise / (theta * signal))^2), with `signal_pm` the
    full-coverage shift on tissue and `noise_pm` defaulting to
    `effective_noise_pm()`.  This is the sizing used in the paper, the
    figures and `studies/audit_values.py`.  With theta = 1 it gives the sites
    needed to detect full coverage.
    """
    th = USABLE_THETA if theta is None else theta
    noise = effective_noise_pm() if noise_pm is None else noise_pm
    return int(math.ceil((N_SIGMA * noise / (th * abs(signal_pm))) ** 2))


MEASURED_LAYER_SERIES = {
    2.0: {"layer_shift_pm": 117.27, "dn_full": 1.6964e-2, "f_ratio": 0.4648},
    3.0: {"layer_shift_pm": 156.02, "dn_full": 1.1309e-2, "f_ratio": 0.6182},
    5.0: {"layer_shift_pm": 221.67, "dn_full": 6.7854e-3, "f_ratio": 0.8779},
    7.0: {"layer_shift_pm": 276.13, "dn_full": 4.8467e-3, "f_ratio": 1.0928},
}

#: The reference thickness whose site density a grafted brush preserves.
BRUSH_REFERENCE_NM = 3.0


def measured_layer_tradeoff(sites_scale_with_thickness: bool = False) -> list:
    """The thickness trade-off, straight from the measured layer shifts.

    signal(d) = layer_shift(d) * dn_full / dn_applied

    At FIXED areal site density `dn_full` dilutes as the layer thickens, and
    that dilution is what the trade-off turns on.  For a grafted brush
    functionalised through its volume the site density per unit volume is
    preserved instead, so `dn_full` stays at its reference value and only the
    measured shift grows.

    No decay length enters either branch.  The layer shifts were measured in
    water, so each signal is multiplied by `tissue_gain`, and the coverage
    floor and site count use `effective_noise_pm`, as in the paper.
    """
    dn_applied = MEASURED["dn_applied"]
    dn_ref = MEASURED_LAYER_SERIES[BRUSH_REFERENCE_NM]["dn_full"]
    sigma = effective_noise_pm()
    gain = tissue_gain()
    rows = []
    for d in sorted(MEASURED_LAYER_SERIES):
        r = MEASURED_LAYER_SERIES[d]
        dn = dn_ref if sites_scale_with_thickness else r["dn_full"]
        signal = r["layer_shift_pm"] * dn / dn_applied * gain
        theta = N_SIGMA * sigma / signal
        rows.append({
            "layer_nm": d,
            "layer_shift_pm": r["layer_shift_pm"],
            "dn_full_coverage": dn,
            "signal_pm": signal,
            "theta_single_site": theta,
            "sites_for_theta_25": sites_for_signal(signal),
        })
    return rows


def print_measured_layer_tradeoff() -> None:
    """Print the thickness trade-off from the measured layer shifts."""
    base = tissue_signal_pm()
    print()
    print("  ACTIVE LAYER THICKNESS, MEASURED AT FOUR POINTS")
    print("  No decay model. signal = layer_shift * dn_full / dn_applied, on")
    print("  tissue, and every layer_shift is a simulation.")
    for scaled in (False, True):
        label = ("grafted brush, sites scale with thickness"
                 if scaled else "fixed areal site density")
        print()
        print(f"  {label}")
        print(f"    {'layer':>7}{'shift(pm)':>11}{'dn_full':>12}"
              f"{'signal':>10}{'theta':>9}{'sites':>8}{'vs 3nm':>9}")
        for r in measured_layer_tradeoff(scaled):
            print(f"    {r['layer_nm']:>6.1f} {r['layer_shift_pm']:>10.2f} "
                  f"{r['dn_full_coverage']:>11.4e} {r['signal_pm']:>9.1f} "
                  f"{100 * r['theta_single_site']:>8.0f}%"
                  f"{r['sites_for_theta_25']:>8d}"
                  f"{100 * (r['signal_pm'] / base - 1):>+8.1f}%")
    fixed = {r["layer_nm"]: r for r in measured_layer_tradeoff(False)}
    brush = {r["layer_nm"]: r for r in measured_layer_tradeoff(True)}
    print()
    print("  Fixed density: the signal falls monotonically with thickness,")
    print("  because dilution outruns the growth in the measured shift. The")
    print("  THINNEST layer the chemistry allows wins: 2 nm beats the built")
    print(f"  3 nm by {100 * (fixed[2.0]['signal_pm'] / base - 1):.0f}% and "
          f"needs {fixed[2.0]['sites_for_theta_25']} sites instead of "
          f"{fixed[3.0]['sites_for_theta_25']}.")
    print()
    print("  Grafted brush: the dilution is cancelled by construction, so the")
    print("  signal follows the measured shift and rises with thickness. At")
    print(f"  7 nm it is {brush[7.0]['signal_pm']:.0f} pm, "
          f"{100 * (brush[7.0]['signal_pm'] / base - 1):.0f}% above the built "
          "layer. That gain is real")
    print("  ONLY if the brush carries binding sites through its whole volume,")
    print("  which is a claim about surface chemistry that this study does not")
    print("  test. It is the difference between the two tables, and it is the")
    print("  single largest lever in the design.")


def print_layer_tradeoff() -> None:
    """The thickness trade-off.  Measured first, modelled second.

    The modelled table below runs on `overlap_from_layer(DELTA_SENSING_NM)`,
    and the decay length is known only to 12-21 nm because the exponential
    form does not fit the four-point series well.
    `print_measured_layer_tradeoff` needs no decay length and is preferred.
    The model is useful for thicknesses outside the measured set, where it is
    an extrapolation.
    """
    print_measured_layer_tradeoff()

    m7 = MEASURED_7NM
    print()
    print("  THE SAME THING MODELLED, FOR COMPARISON AND FOR THICKNESSES")
    print("  OUTSIDE THE MEASURED SET.  Runs on the decay length, which is")
    print("  uncertain to 12-21 nm; treat every row as indicative.")
    print("  The 7 nm brush trio below is an ABSORBER-boundary run.")
    print(f"    predicted overlap at 7 nm     {m7['overlap_predicted']:.4f}"
          "   (from the 3 nm point alone)")
    print(f"    MEASURED overlap at 7 nm      {m7['overlap']:.4f}"
          f"   ({m7['prediction_error_percent']:+.2f}%)")
    print()
    for scaled in (False, True):
        label = ("sites scale with thickness (grafted brush)" if scaled
                 else "fixed areal site density (same chemistry, thicker)")
        print(f"  {label}")
        print(f"    {'layer':>7}{'overlap':>10}{'dn_full':>12}"
              f"{'signal':>10}{'theta':>9}{'sites':>8}")
        for d in (2.0, 3.0, 5.0, 7.0, 10.0):
            r = signal_from_layer(d, sites_scale_with_thickness=scaled)
            print(f"    {d:>6.1f} {r['overlap']:>9.4f} "
                  f"{r['dn_full_coverage']:>11.4e} {r['signal_pm']:>9.1f} "
                  f"{100 * r['theta_single_site']:>8.0f}%"
                  f"{r['sites_for_theta_25']:>8d}")
        print()
    print("  AT 7 nm, STRAIGHT FROM THE ABSORBER-BOUNDARY TRIO (no decay model;")
    print("  the PML value is the 7 nm row of the measured table above)")
    base = tissue_signal_pm()
    for scaled in (False, True):
        r = measured_signal_7nm(sites_scale_with_thickness=scaled)
        tag = "brush, sites scale" if scaled else "fixed areal density"
        print(f"    {tag:<22}{r['signal_pm']:>8.1f} pm"
              f"   theta {100 * r['theta_single_site']:>5.0f}%"
              f"   sites {r['sites_for_theta_25']:>4d}"
              f"   vs 3 nm {100 * (r['signal_pm'] / base - 1):+6.1f}%")
    print()
    print("  At fixed areal site density the signal falls monotonically with")
    print("  thickness. The measured table above says the same thing without")
    print("  needing a decay length, and is preferred.")


def snr(overlap: float, n_independent: int = 1,
        speckle_pm: float | None = None,
        s_bulk: float | None = None,
        delta_n_layer: float | None = None,
        speckle_scales_with_tissue: bool = False) -> dict:
    """
    Signal-to-speckle for a given overlap and number of independent sites.

    TWO MODELS, AND THEY DISAGREE BY (1-f) -- REPORT BOTH
    ------------------------------------------------------
    Signal is linear in overlap under both: Δλ = S_bulk * Δn_layer * f, the
    standard adlayer model.  What they disagree about is the NOISE.

      (A) speckle_scales_with_tissue=False  -- CONSERVATIVE, the default.
          Tissue speckle stays at its measured value however much the mode is
          confined.  S/N ∝ f.

      (B) speckle_scales_with_tissue=True   -- OPTIMISTIC.
          Speckle enters through the mode's overlap with the TISSUE, so
          confining the mode into the gap should reduce it in proportion to
          (1 - f), renormalised to our measured point.  S/N ∝ f/(1-f), which
          is the design "leverage".

    They differ by 1.9x at f = 0.495, so both are available.  The truth is
    between them: some of the non-layer mode weight sits in the METAL, not the
    tissue, and that part never contributes speckle.  A permittivity-weighted
    mode partition on the geometry of interest would settle it.

    Speckle averages down as 1/sqrt(N) over N sites that sample statistically
    INDEPENDENT tissue -- Goodman, *J. Opt. Soc. Am.* 66(11):1145 (1976),
    doi:10.1364/JOSA.66.001145.  The independence condition is the part that
    bites; see `array_pitch_required`.
    """
    s_bulk = s_bulk if s_bulk is not None else MEASURED["s_bulk_nm_per_riu"]
    speckle = speckle_pm if speckle_pm is not None else MEASURED["speckle_pm"]
    # Defaulted from MEASURED rather than a literal, so it always matches the
    # current chemistry (accessible sites only).
    if delta_n_layer is None:
        delta_n_layer = MEASURED["dn_full_coverage"]

    signal_pm = delta_n_layer * s_bulk * overlap * 1000.0

    if speckle_scales_with_tissue:
        # Renormalise so the curve passes through the measured point.
        speckle = speckle * (1.0 - overlap) / (1.0 - MEASURED["overlap"])

    noise_pm = speckle / np.sqrt(max(n_independent, 1))
    ratio = signal_pm / noise_pm
    theta = N_SIGMA / ratio if ratio > 0 else float("inf")
    return {
        "overlap": overlap,
        "n_independent": n_independent,
        "signal_pm": signal_pm,
        "noise_pm": noise_pm,
        "snr": ratio,
        "theta_floor": theta,
        # "detectable" meant theta_floor <= 1: the sensor responds only when
        # EVERY site is occupied.  That is saturation, not detection, and it
        # is not a usable instrument.  Kept for compatibility, but read
        # `usable` instead.
        "detectable": bool(theta <= 1.0) if ratio > 0 else False,
        "usable": bool(theta <= 0.5) if ratio > 0 else False,
    }


def array_pitch_required(cfg: StudyConfig | None = None,
                         conservative: bool = True) -> dict:
    """
    How far apart must two hot spots be to sample independent tissue?

    TWO ANSWERS, AND THEY DISAGREE -- QUOTE BOTH
    --------------------------------------------
    OUR MODEL says the index correlation length is the fitted l_c = 1645 nm,
    and `optical_properties` already grades a box as holding independent
    volumes at (L/l_c)^3 >= 27, i.e. about 3 correlation lengths -> ~5 um.

    THE MEASUREMENT says more.  Schmitt & Kumar, *Optics Letters* 21(16):1310
    (1996), doi:10.1364/OL.21.001310, measured the index fluctuation spectrum
    of biological tissue directly and found a fractal power law over 0.5-5
    /um with an OUTER SCALE of 4-10 um, above which correlations vanish.
    Their per-tissue correlation lengths run 5.2-11.5 um.

    Our fitted 1.6 um is well below their range, so a pitch chosen from our
    model alone would be optimistic by a factor of 3 or so.  15 um is used
    for design margin.

    This matters more than it looks: a DENSE metasurface at 0.2-1 um pitch
    does NOT give N independent samples however many elements it has.  The
    array has to be sparse.
    """
    cfg = cfg or StudyConfig()
    l_c = cfg.tissue.l_c_um
    ours = 3.0 * l_c
    theirs = 15.0
    pitch = theirs if conservative else ours
    return {
        "l_c_um": l_c,
        "pitch_from_our_model_um": ours,
        "pitch_from_schmitt_kumar_um": theirs,
        "recommended_pitch_um": pitch,
        "note": ("Schmitt & Kumar 1996 measured an outer scale of 4-10 um in "
                 "real tissue; our fitted l_c of "
                 f"{l_c * 1000:.0f} nm implies only {ours:.1f} um. Use the "
                 "measured outer scale, not the fitted correlation length."),
    }


#: Coverage floor at or below which the sensor has real dynamic range.
#: Above ~0.5 a Langmuir sensor only responds once it is nearly saturated,
#: so it cannot report a concentration -- only "full".  0.25 leaves the
#: response on the usable part of the isotherm.
USABLE_THETA = 0.25


def sites_for_theta(target_theta: float = USABLE_THETA,
                    overlap: float | None = None, **kw) -> dict:
    """
    Independent sites needed to reach a target COVERAGE floor.

    This is the right sizing target, and `sites_needed` (which targets a
    signal-to-noise ratio) is not.  A 3:1 S/N at full coverage corresponds to
    theta_floor = 1.0 -- a sensor that responds only when every site is
    occupied.  Through the Langmuir inversion, and on the TOTAL basis the
    regulation uses, that is 3.9e6 mg/kg -- it misses the EU limit by nearly
    eight orders of magnitude, even though "3:1" sounds like success.

    This function runs on the adlayer model in `snr` (water signal, speckle
    alone), which is useful for exploring overlaps that were not simulated.
    For the measured design points use `sites_for_signal`, which sizes from
    the signal on tissue against the effective noise, as in the paper.
    """
    return sites_needed(N_SIGMA / target_theta, overlap, **kw)


def sites_needed(target_snr: float = 3.0, overlap: float | None = None,
                 **kw) -> dict:
    """
    How many independent sites to reach a target signal-to-speckle.

    Prefer `sites_for_theta`: S/N is not the criterion that decides whether
    the instrument works.  See that function's docstring.
    """
    overlap = overlap if overlap is not None else MEASURED["overlap"]
    one = snr(overlap, 1, **kw)
    need = (target_snr / one["snr"]) ** 2 if one["snr"] > 0 else float("inf")
    n = int(np.ceil(need))
    pitch = array_pitch_required()["recommended_pitch_um"]
    side = np.sqrt(n) * pitch
    return {"overlap": overlap, "snr_single": one["snr"],
            "n_sites": n, "pitch_um": pitch,
            "array_side_mm": side / 1000.0,
            "array_side_um": side}


# ==========================================================================
#  The options, ranked
# ==========================================================================
def design_points() -> dict:
    """The two measured designs, with their full-coverage signals on tissue.

    as_built   3 nm recognition layer, signal measured on tissue
               (`tissue_signal_pm`).
    brush      7 nm carboxybetaine brush functionalised through its volume on
               the same geometry, from the PML layer series with the site
               density per unit volume preserved (`measured_layer_tradeoff`).
    """
    brush = {r["layer_nm"]: r for r in measured_layer_tradeoff(True)}[7.0]
    return {
        "as_built": {"layer_nm": 3.0,
                     "f": MEASURED_LAYER_SERIES[3.0]["f_ratio"],
                     "signal_pm": tissue_signal_pm()},
        "brush": {"layer_nm": 7.0,
                  "f": MEASURED_LAYER_SERIES[7.0]["f_ratio"],
                  "signal_pm": brush["signal_pm"]},
    }


def options(cfg: StudyConfig | None = None) -> list[dict]:
    """
    Each candidate design, with its signal, noise and coverage floor.

    Signals are the measured full-coverage shifts on tissue (`design_points`)
    and the noise is `effective_noise_pm()` averaged over N independent
    sites, so every number agrees with the paper.  The array rows use the
    smallest N that reaches the usable coverage floor (`sites_for_signal`),
    computed rather than hardcoded.
    """
    cfg = cfg or StudyConfig()
    pts = design_points()
    base, brush = pts["as_built"], pts["brush"]
    n_base = sites_for_signal(base["signal_pm"])
    n_brush = sites_for_signal(brush["signal_pm"])

    rows = [
        {
            "name": "as built, single site",
            "design": base, "n": 1,
            "change": "none",
            "test": "measured (paired_run.py, tissue_signal study)",
            "evidence": "this project, 3 nm layer under PML",
        },
        {
            # 7 nm of ACTIVE layer on the UNCHANGED geometry: the
            # carboxybetaine brush carries the binding handles, so the 4 nm
            # antifouling layer stops being inactive volume.  No lithography
            # change and no smaller tip fillet are needed.
            "name": "brush chemistry (7 nm active layer, same geometry)",
            "design": brush, "n": 1,
            "change": "recognition chemistry through the full brush volume",
            "test": "optics measured (layer_series.py); chemistry untested",
            "evidence": ("Vaisocherova et al., Anal. Chem. 80:7894 (2008), "
                         "doi:10.1021/ac8015888"),
        },
        {
            "name": f"as built + sparse array, {n_base} sites",
            "design": base, "n": n_base,
            "change": f"{n_base} antennas at 15 um pitch, ensemble readout",
            "test": "no new FDTD, statistics on the stored runs",
            "evidence": ("Goodman, JOSA 66:1145 (1976), "
                         "doi:10.1364/JOSA.66.001145; independence from "
                         "Schmitt & Kumar, Opt. Lett. 21:1310 (1996), "
                         "doi:10.1364/OL.21.001310"),
        },
        {
            "name": f"brush chemistry + {n_brush}-site array",
            "design": brush, "n": n_brush,
            "change": "both of the above",
            "test": "no new FDTD",
            "evidence": "combined",
        },
    ]
    noise_one = effective_noise_pm()
    for r in rows:
        d = r["design"]
        noise = noise_one / np.sqrt(r["n"])
        ratio = d["signal_pm"] / noise
        r.update({
            "overlap": d["f"],
            "signal_pm": d["signal_pm"],
            "noise_pm": noise,
            "snr": ratio,
            "theta_floor": N_SIGMA / ratio,
        })
    return rows


def print_options(cfg: StudyConfig | None = None) -> None:
    """The design table, on the same basis as the paper."""
    cfg = cfg or StudyConfig()
    m = MEASURED
    sig = tissue_signal_pm()
    noise = effective_noise_pm()

    print("  MEASURED BASELINE (PML boundary, 3 nm layer)")
    print(f"    bulk sensitivity        {m['s_bulk_nm_per_riu']:8.1f} nm/RIU")
    print(f"    layer response ratio f  {m['overlap']:8.4f}")
    print(f"    signal, full coverage   {m['signal_pm']:8.1f} pm  (water)")
    print(f"    signal, full coverage   {sig:8.1f} pm  (tissue)")
    print(f"    tissue speckle          {m['speckle_pm']:8.1f} pm  "
          f"[{m['speckle_ci_pm'][0]:.0f}, {m['speckle_ci_pm'][1]:.0f}]")
    print(f"    effective noise         {noise:8.1f} pm  "
          "(speckle and binding-shift scatter)")
    _th_now = N_SIGMA * noise / sig
    print(f"    signal / noise          {sig / noise:8.2f}")
    print(f"    coverage floor          {_th_now * 100:8.1f}% of sites  "
          "(a single site cannot respond")
    print("                                        before full coverage)")
    print(f"    sites for full coverage {sites_for_signal(sig, 1.0):8d}")
    print(f"    improvement needed      {_th_now / USABLE_THETA:8.1f}x  "
          f"(to reach a usable theta of {USABLE_THETA:.2f})")
    print()

    g = gap_floor_nm(cfg)
    print("  GEOMETRY NOTE: NOMINAL AND PHYSICAL GAP ARE NOT THE SAME")
    print(f"    tip radius {g['tip_radius_nm']:.0f} nm at a "
          f"{g['apex_deg']:.0f} degree apex pulls each tip back "
          f"{g['setback_per_tip_nm']:.1f} nm,")
    print(f"    so gap_nm = {cfg.bowtie.gap_nm:.0f} is a PHYSICAL gap of "
          f"{cfg.bowtie.gap_nm + g['floor_nm']:.0f} nm.")
    print(f"    With the sensing decay length measured at "
          f"{DELTA_SENSING_NM:.1f} nm, narrowing")
    print("    the gap is NOT the design lever; thickening the active layer")
    print("    is, and it needs no fabrication change.")
    print()

    from noise_floor import langmuir_concentration, molar_to_mg_per_kg
    K = cfg.regulatory.langmuir_K_per_M
    limit = cfg.regulatory.eu_limit_mg_per_kg
    # DIRECT CONTACT is the regime this sensor is proposed for: pressed on
    # intact muscle, no digestion step.  Using `free_fraction` (0.84, which
    # assumes a release step) here would flatter the design by ~800x.
    free_fraction = cfg.regulatory.free_fraction_direct_contact

    print(f"  {'option':<52} {'f':>6} {'N':>4} {'S/N':>6} "
          f"{'theta':>7} {'C_floor':>10}   verdict")
    for r in options(cfg):
        th = r["theta_floor"]
        if th >= 1.0:
            cf_s, verdict = "       n/a", "needs more than full coverage"
        else:
            # The Langmuir inversion returns the FREE cadmium concentration.
            # The EU limit is a TOTAL.  Comparing them directly would be a
            # units error of 1/f, up to nine orders of magnitude, so divide by
            # the free fraction as `noise_floor.compute_noise_floor` does.
            cf_free = molar_to_mg_per_kg(langmuir_concentration(th, K))
            cf = cf_free / max(free_fraction, 1e-12)
            cf_s = f"{cf:10.4g}"
            if cf > limit:
                verdict = f"FAILS EU limit by {cf / limit:.3g}x"
            elif th > 0.5:
                verdict = "clears EU, but saturation-only"
            else:
                verdict = f"USABLE: {limit / cf:.0f}x margin on EU limit"
        print(f"  {r['name']:<52} {r['overlap']:6.3f} {r['n']:4d} "
              f"{r['snr']:6.2f} {th * 100:6.1f}% {cf_s}   {verdict}")
    print()
    print("    f is the measured layer-to-water response ratio (not bounded")
    print("    by 1).  theta = coverage needed for a 3-sigma response.")
    print(f"    C_floor is that coverage inverted through a Langmuir isotherm "
          f"at K = {K:.2e} /M,")
    print(f"    then divided by the direct-contact free fraction "
          f"{free_fraction:.0e} to put it")
    print(f"    on the same TOTAL basis as the EU limit of {limit} mg/kg.")
    print("    That division is the whole ballgame: the isotherm sees FREE")
    print("    cadmium, the regulation counts TOTAL, and in intact muscle")
    print("    almost all of it is locked to metallothionein.")
    print("    NOTE: 'clears 3:1' is NOT the criterion.  A design can clear")
    print("    3:1 at full coverage and still miss the regulation by 8")
    print("    orders of magnitude,")
    print("    because Langmuir is steeply nonlinear near saturation.  Above")
    print("    theta ~ 0.5 the sensor only answers when it is already full:")
    print("    no dynamic range, and no way to report a concentration.")
    print("    Noise is the effective noise measured on tissue, averaged over")
    print("    N independent sites.")
    print()

    ap = array_pitch_required(cfg)
    print("  THE ARRAY CONDITION, WHICH IS THE NON-OBVIOUS PART")
    for line in _wrap(ap["note"], 68):
        print(f"    {line}")
    pts = design_points()
    pitch = ap["recommended_pitch_um"]
    n_base = sites_for_signal(pts["as_built"]["signal_pm"])
    n_brush = sites_for_signal(pts["brush"]["signal_pm"])
    side_base = np.sqrt(n_base) * pitch
    side_brush = np.sqrt(n_brush) * pitch
    print()
    print(f"    For a USABLE coverage floor (theta <= {USABLE_THETA:.2f}), "
          "the as-built layer")
    print(f"    needs {n_base} independent sites at {pitch:.0f} um pitch = "
          f"{side_base:.0f} x {side_base:.0f} um.")
    print(f"    With the brush chemistry (7 nm active layer, SAME geometry): "
          f"{n_brush} sites = "
          f"{side_brush:.0f} x {side_brush:.0f} um.")
    print(f"    The {n_base / n_brush:.1f}x reduction is "
          "chemistry, not lithography. Neither")
    print("    option reaches the usable floor at a single site, so an array")
    print("    is required either way.")
    print("    A DENSE metasurface does not work: at 0.2-1 um pitch the hot")
    print("    spots sample the same tissue and N does not grow.")
    print()
    print_inventory_gate(cfg)


def _wrap(text: str, width: int = 68) -> list[str]:
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


# ==========================================================================
#  The gate that sits BEHIND every optical number above
# ==========================================================================
def inventory_gate(cfg: StudyConfig | None = None,
                   theta: float = USABLE_THETA,
                   pitch_um: float | None = None,
                   diffusivity_m2_s: float = 5e-10) -> list[dict]:
    """
    Can the cadmium ions actually REACH the sites the optics needs filled?

    Everything above this point is optics: overlap, speckle, array size.  None
    of it matters if the ions are not available.  Reaching theta requires
    theta * (sites per antenna) ions delivered to every antenna within the
    contact time, drawn from the tissue each antenna can reach by diffusion.

    The catchment is generous -- a sparse array at 15 um pitch gives each
    antenna ~1e5 um^3 in 60 s -- so TOTAL cadmium is never the limit.  What
    limits it is the FREE IONIC fraction: cadmium bound to metallothionein or
    glutathione is invisible to a surface binder.  `RegulatoryConfig` puts
    that fraction somewhere in 1e-9 to 1e-3 and states plainly that it has
    never been measured in fish muscle by any technique.

    Across that span the required concentration moves by SIX ORDERS OF
    MAGNITUDE, from 1000x below the EU limit to 900x above it.  That is a
    wider swing than every optical change in this module combined, and no
    FDTD run addresses it.  It is the dominant uncertainty, ahead of the
    speckle result.
    """
    from noise_floor import mg_per_kg_to_number_density

    cfg = cfg or StudyConfig()
    reg = cfg.regulatory
    pitch = pitch_um if pitch_um is not None else array_pitch_required(
        cfg)["recommended_pitch_um"]

    area_um2 = 2.0 * (cfg.bowtie.thickness_nm * 1e-3) * (30.0 * 1e-3)
    sites = cfg.surface.binding_site_density_per_nm2 * area_um2 * 1e6
    need = theta * sites

    t = reg.measurement_time_s
    reach_um = np.sqrt(6.0 * diffusivity_m2_s * t) * 1e6
    catchment = min(pitch ** 2 * reach_um,
                    4.0 / 3.0 * np.pi * reach_um ** 3)

    lo, hi = reg.free_fraction_direct_contact_range
    limit = reg.eu_limit_mg_per_kg
    out = []
    for ff in (hi, 1e-5, 1e-7, lo):
        per_unit = mg_per_kg_to_number_density(1.0) * ff
        c_needed = need / (per_unit * catchment)
        out.append({
            "free_fraction": ff,
            "ions_needed_per_antenna": need,
            "catchment_um3": catchment,
            "c_needed_mg_per_kg": c_needed,
            "clears_eu": bool(c_needed <= limit),
            "margin": limit / c_needed if c_needed > 0 else float("inf"),
        })
    return out


def release_step_comparison(cfg: StudyConfig | None = None,
                            theta: float = USABLE_THETA) -> list[dict]:
    """
    The one change that rescues the design -- and it is not optical.

    The Langmuir isotherm responds to FREE cadmium.  The EU limit counts
    TOTAL.  In intact muscle the two differ by the free ionic fraction, which
    `RegulatoryConfig` puts between 1e-9 and 1e-3 and notes has never been
    measured.  Every optical improvement in this module is multiplied by that
    fraction, so a 60x reduction in array size is worth nothing against a
    factor of 1e-6 sitting in front of it.

    A digestion or release step moves the free fraction to ~0.84, and THAT
    single change is worth more than every geometry option here combined.
    """
    from noise_floor import langmuir_concentration, molar_to_mg_per_kg

    cfg = cfg or StudyConfig()
    reg = cfg.regulatory
    c_free = molar_to_mg_per_kg(
        langmuir_concentration(theta, reg.langmuir_K_per_M))
    lo, hi = reg.free_fraction_direct_contact_range
    rows = []
    for label, ff in [
        ("digestion / release step", reg.free_fraction),
        ("direct contact, optimistic end", hi),
        ("direct contact, mid", 1e-5),
        ("direct contact, pessimistic end", lo),
    ]:
        c_tot = c_free / max(ff, 1e-12)
        rows.append({
            "regime": label, "free_fraction": ff,
            "c_total_mg_per_kg": c_tot,
            "clears": bool(c_tot <= reg.eu_limit_mg_per_kg),
            "margin": reg.eu_limit_mg_per_kg / c_tot if c_tot > 0 else 0.0,
        })
    return rows


def print_inventory_gate(cfg: StudyConfig | None = None) -> None:
    cfg = cfg or StudyConfig()
    reg = cfg.regulatory
    limit = reg.eu_limit_mg_per_kg
    rows = inventory_gate(cfg)
    r0 = rows[0]

    print("  THE GATE BEHIND ALL OF THE ABOVE -- AND FDTD CANNOT OPEN IT")
    print(f"    Reaching theta = {USABLE_THETA:.2f} needs "
          f"{r0['ions_needed_per_antenna']:.0f} ions at every antenna.")
    print(f"    A sparse array gives each one ~{r0['catchment_um3']:.0f} um^3 "
          "to draw from in")
    print(f"    {reg.measurement_time_s:.0f} s, so TOTAL cadmium is never the "
          "constraint.  The FREE")
    print("    IONIC fraction is -- and it has never been measured in fish")
    print("    muscle by any technique.")
    print()
    print(f"    {'regime':<32}{'f':>9}{'C_total':>12}   vs EU "
          f"{limit} mg/kg")
    for r in release_step_comparison(cfg):
        v = (f"CLEARS, {r['margin']:.0f}x margin" if r["clears"]
             else f"FAILS by {1 / r['margin']:.3g}x")
        print(f"    {r['regime']:<32}{r['free_fraction']:>9.0e}"
              f"{r['c_total_mg_per_kg']:>12.4g}   {v}")
    cmp_rows = {r["regime"]: r for r in release_step_comparison(cfg)}
    opt = cmp_rows["direct contact, optimistic end"]
    pes = cmp_rows["direct contact, pessimistic end"]
    rel = cmp_rows["digestion / release step"]
    print()
    print("    READ THAT TABLE BEFORE ANY OF THE OPTICS ABOVE.")
    print(f"    In DIRECT CONTACT a usable array still fails by "
          f"{1 / opt['margin']:.0f}x at")
    print(f"    the most optimistic free fraction, and by "
          f"{1 / pes['margin']:.1e}x at the")
    print("    pessimistic end.  No geometry in this module closes that.")
    print(f"    With a DIGESTION OR RELEASE STEP it clears by "
          f"{rel['margin']:.0f}x.")
    print()
    print("    So the decisive change is a sample-preparation step, not an")
    print("    antenna.  A device that must work by direct contact on intact")
    print("    muscle cannot meet the limit on this evidence, so the sample")
    print("    preparation should be decided before further optical runs.")
    print()
    print_affinity_scan(cfg)
    print()
    print_direct_contact(cfg)


# ==========================================================================
#  Binder affinity: the cheapest lever in this file, and the only one that
#  costs nothing to change
# ==========================================================================
def affinity_scan(cfg: StudyConfig | None = None,
                  snr_single: float | None = None,
                  discrimination_ratio: float = 2.0,
                  k_values: tuple[float, ...] = (
                      2.9e7, 1e7, 5e6, 2.68e6, 1e6, 5e5, 2e5)) -> list[dict]:
    """
    Which binding affinity best distinguishes "at the limit" from "under it"?

    THE MISTAKE A STRONG BINDER MAKES
    ---------------------------------
    Intuition says a tighter binder is better, and for a threshold alarm it is.
    For a sensor that must say whether a sample is over or under 0.05 mg/kg it
    is not.  With a release step the free cadmium at the EU limit is ~374 nM,
    and the current binder (K = 2.9e7 /M) is 92% saturated there.  Halving the
    concentration moves coverage only from 0.916 to 0.844, a 0.071 change.
    The sensor is operating on the flat top of its own isotherm.

    Retuning to K ~ 5e6 /M puts the limit near mid-isotherm, where the same
    halving moves coverage by 0.168: 2.4x more signal for the SAME optics,
    and about 5.6x fewer sites, because the site count scales as
    1 / delta_theta^2.

    This is the cheapest change available -- a choice of chelator, not a
    fabrication process, not an FDTD run -- and no simulation in this project
    would ever have surfaced it, because the isotherm never entered the
    optical model.

    `discrimination_ratio` is what the sensor must resolve: 2.0 means telling
    the limit apart from half the limit.  `snr_single` is the single-site
    signal to noise at full coverage, by default the measured as-built value
    on tissue (`tissue_signal_pm() / effective_noise_pm()`).
    """
    from noise_floor import mg_per_kg_to_molar, langmuir_coverage

    cfg = cfg or StudyConfig()
    reg = cfg.regulatory
    c_lim = mg_per_kg_to_molar(reg.eu_limit_mg_per_kg * reg.free_fraction)
    one = (snr_single if snr_single is not None
           else tissue_signal_pm() / effective_noise_pm())

    rows = []
    for K in k_values:
        t_hi = langmuir_coverage(c_lim, K)
        t_lo = langmuir_coverage(c_lim / discrimination_ratio, K)
        d = t_hi - t_lo
        n = int(np.ceil((N_SIGMA / (d * one)) ** 2)) if d > 0 else 10 ** 9
        rows.append({
            "K_per_M": K, "theta_at_limit": t_hi,
            "theta_at_lower": t_lo, "delta_theta": d,
            "n_sites": max(1, n),
            "array_edge_um": np.sqrt(max(1, n)) * array_pitch_required(
                cfg)["recommended_pitch_um"],
            "is_current": abs(K - reg.langmuir_K_per_M) < 1e-6,
        })
    return rows


def print_affinity_scan(cfg: StudyConfig | None = None) -> None:
    cfg = cfg or StudyConfig()
    from noise_floor import mg_per_kg_to_molar

    rows = affinity_scan(cfg)
    best = max(rows, key=lambda r: r["delta_theta"])
    cur = next((r for r in rows if r["is_current"]), None)
    reg = cfg.regulatory
    c_lim_nM = mg_per_kg_to_molar(reg.eu_limit_mg_per_kg
                                  * reg.free_fraction) * 1e9
    print("  THE CHEAPEST LEVER: BINDER AFFINITY")
    print(f"    With a release step, free cadmium at the EU limit is "
          f"~{c_lim_nM:.0f} nM.")
    print("    The task is telling the limit apart from HALF the limit, so")
    print("    what matters is the COVERAGE DIFFERENCE, not the coverage.")
    print()
    print(f"    {'K (/M)':>10}{'th@limit':>10}{'th@half':>9}"
          f"{'d_theta':>9}{'N sites':>9}{'edge um':>9}")
    for r in rows:
        mark = "  <-- current" if r["is_current"] else (
            "  <-- best" if r is best else "")
        print(f"    {r['K_per_M']:>10.2e}{r['theta_at_limit']:>10.3f}"
              f"{r['theta_at_lower']:>9.3f}{r['delta_theta']:>9.3f}"
              f"{r['n_sites']:>9d}{r['array_edge_um']:>9.0f}{mark}")
    if cur:
        print()
        print(f"    The current binder is {cur['K_per_M'] / best['K_per_M']:.0f}x "
              f"TOO STRONG.  It sits at {100 * cur['theta_at_limit']:.0f}% saturation")
        print("    at the limit, on the flat top of its own isotherm.")
        print(f"    Retuning to {best['K_per_M']:.1e} /M cuts the site count "
              f"from {cur['n_sites']} to {best['n_sites']}")
        print(f"    -- a {cur['n_sites'] / best['n_sites']:.1f}x reduction for "
              "the price of a different chelator,")
        print("    with no change to the optics at all.")


# ==========================================================================
#  Direct contact: what the instrument CAN measure
# ==========================================================================
#: Highest binding affinity ever MEASURED for a Cd binder tethered to a
#: surface: 1.47e7 /M, surface-displayed CadR by ITC (Zhang et al., RSC Adv.
#: 5:9111 (2015), doi:10.1039/c4ra07805e).  Solution-phase phytochelatins and
#: metallothionein reach log K 13-15, but NOTHING in the literature retains
#: >1e8 /M once immobilised, and that measurement needed a 16 h incubation.
#: Any design assuming 1e11 on a surface is extrapolating four orders of
#: magnitude past the evidence.  This constant is the cited measurement.
K_IMMOBILISED_CEILING = 1.47e7

#: Fraction of the MT-bound cadmium pool a surface sink can recruit in a 60 s
#: contact.  Cd-MT demetalation has t_1/2 ~ 3 days (Li et al., PNAS
#: 77:6334 (1980), doi:10.1073/pnas.77.11.6334; compiled in Scheller, Irvine
#: & Stillman, Dalton Trans. 47:4049 (2018), doi:10.1039/c7dt03319b), and
#: MT-to-protein Cd transfer runs at k ~ 6 /M/s.  Both give ~1e-4.
BOUND_POOL_RECRUITABLE_60S = 5e-4


def direct_contact_spec(cfg: StudyConfig | None = None,
                        signal_pm: float | None = None,
                        noise_pm: float | None = None,
                        n_sites: tuple[int, ...] | None = None,
                        K: float = K_IMMOBILISED_CEILING) -> list[dict]:
    """
    Direct contact, reframed as the measurement it can actually make.

    WHY THE COMPLIANCE CLAIM CANNOT BE RESCUED
    ------------------------------------------
    Not thermodynamics -- KINETICS, which is the more robust failure because
    it does not depend on the unmeasured free-ion fraction:

      * Cd-MT demetalation half-life is ~3 days.  A 60 s contact recruits
        ~0.02-0.05% of the bound pool, whatever the binder's affinity.
      * Four hours of full gastric digestion (pepsin, pH 1.5, pancreatin)
        liberates under 10% of muscle Cd (Milea et al., J. Xenobiot. 15:92
        (2025), doi:10.3390/jox15030092).  A neutral-pH monolayer in 60 s is
        orders of magnitude weaker than that.
      * Selectivity pulls the opposite way from affinity.  Zn sits at 690x
        and Cu at 35x molar excess over Cd at the EU limit (Yap & Al-Mutairi,
        Toxics 10:52 (2022), doi:10.3390/toxics10020052).  The thiolate
        clusters that bind Cd tightly bind Cu(I) about five orders TIGHTER
        still (Chem. Sci. 13:5289 (2022), doi:10.1039/d2sc00676f).

    WHAT IT CAN MEASURE INSTEAD, AND WHY THAT IS WORTH MORE
    -------------------------------------------------------
    A contact sensor responds to FREE IONIC cadmium at the tissue surface.
    That is a real quantity, it is the one this whole project's error bars
    hinge on -- and it has never been measured in fish muscle by DGT,
    ion-selective electrode, Donnan membrane, voltammetry or ultrafiltration.

    So the instrument that "fails" as a compliance device is exactly the
    instrument needed to close the largest gap in its own analysis.  The
    numbers below are its specification as a free-Cd probe.

    USAGE
    -----
    Defaults reproduce the detection limits in the paper: the as-built signal
    on tissue against the effective noise, K at the immobilised ceiling, and
    arrays of (design target, 100, 600, 2400, 9600) sites, where the design
    target is `sites_for_signal(tissue_signal_pm())`.  Pass `signal_pm` for
    another design, for example `design_points()["brush"]["signal_pm"]`.
    """
    from noise_floor import langmuir_concentration, molar_to_mg_per_kg

    cfg = cfg or StudyConfig()
    # The MEASURED as-built signal, not the brush one: the spec quoted for
    # the instrument should rest on numbers already in hand.
    sig = signal_pm if signal_pm is not None else tissue_signal_pm()
    noise = noise_pm if noise_pm is not None else effective_noise_pm()
    one = sig / noise
    if n_sites is None:
        n_sites = (sites_for_signal(sig, noise_pm=noise), 100, 600, 2400, 9600)
    pitch = array_pitch_required(cfg)["recommended_pitch_um"]

    rows = []
    for n in n_sites:
        th = N_SIGMA / (one * np.sqrt(n))
        if th >= 1.0:
            rows.append({"n_sites": n, "theta_floor": th, "lod_nM": None})
            continue
        c_M = langmuir_concentration(th, K)
        rows.append({
            "n_sites": n,
            "array_edge_um": np.sqrt(n) * pitch,
            "theta_floor": th,
            "lod_free_nM": c_M * 1e9,
            "lod_free_mg_per_kg": molar_to_mg_per_kg(c_M),
        })
    return rows


def print_direct_contact(cfg: StudyConfig | None = None) -> None:
    cfg = cfg or StudyConfig()
    rows = direct_contact_spec(cfg)
    print("  DIRECT CONTACT: WHAT THE INSTRUMENT CAN MEASURE")
    print("    The compliance claim fails on KINETICS, not affinity, and that")
    print("    is the robust failure: Cd-MT demetalation t_1/2 is ~3 DAYS, so")
    print(f"    a 60 s contact recruits ~{BOUND_POOL_RECRUITABLE_60S:.0e} of "
          "the bound pool at ANY affinity.")
    print("    Four hours of gastric digestion releases under 10%.  And the")
    print("    best binder ever measured ON A SURFACE is 1.5e7 /M -- what we")
    print("    already assume.  1e11 on gold is not in the literature.")
    print()
    print("    But a contact sensor measures FREE IONIC Cd at the surface,")
    print("    and THAT quantity has never been measured in fish muscle by")
    print("    any technique.  It is the parameter this project's own error")
    print("    bars hinge on.  As that instrument, the spec is:")
    print()
    print(f"    {'N sites':>9}{'edge':>10}{'theta':>9}{'free-Cd LOD':>14}")
    for r in rows:
        if r.get("lod_free_nM") is None:
            continue
        print(f"    {r['n_sites']:>9d}{r['array_edge_um']:>9.0f}u"
              f"{r['theta_floor'] * 100:>8.1f}%"
              f"{r['lod_free_nM']:>11.2f} nM")
    print()
    print("    A free-Cd probe in the nM range shown above would produce the")
    print("    first measurement of the free fraction in muscle.  That result")
    print("    determines whether ANY direct-contact compliance device is")
    print("    possible, and it is reachable with the geometry AS BUILT.")


if __name__ == "__main__":
    import warnings
    from dataclasses import replace as _replace

    from optical_properties import calibrate

    # The correlation length that sets the array pitch is the FITTED one, not
    # the starting guess in config.py -- 1645 nm against 350.
    _cfg = StudyConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            _cfg = _replace(_cfg, tissue=calibrate(
                _cfg.tissue, _cfg.calibration).as_tissue_config(_cfg.tissue))
    print_options(_cfg)
