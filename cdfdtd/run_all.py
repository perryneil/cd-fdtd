"""
run_all.py — the whole study, in the order it should be done.
=============================================================

HOW TO USE THIS
---------------
    python run_all.py                 # everything free, nothing uploaded
    python run_all.py --stage 1       # just one stage
    python run_all.py --estimate      # ask Tidy3D what a run would cost
    python run_all.py --submit        # actually run (costs FlexCredits)

THE STAGES, AND WHY THEY ARE IN THIS ORDER
------------------------------------------
Each stage is cheap and catches errors that would waste the next one, for
example running a large ensemble whose result turns out to depend on the mesh.

  0  CONFIGURATION       Print everything, so the run is self-documenting.

  1  CALIBRATION         Make the synthetic tissue scatter like real muscle.
                         Free.  Analytic.  Seconds.

  2  VALIDITY CHECKS     Is the weak-scattering approximation we just relied on
                         actually valid?  Is the FDTD box small compared with a
                         scattering length, as the two-stage argument requires?
                         Free.  Instant.

  3  CONVERGENCE         Does the tissue's index contrast depend on the voxel
                         size?  If yes, STOP -- every later number is a mesh
                         setting.  Free, no FDTD.

  4  GEOMETRY            Build the sensor.  Check how much of the hot spot the
                         analyte can actually reach.  Free.

  5  INVENTORY           How many cadmium ions are physically available at the
                         regulatory limit, and over what volume must they be
                         gathered?  Free, and it reframes the whole problem.

  6  SIMULATIONS         Build the water reference and the tissue ensemble.
                         Free to build; costs money to run.

  7  ENSEMBLE STATISTICS Separate genuine speckle from finite-box drift.
                         (Demonstrated on synthetic data with a known answer
                         when no stored results are present.)

  8  NOISE FLOOR         All four floors, against the 0.05 mg/kg limit.

THE LETTERED SUB-STAGES
-----------------------
Additional checks.  Each runs with its parent stage number, and all of them
are free except 6b with `--submit --paired`.

 2b  ANISOTROPY          Muscle has a grain and a bowtie is polarisation-
                         driven.  The correlation-function aspect ratio has
                         never been measured in any species, so this inverts
                         the one thing that HAS been (Marquez's mu_s' ratio)
                         to constrain it, and sizes the birefringence a
                         scalar medium cannot carry.

 2c  REDUCED SCATTERING  mu_s' is the number cod and bovine muscle agree on,
                         and the one a reader can compare against anything.

 4b  CONFORMAL SHELLS    Measures the recognition layer against a true signed
                         distance function, in the gap and at worst case.

 5c  CONFORMATIONAL      The two borrowed aptamer numbers, swept as a 2-D
     SWEEP               surface, because they trade off and can cancel.

 5d  PHOTON BUDGET       Why there is no Monte Carlo stage: the analytic
                         bracket settles the question without one.

 6b  PAIRED RUN          Measures the bulk sensitivity and layer response ratio f with
                         the paired protocol.  Run this BEFORE the ensemble.

 8b  FREE CADMIUM        Bioaccessibility is not the free ionic fraction.
                         Decides whether the device needs a release step.

Stages 0-5 and 7-8 never touch the network.  Only stage 6 with `--submit`
spends anything.
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from config import StudyConfig


BANNER = "=" * 76


def hdr(n: int | str, title: str) -> None:
    print()
    print(BANNER)
    print(f"STAGE {n}  --  {title}")
    print(BANNER)


# ==========================================================================
def stage_0_configuration(cfg: StudyConfig, out_dir: str) -> None:
    hdr(0, "CONFIGURATION")
    print(cfg.summary())
    path = os.path.join(out_dir, "config_as_written.json")
    cfg.save(path)
    print()
    print(f"  saved to {path}")
    print()
    print("  THIS IS THE CONFIGURATION AS WRITTEN, NOT AS USED.")
    print("  Two of the tissue parameters printed above -- the index variance")
    print("  and the correlation length -- are STARTING GUESSES that stage 1")
    print("  overwrites by fitting them to the measured optical properties.")
    print("  At the defaults they change by a factor of four and of nearly")
    print("  five respectively, so this file is not the tissue the study")
    print("  simulates. Stage 9 writes results/config_as_used.json, which is")
    print("  the configuration to report.")


def save_calibrated_config(cfg: StudyConfig, out_dir: str) -> str:
    """Write the configuration the study actually ran with."""
    path = os.path.join(out_dir, "config_as_used.json")
    cfg.save(path)
    return path


# ==========================================================================
def stage_1_calibration(cfg: StudyConfig) -> StudyConfig:
    from dataclasses import replace
    from optical_properties import calibrate, sensitivity_to_m

    hdr(1, "CALIBRATING THE TISSUE AGAINST MEASURED OPTICAL PROPERTIES")
    print("  Adjusting the index contrast and blob size until the synthetic")
    print("  tissue scatters light the way muscle does.  Analytic, not FDTD --")
    print("  see optical_properties.py for why simulating a slab would be a")
    print("  bad way to do this.")
    print()
    result = calibrate(cfg.tissue, cfg.calibration)

    if result.hit_bound:
        print()
        print("  *** The optimiser ran to a search bound.  The requested")
        print("      (mu_s, g) pair is not reachable at this m.  Fix that")
        print("      before going further. ***")

    print()
    print("  Sensitivity to the ONE parameter we refused to fit (m):")
    print(f"    {'m':>6}  {'(dn)^2':>12}  {'l_c (nm)':>10}  {'status':>22}")
    m_rows = sensitivity_to_m(cfg.tissue, cfg.calibration)
    for r in m_rows:
        if r.hit_bound:
            status = "HIT SEARCH BOUND"
        elif not r.converged:
            status = "DID NOT CONVERGE"
        else:
            status = "ok"
        print(f"    {r.m_fixed:6.2f}  {r.delta_n_sq:12.4e}  "
              f"{r.l_c_um * 1000:10.1f}  {status:>22}")

    bad_m = [r for r in m_rows if r.hit_bound or not r.converged]
    if bad_m:
        print()
        print(f"  *** {len(bad_m)} row(s) above are NOT VALID RESULTS. ***")
        print("  The optimiser either ran to the edge of its search range or")
        print("  failed to converge, which means the requested (mu_s, g) pair")
        print("  is not reachable at that m.  Those numbers are wherever the")
        print("  optimiser happened to stop and change between runs -- do not")
        print("  read a trend through them, and do not put them in a table.")
        print("  Either widen calibration.fit_bounds_l_c_um or state that the")
        print("  target pair is unreachable below m ~ 1.25.")
    print()
    print("  The fitted blob size moves by roughly an order of magnitude")
    print("  across the plausible range of m.  That is the 3-into-2")
    print("  under-determination made visible.  Report this table; do not let")
    print("  an optimiser pick m for you.")

    return replace(cfg, tissue=result.as_tissue_config(cfg.tissue))


# ==========================================================================
def stage_2_validity(cfg: StudyConfig) -> None:
    from optical_properties import verify_born_validity, check_scale_splitting

    hdr(2, "ARE THE APPROXIMATIONS ACTUALLY VALID?")
    wl = cfg.calibration.calib_wavelength_um

    print("  (a) Weak-scattering (Born) approximation, used by the calibration")
    for k, v in verify_born_validity(cfg.tissue, wl).items():
        print(f"      {k:26s} {v}")

    print()
    print("  (b) Scale splitting: the FDTD box must be much smaller than one")
    print("      scattering length, or Monte Carlo and FDTD cannot be treated")
    print("      as separate stages.")
    checks = check_scale_splitting(cfg.tissue, cfg.fdtd.domain_um, wl)
    for k, v in checks.items():
        print(f"      {k:26s} {v}")

    print()
    print("  (c) Can the box support ensemble statistics at all?")
    print(f"      correlation length         "
          f"{checks['correlation_length_um'] * 1000:.0f} nm")
    print(f"      box                        {checks['fdtd_box_um']:.1f} um")
    print(f"      independent volumes        "
          f"{checks['independent_volumes_in_box']:.2f}   = (L/l_c)^3")
    print()
    for line in _wrap_text(str(checks["box_verdict"]), 66):
        print(f"      {line}")

    if not checks["enough_blobs"]:
        print()
        print("      This propagates: stage 3(b) cannot show the L^(-3/2)")
        print("      averaging law, and the stage 7 speckle residual will be")
        print("      an UNDERESTIMATE because the realisations are not")
        print("      independent.  Read the stage 8 verdict with that in mind.")


def _wrap_text(text: str, width: int) -> list[str]:
    """Wrap a long verdict so it stays readable in a terminal."""
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
def stage_3_convergence(cfg: StudyConfig) -> None:
    from convergence import (
        tissue_voxel_study, print_voxel_study,
        supercell_study, print_supercell_study,
        gap_mesh_study, print_gap_mesh_study,
    )

    hdr(3, "CONVERGENCE -- IS THE ANSWER PHYSICS OR IS IT THE GRID?")
    print("  (a) Does the tissue's index contrast depend on the voxel size?")
    print()
    print_voxel_study(tissue_voxel_study(cfg), cfg.tissue.l_min_um * 1000)

    print()
    print("  (b) How much apparent spread comes purely from the box being")
    print("      finite?")
    print()
    print_supercell_study(supercell_study(cfg, sizes_um=(1.5, 2.0, 3.0, 4.0)))

    print()
    print("  (c) Does the FDTD mesh in the gap actually refine when asked?")
    print()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print_gap_mesh_study(gap_mesh_study(cfg, submit=False))


# ==========================================================================
def stage_4_geometry(cfg: StudyConfig) -> None:
    from sensor import gap_accessibility

    hdr(4, "SENSOR GEOMETRY AND STERIC ACCESSIBILITY")
    b = cfg.bowtie
    print(f"  gold bowtie: side {b.side_nm} nm, gap {b.gap_nm} nm, "
          f"thickness {b.thickness_nm} nm, tip radius {b.tip_radius_nm} nm")
    print(f"  coatings: {cfg.surface.recognition_thickness_nm} nm recognition "
          f"+ {cfg.surface.fouling_thickness_nm} nm fouling")
    print(f"  analyte radius: {cfg.surface.analyte_hydrodynamic_radius_nm} nm")
    print()
    print("  How much of the hot spot can the analyte physically reach?")
    print("  Computed on the REAL bowtie (rounded tips, exact signed distance)")
    print("  and on the PHYSICAL gap, with and without the coating stack.")
    print()
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for gap_nm in (5, 10, 20, 30, 50):
            rows.append(gap_accessibility(cfg, gap_nm=gap_nm))

    print(f"  {'set':>5}  {'physical':>9}  {'free':>8}  {'fits?':>5}  "
          f"{'bare metal':>10}  {'with coats':>10}")
    for r in rows:
        mark = ("  <-- current" if r["configured_gap_nm"] == b.gap_nm else "")
        print(f"  {r['configured_gap_nm']:4.0f}n  {r['physical_gap_nm']:8.1f}n  "
              f"{r['free_channel_nm']:7.1f}n  "
              f"{'yes' if r['analyte_fits'] else 'NO':>5}  "
              f"{100 * r['reachable_bare']:9.1f}%  "
              f"{100 * r['reachable_coated']:9.1f}%{mark}")
    print()
    cur = next(r for r in rows if r["configured_gap_nm"] == b.gap_nm)
    print(f"  Hot-spot region: {cur['region_note']}")
    print(f"  Coatings consume {2 * cur['coating_nm_per_side']:.0f} nm of every "
          f"gap ({cur['coating_nm_per_side']:.0f} nm per side);")
    print(f"  the analyte is {cur['analyte_diameter_nm']:.1f} nm across.")
    print()
    print("  READ THE TWO RIGHT-HAND COLUMNS AGAINST EACH OTHER.")
    print("  The bare metal is reachable at essentially every gap. It is the")
    print("  COATING STACK that excludes the analyte, and it does so")
    print(f"  catastrophically once the free channel approaches "
          f"{cur['analyte_diameter_nm']:.1f} nm.")
    print(f"  At the current design the coatings cost a factor of "
          f"{cur['exclusion_factor']:.1f}x in accessible hot-spot volume.")
    print()
    print("  That is a more useful finding than 'a tighter gap is worse',")
    print("  because coating thickness is a design variable and the gap is")
    print("  constrained by fabrication. Thinning the antifouling layer buys")
    print("  more accessible hot spot than widening the gap does.")
    if not cur["analyte_fits"]:
        print()
        print("  *** THE ANALYTE DOES NOT FIT THROUGH THE COATED GAP AT THE")
        print("      CURRENT DESIGN. Nothing downstream of this is meaningful.")


# ==========================================================================
def stage_5_inventory(cfg: StudyConfig) -> None:
    from noise_floor import (
        inventory, mg_per_kg_to_molar, mg_per_kg_to_number_density,
    )

    hdr(5, "THE CADMIUM INVENTORY -- HOW MANY IONS ARE THERE, REALLY?")
    lim = cfg.regulatory.eu_limit_mg_per_kg
    reg = cfg.regulatory
    total_per_um3 = mg_per_kg_to_number_density(
        lim, reg.tissue_density_kg_per_m3)
    print(f"  At the {lim} mg/kg regulatory limit, TOTAL cadmium:")
    print("    as an extracted solution   "
          f"{mg_per_kg_to_molar(lim) * 1e9:.1f} nM"
          "   <-- assumes a digestion step")
    print("    as ions in intact tissue   "
          f"{total_per_um3:.1f} per cubic micron")
    print()
    print(f"  The inventory below is FREE cadmium: the total above times the")
    print(f"  free fraction of {reg.free_fraction:.2f}, i.e. "
          f"{total_per_um3 * reg.free_fraction:.1f} per cubic micron.")
    print("  Everything downstream uses the free number.")
    print()
    print(inventory(cfg))


# ==========================================================================
def stage_5b_recognition(cfg: StudyConfig) -> float:
    """Derive the recognition-layer index change from published values."""
    from recognition_layer import (
        estimate_index_change, max_plausible_site_density,
        DN_DC_ML_PER_G, CD_APTAMER_KD_M, CD_APTAMER_K_PER_M,
    )
    from references import cite

    hdr("5b", "THE RECOGNITION LAYER, FROM PUBLISHED MEASUREMENTS")
    print("  Published inputs:")
    print(f"    DNA dn/dc          {DN_DC_ML_PER_G['dna']} mL/g")
    print(f"    protein dn/dc      {DN_DC_ML_PER_G['protein']} mL/g")
    print(f"    Cd aptamer Kd      {CD_APTAMER_KD_M * 1e9:.1f} nM "
          f"-> K = {CD_APTAMER_K_PER_M:.2e} /M")
    print()
    print(f"    {cite('malvern_dndc')}")
    print(f"    {cite('wu2014')}")
    print(f"    {cite('defeijter1978')}")
    print()

    ceiling = max_plausible_site_density(
        cfg.tissue.n0, DN_DC_ML_PER_G["dna"],
        cfg.surface.recognition_thickness_nm, 9500.0,
    )
    print(f"  Physical ceiling on site density: {ceiling:.3f} /nm^2 "
          f"({ceiling * 1e14:.1e} /cm^2)")
    print("  Configured:                       "
          f"{cfg.surface.binding_site_density_per_nm2:.3f} /nm^2  "
          + ("OVER THE CEILING" if cfg.surface.binding_site_density_per_nm2
             > ceiling else "OK"))
    print()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = estimate_index_change(cfg.surface, n_solvent=cfg.tissue.n0)
    print(str(est).replace("\n", "\n  "))
    print()
    print("  Stage 8 uses the THREE-TERM value below.  The mass-only figure")
    print("  is shown above as the pessimistic bound.")

    # Convert the index change into an approximate resonance shift.
    # A bowtie's bulk sensitivity is of order 200-400 nm/RIU; the layer only
    # fills part of the mode, so the bulk value is scaled by the layer response
    # ratio f (`TransductionConfig.mode_overlap`).
    bulk_sensitivity_nm_per_riu = cfg.transduction.bulk_sensitivity_nm_per_riu
    overlap = cfg.transduction.mode_overlap
    shift_mass = est.delta_n_mass_only * bulk_sensitivity_nm_per_riu * overlap
    shift_nm = (est.delta_n_with_conformation
                * bulk_sensitivity_nm_per_riu * overlap)
    print()
    print("  Approximate resonance shift at full coverage")
    print(f"    (x {bulk_sensitivity_nm_per_riu:.0f} nm/RIU bulk sensitivity, "
          f"x {overlap:.2f} layer response ratio f)")
    print(f"    mass only, pessimistic bound   {shift_mass:.4f} nm")
    print(f"    all three terms                {shift_nm:.4f} nm"
          "   <-- used by stage 8")
    print(f"    ratio                          "
          f"{shift_nm / max(shift_mass, 1e-12):.1f}x")
    print()
    t = cfg.transduction
    if t.measured:
        print("  Both the bulk sensitivity and the response ratio above are")
        print("  measured by `paired_run.py`. The shift at full coverage is")
        print(f"  {shift_nm * 1000:.0f} pm.")
    else:
        print("  The response ratio above is a prior estimate. The paired")
        print("  bound/unbound FDTD run (stage 6b, paired_run.py) measures it.")
    return shift_nm


# ==========================================================================
def stage_6_simulations(
    cfg: StudyConfig, estimate: bool, submit: bool, out_dir: str,
    over_budget: bool = False,
) -> None:
    from simulation import (
        build_sensor_simulation, report, estimate_cost,
        required_run_time_ps, check_mesh_vs_cutoff, estimate_campaign_cost,
        grids_match, metal_present_check, source_clearance_check,
    )
    from ensemble import run_ensemble

    hdr(6, "BUILDING THE SIMULATIONS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        print("  Aqueous reference (the baseline every ratio is quoted "
              "against):")
        sim_water = build_sensor_simulation(cfg, realization=None)
        print("   ", str(report(sim_water)).replace("\n", "\n    "))

        print()
        print(f"  Tissue ensemble ({cfg.ensemble.n_seeds} realisations):")
        _, sims = run_ensemble(cfg, submit=False, verbose=False)
        print("   ", str(report(sims[0])).replace("\n", "\n    "))

    print()
    print("  Do the water reference and the tissue runs share a mesh?")
    gm = grids_match(sim_water, sims[0])
    print(f"    water  {gm['shape_a']}")
    print(f"    tissue {gm['shape_b']}")
    for line in _wrap_text(gm["verdict"], 68):
        print(f"    {line}")
    if not gm["ok"]:
        print("    *** Every ratio between these two runs is suspect. ***")

    print()
    print("  Is there actually any gold in the box?")
    mp = metal_present_check(cfg, sim_water)
    for line in _wrap_text(mp["verdict"], 68):
        print(f"    {line}")
    if not mp["ok"]:
        print("    *** NOTHING BELOW IS A PLASMONIC RESULT. DO NOT SUBMIT. ***")

    print()
    print("  Is the source far enough from the absorbing boundary?")
    sc = source_clearance_check(cfg)
    for line in _wrap_text(sc["verdict"], 68):
        print(f"    {line}")

    print()
    print("  Run time, derived from physics rather than guessed:")
    rt = required_run_time_ps(cfg)
    print(f"    source pulse finishes        {rt['source_fs']:.1f} fs")
    print(f"    mode decay time (Q={rt['expected_q']:.0f})       "
          f"{rt['mode_tau_fs']:.1f} fs")
    print(f"    decay to shutoff             {rt['decay_fs']:.1f} fs")
    print(f"    photon transit across box    {rt['box_transit_fs']:.1f} fs  "
          f"(no cavity: mfp >> box, so no slow ringdown)")
    print(f"    -> recommended               {rt['recommended_ps']:.3f} ps")
    print(f"       configured                {rt['configured_ps']:.3f} ps  "
          f"(covers Q up to {rt['max_q_supported']:.0f})")
    if rt["too_short"]:
        print("    *** TOO SHORT - resonances will be truncation-broadened ***")
    elif rt["wasteful"]:
        print("    *** MORE THAN 3x WHAT IS NEEDED - cost is linear in this ***")

    print()
    print("  Mesh against the tissue cutoff:")
    mc = check_mesh_vs_cutoff(cfg, sim_water)
    for line in _wrap_text(mc["verdict"], 68):
        print(f"    {line}")

    camp = None
    if estimate or submit:
        print()
        print("  Asking Tidy3D for a cost estimate (uploads, does not run)...")
        cost = estimate_cost(sim_water, task_name="cd_fdtd_estimate")
        if cost is not None:
            camp = estimate_campaign_cost(cfg, cost)
            print(f"    per ensemble-mesh simulation   {cost:.2f} FlexCredits"
                  f"   ({camp['base_cells']:,} cells x "
                  f"{camp['base_steps']:,} steps)")
            print()
            print("    THE WHOLE CAMPAIGN, each run priced at its OWN mesh:")
            print(f"      {'':3s}   {'run':34s} {'rel':>6} {'each':>7} "
                  f"{'total':>8}")
            for ln in camp["lines"]:
                print(f"      {ln['n']:3d} x {ln['label']:34s} "
                      f"{ln['rel']:5.2f}x {ln['each']:7.2f} {ln['cost']:8.2f}")
            print(f"      {camp['n_runs']:3d}   {'SUBTOTAL':34s} "
                  f"{'':6s} {'':7s} {camp['total']:8.2f}")
            print(f"      {'':3s}   {'x contingency %.2f' % camp['contingency']:34s} "
                  f"{'':6s} {'':7s} {camp['projected']:8.2f} FlexCredits")
            print(f"      {'':3s}   {'your ceiling':34s} "
                  f"{'':6s} {'':7s} {camp['ceiling']:8.2f}")
            print()
            if camp["over_budget"]:
                print(f"    *** OVER BUDGET by "
                      f"{-camp['headroom']:.2f} FlexCredits. ***")
                print("    Cheapest levers, in order of how much they save per")
                print("    unit of scientific damage:")
                print("      - drop the finest point from fdtd.mesh_study_dl_nm")
                print("      - shrink fdtd.mesh_study_domain_um (water only)")
                print("      - lower ensemble.n_seeds, then add seeds later if")
                print("        the running standard error has not settled")
            else:
                print(f"    Within budget, {camp['headroom']:.2f} FlexCredits "
                      f"of headroom.")
            print()
            print("    Tidy3D's per-simulation figure is a worst case based on")
            print("    the full run time; runs that hit the decay shutoff early")
            print("    cost less, and web.real_cost(task_id) reports what you")
            print("    were actually charged.  The per-run WEIGHTS above are a")
            print("    model (cells x time steps) -- hence the contingency.")
        else:
            print("    (no estimate available -- check your Tidy3D API key)")
            if submit:
                print("    *** The budget guard cannot run without an "
                      "estimate. ***")
                print("    Submitting now spends FlexCredits with no ceiling "
                      "check.")

    if submit and not mp["ok"]:
        print()
        print("  REFUSING TO SUBMIT: there is no metal in the simulation.")
        print("  Spending FlexCredits on a plasmonic study with no plasmon is")
        print("  never the right move. Fix the structure order first.")
        return

    if submit and camp is not None and camp["over_budget"] and not over_budget:
        print()
        print("  REFUSING TO SUBMIT: the projected cost "
              f"({camp['projected']:.2f}) exceeds your ceiling "
              f"({camp['ceiling']:.2f}).")
        print("  Lower the cost with one of the levers above, raise")
        print("  BudgetConfig.flexcredits, or pass --over-budget if you have")
        print("  decided to spend it anyway.  Nothing has been run.")
        return

    if submit:
        from ensemble import print_seed_results, save_seed_results

        print()
        # What THIS command spends, which is not the campaign projection
        # printed above.  Conflating the two is how a one-simulation smoke
        # test reads as an eighteen-credit commitment.
        n = cfg.ensemble.n_seeds
        if camp is not None:
            each = camp["lines"][0]["each"]
            print(f"  SUBMITTING {n} tissue simulation(s) -- about "
                  f"{n * each:.2f} FlexCredits for THIS command.")
            print(f"  (the {camp['projected']:.2f} figure above is the whole "
                  f"campaign, most of which")
            print("   this command does not run.)")
        else:
            print(f"  SUBMITTING {n} tissue simulation(s). This spends "
                  f"FlexCredits.")
        results, _ = run_ensemble(
            cfg, submit=True,
            out_dir=os.path.join(out_dir, "ensemble"),
            results_path=os.path.join(out_dir, "seed_results.json"),
        )
        np.save(os.path.join(out_dir, "seed_results.npy"),
                np.array([(r.seed, r.resonance_um, r.fwhm_um,
                           r.mean_index_local) for r in results]))
        json_path = os.path.join(out_dir, "seed_results.json")
        save_seed_results(results, json_path)
        print(f"    {len(results)} result(s) saved to {json_path}")
        print()
        print("  WHAT CAME BACK -- the whole point of paying for it:")
        print()
        verdict = print_seed_results(results, cfg)
        print()
        print("  Real cost, as charged rather than estimated:")
        print("    python -c \"import tidy3d.web as w; "
              "print(w.real_cost('TASK_ID'))\"")
        print("    Task IDs are in the Tidy3D web console; the estimate is a")
        print("    worst case, so the charge is usually lower.")
    else:
        print()
        print("  Nothing uploaded.  Pass --submit to run, --estimate to cost.")


# ==========================================================================
def stage_7_statistics(cfg: StudyConfig, out_dir: str = "results") -> float:
    from ensemble import (
        _synthetic_ensemble, analyse_ensemble, production_box_drift,
    )

    hdr(7, "ENSEMBLE STATISTICS -- SEPARATING SPECKLE FROM BOX DRIFT")

    # USE REAL DATA WHEN THERE IS ANY.
    #
    # If a completed ensemble exists on disk, analyse it and pass its speckle
    # residual to stage 8. Otherwise fall back to the synthetic demonstration,
    # which verifies the METHOD but is not a result.
    real_path = os.path.join(out_dir, "seed_results.json")
    if os.path.exists(real_path):
        import json

        from ensemble import SeedResult, analyse_ensemble as _an
        with open(real_path) as fh:
            raw = json.load(fh)
        real = [SeedResult(**r) for r in raw]
        usable = [r for r in real if r.fit_ok]
        print(f"  REAL DATA: {len(real)} seed(s) in {real_path}, "
              f"{len(usable)} usable.")
        if len(usable) >= 5:
            print()
            res = _an(real, bootstrap_resamples=cfg.ensemble.bootstrap_resamples)
            print(res.summary())
            print()
            print("  This is measured, not demonstrated. Stage 8 uses it.")
            return res.sigma_speckle_nm
        print(f"  Too few usable seeds to estimate a spread (need 5+).")
        print("  Falling back to the synthetic demonstration below; stage 8")
        print("  will therefore be illustrative, NOT a result.")
        print()
    else:
        print(f"  No real ensemble at {real_path} yet.")
        print()

    print("  Demonstrated here on SYNTHETIC data with a known answer, so the")
    print("  method is verified before it is applied to real results.")
    print("  *** STAGE 8 IS ILLUSTRATIVE UNTIL A REAL ENSEMBLE EXISTS. ***")
    print()
    print("  The drift the demonstration has to beat is MEASURED on the")
    print("  production box, not assumed:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = production_box_drift(cfg)
    print(f"    mean-index drift, {d['domain_um'][0]:.0f} x "
          f"{d['domain_um'][1]:.0f} x {d['domain_um'][2]:.1f} um box   "
          f"{d['drift_rms']:.3e}")
    print(f"    from {d['n_seeds']} realisations at {d['voxel_nm']:.0f} nm "
          f"voxels (+/-{100 * d['rel_error']:.0f}%)")
    print(f"    equivalent resonance wobble          "
          f"{d['drift_rms'] * cfg.transduction.bulk_sensitivity_nm_per_riu * 1000:.0f} pm")
    print()
    print("  The demonstration uses this measured drift rather than an assumed")
    print("  value, because a regression that recovers speckle from weak drift")
    print("  is not guaranteed to recover it from strong drift.")
    print()
    true_pm = 40.0
    true_slope = cfg.transduction.bulk_sensitivity_nm_per_riu
    synth = _synthetic_ensemble(
        n=cfg.ensemble.n_seeds, bulk_sensitivity=true_slope,
        speckle_rms_nm=true_pm / 1000,
        index_drift_rms=d["drift_rms"],
    )
    res = analyse_ensemble(
        synth, bootstrap_resamples=cfg.ensemble.bootstrap_resamples
    )
    print(res.summary())
    print()
    print(f"  TRUE speckle spread          {true_pm:.2f} pm")
    print(f"  recovered                    {res.sigma_speckle_nm * 1000:.2f} pm")
    naive = analyse_ensemble(synth, regress_out_mean_index=False,
                             bootstrap_resamples=500)
    print(f"  naive (no regression)        {naive.sigma_raw_nm * 1000:.2f} pm"
          "   -- overestimates by "
          f"{naive.sigma_raw_nm * 1000 / true_pm:.2f}x")
    return res.sigma_speckle_nm


# ==========================================================================
def stage_8_noise_floor(
    cfg: StudyConfig, sigma_nm: float, shift_nm: float = 1.2
) -> None:
    from noise_floor import (
        compute_noise_floor, c_floor_vs_affinity,
        free_fraction_sweep, print_free_fraction_sweep,
    )

    hdr(8, "THE NOISE FLOOR AGAINST THE REGULATORY LIMIT")
    print("  Using the speckle spread from stage 7 and the shift at full")
    print("  coverage derived in stage 5b.")
    print()
    result = compute_noise_floor(
        cfg, sigma_lambda_nm=sigma_nm, shift_per_full_coverage_nm=shift_nm,
    )
    print(result)
    print()
    print("  And because the concentration floor depends on the binder:")
    Ks, cs = c_floor_vs_affinity(cfg, result.theta_floor)
    lim = cfg.regulatory.eu_limit_mg_per_kg
    print(f"    {'K (1/M)':>10}   {'C_floor (mg/kg, total Cd)':>26}")
    for K, c in list(zip(Ks, cs))[::10]:
        flag = "  meets the limit" if c < lim else "  FAILS the limit"
        print(f"    {K:10.2e}   {c:26.6f}{flag}")
    print()
    print("  Reading: quoting a single C_floor silently fixes the binding")
    print("  affinity.  The curve lets a reader place their own binder on it.")

    print()
    print("  HOW MUCH CADMIUM MUST THE RELEASE CHEMISTRY LIBERATE?")
    print("  " + "-" * 66)
    print("  The free fraction is not a number the literature agrees on, so")
    print("  instead of guessing it we invert the question:")
    print()
    rows = free_fraction_sweep(cfg, sigma_nm, shift_nm)
    print_free_fraction_sweep(rows, cfg.regulatory.eu_limit_mg_per_kg)


# ==========================================================================
def stage_9_provenance(cfg: StudyConfig, out_dir: str) -> None:
    from references import provenance_table, unsourced_parameters, bibliography

    hdr(9, "PROVENANCE -- WHERE EVERY NUMBER CAME FROM")
    print(provenance_table())
    print()
    print(unsourced_parameters())
    path = os.path.join(out_dir, "bibliography.txt")
    with open(path, "w") as fh:
        fh.write(bibliography())
    print()
    print(f"  bibliography written to {path}")

    cfg_path = save_calibrated_config(cfg, out_dir)
    print(f"  configuration AS USED written to {cfg_path}")
    print()
    print("  That file, not the stage 0 one, is the configuration to report:")
    print("  it carries the FITTED index variance")
    print(f"  ({cfg.tissue.delta_n_sq:.4e}) and correlation length")
    print(f"  ({cfg.tissue.l_c_um * 1000:.1f} nm), not the starting guesses.")


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Heterogeneous-tissue FDTD study for plasmonic Cd sensing."
    )
    ap.add_argument("--stage", type=int, default=None,
                    help="run only this stage (0-9); lettered sub-stages run "
                         "with their parent number")
    ap.add_argument("--estimate", action="store_true",
                    help="ask Tidy3D what a run would cost (uploads, "
                         "does not solve)")
    ap.add_argument("--submit", action="store_true",
                    help="ACTUALLY RUN the simulations -- spends FlexCredits")
    ap.add_argument("--paired", action="store_true",
                    help="with --submit, ALSO run the paired bound/unbound "
                         "trio that measures the layer response ratio (about 11 "
                         "FlexCredits). Without this, --submit runs only the "
                         "ensemble and stage 6b just prints the plan.")
    ap.add_argument("--over-budget", action="store_true",
                    help="submit even when the projected cost exceeds "
                         "BudgetConfig.flexcredits (you are choosing to "
                         "spend it)")
    ap.add_argument("--out", default="results",
                    help="output directory (default: results)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="override the number of ensemble realisations")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = StudyConfig()
    if args.seeds:
        from dataclasses import replace
        cfg = replace(cfg, ensemble=replace(cfg.ensemble, n_seeds=args.seeds))

    def want(n: int) -> bool:
        return args.stage is None or args.stage == n

    if want(0):
        stage_0_configuration(cfg, args.out)
    # Calibration is not optional for any later stage.  Running `--stage 2`
    # against the UNCALIBRATED starting guesses reports numbers that are
    # properties of config.py's placeholder values, not of the tissue the
    # study actually uses.  So: always calibrate from stage 1 onward, and only
    # print the calibration report when it was asked for.
    if args.stage is None or args.stage >= 1:
        if want(1) or args.stage is None:
            cfg = stage_1_calibration(cfg)
        else:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                cfg = stage_1_calibration(cfg)
            print()
            print("  (tissue calibrated first -- delta_n_sq = "
                  f"{cfg.tissue.delta_n_sq:.4e}, l_c = "
                  f"{cfg.tissue.l_c_um * 1000:.1f} nm. Run --stage 1 for the "
                  "full report.)")
    if want(2):
        stage_2_validity(cfg)
        stage_2b_anisotropy(cfg)
        stage_2c_mu_s_prime(cfg)
    if want(3):
        stage_3_convergence(cfg)
    if want(4):
        stage_4_geometry(cfg)
        stage_4b_coating(cfg)
    if want(5):
        stage_5_inventory(cfg)
    shift_nm = 1.2
    if want(5) or args.stage is None:
        shift_nm = stage_5b_recognition(cfg)
    if want(5):
        stage_5c_conformational(cfg)
        stage_5d_photon_budget(cfg)
    if want(6):
        stage_6_simulations(cfg, args.estimate, args.submit, args.out,
                            over_budget=args.over_budget)
        stage_6b_paired(cfg, args.submit and args.paired)

    sigma = 0.040
    if want(7):
        sigma = stage_7_statistics(cfg, args.out)
    elif args.stage is not None and args.stage == 8:
        print()
        print("  *** STAGE 8 IS RUNNING ON PLACEHOLDER INPUTS. ***")
        print(f"  Stage 7 did not run, so the tissue wobble is the fallback")
        print(f"  {sigma * 1000:.0f} pm rather than a measured value, and stage")
        print(f"  5b did not run, so the shift at full coverage is the")
        print(f"  fallback {shift_nm:.1f} nm rather than the {0.4347:.4f} nm")
        print("  the recognition-layer calculation actually gives. Every")
        print("  number below is therefore wrong by roughly that ratio.")
        print("  Run the full pipeline, or --stage 5 and --stage 7 first.")
    if want(8):
        stage_8_noise_floor(cfg, sigma, shift_nm)
        stage_8b_free_cadmium(cfg, sigma, shift_nm)
    if want(9):
        stage_9_provenance(cfg, args.out)

    print()
    print(BANNER)
    print("DONE.")
    if not args.submit:
        print("Nothing was uploaded and nothing was spent.")
        print("Next: --estimate to cost the ensemble, then --submit to run it.")
    print(BANNER)


# ==========================================================================
#  The lettered sub-stages
# ==========================================================================
def stage_2b_anisotropy(cfg: StudyConfig) -> None:
    """Fish muscle is not isotropic, and a bowtie is not blind to it."""
    from anisotropy import print_anisotropy_study, _self_test

    hdr("2b", "ANISOTROPY -- THE TISSUE HAS A GRAIN")
    st = _self_test(cfg)
    print(f"  Self-test against the isotropic module at aspect 1: "
          f"{'PASS' if st['ok'] else 'FAIL'} "
          f"(mu_s to {st['rel_error_mu_s']:.1e}, g to "
          f"{st['abs_error_g']:.1e})")
    print()
    print_anisotropy_study(cfg)


def stage_2c_mu_s_prime(cfg: StudyConfig) -> None:
    """The one optical number two independent species agree on."""
    from optical_properties import check_mu_s_prime

    hdr("2c", "REDUCED SCATTERING -- THE CROSS-SPECIES CHECK")
    r = check_mu_s_prime(cfg)
    print(f"  achieved mu_s'          {r['achieved_per_cm']:8.2f} /cm")
    print(f"  target                  {r['target_per_um'] * 1e4:8.2f} /cm")
    print(f"  implied by (mu_s, g)    "
          f"{r['implied_by_targets_per_um'] * 1e4:8.2f} /cm")
    print(f"  cod band (Vraalstad 2025)      1.20 -  6.00 /cm   "
          f"{'IN' if r['in_cod_band'] else 'out'}")
    print(f"  bovine band (Van Beers 2018)   4.00 -  9.00 /cm   "
          f"{'IN' if r['in_bovine_band'] else 'out'}")
    print()
    for line in _wrap_text(r["verdict"], 68):
        print(f"  {line}")

    # The calibration fits parameters for an INFINITE medium.  The box is not
    # infinite, and in the mass-fractal regime a large slice of the variance
    # lives at spatial frequencies below what a 5 um box can hold.  So the
    # tissue the solver actually sees is weaker than the tissue that was
    # calibrated, and both numbers should be on the page.
    from tissue import band_limited_variance
    t = cfg.tissue
    k_box = 2 * np.pi / max(cfg.fdtd.domain_um)
    k_nyq = np.pi / (cfg.fdtd.tissue_voxel_nm * 1e-3)
    var_band = band_limited_variance(
        t.delta_n_sq, t.l_c_um, t.m, t.l_min_um, k_box, k_nyq)
    frac = var_band / t.delta_n_sq if t.delta_n_sq else float("nan")

    print()
    print("  AND THE VARIANCE THE BOX ACTUALLY CARRIES:")
    print(f"    calibrated (infinite medium)   {t.delta_n_sq:.4e}  "
          f"(RMS {t.delta_n_sq ** 0.5:.5f})")
    print(f"    realisable in a {max(cfg.fdtd.domain_um):.1f} um box      "
          f"{var_band:.4e}  (RMS {var_band ** 0.5:.5f})")
    print(f"    fraction retained              {100 * frac:.1f}%")
    print()
    print("  The calibration solves for an INFINITE medium; the box is not")
    print("  infinite, so variance in blobs bigger than the box is missing")
    print("  and the tissue the solver sees is weaker than the one that was")
    print(f"  calibrated -- its effective mu_s is about "
          f"{100 * (1 - frac):.1f}% lower.")
    print()
    if frac >= 0.90:
        print(f"  At {100 * frac:.1f}% this is a small correction and the")
        print("  calibrated numbers are quotable as they stand. Mention the")
        print("  figure once; do not build an argument on it.")
    else:
        print(f"  At {100 * frac:.1f}% this is NOT a small correction. Quote")
        print("  mu_s and g with the qualifier, or enlarge the box.")
    print()
    print("  Note that stage 3(a) reports a much lower retained fraction. That")
    print("  study deliberately uses a small 0.6 um box so the finest voxel")
    print("  stays affordable -- its low-k loss is a property of THAT box, not")
    print("  of the production one, and the two figures are not in conflict.")
    print()
    print("  Either way the missing power sits at scales far larger than the")
    print("  antenna, so inside the box it acts as a slowly varying mean index")
    print("  offset -- exactly the quantity stage 7 regresses out.")


def stage_4b_coating(cfg: StudyConfig) -> None:
    """Is the recognition layer where we think it is?"""
    from coating import print_coating_check

    hdr("4b", "THE CONFORMAL SHELLS, MEASURED AGAINST A TRUE SDF")
    print_coating_check(cfg)


def stage_5c_conformational(cfg: StudyConfig) -> None:
    """The two borrowed numbers, swept."""
    from conformational_sweep import print_conformational_sweep

    hdr("5c", "CONFORMATIONAL TERMS -- TRANSFERABILITY, SWEPT")
    print_conformational_sweep(cfg)


def stage_5d_photon_budget(cfg: StudyConfig) -> None:
    """The transport stage, done analytically instead of by Monte Carlo."""
    from photon_budget import print_photon_budget

    hdr("5d", "PHOTON BUDGET -- WHY THERE IS NO MONTE CARLO STAGE")
    print_photon_budget(cfg)


def stage_8b_free_cadmium(cfg: StudyConfig, sigma_nm: float,
                          shift_nm: float) -> None:
    """Bioaccessibility is not the free ionic fraction, and the gap is huge.

    Runs after stage 8 because the sweep needs the resonance precision from
    stage 7 and the shift-per-coverage from stage 5b -- the same two inputs
    the noise floor itself uses, so the two are directly comparable.
    """
    from noise_floor import free_fraction_sweep, print_free_fraction_sweep

    hdr("8b", "FREE IONIC CADMIUM -- THE NUMBER NOBODY HAS MEASURED")
    r = cfg.regulatory
    lo, hi = r.free_fraction_direct_contact_range
    print("  Two regimes, and they are not substitutes for each other:")
    print()
    print(f"    with a digestion / release step   {r.free_fraction:.3f}")
    print("      He, Ke & Wang, J. Agric. Food Chem. 58:3517 (2010),")
    print("      doi:10.1021/jf100227n -- in-vitro digestion bioaccessibility,")
    print("      raw muscle of two farmed marine fish. MEASURED.")
    print()
    print(f"    direct tissue contact            "
          f"{r.free_fraction_direct_contact:.1e}")
    print(f"      plausible span                 {lo:.0e} to {hi:.0e}")
    print("      DERIVED, not measured. Free ionic Cd(II) in fish muscle has")
    print("      never been measured -- not by DGT, ion-selective electrode,")
    print("      Donnan membrane, voltammetry or ultrafiltration. The bound")
    print("      comes from Cd-glutathione stability constants (Watly et al.,")
    print("      Inorg. Chem. 60:4657 (2021), doi:10.1021/acs.inorgchem."
          "0c03639)")
    print("      at a nominal 1 mM cytosolic GSH, with the lower end allowing")
    print("      for cysteine-rich ligands (Quinn & Wilcox, Metallomics")
    print("      16:mfae041 (2024), doi:10.1093/mtomcs/mfae041).")
    print()
    print("  Three to nine orders of magnitude apart. The requirement across")
    print("  the whole range:")
    print()
    required = None
    try:
        rows = free_fraction_sweep(cfg, sigma_nm, shift_nm)
        print_free_fraction_sweep(rows, r.eu_limit_mg_per_kg)
        from noise_floor import required_free_fraction
        required = required_free_fraction(rows)
    except Exception as exc:
        print(f"    (sweep unavailable: {exc})")

    if required is not None and required == required:
        print()
        print("  THE COMPARISON THAT DECIDES THE OPERATING MODE:")
        print(f"    required                       {100 * required:.2f}%")
        print(f"    digestion / release step       "
              f"{100 * r.free_fraction:.1f}%   "
              f"{'CLEARS' if r.free_fraction >= required else 'FAILS'}")
        print(f"    direct contact, best case      "
              f"{100 * hi:.4f}%   "
              f"{'CLEARS' if hi >= required else 'FAILS'}")
        print(f"    direct contact, worst case     "
              f"{100 * lo:.1e}%   "
              f"{'CLEARS' if lo >= required else 'FAILS'}")
        if hi < required:
            print()
            print(f"    Direct tissue contact fails by a factor of at least")
            print(f"    {required / hi:.0f}x, and by up to {required / lo:.0e}x. "
                  f"That is not a")
            print("    margin a better antenna closes -- it is a statement")
            print("    about cadmium speciation in muscle. THIS DEVICE NEEDS A")
            print("    RELEASE STEP.")
    print()
    print("  AND A SECOND REASON THE DIGESTION NUMBER IS THE WRONG ONE HERE.")
    print("  A sensor pressed against intact tissue meets INTERSTITIAL fluid,")
    print("  not cytosol. Every speciation measurement above is cytosolic.")
    print("  Nothing in the literature resolves that, and no model can.")
    print()
    print("  If the device only clears the EU limit at 0.84, its claim is")
    print("  conditional on a release step.")


def stage_6b_paired(cfg: StudyConfig, submit: bool) -> None:
    """Measure the layer response ratio f with the paired bound/unbound protocol.

    NOTE THE SEPARATE GATE.  This stage does NOT submit on a bare `--submit`;
    it needs `--paired` as well, so that `--stage 6 --submit --seeds 1` (a
    one-simulation smoke test) does not also launch the paired trio at the
    fine mesh.
    """
    from paired_run import print_paired_run

    hdr("6b", "PAIRED BOUND/UNBOUND -- LAYER RESPONSE RATIO")
    print_paired_run(cfg, submit=submit)
    if not submit:
        print()
        print("    (this stage did not submit: it needs --paired as well as")
        print("     --submit, so a cheap smoke test cannot trigger it by")
        print("     accident)")


if __name__ == "__main__":
    main()
