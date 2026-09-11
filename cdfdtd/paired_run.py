"""
paired_run.py — the paired protocol: bulk sensitivity and layer response ratio.
==================================================================================

    python paired_run.py                        # build offline, report the cost
    python paired_run.py --boundary pml --submit           # 3 nm layer
    python paired_run.py --redesign --boundary pml --submit  # 7 nm active layer

WHAT IT MEASURES
----------------
The resonance shift produced by a change in the recognition layer index is

    shift  =  delta_n  x  (bulk sensitivity, nm/RIU)  x  f

where f is the layer response relative to the water response. f is a property
of the actual mode of the actual antenna, so it is measured rather than
computed from a formula.

THE MEASUREMENT
---------------
Build the same geometry three times:

    unbound    recognition layer in its unbound state
    bound      recognition layer in its cadmium-bound state (index step dn)
    bulk_hi    unbound layer, background water index raised by 5e-3

Then

    layer shift           =  lambda_bound   - lambda_unbound
    bulk shift            =  lambda_bulk_hi - lambda_unbound
    S_bulk                =  bulk shift / 5e-3
    f                     =  layer shift / (S_bulk * dn)

All three runs share one mesh, one fitter and one geometry, so systematic
errors largely cancel in the ratio. The unbound run doubles as the low point
of the bulk step, so three simulations are enough. Note that the bulk run
raises only the water index; f is therefore a layer-to-water ratio and can
exceed 1 (see `layer_series.py`).

The full-coverage signal does not depend on S_bulk: f * S_bulk * dn_full
reduces to layer shift * (dn_full / dn).

COST
----
These runs use the finest mesh in the convergence series, because they must
resolve a 3 nm shell; `simulation.estimate_campaign_cost` prices them that
way. Results already on disk are reused. Nothing is submitted unless
`--submit` is given.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from config import StudyConfig


@dataclass
class ModeOverlapResult:
    """What the paired runs measured."""

    lambda_unbound_nm: float
    lambda_bound_nm: float
    measured_shift_pm: float

    delta_n_layer: float
    bulk_sensitivity_nm_per_riu: float
    predicted_bulk_shift_pm: float

    mode_overlap: float
    guessed_overlap: float

    fwhm_unbound_nm: float
    fwhm_bound_nm: float
    decay_ok: bool

    @property
    def ratio_to_guess(self) -> float:
        return (self.mode_overlap / self.guessed_overlap
                if self.guessed_overlap else float("nan"))

    def __str__(self) -> str:
        return (
            "MODE OVERLAP, MEASURED\n"
            f"  resonance, unbound layer     {self.lambda_unbound_nm:10.4f} nm\n"
            f"  resonance, bound layer       {self.lambda_bound_nm:10.4f} nm\n"
            f"  measured shift               {self.measured_shift_pm:10.2f} pm\n"
            f"  layer index change           {self.delta_n_layer:10.4e}\n"
            f"  bulk sensitivity             "
            f"{self.bulk_sensitivity_nm_per_riu:10.2f} nm/RIU\n"
            f"  shift if the layer filled the mode  "
            f"{self.predicted_bulk_shift_pm:10.2f} pm\n"
            f"  MODE OVERLAP                 {self.mode_overlap:10.4f}\n"
            f"  prior estimate               {self.guessed_overlap:10.4f}  "
            f"({self.ratio_to_guess:.2f}x)\n"
            f"  linewidths                   "
            f"{self.fwhm_unbound_nm:.3f} / {self.fwhm_bound_nm:.3f} nm\n"
            f"  fields decayed before the run ended: "
            f"{'yes' if self.decay_ok else 'NO -- linewidths are truncation-broadened'}"
        )


def build_pair(cfg: StudyConfig, fine_mesh: bool = True,
               dl_gap_nm: float | None = None) -> dict:
    """
    The four simulations, built offline.

    Two for the layer measurement (unbound / bound) and two for the bulk
    sensitivity (background index n0 and n0 + `d_bulk`).  Returns them
    unsubmitted, with their cost weight, so you can see the bill first.
    """
    from simulation import build_sensor_simulation, simulation_work

    c = cfg
    if dl_gap_nm is not None:
        # Explicit mesh, for the convergence check on f itself.  The metal
        # override scales with it so the whole neighbourhood refines
        # together; see convergence.gap_mesh_study for why that matters.
        c = replace(cfg, fdtd=replace(cfg.fdtd, dl_gap_nm=dl_gap_nm,
                                      dl_metal_nm=dl_gap_nm * 2.0))
    elif fine_mesh:
        dl = min(cfg.fdtd.mesh_study_dl_nm)
        c = replace(cfg, fdtd=replace(cfg.fdtd, dl_gap_nm=dl,
                                      dl_metal_nm=dl * 2.0))

    d_bulk = 0.005      # a 5e-3 index step: big enough to fit, small enough
                        # to stay in the linear regime

    sims = {
        "unbound": build_sensor_simulation(c, realization=None, cd_bound=False),
        "bound": build_sensor_simulation(c, realization=None, cd_bound=True),
    }
    # THE BACKGROUND INDEX MUST BE CHANGED THROUGH `background_n`.
    #
    # Changing `cfg.tissue.n0` has no effect when `realization=None`, because
    # the aqueous background is `water_medium()` at a fixed 1.333. The bulk
    # run would then be identical to its baseline and give a zero shift.
    WATER_N = 1.333
    sims["bulk_lo"] = sims["unbound"]
    sims["bulk_hi"] = build_sensor_simulation(
        c, realization=None, cd_bound=False,
        background_n=WATER_N + d_bulk,
    )

    work = simulation_work(c)["work"]
    base = simulation_work(cfg)["work"]
    return {
        "sims": sims,
        "config": c,
        "d_bulk": d_bulk,
        "n_runs": 3,          # bulk_lo is the same simulation as unbound
        "relative_cost_each": work / base,
        "note": (
            f"3 distinct simulations at {c.fdtd.dl_gap_nm:.1f} nm gap mesh, "
            f"{work / base:.2f}x the ensemble-mesh cost each. The unbound run "
            f"serves as both the layer baseline and the bulk-sensitivity "
            f"low point, so three simulations are needed, not four."
        ),
    }


def run_pair(cfg: StudyConfig, submit: bool = False,
             out_dir: str = "data/paired",
             guessed_overlap: float = 0.15,
             dl_gap_nm: float | None = None) -> ModeOverlapResult | dict:
    """
    Run the pair and compute the mode overlap.

    With `submit=False` this returns the build report and spends nothing.
    """
    import os
    import warnings

    from simulation import run, check_decay, hdf5_looks_complete
    from observables import (hotspot_spectrum, fit_resonance,
                             relative_shift_pm, shift_stability)
    from recognition_layer import estimate_index_change

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = build_pair(cfg, dl_gap_nm=dl_gap_nm)
        est = estimate_index_change(cfg.surface, n_solvent=cfg.tissue.n0)

    if not submit:
        return {"built": built, "delta_n_layer": est.delta_n_with_conformation,
                "submitted": False}

    os.makedirs(out_dir, exist_ok=True)
    fits, decays, spectra = {}, {}, {}
    progress_path = os.path.join(out_dir, "progress.json")
    progress = {}
    for name in ("unbound", "bound", "bulk_hi"):
        path = os.path.join(out_dir, f"{name}.hdf5")
        if hdf5_looks_complete(path):
            # RESUME: the result is already on disk, so load it instead of
            # re-running. A crash after two of the three runs never repeats
            # the first two, and re-analysing an existing trio never
            # re-submits anything.
            #
            # The test is `hdf5_looks_complete`, not `os.path.isfile`: an
            # interrupted download leaves a file that exists but is
            # incomplete, and using it would silently corrupt the result.
            import tidy3d as td
            print(f"  {name}: found {path}, loading instead of submitting.")
            data = td.SimulationData.from_file(path)
        elif os.path.isfile(path):
            raise RuntimeError(
                f"{path} exists but is short or unreadable, probably an "
                f"interrupted download. The simulation may already have "
                f"finished on the server. Recover it with "
                f"`python fetch_task.py <task_id> {path}` using the id from "
                f"the original run's log, or delete the file to run it again."
            )
        else:
            data = run(built["sims"][name], task_name=f"paired_{name}",
                       path=path)
        progress[name] = "done"
        with open(progress_path, "w") as fh:
            import json as _json
            _json.dump(progress, fh)
        wl, inten = hotspot_spectrum(data)
        spectra[name] = (wl, inten)
        fits[name] = fit_resonance(wl, inten)
        decays[name] = check_decay(data)

    wl_ref, y_ref = spectra["unbound"]

    # The shifts are measured by whole-line least squares, NOT by differencing
    # two peak fits.  See `observables.relative_shift_pm` for why: on this data
    # peak-fitting methods can disagree by a factor of forty, because the
    # shift is about six percent of one sample spacing.
    stab_layer = shift_stability(wl_ref, y_ref, spectra["bound"][1])
    stab_bulk = shift_stability(wl_ref, y_ref, spectra["bulk_hi"][1])
    measured_pm = stab_layer["shift_pm"]
    bulk_pm = stab_bulk["shift_pm"]

    if not (stab_layer["ok"] and stab_bulk["ok"]):
        raise RuntimeError(
            "The shifts are not resolved: "
            f"layer {stab_layer['verdict']}, bulk {stab_bulk['verdict']}. "
            "Raise FDTDConfig.freq_points and repeat before quoting anything."
        )
    if abs(bulk_pm) < 1e-6:
        raise RuntimeError(
            "The bulk-sensitivity run produced no shift. The background index "
            "step did not take effect; check that `background_n` reaches "
            "`build_sensor_simulation`."
        )

    lam_u = fits["unbound"].wavelength_um * 1000
    lam_b = lam_u + measured_pm / 1000.0

    # UNITS.  `bulk_pm` is picometres and `d_bulk` is dimensionless, so
    # nm/RIU is bulk_pm/1000 divided by d_bulk.
    s_bulk = (bulk_pm / 1000.0) / built["d_bulk"]          # nm per RIU

    # THE DELTA-N THAT DIVIDES THE SHIFT MUST BE THE ONE THE SIMULATION
    # APPLIED.
    #
    # The overlap is f = shift / (S_bulk * dn_applied), where `dn_applied`
    # is the index step between the bound and unbound runs AS BUILT --
    # readable straight off the config that built them (1.4530 -> 1.4543,
    # a step of 0.0013).  It must NOT be the de Feijter full-coverage
    # estimate, which is a different and much larger number; that enters
    # only when converting f into a full-coverage signal below.
    surf = built["config"].surface
    dn_applied = surf.recognition_n_bound - surf.recognition_n_unbound
    if abs(dn_applied) < 1e-12:
        raise RuntimeError(
            "recognition_n_bound equals recognition_n_unbound in the config "
            "that built these runs; the layer measurement is empty."
        )
    predicted_pm = dn_applied * s_bulk * 1000.0            # pm
    overlap = measured_pm / predicted_pm if predicted_pm else float("nan")

    # The full-coverage physical signal is a SEPARATE quantity: the overlap
    # (geometry) times the de Feijter estimate of the real index change at
    # full cadmium coverage (chemistry).  Report both, never conflate them.
    dn = dn_applied
    signal_full_pm = overlap * s_bulk * est.delta_n_with_conformation * 1000.0
    print(f"  applied layer index step        {dn_applied:10.4e}")
    print(f"  de Feijter full-coverage step   "
          f"{est.delta_n_with_conformation:10.4e}")
    print(f"  full-coverage signal            {signal_full_pm:10.1f} pm")

    if not (0.0 < overlap < 1.0):
        # PRINT EVERY INPUT BEFORE RAISING, so it is clear which of the two
        # measurements is responsible.
        print()
        print("  OVERLAP OUT OF RANGE -- the inputs behind it")
        print(f"    resonance, unbound            {lam_u:10.4f} nm")
        print(f"    layer shift (bound-unbound)   {measured_pm:10.2f} pm")
        print(f"    bulk shift (dn = "
              f"{built['d_bulk']:.3f})       {bulk_pm:10.2f} pm")
        print(f"    implied S_bulk                {s_bulk:10.2f} nm/RIU")
        print(f"    layer index step applied      {dn_applied:10.4e}")
        print(f"    shift the layer WOULD give    {predicted_pm:10.2f} pm"
              "   if it filled the mode")
        print(f"    ratio                         {overlap:10.4f}")
        print()
        print("    Two things can put this above one, and they need different")
        print("    fixes:")
        print("      (a) the layer shift is too large for the geometry, or")
        print("      (b) S_bulk is too small because the bulk run raises only")
        print("          `background_n` and leaves the recognition layer's own")
        print("          medium alone -- in which case this ratio is W_L/W_B,")
        print("          not the fraction W_L, and is not bounded by one.")
        print()
        print("    `layer_series.py` distinguishes them by measuring f at four")
        print("    layer thicknesses. The runs are saved before this error.")
        raise RuntimeError(
            f"Mode overlap came out as {overlap:.3f}, outside (0, 1). The "
            f"inputs are printed above; see layer_series.py, which treats f "
            f"as a layer-to-water ratio."
        )

    return ModeOverlapResult(
        lambda_unbound_nm=lam_u,
        lambda_bound_nm=lam_b,
        measured_shift_pm=measured_pm,
        delta_n_layer=dn,
        bulk_sensitivity_nm_per_riu=s_bulk,
        predicted_bulk_shift_pm=predicted_pm,
        mode_overlap=overlap,
        guessed_overlap=guessed_overlap,
        fwhm_unbound_nm=fits["unbound"].fwhm_um * 1000,
        fwhm_bound_nm=fits["bound"].fwhm_um * 1000,
        decay_ok=all(d.get("ok", False) for d in decays.values()),
    )


def overlap_from_field(cfg: StudyConfig, sim_data,
                       shell_nm: float | None = None) -> dict:
    """
    An offline cross-check on the measured overlap, from a stored run.

    The mode overlap is, by definition, the fraction of the mode's sensing
    weight inside the recognition shell:

        overlap  =  Int_shell  eps |E|^2 dV
                    ------------------------
                    Int_dielectric eps |E|^2 dV

    MONITORS USED
    -------------
    `build_monitors` stores the near field under `simulation.FIELD_MONITOR`
    ("near_field") together with a permittivity map, in every run except the
    light ensemble ones.

    THE METAL IS EXCLUDED, NOT WEIGHTED.
    ------------------------------------
    Inside gold, eps is negative and the plain eps|E|^2 has the wrong sign;
    the correct energy density there needs the Brillouin factor d(w eps)/dw.
    We sidestep it: an index step applied to a dielectric cannot act inside
    the metal, so the metal contributes to neither the numerator nor the
    denominator of the quantity we are checking.  Voxels with Re(eps) <= 0
    are dropped.

    WHAT THIS CANNOT SETTLE
    -----------------------
    The near-field monitor is a small box around the antenna.  Sensing weight
    in water beyond it is missing from the denominator, so this OVERESTIMATES
    the shell fraction.  The bias has a known sign, which is what makes it
    useful: if the integral comes out at or below the smaller candidate
    despite a bias pushing the other way, that is informative.  If it lands
    near the larger one, this test is inconclusive rather than confirming.
    """
    import numpy as np

    from coating import bowtie_sdf_2d
    from simulation import FIELD_MONITOR, EPS_MONITOR

    shell = (shell_nm if shell_nm is not None
             else cfg.surface.recognition_thickness_nm) * 1e-3

    try:
        fld = sim_data[FIELD_MONITOR]
    except Exception as exc:
        return {"ok": False, "reason": f"no {FIELD_MONITOR} monitor: {exc}"}

    try:
        comps, coords = [], None
        for c in ("Ex", "Ey", "Ez"):
            arr = getattr(fld, c, None)
            if arr is None:
                continue
            if coords is None:
                coords = {k: np.array(arr.coords[k]) for k in ("x", "y", "z")}
            a = np.asarray(arr)
            if a.ndim == 4:            # x, y, z, f -> take the middle frequency
                a = a[..., a.shape[-1] // 2]
            comps.append(np.abs(a) ** 2)
        if not comps or coords is None:
            return {"ok": False, "reason": "field monitor carries no E data"}
        energy = np.sum(comps, axis=0)
    except Exception as exc:
        return {"ok": False, "reason": f"could not read the field: {exc}"}

    x, y, z = coords["x"], coords["y"], coords["z"]
    if energy.shape != (x.size, y.size, z.size):
        return {"ok": False,
                "reason": f"field {energy.shape} does not match the grid "
                          f"{(x.size, y.size, z.size)}"}

    X, Y = np.meshgrid(x, y, indexing="ij")
    a2d = bowtie_sdf_2d(cfg.bowtie, X, Y, pad_um=0.0)[:, :, None]
    half_t = cfg.bowtie.thickness_nm * 1e-3 / 2.0
    b1d = np.abs(z)[None, None, :] - half_t

    # THE SHELL WRAPS THE METAL, INCLUDING ITS TOP AND BOTTOM FACES.
    #
    # A mask built from the in-plane distance alone,
    #     (a2d > 0) & (a2d <= shell) & (|z| <= half_t + shell)
    # would keep only the coating around the RIM: every voxel directly above
    # or below the metal has a2d < 0. The two flat faces carry a comparable
    # area, so the full 3-D distance is used instead.
    #
    # The bowtie is a polygon extruded in z, so its true signed distance is
    # the standard extrusion formula built from the in-plane distance `a` and
    # the out-of-plane distance `b`.
    d3d = (np.minimum(np.maximum(a2d, b1d), 0.0)
           + np.sqrt(np.maximum(a2d, 0.0) ** 2 + np.maximum(b1d, 0.0) ** 2))
    in_shell = (d3d > 0) & (d3d <= shell)

    # Drop the metal by permittivity where we have it, by geometry otherwise.
    try:
        eps_arr = sim_data[EPS_MONITOR]
        e = np.asarray(getattr(eps_arr, "eps_xx"))
        if e.ndim == 4:
            e = e[..., 0]
        dielectric = np.real(e) > 0
        eps_source = EPS_MONITOR
    except Exception:
        dielectric = d3d > 0
        eps_source = "geometry (no permittivity monitor)"
    if dielectric.shape != energy.shape:
        dielectric = d3d > 0
        eps_source = "geometry (permittivity grid mismatched)"

    in_shell = in_shell & dielectric
    total = float(energy[dielectric].sum())
    inside = float(energy[in_shell].sum())

    # CAN THIS GRID EVEN SEE THE SHELL?
    #
    # The monitor samples the simulation mesh, which is sub-nanometre in the
    # gap and tens of nanometres in the bulk.  A 3 nm shell resolved by three
    # voxels near the hot spot and by a fraction of one voxel around the rim
    # is not one measurement -- it is two, averaged without saying so.  A
    # coarse cell straddling the shell is classified by its centre, which
    # loses the shell rather than smearing it, so the fraction comes out low
    # for a reason that has nothing to do with the mode.  The spacing is
    # reported so the number can be weighted accordingly.
    def _spacing(v):
        return np.diff(v) if v.size > 1 else np.array([np.nan])
    dx, dy, dz = _spacing(x), _spacing(y), _spacing(z)
    finest = float(np.nanmin([dx.min(), dy.min(), dz.min()]))
    coarsest = float(np.nanmax([dx.max(), dy.max(), dz.max()]))

    return {
        "ok": True,
        "overlap_energy_fraction": inside / total if total > 0 else float("nan"),
        "shell_nm": shell * 1000,
        "shell_voxels": int(in_shell.sum()),
        "dielectric_voxels": int(dielectric.sum()),
        "grid_finest_nm": finest * 1000,
        "grid_coarsest_nm": coarsest * 1000,
        "voxels_across_shell_finest": shell / finest if finest > 0 else 0.0,
        "voxels_across_shell_coarsest": (shell / coarsest
                                         if coarsest > 0 else 0.0),
        "metal_excluded_by": eps_source,
        "note": ("|E|^2 over a finite near-field box, metal excluded, shell "
                 "taken as the full 3-D wrap. The finite box biases this "
                 "UPWARD, so a low value is the informative outcome."),
    }


def print_paired_run(cfg: StudyConfig, submit: bool = False,
                     guessed_overlap: float = 0.15,
                     out_dir: str = "data/paired",
                     dl_gap_nm: float | None = None) -> None:
    """The plan, or the result."""
    res = run_pair(cfg, submit=submit, guessed_overlap=guessed_overlap,
                   out_dir=out_dir, dl_gap_nm=dl_gap_nm)

    if isinstance(res, dict):
        b = res["built"]
        print("  The mode-overlap factor scales the final answer linearly.")
        print()
        print(f"    layer index change on binding   {res['delta_n_layer']:.4e}")
        print(f"    prior estimate of the overlap   {guessed_overlap:.3f}")
        print()
        for line in _wrap(b["note"], 68):
            print(f"    {line}")
        print()
        print("    Run it with --submit, before the tissue ensemble: the")
        print("    ensemble measures the spread around the central value this")
        print("    pair fixes.")
        return

    print(str(res).replace("\n", "\n  "))
    print()
    r = res.ratio_to_guess
    if not np.isfinite(r):
        print("  The measurement did not produce a finite ratio. Check that")
        print("  both resonances were actually found inside the band.")
    elif 0.7 <= r <= 1.4:
        print(f"  The measured overlap is {r:.2f}x the prior estimate.")
    else:
        print(f"  The measured overlap is {r:.2f}x the prior estimate. Every")
        print("  number downstream of stage 5 scales with this factor, so")
        print("  re-run stages 5 and 8 with the measured value.")
    if not res.decay_ok:
        print()
        print("  WARNING: the fields had not decayed when a run ended, so the")
        print("  linewidths above are truncation-broadened. Raise")
        print("  FDTDConfig.run_time_ps or expected_q and repeat.")


def _wrap(text: str, width: int = 68) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# ==========================================================================
#  The 7 nm variant: recognition chemistry through the whole brush
# ==========================================================================
#: Same geometry as the baseline, but the recognition chemistry occupies the
#: full 7 nm stack instead of a 3 nm shell on a 4 nm inactive brush.  Same
#: fabrication, same band, one layer change.
REDESIGN_ACTIVE_LAYER_NM = 7.0


def redesign_config(cfg: StudyConfig) -> StudyConfig:
    """The brush-chemistry variant: unchanged geometry, one 7 nm active
    layer in place of the 3 nm recognition shell plus 4 nm dead brush."""
    return replace(
        cfg,
        surface=replace(cfg.surface,
                        recognition_thickness_nm=REDESIGN_ACTIVE_LAYER_NM,
                        fouling_thickness_nm=0.0),
    )


def print_redesign_preflight(cfg: StudyConfig) -> bool:
    """
    Offline checks on the 7 nm variant before anything is submitted.

    Returns True only if every gate passes.  `--submit` refuses to run when
    any gate fails.
    """
    import numpy as np
    from simulation import (build_sensor_simulation, report,
                            metal_present_check, simulation_work)

    c = redesign_config(cfg)
    half = np.radians(c.bowtie.apex_angle_deg) / 2.0
    setback = c.bowtie.tip_radius_nm * (1.0 / np.sin(half) - 1.0)
    physical = c.bowtie.gap_nm + 2.0 * setback

    print("  REDESIGN PREFLIGHT -- all of this is free")
    print(f"    tip radius                  {c.bowtie.tip_radius_nm:.1f} nm")
    print(f"    nominal gap (config)        {c.bowtie.gap_nm:.1f} nm")
    print(f"    PHYSICAL tip-to-tip gap     {physical:.1f} nm")
    print(f"    active layer                "
          f"{c.surface.recognition_thickness_nm:.1f} nm "
          "(chemistry through the brush, no separate fouling shell)")
    print(f"    band                        {c.calibration.band_um} um")
    print()

    ok = True

    # Gate 1: the active layer is what was asked for, and the geometry is
    # UNCHANGED from the baseline -- that is what makes the comparison
    # clean and the decay-length prediction testable.
    if abs(c.surface.recognition_thickness_nm - REDESIGN_ACTIVE_LAYER_NM) > 1e-9:
        ok = False
        print("    FAIL: active layer thickness does not match the target.")
    if (c.bowtie.gap_nm != cfg.bowtie.gap_nm
            or c.bowtie.tip_radius_nm != cfg.bowtie.tip_radius_nm):
        ok = False
        print("    FAIL: the geometry changed. The brush trio must run on")
        print("    the same geometry as the baseline run.")

    # Gate 2: the mesh actually resolves the gap.  A requested cell size is
    # only a request (see convergence.gap_mesh_study).
    built = build_pair(c)
    sim = built["sims"]["unbound"]
    rep = report(sim)
    cells_across = physical / rep.min_step_nm
    layer_cells = c.surface.recognition_thickness_nm / rep.min_step_nm
    print(f"    achieved minimum cell       {rep.min_step_nm:.3f} nm")
    print(f"    cells across the gap        {cells_across:.1f}")
    print(f"    cells across the layer      {layer_cells:.1f}")
    if layer_cells < 3.0:
        print("    WARNING: fewer than 3 cells across the active layer. The")
        print("    overlap is measured FROM that layer, so a thin mesh biases")
        print("    it directly.")
    if cells_across < 4.0:
        ok = False
        print("    FAIL: fewer than 4 cells across the physical gap, so the")
        print("    simulated geometry would not match the requested one.")
        print("    Lower FDTDConfig.mesh_study_dl_nm and re-check.")

    # Gate 3: the metal actually exists in the built simulation (structure
    # ordering can otherwise override the gold with a coating medium).
    mp = metal_present_check(c, sim=sim)
    if not mp.get("ok", False):
        ok = False
        print(f"    FAIL: metal check: {mp}")
    else:
        print("    metal present in the gap    yes")

    # Gate 4: cost, stated before anyone agrees to it.
    rel = built["relative_cost_each"]
    print(f"    cost per run                {rel:.2f}x the ensemble mesh")
    print(f"    runs                        3 "
          "(unbound doubles as the bulk low point)")
    print()
    if ok:
        print("    ALL GATES PASS. Run again with --submit to spend.")
    else:
        print("    A GATE FAILED. --submit will refuse until it passes.")
    return ok


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(
        description="The paired trio: bulk sensitivity and mode overlap "
                    "from three simulations. Free without --submit.")
    ap.add_argument("--submit", action="store_true",
                    help="Run on the Tidy3D cloud solver (uses credits), "
                         "except for runs whose .hdf5 already exists in the "
                         "output directory, which are loaded instead.")
    ap.add_argument("--redesign", action="store_true",
                    help="7 nm variant: recognition chemistry through the "
                         "full 7 nm active layer on the UNCHANGED geometry.")
    ap.add_argument("--mesh", type=float, default=None,
                    help="Gap mesh in nm for this trio, overriding the "
                         "config default. Use it to test whether the mode "
                         "overlap is mesh converged: the default trio uses "
                         "1.0 nm, so --mesh 2.0 gives a second point at "
                         "about half the cost.")
    ap.add_argument("--boundary", choices=["absorber", "pml"], default=None,
                    help="Override fdtd.boundary for this trio. Use "
                         "--boundary pml (used for all values in the paper); "
                         "the adiabatic absorber does not converge in "
                         "boundary_test.py. Output goes to a separate "
                         "directory for each boundary.")
    ap.add_argument("--out", default=None,
                    help="Output directory. Defaults to data/paired, or "
                         "data/paired_redesign with --redesign, with the "
                         "boundary appended when --boundary is given.")
    args = ap.parse_args()

    if args.submit:
        from simulation import require_tidy3d
        require_tidy3d()

    base = StudyConfig()
    if args.boundary and args.boundary != base.fdtd.boundary:
        from dataclasses import replace as _replace

        # A DIFFERENT BOUNDARY IS A DIFFERENT RUN, SO IT GETS ITS OWN
        # DIRECTORY.  run_pair loads any .hdf5 already present instead of
        # resubmitting, so sharing a directory would silently return
        # absorber results for a PML request.
        print(f"  boundary overridden to {args.boundary} "
              f"(config default is {base.fdtd.boundary})")
        base = _replace(base, fdtd=_replace(base.fdtd,
                                            boundary=args.boundary))
    _bsuffix = f"_{args.boundary}" if args.boundary else ""

    if args.redesign:
        out_dir = args.out or os.path.join("data",
                                           "paired_redesign" + _bsuffix)
        gates_ok = print_redesign_preflight(base)
        if args.submit and not gates_ok:
            raise SystemExit("Refusing to submit: a preflight gate failed.")
        if args.submit:
            print()
            res = run_pair(redesign_config(base), submit=True,
                           out_dir=out_dir, dl_gap_nm=args.mesh)
            print(str(res).replace("\n", "\n  "))
    else:
        suffix = f"_mesh{args.mesh:g}" if args.mesh else ""
        out_dir = args.out or os.path.join(
            "data", f"paired{suffix}{_bsuffix}")
        if args.mesh:
            print(f"  Gap mesh overridden to {args.mesh:g} nm. Output goes to "
                  f"{out_dir}, so the default 1.0 nm trio is not "
                  "overwritten.")
        print_paired_run(base, submit=args.submit, out_dir=out_dir,
                         dl_gap_nm=args.mesh)
