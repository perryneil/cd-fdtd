"""
ensemble.py — running many tissue realizations and analysing the spread.
==========================================================================

WHY MANY RUNS
-------------
Every fish is different, and every patch of the same fish is different.  One
simulation tells you what happens in one particular random tissue, which is
not what anybody wants to know.

So: generate many statistically identical but individually different tissues,
run each one, and look at how much the answer moves.  With NO cadmium present
anywhere, that movement is pure noise imposed by the tissue.  If it is larger
than the signal a regulated amount of cadmium would produce, the design cannot
work.  That is the paper's central claim, and this module produces the number
behind it.

A FINITE-BOX EFFECT THAT MUST BE REMOVED
----------------------------------------
Each simulation box is finite.  A finite box of
random material has its own average refractive index, and that average drifts
from one random draw to the next — smaller boxes drift more, larger boxes
less.  It is a pure finite-size effect.

A plasmonic resonance moves when the average index around it moves.  That is
ordinary, well-understood bulk sensitivity — it is what every homogeneous
model already predicts, and it is not "heterogeneity" in any interesting sense.

So if you simply measure the spread of resonance wavelengths across your
ensemble, a large part of what you measure is a statement about the box size
you happened to choose, not a heterogeneity-induced noise floor.

THE FIX
-------
For every realisation, record two things: the resonance wavelength, and the
average refractive index the mode actually samples.  Then fit a straight line
of resonance against average index.

  * The SLOPE is the ordinary bulk sensitivity.  Interesting as a check --
    it should agree with what a homogeneous model predicts -- but not new.

  * The SCATTER ABOUT THE LINE is what is left when bulk drift is removed.
    THAT is genuine near-field speckle: the resonance moving because the
    tissue's fine structure rearranges the hot spot, not because the average
    went up or down.

The residual is the number that should be propagated to a noise floor.  This
module reports both, plus the fraction of the raw variance that was bulk
drift, so the separation can be shown explicitly.

HOW MANY SEEDS
--------------
Estimating a spread is much harder than estimating an average.  The relative
error on an estimated standard deviation is about 1/sqrt(2(N-1)):

        N = 25   ->  14%
        N = 50   ->  10%
        N = 100  ->   7%

Since the noise floor is proportional to that spread, N = 25 gives a headline
number with a 14% error bar.  This module also computes a bootstrap confidence
interval, which costs nothing and is far more informative than watching a
running variance "settle" -- running variances settle deceptively.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from config import StudyConfig
from tissue import generate_tissue
from observables import fit_resonance, hotspot_spectrum
from simulation import build_sensor_simulation, check_decay


# ==========================================================================
#  One realisation's worth of results
# ==========================================================================
@dataclass
class SeedResult:
    """What we learn from a single random tissue."""

    seed: int
    resonance_um: float
    fwhm_um: float
    mean_index_local: float
    """Average refractive index in the region the mode samples.  This is the
    regressor that separates bulk drift from speckle."""

    mean_index_box: float
    decay_ok: bool
    fit_ok: bool
    note: str = ""


# ==========================================================================
#  The ensemble
# ==========================================================================
@dataclass
class EnsembleResult:
    """Summary statistics of one ensemble."""

    seeds: list[int]
    resonances_nm: np.ndarray
    fwhm_nm: np.ndarray
    mean_index: np.ndarray

    # --- raw spread -----------------------------------------------------
    sigma_raw_nm: float
    """Standard deviation of the resonance wavelength across the ensemble.
    DO NOT propagate this to a noise floor -- part of it is finite-box
    artefact."""

    # --- after removing bulk index drift --------------------------------
    bulk_sensitivity_nm_per_riu: float
    """Slope of resonance against average index.  A sanity check: it should
    match what a homogeneous model gives for the same geometry."""

    sigma_speckle_nm: float
    """*** THE NUMBER TO USE. ***  Scatter about the regression line: the
    resonance wobble that remains once ordinary bulk-index drift is accounted
    for.  This is genuine near-field speckle."""

    bulk_variance_fraction: float
    """What fraction of the raw variance was just finite-box index drift.  If
    this is large, quoting sigma_raw would have been mostly a statement about
    your box size."""

    # --- uncertainty on the uncertainty ---------------------------------
    sigma_speckle_ci95: tuple[float, float]
    """Bootstrap 95% confidence interval on sigma_speckle."""

    n_used: int
    n_discarded: int
    warnings_: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lo, hi = self.sigma_speckle_ci95
        lines = [
            "ENSEMBLE RESULT",
            "=" * 66,
            f"  realisations used            {self.n_used}"
            + (f"   ({self.n_discarded} discarded)" if self.n_discarded else ""),
            f"  mean resonance               "
            f"{np.mean(self.resonances_nm):.3f} nm",
            f"  mean linewidth               {np.mean(self.fwhm_nm):.3f} nm",
            "",
            f"  raw spread sigma_raw         "
            f"{self.sigma_raw_nm * 1000:.2f} pm   <-- do NOT use this",
            f"  bulk index sensitivity       "
            f"{self.bulk_sensitivity_nm_per_riu:.1f} nm/RIU",
            f"  of the raw variance,         "
            f"{100 * self.bulk_variance_fraction:.1f}% was finite-box "
            f"index drift",
            "",
            f"  SPECKLE RESIDUAL             "
            f"{self.sigma_speckle_nm * 1000:.2f} pm   <-- use this one",
            f"    bootstrap 95% CI           "
            f"[{lo * 1000:.2f}, {hi * 1000:.2f}] pm",
            f"    relative width of CI       "
            f"{100 * (hi - lo) / (2 * self.sigma_speckle_nm):.0f}%",
        ]
        if self.warnings_:
            lines += ["", "  WARNINGS:"] + [f"    - {w}" for w in self.warnings_]
        return "\n".join(lines)


def analyse_ensemble(
    results: list[SeedResult],
    regress_out_mean_index: bool = True,
    bootstrap_resamples: int = 5000,
    rng_seed: int = 0,
) -> EnsembleResult:
    """
    Turn a list of per-seed results into the statistics that matter.

    This is pure post-processing on numbers you already have, so it is cheap
    and can be re-run with different choices without re-simulating anything.
    """
    good = [r for r in results if r.fit_ok and np.isfinite(r.resonance_um)]
    discarded = len(results) - len(good)
    warnings_: list[str] = []

    if discarded:
        warnings_.append(
            f"{discarded} realisation(s) discarded because the resonance fit "
            f"failed.  Check that the band covers the resonance."
        )
    n_not_decayed = sum(1 for r in good if not r.decay_ok)
    if n_not_decayed:
        warnings_.append(
            f"{n_not_decayed} realisation(s) had fields still ringing when "
            f"the run ended.  Their linewidths are truncation-broadened; "
            f"increase fdtd.run_time_ps."
        )

    if len(good) < 3:
        raise ValueError(
            f"Only {len(good)} usable realisations; cannot estimate a spread."
        )

    lam = np.array([r.resonance_um * 1000 for r in good])      # nm
    fwhm = np.array([r.fwhm_um * 1000 for r in good])
    nbar = np.array([r.mean_index_local for r in good])

    sigma_raw = float(np.std(lam, ddof=1))

    # --- the regression that separates bulk drift from speckle ----------
    if regress_out_mean_index and np.std(nbar) > 1e-12:
        slope, intercept = np.polyfit(nbar, lam, 1)
        residual = lam - (slope * nbar + intercept)
        sigma_speckle = float(np.std(residual, ddof=2))
        bulk_fraction = float(
            max(0.0, 1.0 - (sigma_speckle ** 2) / (sigma_raw ** 2))
        ) if sigma_raw > 0 else 0.0
    else:
        slope = float("nan")
        residual = lam - lam.mean()
        sigma_speckle = sigma_raw
        bulk_fraction = 0.0
        if regress_out_mean_index:
            warnings_.append(
                "The mean index barely varied across realisations, so bulk "
                "drift could not be regressed out.  Either the box is large "
                "enough that drift is negligible (good) or the mean index was "
                "not recorded correctly (check that)."
            )

    if bulk_fraction > 0.5:
        warnings_.append(
            f"{100 * bulk_fraction:.0f}% of the raw variance was finite-box "
            f"index drift.  Quoting sigma_raw would have been mostly a "
            f"statement about your box size.  Consider a larger supercell and "
            f"show the residual is stable as it grows."
        )

    # --- bootstrap confidence interval on the residual spread -----------
    rng = np.random.default_rng(rng_seed)
    n = len(residual)
    boots = np.empty(bootstrap_resamples)
    for b in range(bootstrap_resamples):
        idx = rng.integers(0, n, n)
        boots[b] = np.std(residual[idx], ddof=1)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    rel_ci = (ci[1] - ci[0]) / (2 * sigma_speckle) if sigma_speckle > 0 else 0
    if rel_ci > 0.30:
        warnings_.append(
            f"The confidence interval on the speckle spread is +/-"
            f"{100 * rel_ci / 2:.0f}%.  If the noise floor lands near the "
            f"regulatory limit, this interval decides the answer -- run more "
            f"seeds."
        )

    return EnsembleResult(
        seeds=[r.seed for r in good],
        resonances_nm=lam, fwhm_nm=fwhm, mean_index=nbar,
        sigma_raw_nm=sigma_raw,
        bulk_sensitivity_nm_per_riu=float(slope),
        sigma_speckle_nm=sigma_speckle,
        bulk_variance_fraction=bulk_fraction,
        sigma_speckle_ci95=ci,
        n_used=len(good), n_discarded=discarded,
        warnings_=warnings_,
    )


# ==========================================================================
#  Running the ensemble
# ==========================================================================
def seed_list(cfg: StudyConfig) -> list[int]:
    """Derive every realisation's seed from one master integer, so the whole
    ensemble is reproducible from a single number in the config."""
    rng = np.random.default_rng(cfg.ensemble.base_seed)
    return [int(s) for s in rng.integers(0, 2 ** 31 - 1, cfg.ensemble.n_seeds)]


def local_index_region(cfg: StudyConfig) -> tuple[tuple, tuple, tuple]:
    """
    The box over which the 'average index the mode samples' is computed.

    Chosen to be a few times the near-field decay length around the gap: big
    enough that it is not dominated by a handful of voxels, small enough that
    it is genuinely what the mode sees rather than the whole simulation box.
    """
    from config import NM
    gap = cfg.bowtie.gap_nm * NM
    span = max(4 * gap, 0.10)             # at least 100 nm
    return ((-span, span), (-span, span), (-span, span))


def iter_ensemble_simulations(cfg: StudyConfig):
    """
    Yield one (seed, realization, simulation) at a time.

    WHY A GENERATOR AND NOT A LIST
    ------------------------------
    Each tissue realisation is a three-dimensional array of refractive index.
    At the production voxel size that is a couple of hundred megabytes, and the
    Simulation object holds another copy inside its CustomMedium.  Building
    forty of them at once needs many gigabytes and will simply be killed by the
    operating system.

    Yielding them one at a time means only one realisation is ever alive.  The
    cost is that you cannot hand the whole set to `web.Batch` for parallel
    submission -- see `run_ensemble` for how that trade-off is handled.
    """
    for seed in seed_list(cfg):
        real = generate_tissue(cfg.tissue, cfg.fdtd, seed=seed)
        yield seed, real, build_sensor_simulation(
            cfg, realization=real, light_monitors=True)


def run_ensemble(
    cfg: StudyConfig,
    submit: bool = False,
    out_dir: str = "data/ensemble",
    verbose: bool = True,
    peek: int = 1,
    keep_raw: bool = True,
    resume: bool = True,
    results_path: str | None = None,
) -> tuple[list[SeedResult], list]:
    """
    Build (and optionally run) one simulation per random tissue.

    OFFLINE BY DEFAULT
    ------------------
    With `submit=False` this builds only the first `peek` simulations -- enough
    to report cell counts and confirm the ensemble is well posed -- and does
    not contact the cloud or spend anything.  It deliberately does NOT build
    all of them, because holding forty three-dimensional index maps in memory
    at once will exhaust the machine.

    With `submit=True` it walks the ensemble one realisation at a time,
    uploading and solving each before moving on.  Memory stays flat.  The cost
    is that runs are sequential rather than batched; if you have the memory and
    want parallel submission, collect `iter_ensemble_simulations` into a list
    yourself and hand it to `tidy3d.web.Batch`.

    An ensemble is `n_seeds` times the cost of one simulation.  Estimate first.

    DISK
    ----
    Each saved `.hdf5` carries the whole `Simulation` object, and for a tissue
    run that includes the CUSTOM MEDIUM -- a 500 x 500 x 250 index array, about
    250 MB on its own (roughly 500 MB per file in total).  Tidy3D's data-size
    estimate covers monitor output only, not the medium stored with it.  A
    run without tissue is a few MB.

    RESUMING
    --------
    A forty-seed run takes hours, so the loop is built to survive
    interruption:

      * `resume=True` (default) skips any seed whose `.hdf5` is already on
        disk, re-reading it instead of re-running it.
      * the running record is written to `results_path` after EVERY seed, so
        an interrupted run keeps everything it has done.

    Twenty-two seeds is therefore roughly twelve gigabytes.  `keep_raw=False`
    extracts the numbers and deletes each file once it has been read, which
    costs nothing scientific -- the per-seed record in `seed_results.json` is
    what the analysis uses -- but does mean the spectra cannot be re-fitted
    later without re-running.  Keep them if you have the room.

    Returns
    -------
    (results, sample_simulations)
        `results` is empty when `submit=False`.
    """
    os.makedirs(out_dir, exist_ok=True)
    region = local_index_region(cfg)
    seeds = seed_list(cfg)

    if not submit:
        sample = []
        for i, (_, _, sim) in enumerate(iter_ensemble_simulations(cfg)):
            sample.append(sim)
            if i + 1 >= max(1, peek):
                break
        if verbose:
            print(f"  {len(seeds)} realisations configured; "
                  f"{len(sample)} built as a sample.")
            print("  Nothing uploaded.  Pass submit=True to run them all.")
        return [], sample

    import gc

    from simulation import run as run_sim

    results: list[SeedResult] = []
    if results_path is None:
        results_path = os.path.join(out_dir, "seed_results.json")

    for i, (seed, real, sim) in enumerate(iter_ensemble_simulations(cfg)):
        raw_path = os.path.join(out_dir, f"seed_{seed}.hdf5")
        # `hdf5_looks_complete`, not `os.path.exists`: a half-downloaded file
        # exists but is incomplete, and reusing it would corrupt the ensemble.
        from simulation import hdf5_looks_complete
        reused = resume and hdf5_looks_complete(raw_path)
        if resume and not reused and os.path.exists(raw_path):
            print(f"    {raw_path} is short or unreadable (dropped "
                  f"download); it will be re-run. It can be recovered with "
                  f"fetch_task.py if you have the task id.")
        if verbose:
            print(f"  [{i + 1}/{len(seeds)}] seed {seed} "
                  f"{'(on disk, re-reading)' if reused else '...'}", flush=True)

        mean_local = real.mean_index_in_region(*region)
        mean_box = real.mean_index
        del real                       # free the index map before solving

        data = None
        try:
            if reused:
                import tidy3d as td
                data = td.SimulationData.from_file(raw_path)
            else:
                data = run_sim(sim, task_name=f"cd_fdtd_seed_{seed}",
                               path=raw_path, verbose=False)
            wl, intensity = hotspot_spectrum(data)
            fit = fit_resonance(wl, intensity)
            decay = check_decay(data)
            results.append(SeedResult(
                seed=seed,
                resonance_um=fit.wavelength_um,
                fwhm_um=fit.fwhm_um,
                mean_index_local=mean_local,
                mean_index_box=mean_box,
                decay_ok=bool(decay.get("ok", False)),
                fit_ok=bool(np.isfinite(fit.wavelength_um)
                            and np.isfinite(fit.fwhm_um)
                            and fit.fwhm_um > 0
                            and fit.converged),
                note=fit.warning,
            ))
        except Exception as exc:
            results.append(SeedResult(
                seed=seed, resonance_um=float("nan"), fwhm_um=float("nan"),
                mean_index_local=mean_local, mean_index_box=mean_box,
                decay_ok=False, fit_ok=False, note=str(exc),
            ))

        # SAVE AFTER EVERY SEED, so a crash never loses completed results.
        try:
            save_seed_results(results, results_path)
        except OSError:
            pass

        if not keep_raw and not reused:
            try:
                os.remove(raw_path)
            except OSError:
                pass

        # MEMORY.  Each iteration allocates a 62.5-million-voxel index array
        # for the tissue AND loads back a SimulationData that carries the same
        # array inside it -- about half a gigabyte apiece.  Without dropping
        # both references explicitly and collecting, the peak climbs until the
        # OS kills the process.
        del data, sim
        gc.collect()

    return results, []


def print_seed_results(results: list, cfg=None) -> dict:
    """
    Report what came back from a run.

    Prints, for every seed, whether the fields decayed, whether the resonance
    fit converged, the resonance and the linewidth, and grades the two things
    that decide whether a larger campaign is worth running: field decay and a
    measurable linewidth inside the band.  Use it after a one-simulation
    smoke test before launching the full ensemble.
    """
    import numpy as _np

    if not results:
        return {"ok": False, "reason": "no results"}

    lo, hi = (cfg.calibration.band_um if cfg is not None else (0.5, 0.9))
    print(f"  {'seed':>12}  {'resonance':>10}  {'FWHM':>9}  {'Q':>6}  "
          f"{'decay':>6}  {'fit':>5}  {'in band':>8}")

    # A run that never reached the solver (for example a lost network
    # connection) is not a physics failure, and must not be reported as
    # undecayed fields or a missing linewidth.
    def _never_ran(r):
        note = (r.note or "").lower()
        return any(k in note for k in ("connection", "timed out", "timeout",
                                       "network", "http", "giving up"))

    failed_runs = [r for r in results if _never_ran(r)]
    physics = [r for r in results if not _never_ran(r)]

    n_ok = 0
    for r in results:
        wl = r.resonance_um
        fw = r.fwhm_um
        q = (wl / fw) if (fw and _np.isfinite(fw) and fw > 0) else float("nan")
        in_band = bool(_np.isfinite(wl) and lo <= wl <= hi)
        has_width = bool(_np.isfinite(fw) and fw > 0)
        good = r.decay_ok and r.fit_ok and in_band and has_width
        n_ok += bool(good)
        wl_s = f"{wl * 1000:9.3f}n" if _np.isfinite(wl) else "        -"
        fw_s = f"{fw * 1000:8.3f}n" if _np.isfinite(fw) else "       -"
        q_s = f"{q:6.1f}" if _np.isfinite(q) else "     -"
        print(f"  {r.seed:>12}  {wl_s}  {fw_s}  {q_s}  "
              f"{'ok' if r.decay_ok else 'NO':>6}  "
              f"{'ok' if r.fit_ok else 'NO':>5}  "
              f"{'yes' if in_band else 'NO':>8}")
        if r.note:
            print(f"                note: {r.note[:60]}")

    print()
    if failed_runs:
        print(f"  {len(failed_runs)} of {len(results)} run(s) NEVER REACHED THE "
              f"SOLVER:")
        for r in failed_runs:
            print(f"    seed {r.seed}: {r.note[:60]}")
        print("  That is an infrastructure failure, not a physics one. Nothing")
        print("  was solved for those seeds, and re-running picks them up")
        print("  automatically -- everything already on disk is re-read.")
        print()

    fails = []
    if any(not r.decay_ok for r in physics):
        fails.append(
            "THE FIELDS HAD NOT DECAYED when at least one run ended. "
            "Truncating a still-ringing mode broadens its line, so the "
            "linewidths above are too WIDE, the Q values too LOW, and every "
            "figure of merit downstream too PESSIMISTIC -- the error is in "
            "the safe direction, but it is still an error and it scales the "
            "detection limit. Raise FDTDConfig.run_time_ps or expected_q and "
            "repeat the smoke test BEFORE running the ensemble."
        )
    no_width = [r for r in physics
                if not (_np.isfinite(r.fwhm_um) and r.fwhm_um > 0)]
    if no_width:
        fails.append(
            "NO LINEWIDTH was measurable on at least one run: the spectrum "
            "never falls to half its peak on both sides inside the recorded "
            "band. The peak position alone is not enough -- every figure of "
            "merit in this study is a shift DIVIDED BY a linewidth, and "
            "stage 8's noise floor needs one too. More seeds would only "
            "give more runs with no linewidth. Either widen "
            "CalibrationConfig.band_um so both half-maximum crossings fall "
            "inside it, or retune the antenna so the resonance sits nearer "
            "the middle of the band."
        )
    elif any(not r.fit_ok for r in physics):
        fails.append(
            "The resonance fit did not converge on at least one run. Look at "
            "the spectrum before running anything more."
        )
    out_of_band = [r for r in physics
                   if _np.isfinite(r.resonance_um)
                   and not (lo <= r.resonance_um <= hi)]
    if out_of_band:
        fails.append(
            f"A resonance fell OUTSIDE the {lo * 1000:.0f}-{hi * 1000:.0f} nm "
            f"recorded band. The fitter is then reporting an edge artefact, "
            f"not a mode. Retune the antenna or widen "
            f"CalibrationConfig.band_um."
        )

    qs = [r.resonance_um / r.fwhm_um for r in physics
          if _np.isfinite(r.fwhm_um) and r.fwhm_um > 0]
    q_max = max(qs) if qs else float("nan")
    if cfg is not None and _np.isfinite(q_max):
        supported = cfg.fdtd.expected_q * 1.45
        if q_max > supported:
            fails.append(
                f"The measured Q of {q_max:.0f} exceeds what the configured "
                f"run time supports (about {supported:.0f}). The mode is "
                f"still ringing at shutoff, so the linewidth is truncated. "
                f"Raise FDTDConfig.expected_q to at least {q_max * 1.5:.0f} "
                f"and repeat."
            )

    if fails:
        print("  *** THE PHYSICS OF AT LEAST ONE RUN IS SUSPECT. ***")
        for f in fails:
            for line in _wrap_lines(f, 68):
                print(f"    {line}")
            print()
    else:
        print(f"  ALL CLEAR on {n_ok}/{len(results)} run(s): fields decayed, "
              f"the fit converged with a")
        print("  measurable linewidth, and the resonance sits inside the "
              "recorded band.")
        if cfg is not None and _np.isfinite(q_max):
            print(f"  Measured Q is {q_max:.1f} against a configured ceiling "
                  f"of about {cfg.fdtd.expected_q * 1.45:.0f}.")
        print("  This is what the smoke test existed to establish. Proceed.")

    return {"ok": not fails, "n_ok": n_ok, "n": len(results),
            "q_max": q_max, "failures": fails}


def _wrap_lines(text: str, width: int = 68) -> list:
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


def save_seed_results(results: list, path: str) -> None:
    """
    Write the FULL per-seed record.

    The `.npy` file holds seed, resonance, linewidth and local mean index only.
    The JSON written alongside also keeps `decay_ok`, `fit_ok` and `note`, the
    fields that say whether the other values mean anything, so a run can be
    re-read completely.
    """
    import json as _json

    payload = [
        {"seed": int(r.seed),
         "resonance_um": float(r.resonance_um),
         "fwhm_um": float(r.fwhm_um),
         "mean_index_local": float(r.mean_index_local),
         "mean_index_box": float(r.mean_index_box),
         "decay_ok": bool(r.decay_ok),
         "fit_ok": bool(r.fit_ok),
         "note": str(r.note)}
        for r in results
    ]
    with open(path, "w") as fh:
        _json.dump(payload, fh, indent=2)


# ==========================================================================
#  Self-check: a synthetic ensemble, so the statistics can be tested
#  offline
# ==========================================================================
def production_box_drift(
    cfg,
    n_seeds: int = 16,
    voxel_nm: float = 60.0,
) -> dict:
    """
    The finite-box mean-index drift of the PRODUCTION box, measured.

    WHY
    ---
    `_synthetic_ensemble` needs a drift amplitude.  The supercell study in
    `convergence.py` measures roughly 1e-3 at a 4 um box, about five times the
    2e-4 self-test default, so the demonstration in stage 7 uses this measured
    value instead.

    A demonstration that the regression can recover speckle from weak drift
    does not establish that it can recover it from strong drift.  Since the
    whole point of stage 7 is to verify the method BEFORE real data exists,
    the verification has to use the drift the real data will actually have.

    WHY A COARSE VOXEL IS NOT A COMPROMISE HERE
    -------------------------------------------
    Mean-index drift is a purely LOW-SPATIAL-FREQUENCY quantity: it is the
    box average, which is the k = 0 component of the field.  Voxels only
    control the high-k end.  So a 60 nm voxel measures this just as well as a
    10 nm one and is over two hundred times cheaper -- seconds instead of many
    minutes, which is why this can run inside every stage 7 rather than being
    a study you remember to do separately.

    `oversample=2.0` is essential and not a detail: FFT synthesis on a
    periodic box forces every realisation to share the same mean, so without
    generating large and cropping, this function would measure exactly zero.
    """
    from tissue import generate_tissue

    means = []
    for i in range(n_seeds):
        r = generate_tissue(
            cfg.tissue, cfg.fdtd, seed=10_000 + i,
            domain_um=cfg.fdtd.domain_um, voxel_nm=voxel_nm,
            apply_taper=False, oversample=2.0,
        )
        means.append(float(r.n.mean()))

    arr = np.asarray(means)
    drift = float(arr.std(ddof=1))
    # Relative error on a standard deviation from n samples.
    rel_err = 1.0 / np.sqrt(2.0 * (n_seeds - 1))
    return {
        "drift_rms": drift,
        "n_seeds": n_seeds,
        "voxel_nm": voxel_nm,
        "domain_um": tuple(cfg.fdtd.domain_um),
        "rel_error": float(rel_err),
        "ci68": (drift * (1 - rel_err), drift * (1 + rel_err)),
    }


def _synthetic_ensemble(
    n: int = 40,
    bulk_sensitivity: float = 250.0,     # nm per refractive index unit
    index_drift_rms: float = 2.0e-4,     # finite-box mean-index drift
                                         # -- pass the MEASURED value from
                                         # `production_box_drift`; the default
                                         # here is only for the self-test
    speckle_rms_nm: float = 0.040,       # the thing we want to recover
    seed: int = 7,
) -> list[SeedResult]:
    """
    Fabricate an ensemble whose true speckle spread we know, so we can check
    that the regression recovers it.

    This is the unit test for the most important statistical step in the whole
    project.  If the regression cannot recover a known answer from synthetic
    data, it will not recover an unknown one from real data.
    """
    rng = np.random.default_rng(seed)
    n0 = 1.39
    out = []
    for i in range(n):
        drift = rng.normal(0.0, index_drift_rms)
        speckle = rng.normal(0.0, speckle_rms_nm)
        lam_nm = 750.0 + bulk_sensitivity * drift + speckle
        out.append(SeedResult(
            seed=i, resonance_um=lam_nm / 1000.0, fwhm_um=0.060,
            mean_index_local=n0 + drift, mean_index_box=n0 + drift,
            decay_ok=True, fit_ok=True,
        ))
    return out


def estimator_bias_check(
    n_seeds: int = 40,
    true_speckle_pm: float = 40.0,
    trials: int = 200,
) -> dict:
    """
    Is the speckle estimator unbiased, and how noisy is a single ensemble?

    Repeats the whole synthetic experiment many times and looks at the
    distribution of the estimate.  Two things come out:

      BIAS    should be small and negative -- the standard small-sample bias
              of a standard-deviation estimate, roughly -1/(4N).

      SPREAD  how much a single ensemble's answer can differ from the truth by
              luck alone.  This is the number that tells you how many seeds
              you actually need.  It is much more useful than watching a
              running variance appear to "settle", which it does long before
              it is trustworthy.
    """
    est = []
    for t in range(trials):
        synth = _synthetic_ensemble(
            n=n_seeds, speckle_rms_nm=true_speckle_pm / 1000, seed=t
        )
        r = analyse_ensemble(synth, bootstrap_resamples=10, rng_seed=t)
        est.append(r.sigma_speckle_nm * 1000)
    est = np.array(est)
    return {
        "n_seeds": n_seeds,
        "true_pm": true_speckle_pm,
        "mean_estimate_pm": float(est.mean()),
        "bias_percent": float(100 * (est.mean() - true_speckle_pm)
                              / true_speckle_pm),
        "spread_percent": float(100 * est.std() / est.mean()),
    }


def print_bigbox_preflight(cfg: StudyConfig, depth_um: float,
                           n_seeds: int) -> bool:
    """
    Offline checks before running a larger-box ensemble.

    WHY A LARGER BOX
    ----------------
    The production ensemble uses a domain 2.5 um deep against a fitted
    correlation length of 1645 nm, so the box holds about 1.5 correlation
    lengths along its shortest axis.  Averaging only shrinks a spread when
    there is more than one independent thing to average, so the finite box
    can SUPPRESS the measured variance.  This run measures that finite-box
    correction directly.

    WHAT TO WATCH
    -------------
    Cost scales with cell count, which scales with the VOLUME.  Going from
    2.5 to 4.0 um deep is 1.6x the cells and therefore roughly 1.6x the cost
    per seed, before any change in the number of time steps.  The gate below
    refuses to proceed if the box still holds under three correlation
    lengths, because that is the condition the study is trying to escape.
    """
    from dataclasses import replace

    lc_um = cfg.tissue.l_c_um
    dom = list(cfg.fdtd.domain_um)
    new_dom = (dom[0], dom[1], depth_um)
    ratio_old = min(dom) / lc_um
    ratio_new = min(new_dom) / lc_um
    vol_old = dom[0] * dom[1] * dom[2]
    vol_new = new_dom[0] * new_dom[1] * new_dom[2]

    print("  LARGER-BOX ENSEMBLE PREFLIGHT -- all of this is free")
    print(f"    fitted correlation length   {lc_um * 1000:.0f} nm")
    print(f"    production domain           {dom[0]:.1f} x {dom[1]:.1f} x "
          f"{dom[2]:.1f} um   L/l_c = {ratio_old:.2f}")
    print(f"    proposed domain             {new_dom[0]:.1f} x {new_dom[1]:.1f} "
          f"x {new_dom[2]:.1f} um   L/l_c = {ratio_new:.2f}")
    print(f"    volume ratio                {vol_new / vol_old:.2f}x")
    print(f"    seeds                       {n_seeds}")
    print(f"    cost scaling vs production  ~{vol_new / vol_old * n_seeds / 34:.2f}x"
          " the original 34-seed ensemble")
    print()

    # THE LOCAL COSTS, WHICH ARE NOT THE SOLVER'S COSTS.
    # The cloud does the electromagnetics, but the tissue medium is generated
    # and uploaded from this machine, and both scale with the volume. A
    # 125-million-voxel realization can exhaust memory and appear to hang.
    voxel_nm = cfg.fdtd.tissue_voxel_nm
    nvox = 1
    for L in new_dom:
        nvox *= max(8, int(round(L / (voxel_nm / 1000.0))))
    peak_gb = 32.0 * nvox / 1e9          # measured 32 bytes per voxel
    secs = 2.1e-7 * nvox                 # measured on this machine class
    payload_gb = 4.0 * nvox / 1e9        # float32 permittivity, per simulation
    print("    LOCAL COST PER SEED (memory, time and upload on this machine)")
    print(f"    tissue grid                 {nvox / 1e6:.0f} million voxels "
          f"at {voxel_nm:.0f} nm")
    print(f"    peak memory, generation     ~{peak_gb:.1f} GB")
    print(f"    generation time             ~{secs:.0f} s")
    print(f"    permittivity payload        ~{payload_gb:.2f} GB per simulation "
          "(single precision)")
    print(f"    total generation            ~{secs * n_seeds / 60:.0f} min "
          f"for {n_seeds} seeds")
    print(f"    total transfer              ~{payload_gb * n_seeds:.0f} GB up, "
          "plus results down")
    print()
    if peak_gb > 8.0:
        print(f"    WARNING: {peak_gb:.1f} GB peak will swap on a 16 GB "
              "machine. Reduce the")
        print("    box, or coarsen the tissue voxel, before running this.")
        print()

    ok = True
    if ratio_new < 3.0:
        ok = False
        print(f"    FAIL: the proposed box still holds only {ratio_new:.2f}")
        print("    correlation lengths. Under three, the averaging law is not")
        print("    in force and the new sigma is biased by the same mechanism")
        print("    as the old one. Increase the depth.")
    else:
        print(f"    pass: {ratio_new:.2f} correlation lengths along the short "
              "axis")
        if min(new_dom) != depth_um:
            print(f"    note: the depth is no longer the limiting axis. The "
                  f"{min(new_dom):.1f} um lateral")
            print("    extent now sets L/l_c, so deepening further buys "
                  "nothing. To")
            print("    exceed this ratio, every axis must grow, and cost "
                  "scales with volume.")

    if n_seeds < 20:
        ok = False
        print(f"    FAIL: {n_seeds} seeds gives a relative error on sigma of "
              f"{100 / (2 * (n_seeds - 1)) ** 0.5:.0f}%, too coarse to")
        print("    resolve a correction of order 20%.")
    else:
        print(f"    pass: {n_seeds} seeds, relative error on sigma "
              f"{100 / (2 * (n_seeds - 1)) ** 0.5:.0f}%")

    if cfg.tissue.l_c_um < 1.0:
        ok = False
        print("    FAIL: the tissue model is not calibrated. Calibrate before")
        print("    running, or the box will look adequate when it is not.")
    else:
        print("    pass: tissue model is calibrated")

    print()
    if ok:
        print("    ALL GATES PASS. Run again with --submit to spend.")
        print("    The solver prints its own FlexCredit estimate for each "
              "seed before")
        print("    running it. Watch the first one and stop if the total "
              "would breach")
        print("    the campaign budget; the run resumes from whatever "
              "completed.")
    else:
        print("    GATES FAILED. Nothing will be submitted.")
    return ok


def bigbox_config(cfg: StudyConfig, depth_um: float, n_seeds: int) -> StudyConfig:
    """The production configuration with a deeper domain and more seeds."""
    from dataclasses import replace

    dom = list(cfg.fdtd.domain_um)
    return replace(
        cfg,
        fdtd=replace(cfg.fdtd, domain_um=(dom[0], dom[1], depth_um)),
        ensemble=replace(cfg.ensemble, n_seeds=n_seeds),
    )


def paired_against_production(results: list, production_path: str) -> None:
    """
    Compare a re-run ensemble with the production one, SEED BY SEED.

    WHY THIS IS A PAIRED TEST AND THE BIG-BOX RUN IS NOT
    ----------------------------------------------------
    `generate_tissue` builds the field on a grid derived from `domain_um`, so
    changing the domain changes the tissue even at a fixed seed -- which is
    why a deeper-box ensemble cannot be paired with the production one and
    carries the sampling error of both.

    A boundary change does not touch the grid.  Run at the production depth
    and voxel, seed n reproduces production realisation n bit for bit, so the
    difference between the two runs is the boundary and nothing else.  That
    turns a comparison of two spreads into a comparison of matched pairs,
    which is far more powerful for the same number of simulations.
    """
    import json as _json

    if not os.path.exists(production_path):
        print(f"\n  (no production results at {production_path}; "
              f"skipping the paired comparison)")
        return

    with open(production_path) as fh:
        prod = {r["seed"]: r for r in _json.load(fh) if r.get("fit_ok")}
    new = {r.seed: r for r in results if r.fit_ok}
    common = sorted(set(prod) & set(new))
    if len(common) < 3:
        print(f"\n  (only {len(common)} seeds in common with the production "
              f"ensemble; skipping the paired comparison)")
        return

    a = np.array([prod[s]["resonance_um"] * 1000 for s in common])
    b = np.array([new[s].resonance_um * 1000 for s in common])
    na = np.array([prod[s]["mean_index_local"] for s in common])
    nb = np.array([new[s].mean_index_local for s in common])
    d = a - b

    print()
    print("=" * 70)
    print(f"PAIRED AGAINST THE PRODUCTION ENSEMBLE, {len(common)} shared seeds")
    print("=" * 70)
    tissue_same = float(np.max(np.abs(na - nb)))
    print(f"  max |local index difference|  {tissue_same:.3e}")
    if tissue_same > 1e-9:
        print("  WARNING: the tissue is NOT the same realisation seed for "
              "seed, so this")
        print("  is not a paired test.  Check that the domain and voxel match "
              "production.")
    else:
        print("  pass: identical tissue seed for seed, so every difference "
              "below is the")
        print("        boundary and carries no sampling error.")

    print(f"  resonance shift, mean         {d.mean():+.3f} nm "
          f"(sd {d.std(ddof=1):.3f}, SEM {d.std(ddof=1) / len(d) ** 0.5:.3f})")

    def spread(lam, nbar):
        s, i = np.polyfit(nbar, lam, 1)
        return float(np.std(lam, ddof=1)), float(np.std(lam - (s * nbar + i),
                                                        ddof=2)), float(s)

    ra, sa, sla = spread(a, na)
    rb, sb, slb = spread(b, nb)
    print()
    print("                             production      this run")
    print(f"  raw spread              {ra * 1000:>10.1f} pm  "
          f"{rb * 1000:>10.1f} pm")
    print(f"  bulk sensitivity        {sla:>10.1f}     {slb:>10.1f}  nm/RIU")
    print(f"  speckle residual        {sa * 1000:>10.1f} pm  "
          f"{sb * 1000:>10.1f} pm")
    if sb > 0:
        print(f"  ratio                   {sa / sb:>10.2f}")
        # PITMAN-MORGAN, NOT F.
        #
        # An F test on these two spreads would treat two runs on IDENTICAL
        # tissue as independent samples.  They are not: the residuals here
        # correlate at r ~ 0.8, and ignoring that throws away exactly the
        # power the paired design was built to buy.  The correct test for
        # equal variances in paired samples asks whether the SUM and the
        # DIFFERENCE of the pairs are correlated, which they are if and only
        # if the two variances differ.
        try:
            from scipy.stats import pearsonr, t as _t

            ra_ = a - (sla * na + np.polyfit(na, a, 1)[1])
            rb_ = b - (slb * nb + np.polyfit(nb, b, 1)[1])
            rho = float(pearsonr(ra_, rb_)[0])
            rp = float(pearsonr(ra_ + rb_, ra_ - rb_)[0])
            dfp = len(common) - 3
            tt = rp * np.sqrt(dfp / max(1e-12, 1 - rp ** 2))
            pv = float(2 * _t.sf(abs(tt), dfp))
            print(f"  residual correlation    {rho:>10.2f}   "
                  f"<- the pairing this test exploits")
            print(f"  Pitman-Morgan p         {pv:>10.4f}   "
                  f"(paired test for equal variances)")
        except Exception as exc:
            print(f"  (Pitman-Morgan test unavailable: {exc})")
    print()
    print("  Read it this way. A large p does not mean the two runs are")
    print("  interchangeable: use the run whose boundary converges under a")
    print("  depth sweep, and let the interval carry the uncertainty. The check")
    print("  that decides which run to believe is the bulk sensitivity above,")
    print("  which should agree with the paired-protocol value measured by a")
    print("  completely independent route.")


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(
        description="Ensemble self-tests, and the larger-box run.")
    _ap.add_argument("--bigbox", action="store_true",
                     help="Preflight (and with --submit, run) the larger-box "
                          "ensemble that measures the finite-box correction.")
    _ap.add_argument("--depth", type=float, default=4.0,
                     help="Domain depth in micrometres for --bigbox "
                          "(default 4.0).")
    _ap.add_argument("--seeds", type=int, default=24,
                     help="Number of tissue realizations for --bigbox "
                          "(default 24).")
    _ap.add_argument("--submit", action="store_true",
                     help="Upload and run (uses Tidy3D credits). Without "
                          "this nothing leaves the machine.")
    _ap.add_argument("--out", default=None,
                     help="Output directory (default data/ensemble_bigbox "
                          "for --bigbox, data/ensemble_<boundary> for "
                          "--boundary-ensemble).")
    _ap.add_argument("--boundary-ensemble", action="store_true",
                     help="Re-run the ensemble at the PRODUCTION depth under "
                          "a different boundary, to measure what the boundary "
                          "contributes to the noise floor. Because the domain "
                          "is unchanged, each seed reproduces its production "
                          "realisation exactly and the comparison is paired.")
    _ap.add_argument("--boundary", choices=["absorber", "pml"],
                     default="pml",
                     help="Boundary for --boundary-ensemble (default pml).")
    _ap.add_argument("--production-results",
                     default="results/seed_results.json",
                     help="Production seed results, for the paired "
                          "comparison.")
    _args = _ap.parse_args()

    if _args.out is None:
        _args.out = ("data/ensemble_bigbox" if _args.bigbox
                     else f"data/ensemble_{_args.boundary}")

    cfg = StudyConfig()

    if _args.boundary_ensemble:
        import contextlib as _ctx, io as _io, warnings as _warn
        from dataclasses import replace as _replace
        from optical_properties import calibrate as _calibrate

        with _warn.catch_warnings():
            _warn.simplefilter("ignore")
            with _ctx.redirect_stdout(_io.StringIO()):
                _fit = _calibrate(cfg.tissue, cfg.calibration)
        cfg = _replace(cfg, tissue=_fit.as_tissue_config(cfg.tissue))

        _dom = cfg.fdtd.domain_um
        _cfg2 = _replace(
            cfg,
            fdtd=_replace(cfg.fdtd, boundary=_args.boundary),
            ensemble=_replace(cfg.ensemble, n_seeds=_args.seeds),
        )
        print("BOUNDARY ENSEMBLE")
        print(f"  domain            {_dom[0]:g} x {_dom[1]:g} x {_dom[2]:g} um"
              f"   <- production, unchanged on purpose")
        print(f"  boundary          {_args.boundary}"
              f"   (production used {cfg.fdtd.boundary})")
        print(f"  seeds             {_args.seeds}"
              f"   (the first {_args.seeds} of the production seed list)")
        print(f"  correlation length {cfg.tissue.l_c_um * 1000:.0f} nm")
        print()
        print("  The domain is deliberately NOT changed.  Holding the grid "
              "fixed is what")
        print("  makes each seed reproduce its production realisation, so "
              "this measures")
        print("  the boundary alone rather than the boundary plus a fresh "
              "draw of tissue.")
        if not _args.submit:
            print("\n  Nothing uploaded.  Run again with --submit to spend.")
            raise SystemExit(0)

        print(f"\nRunning {_args.seeds} seeds into {_args.out}. "
              f"This spends credits.")
        _res, _ = run_ensemble(_cfg2, submit=True, out_dir=_args.out,
                               results_path=os.path.join(
                                   _args.out, "seed_results.json"))
        _an = analyse_ensemble(
            _res, bootstrap_resamples=cfg.ensemble.bootstrap_resamples)
        print(_an.summary())
        paired_against_production(_res, _args.production_results)
        raise SystemExit(0)

    if _args.bigbox:
        # CALIBRATE FIRST.  config.py carries a starting guess for the
        # correlation length that is several times smaller than the fitted
        # value, and every L/l_c ratio computed from it is wrong in the
        # optimistic direction.
        import contextlib as _ctx, io as _io, warnings as _warn
        from dataclasses import replace as _replace
        from optical_properties import calibrate as _calibrate

        with _warn.catch_warnings():
            _warn.simplefilter("ignore")
            with _ctx.redirect_stdout(_io.StringIO()):
                _fit = _calibrate(cfg.tissue, cfg.calibration)
        _start = cfg.tissue.l_c_um
        cfg = _replace(cfg, tissue=_fit.as_tissue_config(cfg.tissue))
        print(f"  correlation length: starting guess {_start * 1000:.0f} nm, "
              f"calibrated {cfg.tissue.l_c_um * 1000:.0f} nm")
        print()
        _gates = print_bigbox_preflight(cfg, _args.depth, _args.seeds)
        if not _args.submit:
            raise SystemExit(0)
        if not _gates:
            raise SystemExit("Gates failed; refusing to spend.")
        _big = bigbox_config(cfg, _args.depth, _args.seeds)
        print(f"\nRunning {_args.seeds} seeds at depth "
              f"{_args.depth:g} um into {_args.out}. This spends credits.")
        _res, _ = run_ensemble(_big, submit=True, out_dir=_args.out,
                               results_path=os.path.join(
                                   _args.out, "seed_results.json"))
        _an = analyse_ensemble(_res,
                               bootstrap_resamples=cfg.ensemble.bootstrap_resamples)
        print(_an.summary())
        print("\nCompare this sigma with the 533.3 pm of the 2.5 um absorber "
              "ensemble. The difference is the finite-box correction.")
        raise SystemExit(0)

    print("=" * 70)
    print("UNIT TEST: can the regression recover a KNOWN speckle spread?")
    print("=" * 70)
    true_speckle_pm = 40.0
    true_slope = 250.0
    synth = _synthetic_ensemble(
        n=cfg.ensemble.n_seeds,
        bulk_sensitivity=true_slope,
        speckle_rms_nm=true_speckle_pm / 1000,
    )
    res = analyse_ensemble(synth,
                           bootstrap_resamples=cfg.ensemble.bootstrap_resamples)
    print(res.summary())
    print()
    print(f"  TRUE speckle spread was       {true_speckle_pm:.2f} pm")
    print(f"  TRUE bulk sensitivity was     {true_slope:.1f} nm/RIU")
    err = abs(res.sigma_speckle_nm * 1000 - true_speckle_pm) / true_speckle_pm
    print(f"  recovery error                {100 * err:.1f}%")
    print()

    print("=" * 70)
    print("WHAT WOULD HAVE HAPPENED WITHOUT THE REGRESSION")
    print("=" * 70)
    naive = analyse_ensemble(synth, regress_out_mean_index=False,
                             bootstrap_resamples=500)
    over = naive.sigma_raw_nm * 1000 / true_speckle_pm
    print(f"  naive spread                  {naive.sigma_raw_nm * 1000:.2f} pm")
    print(f"  true speckle                  {true_speckle_pm:.2f} pm")
    print(f"  OVERESTIMATE BY               {over:.2f}x")
    print()
    print("  That factor is exactly the finite-box artefact.  It would have")
    print("  inflated the reported noise floor by the same factor, and it")
    print("  would shrink if you simply used a bigger box.")
    print()

    print("=" * 70)
    print("IS THE ESTIMATOR UNBIASED, AND HOW MANY SEEDS DO YOU NEED?")
    print("=" * 70)
    print("  Repeating the synthetic experiment 200 times at each ensemble")
    print("  size, to see how far a single run can stray by luck alone.")
    print()
    print(f"  {'seeds':>6}  {'mean estimate':>15}  {'bias':>8}  "
          f"{'run-to-run spread':>18}")
    for n in (20, 40, 100):
        b = estimator_bias_check(n_seeds=n)
        print(f"  {b['n_seeds']:6d}  {b['mean_estimate_pm']:12.2f} pm  "
              f"{b['bias_percent']:+7.1f}%  {b['spread_percent']:16.1f}%")
    print()
    print("  The estimator is essentially unbiased.  The run-to-run spread is")
    print("  what should size your ensemble: if the noise floor will be")
    print("  compared against a regulatory limit it is within a factor of two")
    print("  of, you need the spread below about 10%, i.e. 40 seeds or more.")
    print()

    print("=" * 70)
    print("BUILDING THE REAL ENSEMBLE (offline, nothing uploaded)")
    print("=" * 70)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        _, sims = run_ensemble(cfg, submit=False, verbose=True)
    from simulation import report
    print()
    print(report(sims[0]))
