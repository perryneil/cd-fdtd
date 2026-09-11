"""
optical_properties.py — making the synthetic tissue behave like real tissue.
============================================================================

THE PROBLEM THIS SOLVES
-----------------------
`tissue.py` can manufacture a random medium with any blob size and contrast we
like.  But "any" is not good enough: we need the one that scatters light the
way fish muscle scatters light.  Otherwise we have simulated an interesting
mathematical object that tells us nothing about fish.

Two numbers describe how a tissue scatters:

    mu_s   the scattering coefficient.  On average, light travels 1/mu_s
           before being scattered once.  Big mu_s = cloudy, milky.
    g      the anisotropy factor, between -1 and 1.  It is the average of
           cos(scattering angle).  g near 1 means light barely changes
           direction (forward scattering); g near 0 means it is thrown in all
           directions.  Soft tissue is strongly forward-scattering, g ~ 0.9.

This module computes mu_s and g *directly from the tissue spectrum*, and then
inverts: given target values of mu_s and g, it finds the blob size and
contrast that produce them.

WHY NOT JUST SIMULATE IT?
-------------------------
You could put a slab of synthetic tissue in an FDTD box, shine light through
it, and measure what comes out.  There are three problems with that:

1.  The FDTD box is 5-10 microns across.  The scattering mean free path in
    muscle is 50-200 microns.  So the box is one to two orders of magnitude
    *thinner* than a single scattering event.  You are trying to measure a
    small difference between two large numbers, and the answer gets swamped by
    light diffracting off the edges of your slab.

2.  A single random realisation does not have a phase function.  It has
    speckle.  Fitting a smooth curve to speckle gives you a number, but not the
    number you wanted.  You would need an ensemble at every trial parameter
    set, which multiplies an already expensive fit.

3.  The map (contrast, blob size, shape) -> (mu_s, g) is three inputs into two
    outputs.  It is under-determined: infinitely many tissues give the same
    mu_s and g.  An optimiser will happily wander along that valley and stop
    somewhere arbitrary.

The fix for (1) and (2) is to use the analytic result instead: under the Born
approximation — valid precisely because the index contrast in tissue is weak —
the scattering follows directly from the power spectrum, with no simulation at
all.  It is instant, exact within the approximation, and free of speckle.

The fix for (3) is to hold the shape parameter `m` fixed (from published
low-coherence backscattering measurements) and fit only the other two.  This
module refuses to fit all three, on purpose.

FDTD still has a job: `verify_born_validity` checks that the contrast we
calibrated to is weak enough for the Born approximation to hold, and
`simulation.py` can build a slab run to spot-check a handful of points.  But
the *fitting* is analytic.

THE PHYSICS IN ONE PARAGRAPH
----------------------------
Weak scattering from a random index field is a diffraction problem.  A wave of
wavelength lambda arriving in a medium of average index n0 has wavevector
k = 2*pi*n0/lambda.  Scattering by an angle theta requires the medium to have
structure at spatial frequency q = 2*k*sin(theta/2) — a grating of that
spacing, in effect.  So the amount of light scattered to angle theta is
proportional to how much structure the tissue has at that spatial frequency,
which is exactly what the power spectrum Phi(q) tells us.  Adding the standard
dipole-radiation factor for polarised light gives the full phase function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from config import TissueConfig, CalibrationConfig
from tissue import whittle_matern_psd


# ==========================================================================
#  Forward problem: spectrum -> (mu_s, g)
# ==========================================================================
@dataclass
class ScatteringProperties:
    """What a given synthetic tissue does to light at one wavelength."""

    wavelength_um: float
    mu_s_per_um: float
    """Scattering coefficient: 1/mu_s is the average distance between
    scattering events."""

    g: float
    """Anisotropy factor = average of cos(theta) over the phase function."""

    @property
    def mu_s_reduced_per_um(self) -> float:
        """The 'reduced' or 'transport' scattering coefficient,
        mu_s' = mu_s (1 - g).

        This is the number that governs how light diffuses over long
        distances, and the one most often tabulated in the biomedical-optics
        literature, because many forward-scattering events are equivalent to
        one isotropic one."""
        return self.mu_s_per_um * (1.0 - self.g)

    @property
    def mean_free_path_um(self) -> float:
        """Average distance between scattering events, microns.

        Compare this with your FDTD box size: the box must be much SMALLER,
        so that the near-field region sees a correlated random index field
        rather than a diffusive soup.  That is the assumption the whole
        scale-splitting argument rests on, and this is how you check it."""
        return 1.0 / self.mu_s_per_um if self.mu_s_per_um > 0 else np.inf

    def __str__(self) -> str:
        return (
            f"at {self.wavelength_um * 1000:.0f} nm:  "
            f"mu_s = {self.mu_s_per_um * 1000:.2f} /mm,  "
            f"g = {self.g:.4f},  "
            f"mu_s' = {self.mu_s_reduced_per_um * 1000:.3f} /mm,  "
            f"mean free path = {self.mean_free_path_um:.1f} um"
        )


def scattering_from_spectrum(
    wavelength_um: float,
    n0: float,
    delta_n_sq: float,
    l_c_um: float,
    m: float,
    l_min_um: float,
    n_theta: int = 2001,
    polarised: bool = True,
) -> ScatteringProperties:
    """
    Compute mu_s and g analytically from the tissue's power spectrum.

    THE CALCULATION, STEP BY STEP
    -----------------------------
    1.  The wavevector in the medium is k = 2*pi*n0/lambda.

    2.  Light scattered by angle theta corresponds to a momentum transfer of
        q = 2*k*sin(theta/2).  (theta = 0 is straight ahead, q = 0; theta = pi
        is straight back, q = 2k.)

    3.  The differential scattering cross-section per unit volume is

            dsigma/dOmega  =  2 * pi * k^4 * Phi(q) * P(theta, phi)

        where Phi is the index power spectrum and P is a polarisation factor.
        For light linearly polarised along x, P = 1 - sin^2(theta) cos^2(phi);
        integrating that over the azimuth phi gives pi*(2 - sin^2 theta).
        Setting `polarised=False` uses the scalar-wave result P = 1 instead,
        which is what many textbook treatments quote.

    4.  Integrate over all solid angle to get mu_s, and integrate weighted by
        cos(theta) to get g.

    WHEN THIS IS VALID
    ------------------
    The Born approximation: the wave that scatters is assumed to be the
    incident wave, unperturbed.  That requires the accumulated phase error
    across one correlation length to be small.  `verify_born_validity` checks
    it.  For soft tissue with an RMS index wobble of ~0.01-0.02 it is
    comfortably satisfied.

    Returns
    -------
    ScatteringProperties
    """
    k = 2.0 * np.pi * n0 / wavelength_um
    theta = np.linspace(0.0, np.pi, n_theta)
    q = 2.0 * k * np.sin(theta / 2.0)

    phi_of_q = whittle_matern_psd(q, delta_n_sq, l_c_um, m, l_min_um)

    if polarised:
        # Azimuthal integral of (1 - sin^2 theta cos^2 phi) over 0..2pi
        azimuthal = np.pi * (2.0 - np.sin(theta) ** 2)
    else:
        azimuthal = 2.0 * np.pi * np.ones_like(theta)

    # dsigma/dOmega * (azimuthal integral) * sin(theta), ready to integrate
    # over theta alone.
    weight = 2.0 * np.pi * k ** 4 * phi_of_q * azimuthal * np.sin(theta)

    mu_s = float(np.trapezoid(weight, theta))
    if mu_s <= 0:
        return ScatteringProperties(wavelength_um, 0.0, 0.0)

    g = float(np.trapezoid(weight * np.cos(theta), theta) / mu_s)
    return ScatteringProperties(wavelength_um, mu_s, g)


def scattering_spectrum(
    tissue_cfg: TissueConfig,
    wavelengths_um: np.ndarray,
) -> list[ScatteringProperties]:
    """mu_s and g across a whole band, so you can check the calibrated tissue
    behaves sensibly everywhere and not just at the one fitted wavelength."""
    return [
        scattering_from_spectrum(
            wl, tissue_cfg.n0, tissue_cfg.delta_n_sq,
            tissue_cfg.l_c_um, tissue_cfg.m, tissue_cfg.l_min_um,
        )
        for wl in np.atleast_1d(wavelengths_um)
    ]


# ==========================================================================
#  Inverse problem: (mu_s, g) -> spectrum
# ==========================================================================
@dataclass
class CalibrationResult:
    """The outcome of fitting the tissue to measured optical properties."""

    delta_n_sq: float
    l_c_um: float
    m_fixed: float
    achieved: ScatteringProperties
    target_mu_s_per_um: float
    target_g: float
    converged: bool
    hit_bound: bool
    """True if the optimiser ran into a search-range edge.  If this is True the
    target pair is NOT reachable at the fixed `m`, and you must say so rather
    than quietly accepting the edge value."""

    message: str

    def as_tissue_config(self, base: TissueConfig) -> TissueConfig:
        """Return a copy of `base` with the fitted parameters substituted in."""
        from dataclasses import replace
        return replace(base, delta_n_sq=self.delta_n_sq, l_c_um=self.l_c_um)

    def __str__(self) -> str:
        flag = "  *** HIT A SEARCH BOUND ***" if self.hit_bound else ""
        return (
            "Calibration result\n"
            f"  fitted (dn)^2        {self.delta_n_sq:.4e}  "
            f"(RMS dn = {self.delta_n_sq ** 0.5:.5f})\n"
            f"  fitted l_c           {self.l_c_um * 1000:.1f} nm\n"
            f"  held fixed:  m       {self.m_fixed}\n"
            f"  target   mu_s        {self.target_mu_s_per_um * 1000:.3f} /mm,"
            f"  g = {self.target_g:.4f}\n"
            f"  achieved mu_s        {self.achieved.mu_s_per_um * 1000:.3f} /mm,"
            f"  g = {self.achieved.g:.4f}\n"
            f"  mean free path       {self.achieved.mean_free_path_um:.1f} um\n"
            f"  converged            {self.converged}{flag}\n"
            f"  {self.message}"
        )


def calibrate(
    tissue_cfg: TissueConfig,
    calib_cfg: CalibrationConfig,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Find the (contrast, blob size) that reproduce the measured mu_s and g.

    WHAT IS FITTED AND WHAT IS NOT
    ------------------------------
    Fitted:      delta_n_sq  (contrast)
                 l_c_um      (blob size)
    Held fixed:  m           (spectrum shape)
                 l_min_um    (inner cutoff)
                 n0          (baseline index)

    Fitting two parameters to two targets is a well-posed problem.  Fitting
    three to two is not, and this function deliberately does not offer it.  If
    you want to know how much `m` matters, run this function at several values
    of `m` and report the spread (see `sensitivity_to_m`), rather than
    letting an optimiser pick `m`.

    THE FIT ITSELF
    --------------
    Both parameters are searched in log space, because they range over orders
    of magnitude and a linear search would waste all its effort at the top end.
    The residuals are relative, so that mu_s (~1e-2) and g (~0.9) get equal
    weight despite differing by two orders of magnitude.

    Returns
    -------
    CalibrationResult.  Always check `.hit_bound` before using it.
    """
    lo_d, hi_d = calib_cfg.fit_bounds_delta_n_sq
    lo_l, hi_l = calib_cfg.fit_bounds_l_c_um

    def residuals(p):
        delta_n_sq = 10.0 ** p[0]
        l_c = 10.0 ** p[1]
        s = scattering_from_spectrum(
            calib_cfg.calib_wavelength_um, tissue_cfg.n0,
            delta_n_sq, l_c, tissue_cfg.m, tissue_cfg.l_min_um,
        )
        # Relative residuals so both targets carry equal weight.
        r_mu = (s.mu_s_per_um - calib_cfg.target_mu_s_per_um) \
            / calib_cfg.target_mu_s_per_um
        r_g = (s.g - calib_cfg.target_g) / max(calib_cfg.target_g, 1e-6)
        return [r_mu, r_g]

    x0 = [np.log10(tissue_cfg.delta_n_sq), np.log10(tissue_cfg.l_c_um)]
    bounds = (
        [np.log10(lo_d), np.log10(lo_l)],
        [np.log10(hi_d), np.log10(hi_l)],
    )

    sol = least_squares(residuals, x0, bounds=bounds, xtol=1e-12, ftol=1e-12)

    delta_n_sq = 10.0 ** sol.x[0]
    l_c = 10.0 ** sol.x[1]
    achieved = scattering_from_spectrum(
        calib_cfg.calib_wavelength_um, tissue_cfg.n0,
        delta_n_sq, l_c, tissue_cfg.m, tissue_cfg.l_min_um,
    )

    tol = 1e-3
    hit_bound = (
        abs(sol.x[0] - bounds[0][0]) < tol or abs(sol.x[0] - bounds[1][0]) < tol
        or abs(sol.x[1] - bounds[0][1]) < tol
        or abs(sol.x[1] - bounds[1][1]) < tol
    )
    resid = np.max(np.abs(residuals(sol.x)))
    converged = bool(sol.success and resid < 1e-3)

    if hit_bound:
        msg = (
            "The optimiser ran to a search-range edge.  The requested "
            "(mu_s, g) pair is not reachable at this m -- report that, or "
            "change m, rather than using this result."
        )
    elif not converged:
        msg = f"Did not fully converge; worst relative residual {resid:.2e}."
    else:
        msg = f"Converged; worst relative residual {resid:.2e}."

    result = CalibrationResult(
        delta_n_sq=delta_n_sq, l_c_um=l_c, m_fixed=tissue_cfg.m,
        achieved=achieved,
        target_mu_s_per_um=calib_cfg.target_mu_s_per_um,
        target_g=calib_cfg.target_g,
        converged=converged, hit_bound=hit_bound, message=msg,
    )
    if verbose:
        print(result)
    return result


def sensitivity_to_m(
    tissue_cfg: TissueConfig,
    calib_cfg: CalibrationConfig,
    m_values: tuple[float, ...] = (1.15, 1.25, 1.35, 1.45, 1.6, 2.0),
) -> list[CalibrationResult]:
    """
    Repeat the calibration at several values of the shape parameter.

    WHY
    ---
    `m` is held fixed rather than fitted, so it is important to know what
    happens if it is wrong.  This shows how much the fitted contrast and blob
    size move as `m` varies over its plausible range.  If the downstream noise
    floor barely moves, the choice of `m` is not critical.  If it moves a lot,
    `m` needs to be measured.
    """
    from dataclasses import replace
    out = []
    for m in m_values:
        res = calibrate(replace(tissue_cfg, m=m), calib_cfg, verbose=False)
        out.append(res)
    return out


# ==========================================================================
#  Sanity checks on the approximations
# ==========================================================================
def verify_born_validity(
    tissue_cfg: TissueConfig, wavelength_um: float
) -> dict[str, float]:
    """
    Check that the weak-scattering assumption behind the analytic result holds.

    PLAIN LANGUAGE
    --------------
    The analytic formulas assume light passes through the tissue essentially
    undisturbed, and only a little bit leaks off sideways.  That is true when
    the index wobbles are small and the blobs are not too large.  The standard
    test is the *phase accumulation*: how much extra phase does a wave pick up
    crossing one blob, compared with crossing the same distance in the average
    medium?

        phase error  ~  2 * k * delta_n * l_c

    If that is well below 1 radian, the Born approximation is safe.  Between
    about 1 and 3 it is marginal.  Above that you should not trust these
    formulas, and should calibrate against FDTD slabs instead — with all the
    difficulties described at the top of this file.

    The second check compares the FDTD domain size against the scattering mean
    free path: the whole scale-splitting argument requires the FDTD box to be
    much smaller, so that it sees correlated randomness rather than diffusion.
    """
    k = 2.0 * np.pi * tissue_cfg.n0 / wavelength_um
    delta_n_rms = np.sqrt(tissue_cfg.delta_n_sq)
    phase = 2.0 * k * delta_n_rms * tissue_cfg.l_c_um

    props = scattering_from_spectrum(
        wavelength_um, tissue_cfg.n0, tissue_cfg.delta_n_sq,
        tissue_cfg.l_c_um, tissue_cfg.m, tissue_cfg.l_min_um,
    )

    if phase < 1.0:
        verdict = "SAFE"
    elif phase < 3.0:
        verdict = "MARGINAL - cross-check against an FDTD slab"
    else:
        verdict = "UNSAFE - the analytic result should not be used"

    return {
        "phase_accumulation_rad": float(phase),
        "verdict": verdict,
        "size_parameter_k_lc": float(k * tissue_cfg.l_c_um),
        "rms_delta_n": float(delta_n_rms),
        "mean_free_path_um": props.mean_free_path_um,
    }


def check_scale_splitting(
    tissue_cfg: TissueConfig,
    fdtd_domain_um: tuple[float, float, float],
    wavelength_um: float,
) -> dict[str, object]:
    """
    Verify the assumption that makes the two-stage (Monte Carlo + FDTD)
    approach legitimate.

    PLAIN LANGUAGE
    --------------
    The study splits the physics in two: light transport over millimetres is
    handled by Monte Carlo, and the near-field around the sensor by FDTD.  That
    split is only allowed if the FDTD box is much smaller than one scattering
    mean free path — otherwise the box would itself contain multiple scattering
    events, and you cannot treat the two stages separately.

    This function computes the ratio and says whether the assumption holds,
    so it can be reported as a number rather than assumed.
    """
    props = scattering_from_spectrum(
        wavelength_um, tissue_cfg.n0, tissue_cfg.delta_n_sq,
        tissue_cfg.l_c_um, tissue_cfg.m, tissue_cfg.l_min_um,
    )
    box = max(fdtd_domain_um)
    ratio = box / props.mean_free_path_um
    return {
        "fdtd_box_um": box,
        "mean_free_path_um": props.mean_free_path_um,
        "box_over_mfp": float(ratio),
        "sub_diffusive": bool(ratio < 0.1),
        "verdict": (
            "OK - box is much smaller than one scattering length, so the "
            "near field sees a correlated random index field, not a "
            "diffusive medium."
            if ratio < 0.1 else
            "PROBLEM - the box is a substantial fraction of a scattering "
            "length.  The scale-splitting argument does not hold; shrink the "
            "box or justify the coupling."
        ),
        "blobs_across_box": float(box / tissue_cfg.l_c_um),
        "independent_volumes_in_box": float((box / tissue_cfg.l_c_um) ** 3),
        "enough_blobs": bool((box / tissue_cfg.l_c_um) ** 3 >= 27),
        "correlation_length_um": float(tissue_cfg.l_c_um),
        "box_verdict": _box_verdict(box, tissue_cfg.l_c_um),
    }


def check_mu_s_prime(cfg, tissue_cfg=None) -> dict:
    """
    Does the calibrated tissue reproduce the REDUCED scattering coefficient?

    WHY THIS CHECK IS NEEDED
    ------------------------
    The calibration fits `mu_s` and `g` separately, because those are what the
    one fish measurement reports.  But `mu_s` and `g` are strongly correlated
    in any inverse-adding-doubling retrieval, and the quantity that actually
    governs transport -- and the quantity two independent datasets agree on --
    is `mu_s' = mu_s (1 - g)`:

        cod, 500-1700 nm             1.2 - 6 /cm   Vraalstad et al. 2025
        bovine longissimus, 721 nm   4 - 9 /cm     Van Beers et al. 2018

    A fit can land exactly on (mu_s, g) and still miss mu_s', because a small
    error in g is amplified by the (1 - g) factor: at g = 0.93, a 1% error in
    g is a 13% error in mu_s'.  So this is not a redundant check.

    Reports the achieved value against `CalibrationConfig.target_mu_s_prime_
    per_um` and says plainly which of the two literature bands it falls in.
    """
    t = tissue_cfg if tissue_cfg is not None else cfg.tissue
    c = cfg.calibration

    sp = scattering_from_spectrum(
        c.calib_wavelength_um, t.n0, t.delta_n_sq, t.l_c_um, t.m, t.l_min_um
    )
    achieved = sp.mu_s_reduced_per_um
    target = c.target_mu_s_prime_per_um
    from_targets = c.target_mu_s_per_um * (1.0 - c.target_g)

    ratio = achieved / target if target > 0 else float("nan")
    cod_band = (1.2e-4, 6.0e-4)          # /um, i.e. 1.2-6 /cm
    bovine_band = (4.0e-4, 9.0e-4)       # /um, i.e. 4-9 /cm

    in_cod = cod_band[0] <= achieved <= cod_band[1]
    in_bovine = bovine_band[0] <= achieved <= bovine_band[1]

    if in_cod and in_bovine:
        where = "inside BOTH the cod and the bovine bands"
    elif in_cod:
        where = "inside the cod band but below the bovine one"
    elif in_bovine:
        where = "inside the bovine band but above the cod one"
    else:
        where = "OUTSIDE both published bands"

    if 0.5 <= ratio <= 2.0 and (in_cod or in_bovine):
        verdict = (
            f"OK. The calibrated tissue delivers mu_s' = "
            f"{achieved * 1e4:.2f} /cm, {where}. Report this alongside "
            f"(mu_s, g) -- it is the number a reader can compare against any "
            f"other tissue."
        )
    else:
        verdict = (
            f"MISMATCH. The calibrated tissue delivers mu_s' = "
            f"{achieved * 1e4:.2f} /cm against a target of "
            f"{target * 1e4:.2f} /cm ({ratio:.2f}x), {where}. The (mu_s, g) "
            f"pair that was fitted implies {from_targets * 1e4:.2f} /cm, so "
            f"if those two disagree the fit did not reach its target. Report "
            f"mu_s' explicitly alongside (mu_s, g)."
        )

    return {
        "achieved_per_um": achieved,
        "achieved_per_cm": achieved * 1e4,
        "target_per_um": target,
        "implied_by_targets_per_um": from_targets,
        "ratio": ratio,
        "in_cod_band": bool(in_cod),
        "in_bovine_band": bool(in_bovine),
        "ok": bool(0.5 <= ratio <= 2.0 and (in_cod or in_bovine)),
        "verdict": verdict,
    }


def _box_verdict(box_um: float, l_c_um: float) -> str:
    """Judge whether the FDTD box can support ensemble statistics at all.

    THE PROBLEM THIS CATCHES
    ------------------------
    Everything downstream -- the speckle spread, the noise floor, the whole
    ensemble argument -- assumes each simulation box is a fresh, largely
    independent sample of tissue.  That requires the box to contain many
    correlation volumes.  The count is (L / l_c)^3, and it falls off a cliff:
    a box three correlation lengths across holds 27 of them, one at 1.8 holds
    six, one at 1.0 holds a single blob.

    Below about one, different random seeds are not sampling different tissue
    -- they are sampling different parts of the SAME blob.  The spread you
    measure is then a statement about that one blob's placement, and enlarging
    the ensemble does not help, because the realisations are not independent.

    This matters especially once the calibration has been fitted to a high
    anisotropy factor.  Strong forward scattering (g ~ 0.93) demands large
    scatterers, so the fitted correlation length grows -- and it can grow past
    the box you can afford to simulate.  That tension is physical and should
    be reported.
    """
    # 27 independent volumes = three correlation lengths across the box.
    # The same threshold gates `enough_blobs`, so the verdict and the warning
    # can never disagree.
    n_vol = (box_um / l_c_um) ** 3
    if n_vol >= 27:
        return (f"OK - the box holds about {n_vol:.0f} independent correlation "
                f"volumes, enough for ensemble statistics.")
    if n_vol >= 3:
        return (f"MARGINAL - only about {n_vol:.1f} independent correlation "
                f"volumes per box. Seeds are partly correlated; the spread is "
                f"an underestimate of the true tissue-to-tissue variation and "
                f"the ensemble converges more slowly than sqrt(N) suggests.")
    return (
        f"INADEQUATE - about {n_vol:.2f} independent correlation volumes per "
        f"box (l_c = {l_c_um * 1000:.0f} nm against a {box_um:.1f} um box). "
        f"Different seeds are sampling the same blob from different angles, "
        f"not different tissue. Any 'speckle spread' measured this way is "
        f"largely a statement about where one blob landed. Fix by enlarging "
        f"the box to at least 3x l_c, by fitting at a larger m (which yields a "
        f"smaller l_c -- see the sensitivity table), or by stating plainly "
        f"that the model resolves sub-correlation-length structure only."
    )


# ==========================================================================
#  Self-check
# ==========================================================================
if __name__ == "__main__":
    from config import StudyConfig

    cfg = StudyConfig()

    print("=" * 70)
    print("STEP 1 - forward: what does the default tissue do to light?")
    print("=" * 70)
    for wl in (0.5, 0.6, 0.7, 0.8, 0.9):
        print("  " + str(scattering_from_spectrum(
            wl, cfg.tissue.n0, cfg.tissue.delta_n_sq,
            cfg.tissue.l_c_um, cfg.tissue.m, cfg.tissue.l_min_um,
        )))

    print()
    print("=" * 70)
    print("STEP 2 - inverse: fit the tissue to the measured targets")
    print("=" * 70)
    result = calibrate(cfg.tissue, cfg.calibration)
    calibrated = result.as_tissue_config(cfg.tissue)

    print()
    print("Round-trip check (feed the fitted parameters back through the")
    print("forward calculation; it must reproduce the targets):")
    check = scattering_from_spectrum(
        cfg.calibration.calib_wavelength_um, calibrated.n0,
        calibrated.delta_n_sq, calibrated.l_c_um,
        calibrated.m, calibrated.l_min_um,
    )
    print("  " + str(check))

    print()
    print("=" * 70)
    print("STEP 3 - are the approximations we relied on actually valid?")
    print("=" * 70)
    for key, val in verify_born_validity(
        calibrated, cfg.calibration.calib_wavelength_um
    ).items():
        print(f"  {key:28s} {val}")
    print()
    for key, val in check_scale_splitting(
        calibrated, cfg.fdtd.domain_um, cfg.calibration.calib_wavelength_um
    ).items():
        print(f"  {key:28s} {val}")

    print()
    print("=" * 70)
    print("STEP 4 - how much does the un-fitted shape parameter m matter?")
    print("=" * 70)
    print(f"  {'m':>6}  {'(dn)^2':>12}  {'l_c (nm)':>10}  {'converged':>10}")
    for res in sensitivity_to_m(cfg.tissue, cfg.calibration):
        print(f"  {res.m_fixed:6.2f}  {res.delta_n_sq:12.4e}  "
              f"{res.l_c_um * 1000:10.1f}  {str(res.converged):>10}")


# ==========================================================================
#  Calibration sensitivity: which target actually sets the blob size?
# ==========================================================================
def calibration_sensitivity(
    cfg,
    mu_s_values_per_um: tuple[float, ...] = (3.0e-3, 4.0e-3, 5.0e-3, 6.0e-3),
    g_values: tuple[float, ...] = (0.90, 0.93, 0.95, 0.97),
    box_um: float = 2.5,
) -> dict:
    """
    How far does the fitted tissue move when the calibration targets move?

    WHY THIS MATTERS MORE THAN IT LOOKS
    -----------------------------------
    `calibrate` fits two parameters, the index contrast and the correlation
    length, to two targets, mu_s and g.  The reduced scattering coefficient
    mu_s' is NOT fitted; `check_mu_s_prime` only reports it afterwards, so
    the model should not be described as anchored to mu_s'.

    The result: the correlation length is almost independent
    of mu_s and is set almost entirely by g.  Over the range of g reported for
    muscle the fitted l_c moves by a factor of about five, and with it the
    ratio of the simulation box to the correlation length, which is what
    decides whether the supercell is large enough.  The anisotropy factor is
    therefore the least well constrained input in the whole chain and the one
    a future measurement should pin down first.

    Returns a dict with two sweeps and the span of each fitted quantity.
    """
    import contextlib
    import io
    import math
    from dataclasses import replace

    def _fit(mu_s, g):
        c = replace(cfg, calibration=replace(
            cfg.calibration, target_mu_s_per_um=mu_s, target_g=g))
        with contextlib.redirect_stdout(io.StringIO()):
            t = calibrate(c.tissue, c.calibration).as_tissue_config(c.tissue)
        return {
            "target_mu_s_per_um": mu_s,
            "target_g": g,
            "l_c_nm": t.l_c_um * 1000.0,
            "rms_dn": math.sqrt(t.delta_n_sq),
            "box_over_l_c": box_um / t.l_c_um,
        }

    g0 = cfg.calibration.target_g
    mu0 = cfg.calibration.target_mu_s_per_um
    by_mu_s = [_fit(m, g0) for m in mu_s_values_per_um]
    by_g = [_fit(mu0, g) for g in g_values]
    lc_mu = [r["l_c_nm"] for r in by_mu_s]
    lc_g = [r["l_c_nm"] for r in by_g]
    return {
        "by_mu_s": by_mu_s,
        "by_g": by_g,
        "l_c_span_from_mu_s_nm": (min(lc_mu), max(lc_mu)),
        "l_c_span_from_g_nm": (min(lc_g), max(lc_g)),
        "g_dominates": (max(lc_g) - min(lc_g)) > (max(lc_mu) - min(lc_mu)),
        "box_um": box_um,
    }


def print_calibration_sensitivity(cfg, **kw) -> None:
    """Human readable form of `calibration_sensitivity`."""
    r = calibration_sensitivity(cfg, **kw)
    print("  WHICH CALIBRATION TARGET SETS THE BLOB SIZE?")
    print("    mu_s and g are fitted.  mu_s' is only checked afterwards.")
    print()
    print(f"    {'mu_s (/mm)':>11} {'g':>6} {'l_c (nm)':>10} {'RMS dn':>9}"
          f" {'L/l_c':>7}")
    for row in r["by_mu_s"]:
        print(f"    {row['target_mu_s_per_um']*1000:>11.1f}"
              f" {row['target_g']:>6.2f} {row['l_c_nm']:>10.0f}"
              f" {row['rms_dn']:>9.5f} {row['box_over_l_c']:>7.2f}")
    print()
    for row in r["by_g"]:
        print(f"    {row['target_mu_s_per_um']*1000:>11.1f}"
              f" {row['target_g']:>6.2f} {row['l_c_nm']:>10.0f}"
              f" {row['rms_dn']:>9.5f} {row['box_over_l_c']:>7.2f}")
    print()
    lo, hi = r["l_c_span_from_mu_s_nm"]
    print(f"    varying mu_s moves l_c from {lo:.0f} to {hi:.0f} nm")
    lo, hi = r["l_c_span_from_g_nm"]
    print(f"    varying g    moves l_c from {lo:.0f} to {hi:.0f} nm")
    if r["g_dominates"]:
        print("    The anisotropy factor g dominates.  It is the least well")
        print("    constrained input in the chain, and it is what decides")
        print(f"    whether a {r['box_um']:.1f} um box holds enough")
        print("    correlation lengths for the finite box correction to hold.")
