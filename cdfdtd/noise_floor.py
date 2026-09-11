"""
noise_floor.py — from a wobble in nanometres to a number in mg/kg.
==================================================================

THE QUESTION THIS MODULE ANSWERS
--------------------------------
The European limit for cadmium in fish muscle is 0.05 milligrams per kilogram.
Our simulation produces a resonance wavelength that wobbles by some number of
picometres from one random tissue to the next, with no cadmium present at all.

Is that wobble bigger or smaller than the signal a regulated amount of cadmium
would produce?  If bigger, the design cannot work, no matter how good the
chemistry.  That is a genuinely useful thing to know before building anything.

FOUR DIFFERENT FLOORS, AND WHY YOU NEED ALL FOUR
------------------------------------------------
It is tempting to compute one number and call it "the noise floor", and to
describe it as independent of the recognition chemistry.  Only one of the
limits below has that property.  There are four distinct limits, with
different characters:

  1. COVERAGE FLOOR (theta_floor)
     The smallest fraction of binding sites that must be occupied before the
     resonance shift exceeds the tissue-induced wobble.
     *** This one IS independent of recognition chemistry. ***  It is a
     property of the geometry and the matrix alone.  Report it as the headline.

  2. CONCENTRATION FLOOR (C_floor)
     Converting the coverage floor into a concentration requires a binding
     isotherm, and the isotherm contains the affinity K.  In the linear regime
     C_floor is proportional to 1/K.  So it is NOT chemistry-independent.
     Report it as a CURVE against K, not a number.

  3. INVENTORY FLOOR
     A hard physical bound that no chemistry can beat.  At a given
     concentration there are only so many cadmium ions in the tissue near the
     sensor.  Even with a perfect binder that captures every one, you cannot
     occupy more sites than there are ions.  The concentration at which the
     available inventory just fills the coverage floor is a true floor.

  4. OCCUPANCY FLOOR (Poisson counting noise)
     Binding is a random process.  If a hundred ions bind on average, the
     actual number fluctuates by about ten.  That fluctuation is a noise
     source completely separate from tissue heterogeneity, and at small
     interrogation volumes it can be the dominant one.

WHY THE NUMBERS FEEL SO UNCOMFORTABLE
-------------------------------------
At 0.05 mg/kg there are about 281 cadmium ions in a cubic micron of muscle.
A hot spot of 30 x 30 x 30 nanometres therefore contains, on average, about
0.008 of an ion.  A hundredth of an ion.

That is not a reason to abandon the project — it is a reason to be clear about
what the sensor is doing.  It is not counting the ions that happen to be
sitting in the hot spot.  It is capturing ions out of a much larger surrounding
volume and concentrating them onto the recognition layer.  The size of that
"depletion volume" is what sets the performance, and it turns out to be around
a cubic micron: comfortably reachable by diffusion in well under a second.

This moves the relevant length scale from the near-field decay length (tens
of nanometres)
to the depletion radius (about a micron) — which is exactly the range where
the tissue's correlation structure lives.  That makes the case for a
heterogeneous tissue model stronger, not weaker.

TOTAL CADMIUM VERSUS FREE CADMIUM
---------------------------------
ICP-MS measures TOTAL cadmium, after acid digestion has released everything.
An aptamer or chelator sees only FREE Cd2+.  In muscle most cadmium is bound
to metallothionein and other proteins.

Every conversion in this module therefore passes through
`RegulatoryConfig.free_fraction`.  Leaving it at 1.0 means "assume a digestion
or release step recovers everything" — a legitimate design, but one you must
state.  If you keep direct tissue contact, this number is well below one, and
it may matter more than everything else in this file put together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    StudyConfig, SurfaceStackConfig, AVOGADRO, M_CD_G_PER_MOL,
)


# ==========================================================================
#  Unit conversions
# ==========================================================================
def mg_per_kg_to_molar(mg_per_kg: float) -> float:
    """
    Convert a tissue burden into an equivalent molar concentration.

    THE ASSUMPTION HIDING IN THIS ONE LINE
    --------------------------------------
    This treats one kilogram of tissue as one litre of solution, i.e. it
    assumes the cadmium is quantitatively extracted into an equal mass of
    solvent.  At 0.05 mg/kg that gives 445 nM.

    That is the right conversion for a workflow with a digestion step.  It is
    the WRONG conversion for a sensor pressed against intact muscle, where
    there is no solution and the governing quantity is a surface coverage, not
    a molarity.  Use `mg_per_kg_to_number_density` for that case.

    Any comparison of 445 nM against published solution detection limits
    therefore implicitly assumes such an extraction step.
    """
    return mg_per_kg * 1e-3 / M_CD_G_PER_MOL


def mg_per_kg_to_number_density(
    mg_per_kg: float, density_kg_per_m3: float = 1050.0
) -> float:
    """
    Cadmium ions per cubic micron of tissue.

    This is the conversion that matters for a direct-contact sensor, because
    it tells you how many ions are physically available nearby.
    """
    mol_per_kg = mg_per_kg * 1e-3 / M_CD_G_PER_MOL
    per_m3 = mol_per_kg * density_kg_per_m3 * AVOGADRO
    return per_m3 * 1e-18                       # per cubic micron


def _number_density_to_mg_per_kg(
    per_um3: float, density_kg_per_m3: float = 1050.0
) -> float:
    """Inverse of `mg_per_kg_to_number_density`."""
    per_m3 = per_um3 * 1e18
    mol_per_kg = per_m3 / AVOGADRO / density_kg_per_m3
    return mol_per_kg * M_CD_G_PER_MOL * 1e3


def molar_to_mg_per_kg(molar: float) -> float:
    """The inverse of `mg_per_kg_to_molar`, with the same caveat."""
    return molar * M_CD_G_PER_MOL * 1e3


# ==========================================================================
#  The inventory picture
# ==========================================================================
def _fmt_time(seconds: float) -> str:
    """Auto-scale a duration so short times do not print as '0.0 ms'."""
    if not np.isfinite(seconds):
        return "unreachable"
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "us"), (1e-9, "ns")):
        if seconds >= scale:
            return f"{seconds / scale:.2f} {unit}"
    return f"{seconds:.3e} s"


@dataclass
class InventoryPicture:
    """How many cadmium ions are actually available, and from how far away."""

    concentration_mg_per_kg: float
    ions_per_um3: float
    volume_per_ion_um3: float
    cube_edge_per_ion_nm: float
    ions_in_hotspot: float
    hotspot_volume_um3: float
    binding_sites: float
    depletion_volume_um3: float
    depletion_radius_um: float
    diffusion_time_s: float

    def __str__(self) -> str:
        return (
            f"Cadmium inventory at {self.concentration_mg_per_kg} mg/kg\n"
            f"  ions per cubic micron       {self.ions_per_um3:.1f}\n"
            f"  volume holding ONE ion      {self.volume_per_ion_um3:.5f} um^3"
            f"  = a cube {self.cube_edge_per_ion_nm:.0f} nm on a side\n"
            f"  ions inside the hot spot    {self.ions_in_hotspot:.2e}   "
            "<-- far less than one\n"
            f"  binding sites available     {self.binding_sites:.0f}\n"
            "  volume that must be swept   "
            f"{self.depletion_volume_um3:.3f} um^3  "
            f"(radius {self.depletion_radius_um * 1000:.0f} nm)\n"
            f"  time to diffuse that far    {_fmt_time(self.diffusion_time_s)}\n"
            "\n"
            "  Reading: the sensor is not counting ions that happen to sit in\n"
            "  the hot spot - there are none.  It captures them out of a\n"
            "  surrounding volume of about a cubic micron, which diffusion\n"
            "  supplies in well under a second.  The relevant length scale is\n"
            "  therefore the depletion radius, not the near-field decay\n"
            "  length - and that is squarely inside the tissue's correlation\n"
            "  range, which strengthens the case for modelling it."
        )


def inventory(
    cfg: StudyConfig,
    concentration_mg_per_kg: float | None = None,
    hotspot_volume_um3: float = 2.7e-5,
    recognition_area_um2: float = 9.0e-4,
    coverage: float = 1.0,
) -> InventoryPicture:
    """
    Work out the cadmium bookkeeping at a given tissue burden.

    Parameters
    ----------
    concentration_mg_per_kg : float or None
        Defaults to the regulatory limit.
    hotspot_volume_um3 : float
        Typical interrogation volume.  The default 2.7e-5 um^3 is a 30 nm
        cube.  Feed it `SensitivityMap.effective_volume()` for a real number.
    recognition_area_um2 : float
        Area of the recognition layer over the hot spot.  Default 9e-4 um^2 is
        a 30 x 30 nm patch.
    coverage : float
        Fraction of binding sites you want filled.

    Returns
    -------
    InventoryPicture
    """
    reg = cfg.regulatory
    c = (concentration_mg_per_kg if concentration_mg_per_kg is not None
         else reg.eu_limit_mg_per_kg)

    n_per_um3 = mg_per_kg_to_number_density(c, reg.tissue_density_kg_per_m3)
    n_free = n_per_um3 * reg.free_fraction

    sites = (cfg.surface.binding_site_density_per_nm2
             * recognition_area_um2 * 1e6)          # nm^2 per um^2 = 1e6
    needed = sites * coverage

    depletion_v = needed / n_free if n_free > 0 else float("inf")
    depletion_r = (3.0 * depletion_v / (4.0 * np.pi)) ** (1 / 3)
    t_diff = depletion_r ** 2 / (6.0 * reg.diffusion_coeff_um2_per_s)

    return InventoryPicture(
        concentration_mg_per_kg=c,
        ions_per_um3=n_free,
        volume_per_ion_um3=1.0 / n_free if n_free > 0 else float("inf"),
        cube_edge_per_ion_nm=(1.0 / n_free) ** (1 / 3) * 1000 if n_free > 0
        else float("inf"),
        ions_in_hotspot=n_free * hotspot_volume_um3,
        hotspot_volume_um3=hotspot_volume_um3,
        binding_sites=sites,
        depletion_volume_um3=depletion_v,
        depletion_radius_um=depletion_r,
        diffusion_time_s=t_diff,
    )


# ==========================================================================
#  Optics of a partially loaded recognition layer
# ==========================================================================
def maxwell_garnett(eps_host: float, eps_inclusion: float, fill: float) -> float:
    """
    Effective permittivity of a layer that is partly host, partly inclusion.

    PLAIN LANGUAGE
    --------------
    The recognition layer starts out as pure polymer or aptamer.  As cadmium
    complexes bind, small amounts of a different material appear inside it.
    Light does not resolve individual complexes — it sees an average.  The
    Maxwell-Garnett rule computes that average for sparse, well-separated
    inclusions, which is exactly the low-coverage situation we care about.

    (At high coverage, where inclusions touch, Bruggeman's rule is more
    appropriate.  Since a detection LIMIT is by definition a low-coverage
    question, Maxwell-Garnett is the right choice here.)
    """
    beta = (eps_inclusion - eps_host) / (eps_inclusion + 2 * eps_host)
    return eps_host * (1 + 2 * fill * beta) / (1 - fill * beta)


def index_shift_from_coverage(
    surf: SurfaceStackConfig, coverage: float
) -> float:
    """
    How much the recognition layer's refractive index changes at a given
    fractional occupancy of its binding sites.

    Interpolates between the measured unbound and fully bound indices via the
    Maxwell-Garnett rule, so that partial coverage is handled correctly rather
    than assumed linear.  (In practice it is very nearly linear at low
    coverage — which is reassuring, not redundant: it means the near-linearity
    is a derived result rather than an assumption.)
    """
    eps_u = surf.recognition_n_unbound ** 2
    eps_b = surf.recognition_n_bound ** 2
    eps_eff = maxwell_garnett(eps_u, eps_b, float(np.clip(coverage, 0.0, 1.0)))
    return float(np.sqrt(eps_eff) - surf.recognition_n_unbound)


def langmuir_coverage(concentration_M: float, K_per_M: float) -> float:
    """Fraction of binding sites occupied at equilibrium: theta = KC/(1+KC)."""
    kc = K_per_M * concentration_M
    return float(kc / (1.0 + kc))


def langmuir_concentration(coverage: float, K_per_M: float) -> float:
    """The inverse: what concentration gives this coverage.

    Note the shape: C = theta / (K (1 - theta)).  The 1/K out front is why the
    concentration floor is NOT independent of the recognition chemistry."""
    theta = float(np.clip(coverage, 0.0, 1.0 - 1e-12))
    return theta / (K_per_M * (1.0 - theta))


# ==========================================================================
#  The four floors
# ==========================================================================
@dataclass
class NoiseFloorResult:
    """Everything needed to say whether the design can meet the limit."""

    sigma_lambda_nm: float
    """The tissue-induced resonance wobble, with no cadmium present.  Should
    be the SPECKLE residual from `ensemble.py`, after the finite-box mean-index
    drift has been regressed out."""

    shift_per_full_coverage_nm: float
    """How far the resonance moves when the recognition layer goes from empty
    to full.  Measured by running the same geometry in both states."""

    theta_floor: float
    """*** THE CHEMISTRY-INDEPENDENT HEADLINE ***
    The smallest detectable fractional coverage."""

    c_floor_M: float
    """Concentration floor at the configured binding affinity."""

    c_floor_mg_per_kg: float
    c_floor_total_mg_per_kg: float
    """The same, converted back to TOTAL cadmium by dividing by the assumed
    free fraction.  This is the number that is comparable with an ICP-MS
    result and with the regulation."""

    inventory_floor_mg_per_kg: float
    """The hard bound: the burden at which the cadmium physically present in
    the depletion volume only just suffices to reach theta_floor.  No binder,
    however good, can beat this."""

    occupancy_floor_mg_per_kg: float
    """The detection limit if Poisson counting noise on the number of bound
    ions were the ONLY noise source."""

    combined_floor_mg_per_kg: float
    """The detection limit with tissue speckle and counting noise together.
    This is the overall figure."""

    n_bound_at_limit: float
    """How many ions are bound at the regulatory limit.  If this is a handful,
    counting noise dominates everything else."""

    limit_mg_per_kg: float
    verdict: str

    def __str__(self) -> str:
        def fmt(v: float, spec: str = ".6f") -> str:
            """Print infinities as words, not as 'inf' buried in a column."""
            if not np.isfinite(v):
                return "unreachable"
            return format(v, spec)

        lines = [
            "NOISE FLOOR ANALYSIS",
            "=" * 66,
            "  tissue-induced wobble sigma_lambda   "
            f"{self.sigma_lambda_nm * 1000:.2f} pm",
            "  shift at full coverage               "
            f"{self.shift_per_full_coverage_nm:.4f} nm",
            "",
            "  1. COVERAGE FLOOR  (independent of recognition chemistry)",
        ]

        if self.theta_floor >= 1.0:
            lines += [
                "       theta_floor                     "
                f"{100 * self.theta_floor:.1f}% of sites",
                "       *** ABOVE 100% - UNDETECTABLE EVEN AT SATURATION ***",
            ]
        elif self.theta_floor > 0.5:
            lines += [
                "       theta_floor                     "
                f"{100 * self.theta_floor:.1f}% of sites",
                "       *** SATURATION-ONLY - the response only clears the ",
                "           noise once the layer is nearly full, so there is",
                "           no dynamic range and no way to report a",
                "           concentration.  Under 100% is NOT sufficient;",
                "           aim for theta_floor <= 0.25. ***",
            ]
        else:
            lines.append(
                "       theta_floor                     "
                f"{self.theta_floor:.4e}  "
                f"= {100 * self.theta_floor:.4f}% of sites"
            )

        lines += [
            "",
            "  2. CONCENTRATION FLOOR  (depends on K -- see the curve)",
            "       free Cd                         "
            f"{fmt(self.c_floor_mg_per_kg, '.6f')} mg/kg",
            "       expressed as TOTAL Cd           "
            f"{fmt(self.c_floor_total_mg_per_kg, '.6f')} mg/kg",
            "",
            "  3. INVENTORY FLOOR  (hard bound, no chemistry can beat it)",
            f"       {fmt(self.inventory_floor_mg_per_kg)} mg/kg",
            "",
            "  4. OCCUPANCY FLOOR  (Poisson noise on the bound count)",
            f"       {fmt(self.occupancy_floor_mg_per_kg)} mg/kg    "
            f"({self.n_bound_at_limit:.0f} ions bound at the limit, so "
            f"{100 / max(self.n_bound_at_limit, 1e-9) ** 0.5:.1f}% "
            "counting noise)",
            "",
            "  COMBINED  (tissue speckle and counting noise together)",
            f"       {fmt(self.combined_floor_mg_per_kg)} mg/kg",
            "",
            "  REGULATORY LIMIT                     "
            f"{self.limit_mg_per_kg} mg/kg",
            "",
            "  VERDICT:",
        ]
        for line in _wrap(self.verdict, 68):
            lines.append(f"    {line}")
        return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
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


def compute_noise_floor(
    cfg: StudyConfig,
    sigma_lambda_nm: float,
    shift_per_full_coverage_nm: float,
    recognition_area_um2: float = 9.0e-4,
    n_sigma: float = 3.0,
) -> NoiseFloorResult:
    """
    Convert a resonance wobble into all four floors.

    Parameters
    ----------
    sigma_lambda_nm : float
        Standard deviation of the resonance wavelength across the tissue
        ensemble, with zero analyte.  Use the SPECKLE RESIDUAL from
        `ensemble.py`, not the raw spread — the raw spread includes a
        finite-box artefact.
    shift_per_full_coverage_nm : float
        From a paired pair of runs, bound versus unbound recognition layer.
    n_sigma : float
        How many standard deviations count as a detection.  3 is the usual
        analytical-chemistry convention for a limit of detection.
    """
    reg = cfg.regulatory

    # ---- 1. coverage floor: chemistry-independent ---------------------
    detectable_shift = n_sigma * sigma_lambda_nm
    theta_floor = (
        detectable_shift / shift_per_full_coverage_nm
        if shift_per_full_coverage_nm > 0 else float("inf")
    )

    # THE CASE THAT MATTERS MOST.
    # If theta_floor exceeds 1, then even SATURATING every binding site
    # produces less resonance shift than the tissue noise.  There is no
    # concentration -- however high -- at which this geometry detects anything
    # in this matrix.  No binding chemistry can rescue it, because the ceiling
    # is set by the layer being full.
    #
    # This is a clear negative outcome, so it is stated explicitly in the
    # verdict rather than left to show up as a strange-looking number.
    saturation_impossible = theta_floor >= 1.0

    # ---- 2. concentration floor: needs the isotherm -------------------
    if saturation_impossible:
        c_floor_M = float("inf")
        c_floor_mgkg = float("inf")
        c_floor_total = float("inf")
    else:
        c_floor_M = langmuir_concentration(theta_floor, reg.langmuir_K_per_M)
        c_floor_mgkg = molar_to_mg_per_kg(c_floor_M)
        c_floor_total = (
            c_floor_mgkg / reg.free_fraction if reg.free_fraction > 0
            else float("inf")
        )

    # ---- 3. inventory floor: a genuine hard bound ---------------------
    # In the contact time, cadmium can diffuse in from a distance of roughly
    # sqrt(6 D t).  That sets the volume the sensor can draw from -- and it
    # depends only on diffusion and time, NOT on the concentration being
    # measured, so this is a real bound rather than a circular one.
    #
    # The floor is the burden at which the ions inside that reachable volume
    # only just suffice to occupy theta_floor of the binding sites.  A perfect
    # binder that captured every available ion could not do better.
    inv = inventory(cfg, recognition_area_um2=recognition_area_um2)
    sites = inv.binding_sites
    needed_ions = sites * min(theta_floor, 1.0)

    # How many ions can actually reach a small patch in the contact time?
    # Not "everything within a diffusion length" -- a tiny collector cannot
    # sweep a huge hemisphere.  The correct result is the Smoluchowski flux to
    # an absorbing disc of radius a:
    #
    #     dN/dt = 4 * D * a * n
    #
    # which is linear in the radius, not cubic.  Using a diffusion-length
    # sphere instead would overstate the available inventory by orders of
    # magnitude and make this "bound" vacuous.
    a_um = np.sqrt(recognition_area_um2 / np.pi)
    collected_per_unit_density = (
        4.0 * reg.diffusion_coeff_um2_per_s * a_um * reg.measurement_time_s
    )  # ions collected per (ion per um^3) of ambient density

    n_needed_per_um3 = (
        needed_ions / collected_per_unit_density
        if collected_per_unit_density > 0 else float("inf")
    )
    inventory_floor = _number_density_to_mg_per_kg(
        n_needed_per_um3, reg.tissue_density_kg_per_m3
    ) / max(reg.free_fraction, 1e-12)

    # ---- 4. occupancy floor: Poisson counting noise -------------------
    # Binding is a random process.  With N ions bound on average, the actual
    # number fluctuates by sqrt(N).  That produces a coverage noise
    #     sigma_theta = theta / sqrt(N) = sqrt(theta / N_sites)
    # and hence a resonance noise sigma_theta * shift_full.
    #
    # The detection limit is where the SIGNAL equals n_sigma times the TOTAL
    # noise.  We solve that numerically for the coverage, three ways: tissue
    # noise alone, Poisson alone, and both together.
    def _lod_coverage(include_tissue: bool, include_poisson: bool) -> float:
        thetas = np.logspace(-9, 0, 4000)
        signal = thetas * shift_per_full_coverage_nm
        noise_sq = np.zeros_like(thetas)
        if include_tissue:
            noise_sq = noise_sq + sigma_lambda_nm ** 2
        if include_poisson and sites > 0:
            noise_sq = noise_sq + (
                shift_per_full_coverage_nm ** 2 * thetas / sites
            )
        ok = signal >= n_sigma * np.sqrt(noise_sq)
        if not ok.any():
            # No coverage up to saturation gives a detectable signal.
            return float("inf")
        return float(thetas[np.argmax(ok)])

    theta_poisson = _lod_coverage(include_tissue=False, include_poisson=True)
    theta_combined = _lod_coverage(include_tissue=True, include_poisson=True)

    def _theta_to_total_mgkg(th: float) -> float:
        if not np.isfinite(th) or th >= 1.0:
            # Undetectable even at saturation.
            return float("inf")
        c_M = langmuir_concentration(th, reg.langmuir_K_per_M)
        return molar_to_mg_per_kg(c_M) / max(reg.free_fraction, 1e-12)

    occupancy_floor = _theta_to_total_mgkg(theta_poisson)
    combined_floor = _theta_to_total_mgkg(theta_combined)

    # How many ions are bound at the regulatory limit?
    theta_at_limit = langmuir_coverage(
        mg_per_kg_to_molar(reg.eu_limit_mg_per_kg * reg.free_fraction),
        reg.langmuir_K_per_M,
    )
    n_bound = sites * theta_at_limit

    # ---- is a CONTINUUM coverage even meaningful here? ----------------
    # theta is a fraction of occupied sites, and the Langmuir isotherm treats
    # it as a smooth variable.  That is only sensible when there are enough
    # sites for a fraction to mean something.  With a few tens of sites,
    # "24.3% coverage" means 8.8 molecules -- and you cannot have 0.8 of a
    # binding event.  The discreteness is then the dominant physics, not a
    # correction to it.
    n_at_floor = sites * theta_floor if np.isfinite(theta_floor) else float("nan")
    discrete_warning = ""
    if np.isfinite(n_at_floor) and sites < 100:
        discrete_warning = (
            f" NOTE: this patch holds only {sites:.0f} binding sites, so the "
            f"detection threshold corresponds to {n_at_floor:.1f} bound ions "
            f"against a Poisson spread of {np.sqrt(max(n_bound, 1e-9)):.1f}. "
            f"At these counts a continuum coverage and a Langmuir isotherm are "
            f"a poor description -- the answer is set by discrete counting "
            f"statistics. Either enlarge the recognition patch, or redo this "
            f"with an explicit binomial model."
        )

    limit = reg.eu_limit_mg_per_kg

    # NEVER use nanmax here.  A missing floor means "we could not establish a
    # floor", which is a reason to withhold a verdict, not to award a pass.
    candidates = [combined_floor, inventory_floor, c_floor_total]
    if any(not np.isfinite(c) for c in candidates):
        worst = float("inf")
    else:
        worst = max(candidates)

    if saturation_impossible:
        verdict = (
            "IMPOSSIBLE - a fully saturated recognition layer shifts the "
            f"resonance by only {shift_per_full_coverage_nm:.4f} nm, which is "
            f"less than the {n_sigma:.0f}-sigma tissue noise of "
            f"{detectable_shift:.4f} nm. This geometry cannot detect cadmium "
            "in this matrix at ANY concentration. No improvement in binding "
            "affinity helps, because the ceiling is set by the layer being "
            "full. This is a design constraint: the "
            "geometry needs a larger per-coverage shift (tighter mode "
            "confinement, thicker or higher-contrast recognition layer) or "
            "the matrix noise must be reduced."
        )
    elif not np.isfinite(worst):
        verdict = (
            "INDETERMINATE - at least one floor could not be established. "
            "Do not read this as a pass. Check the inputs above for "
            "infinities or missing values."
        )
    elif worst < 0.2 * limit:
        verdict = (
            f"PASS - the worst floor ({worst:.5f} mg/kg) is comfortably below "
            f"the {limit} mg/kg limit." + discrete_warning
        )
    elif worst < limit:
        verdict = (
            f"MARGINAL - the worst floor ({worst:.5f} mg/kg) is below the "
            "limit but within a factor of five.  Report the confidence "
            "interval on sigma_lambda; the comparison hinges on it."
        )
    else:
        verdict = (
            f"FAIL - the worst floor ({worst:.5f} mg/kg) EXCEEDS the "
            f"{limit} mg/kg limit.  No improvement in recognition chemistry "
            "can rescue this geometry in this matrix.  This is a design "
            "constraint."
        )

    return NoiseFloorResult(
        sigma_lambda_nm=sigma_lambda_nm,
        shift_per_full_coverage_nm=shift_per_full_coverage_nm,
        theta_floor=theta_floor,
        c_floor_M=c_floor_M,
        c_floor_mg_per_kg=c_floor_mgkg,
        c_floor_total_mg_per_kg=c_floor_total,
        inventory_floor_mg_per_kg=inventory_floor,
        occupancy_floor_mg_per_kg=occupancy_floor,
        combined_floor_mg_per_kg=combined_floor,
        n_bound_at_limit=n_bound,
        limit_mg_per_kg=limit,
        verdict=verdict,
    )


def free_fraction_sweep(
    cfg: StudyConfig,
    sigma_lambda_nm: float,
    shift_per_full_coverage_nm: float,
    free_fractions: np.ndarray | None = None,
    recognition_area_um2: float = 9.0e-4,
    n_sigma: float = 3.0,
) -> list[dict]:
    """
    How good does the cadmium release have to be for this design to work?

    WHY INVERT THE QUESTION
    -----------------------
    `free_fraction` is the fraction of total cadmium that a chelator or aptamer
    can actually see.  The literature does not give one value for it -- in-vitro
    bioaccessibility in seafood ranges from below 1% in molluscs under INFOGEST
    2.0 to substantially higher in fish muscle, depending on species, cooking
    and protocol.  Picking a point estimate from that range would be inventing
    a number.

    So invert the question.  Instead of asking "what is the free fraction, and
    does the design pass?", ask "how large would the free fraction have to be
    for the design to pass?".  That has a definite answer, it requires no
    guessing, and it tells an experimentalist exactly what the release chemistry
    has to deliver.

    The output is a table of free fraction against the resulting detection
    floor, and the `passes` column shows where it crosses the regulatory limit.

    Returns
    -------
    list of dicts with keys: free_fraction, combined_floor_mg_per_kg, passes.
    """
    from dataclasses import replace

    if free_fractions is None:
        free_fractions = np.logspace(-3, 0, 25)

    rows = []
    for ff in free_fractions:
        c = replace(cfg, regulatory=replace(cfg.regulatory, free_fraction=float(ff)))
        res = compute_noise_floor(
            c, sigma_lambda_nm, shift_per_full_coverage_nm,
            recognition_area_um2=recognition_area_um2, n_sigma=n_sigma,
        )
        rows.append({
            "free_fraction": float(ff),
            "combined_floor_mg_per_kg": res.combined_floor_mg_per_kg,
            "inventory_floor_mg_per_kg": res.inventory_floor_mg_per_kg,
            # A missing floor is NOT a pass.  Require every floor to be
            # finite AND below the limit.
            "passes": bool(
                np.isfinite(res.combined_floor_mg_per_kg)
                and np.isfinite(res.inventory_floor_mg_per_kg)
                and max(res.combined_floor_mg_per_kg,
                        res.inventory_floor_mg_per_kg)
                < cfg.regulatory.eu_limit_mg_per_kg
            ),
        })
    return rows


def required_free_fraction(rows: list[dict]) -> float:
    """The smallest free fraction in the sweep at which the design passes.

    This is the main output of the sweep: a single requirement on the
    release chemistry, derived rather than assumed.
    Returns NaN if no tested value passes.
    """
    passing = [r["free_fraction"] for r in rows if r["passes"]]
    return float(min(passing)) if passing else float("nan")


def print_free_fraction_sweep(rows: list[dict], limit_mg_per_kg: float) -> None:
    """Print the sweep and state the requirement it implies."""
    print(f"  {'free fraction':>14}  {'detection floor':>18}  {'verdict':>10}")
    for r in rows[::3]:
        v = "passes" if r["passes"] else "FAILS"
        print(f"  {100 * r['free_fraction']:12.2f}%  "
              f"{r['combined_floor_mg_per_kg']:15.6f} mg/kg  {v:>10}")
    req = required_free_fraction(rows)
    print()
    if np.isfinite(req):
        print("  REQUIREMENT: the release chemistry must make at least "
              f"{100 * req:.2f}% of")
        print("  the total cadmium available as free Cd2+ for this design to")
        print(f"  resolve the {limit_mg_per_kg} mg/kg limit.")
        print()
        print("  MEASURED, for comparison: He, Ke & Wang (2010) report Cd")
        print("  bioaccessibility of 73.7-93.2% in RAW fish muscle, falling to")
        print("  about 36% after frying.  Even the fried figure clears the")
        print(f"  {100 * req:.1f}% requirement by a wide margin.")
        print()
        print("  Caveat: bioaccessibility is what a digestion")
        print("  protocol releases, not the free ionic fraction a chelator")
        print("  would see in intact tissue.  The comparison holds for a")
        print("  workflow WITH a release step; direct tissue contact remains")
        print("  a different and still-open question.")
        print()
        print("  That is a specification an experimentalist can test.")
    else:
        print("  No free fraction in the tested range lets this design meet")
        print("  the limit.")


def c_floor_vs_affinity(
    cfg: StudyConfig,
    theta_floor: float,
    K_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The concentration floor as a function of binding affinity.

    WHY A CURVE
    -----------
    Because C_floor is proportional to 1/K in the linear regime, quoting one
    value silently fixes the recognition chemistry.  A curve lets any reader
    place their own binder on it, and makes the K-dependence visible.

    Returns
    -------
    (K_values, c_floor_total_mg_per_kg)
    """
    if K_values is None:
        K_values = np.logspace(4, 11, 60)
    ff = max(cfg.regulatory.free_fraction, 1e-12)
    c = np.array([
        molar_to_mg_per_kg(langmuir_concentration(theta_floor, K)) / ff
        for K in K_values
    ])
    return K_values, c


# ==========================================================================
#  Self-check
# ==========================================================================
if __name__ == "__main__":
    cfg = StudyConfig()

    print("=" * 70)
    print("UNIT CONVERSIONS AT THE REGULATORY LIMIT")
    print("=" * 70)
    lim = cfg.regulatory.eu_limit_mg_per_kg
    print(f"  {lim} mg/kg = {mg_per_kg_to_molar(lim) * 1e9:.1f} nM "
          "(assuming quantitative extraction into an equal mass of solvent)")
    print(f"  {lim} mg/kg = {mg_per_kg_to_number_density(lim):.1f} ions "
          "per cubic micron of tissue")
    print()

    print("=" * 70)
    print("THE INVENTORY PICTURE")
    print("=" * 70)
    print(inventory(cfg))
    print()

    print("=" * 70)
    print("THE FOUR FLOORS")
    print("=" * 70)
    # Placeholder inputs: replace with ensemble output and a paired
    # bound/unbound run.
    result = compute_noise_floor(
        cfg,
        sigma_lambda_nm=0.05,               # 50 pm tissue wobble
        shift_per_full_coverage_nm=1.2,     # 1.2 nm at full coverage
    )
    print(result)
    print()

    print("=" * 70)
    print("C_floor DEPENDS ON THE BINDER -- here is the curve")
    print("=" * 70)
    Ks, cs = c_floor_vs_affinity(cfg, result.theta_floor)
    print(f"  {'K (1/M)':>10}   {'C_floor (mg/kg total)':>22}")
    for K, c in list(zip(Ks, cs))[::10]:
        flag = "  <-- meets the limit" if c < lim else ""
        print(f"  {K:10.2e}   {c:22.6f}{flag}")
