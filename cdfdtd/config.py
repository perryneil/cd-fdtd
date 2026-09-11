"""
config.py — every number the study depends on, in one place.
=============================================================

WHY THIS FILE EXISTS
--------------------
Scientific code goes wrong most often not because the physics is wrong but
because a number got changed in one script and not in another.  Everything
adjustable in this project lives here, in plain dataclasses, so that a run is
fully described by one object you can print, save to JSON, and put in a paper's
supplementary material.

Nothing in this file does any computation.  It is a description of *what* to
simulate, not *how*.

UNITS — read this before changing anything
------------------------------------------
Tidy3D works in **microns** (um) for length and **Hz** for frequency, and it
sets the speed of light to 1 in those units.  To avoid a whole class of silent
bugs, this project follows one rule:

    *   Every length stored in a config is in **microns**.
    *   Every length that a human would naturally say in nanometres has a
        field name ending in `_nm`, and is converted to microns exactly once,
        by the `.um` helper on that field.

Concentrations are in mol/L unless a name says otherwise.  Tissue burdens are
in mg/kg because that is the unit the European regulation is written in.

HOW TO USE
----------
    from config import StudyConfig
    cfg = StudyConfig()                    # sensible defaults
    cfg = StudyConfig(tissue=TissueConfig(l_c_um=0.4))   # override a piece
    print(cfg.summary())                   # human-readable dump
    cfg.save("run_2026_08_25.json")        # provenance for the paper
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Literal

# --------------------------------------------------------------------------
# Physical constants.  Kept here rather than importing scipy.constants so that
# a reader can see exactly what value was used.
# --------------------------------------------------------------------------
C_LIGHT_UM_S = 2.99792458e14      # speed of light, microns per second
AVOGADRO = 6.02214076e23          # per mole
M_CD_G_PER_MOL = 112.414          # molar mass of cadmium

NM = 1e-3                          # 1 nanometre expressed in microns


# ==========================================================================
#  1. The tissue
# ==========================================================================
@dataclass(frozen=True)
class TissueConfig:
    """
    Describes fish muscle as a *continuous random medium* (CRM).

    PLAIN LANGUAGE
    --------------
    We are not drawing individual muscle fibres.  Instead we say: the
    refractive index of muscle wobbles around an average value, and the
    wobbles have a characteristic size.  Two points close together tend to
    have similar index; two points far apart do not.  The mathematical object
    that captures "how similar, how far apart" is a *correlation function*,
    and the family we use is called Whittle-Matern.

    It has exactly three knobs:

        delta_n_sq   how strong the wobbles are          (variance of index)
        l_c_um       how big the wobbles are             (correlation length)
        m            the *shape* of the wobble spectrum  (fractal-ness)

    plus one baseline:

        n0           the average refractive index of muscle

    THE SUBTLE ONE: `m`
    -------------------
    For m > 1.5 the wobbles have a well-defined total strength.  For m < 1.5
    the medium is a *mass fractal* of dimension 2m, and the mathematics says
    the total wobble strength is infinite — structure keeps appearing as you
    zoom in, forever.  Real tissue does not do that: below a few nanometres
    there is just water and protein, not "structure".  So we impose an
    explicit smallest scale, `l_min_um`.

    This matters enormously and is easy to get wrong.  If you do *not* set an
    inner cutoff, the computer silently uses the voxel size as the cutoff, and
    then your answer changes every time you change the mesh.  See
    `convergence.py`, which exists precisely to check this.
    """

    # --- the four physical parameters -----------------------------------
    n0: float = 1.35
    """Baseline refractive index of fish white muscle.

    SOURCE: Vraalstad et al., *Food Bioproc. Technol.* 18:9392 (2025),
    doi:10.1007/s11947-025-03985-5, measured 1.36-1.38 at 589 nm on salt-cured
    cod at water contents of 55.4% down to 45.4%, with the index rising as
    water content falls.

    EXTRAPOLATION, FLAGGED.  There is NO measurement of raw fish muscle
    refractive index, in any species.  The one fish measurement that exists is
    on dried salt-cured cod at 45-56% water; raw fillet is 78-82% water, well
    outside that band, and the index falls as water content rises.  So this
    number is a derivation, not a measurement.

    THE DERIVATION
    --------------
    Two-component volume mixing, anchored on the cod value:

        n(C_w)  =  C_w * n_water  +  (1 - C_w) * n_solid

        at C_w = 0.50, n = 1.37 (Vraalstad midpoint)
        with n_water = 1.333  ->  n_solid = 2*1.37 - 1.333 = 1.407
        at C_w = 0.80         ->  n = 0.80*1.333 + 0.20*1.407 = 1.348

    Hence 1.35, derived from the cod anchor rather than measured.

    A NEGATIVE RESULT WORTH KNOWING.  Shvachkina et al., *J. Biomed.
    Photonics Eng.* 4(1):010302 (2018), doi:10.18287/JBPE18.04.010302, give
    n against water content for TENDON.  Extrapolating their slope to 80%
    water gives about 1.30, which is clearly too low.  Do not transfer it --
    tendon's solid phase is collagen, muscle's is myofibrillar protein.

    CORROBORATION FROM MAMMALS.  Bolin et al., *Appl. Opt.* 28(12):2297-2303
    (1989), doi:10.1364/AO.28.002297, measured 1.38-1.41 at 632.8 nm across
    mammalian tissues including bovine muscle, with species not a significant
    factor.  Those are lower-water tissues, so 1.35 for raw fillet sits
    sensibly below them.

    DISPERSION IS NOT MEASURED EITHER.  The wavelength dependence published
    for cod is Segelstein's WATER dispersion rigidly shifted by the 589 nm
    offset, not a measurement.  If the study needs dispersion, say that is
    what is being assumed.

    Still notably above the n = 1.33 the aqueous sensor literature assumes,
    which is the point: that offset alone shifts a resonance, and its size is
    one of this study's results."""

    delta_n_sq: float = 4.0e-4
    """Variance of the index fluctuations, i.e. (delta n)^2.

    4e-4 corresponds to an RMS index wobble of 0.02, a typical soft-tissue
    figure.  This is one of the two parameters that `optical_properties.py`
    solves for during calibration — the value here is only a starting guess."""

    l_c_um: float = 0.35
    """Correlation length in microns: the size of a typical index blob.

    For muscle this sits between the myofibril spacing (1-2 um) and the
    organelle/lipid droplet scale (0.3-1 um).  Also solved for during
    calibration."""

    m: float = 1.35
    """Whittle-Matern shape parameter.

    m < 1.5  -> mass fractal of dimension 2m (the usual tissue regime)
    m = 2    -> ordinary exponential correlation
    m -> inf -> Gaussian correlation

    Held FIXED during calibration, because (delta_n_sq, l_c, m) -> (mu_s, g)
    is a 3-into-2 map and would otherwise be under-determined.  Fix it from
    published low-coherence backscattering measurements, then check how much
    your answer moves when you vary it."""

    # --- numerical parameters ---------------------------------------------
    l_min_um: float = 0.030
    """Inner cutoff: the smallest structure size the tissue actually has.

    30 nm is a defensible physical choice AND an affordable one: the
    convergence study (`convergence.py`, study 1) shows it converges at 5 nm
    voxels, whereas a 5 nm cutoff needs voxels finer than 2.5 nm and costs
    eight times more per halving.  Below the myofilament lattice it is
    arguable that muscle has composition but no optically relevant index
    STRUCTURE at all.  Below this the synthesised index field is smoothed.  THIS IS A
    PHYSICAL PARAMETER, not a numerical one — that is the whole point.  If
    your results depend on the mesh, it is because this was left implicit."""

    n_clip_min: float = 1.30
    n_clip_max: float = 1.55
    """A Gaussian random field has unbounded tails, so a handful of voxels
    would otherwise be assigned absurd indices.  We clip, and the generator
    reports what fraction was clipped so it can be reported.  If more
    than ~0.1% is clipped, your delta_n_sq is too large for a Gaussian model."""

    anisotropy_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Stretch factors applied to the correlation length along x, y, z.

    Fish muscle is NOT isotropic — myofibrils are aligned.  (1, 1, 3) would
    mean blobs three times longer along z than across it.  Left at (1,1,1) by
    default so the baseline matches the isotropic literature; `anisotropy.py`
    quantifies the effect of a non-unit aspect ratio."""

    mu_a_per_um: float = 2.0e-5
    """Absorption coefficient, per micron. 2e-5/um = 0.02/mm, a reasonable
    near-infrared soft-tissue value.  In the visible near 540-580 nm, myoglobin
    pushes this up by more than an order of magnitude — which is an argument
    for designing the sensor in the 700-900 nm window."""


# ==========================================================================
#  2. Calibration targets
# ==========================================================================
@dataclass(frozen=True)
class CalibrationConfig:
    """
    What the synthetic tissue has to reproduce before we trust it.

    PLAIN LANGUAGE
    --------------
    We invented a mathematical tissue.  How do we know it behaves like real
    fish muscle?  We check that it scatters light the same way.  Two numbers
    describe that:

        mu_s   scattering coefficient — how often light scatters per unit
               distance.  Big mu_s = cloudy.
        g      anisotropy factor — whether scattering nudges light forward
               (g near 1) or throws it backwards (g near 0).  Tissue is
               strongly forward-scattering, g ~ 0.9.

    `optical_properties.py` adjusts (delta_n_sq, l_c) until the mathematical
    tissue produces these numbers.
    """

    target_mu_s_per_um: float = 5.0e-3
    """Target scattering coefficient at `calib_wavelength_um`, per micron.
    5e-3 /um = 5 /mm.

    SOURCE: Vraalstad et al., *Food Bioproc. Technol.* 18:9392 (2025),
    doi:10.1007/s11947-025-03985-5, measured mu_s = 4-6 /mm on cod muscle
    over 500-1700 nm by double integrating sphere with inverse adding-
    doubling.  5 /mm is the midpoint.

    This is half a generic soft-tissue figure of 10 /mm, which doubles the
    scattering mean free path to ~200 um and therefore STRENGTHENS the
    scale-splitting argument.

    THIS NUMBER IS CONTESTED
    ------------------------
    Van Beers, Kokawa, Aernouts, Watte, De Smet & Saeys, *Meat Science*
    136:50-58 (2018), doi:10.1016/j.meatsci.2017.10.010, measured BOVINE
    skeletal muscle by the same technique, in the same laboratory.  At 800 nm
    they report mu_s = 185 /cm (inner biceps femoris), 164 /cm (outer BF) and
    158 /cm (longissimus) -- that is 15.8-18.5 /mm, about THREE TIMES the cod
    figure.  Their g values (0.927, 0.941, 0.957) corroborate the cod range
    nicely; only mu_s disagrees.

    The likely reason is that the Vraalstad samples were DRIED SALT-CURED cod
    at 45-56% water, not raw fillet at 78-82%.  There is no measurement of
    raw fish muscle mu_s, in any species.

    The two datasets agree far better in the REDUCED coefficient
    mu_s' = mu_s(1-g): cod gives 1.2-6 /cm, bovine longissimus about 6.8 /cm.
    That is the quantity to calibrate against -- see
    `target_mu_s_prime_per_um` below and `optical_properties.check_mu_s_prime`.

    (Note the DOI: 10.1016/j.meatsci.2017.10.009 is a different paper about
    nitrite in Chinese meat products.  The Van Beers paper is ...10.010.)"""

    target_mu_s_prime_per_um: float = 6.0e-4
    """Target REDUCED scattering coefficient, per micron.  6e-4 /um = 6 /cm.

    WHY THIS IS THE BETTER TARGET
    -----------------------------
    `mu_s` and `g` are strongly correlated in any inverse-adding-doubling
    retrieval: the measurement constrains the transport behaviour, and
    splitting it into mu_s and g requires extra information the experiment
    only weakly supplies.  `mu_s' = mu_s(1-g)` is what actually governs light
    transport at depth and it is what the two independent datasets agree on:

        cod, 500-1700 nm             1.2 - 6 /cm   (Vraalstad et al. 2025)
        bovine longissimus, 721 nm   4 - 9 /cm     (Van Beers et al. 2018)

    6 /cm sits in the overlap.  `optical_properties.check_mu_s_prime` reports
    what the calibrated tissue actually delivers against this, so a fit that
    hits (mu_s, g) but misses mu_s' is caught rather than quietly accepted.

    Calibrate on (mu_s, g) to reproduce the one fish measurement that
    exists, and report mu_s' as well, because that is the number the two
    species agree on and the number a reader can compare against anything
    else."""

    target_g: float = 0.93
    """Target anisotropy factor.

    SOURCE: same paper, measured g = 0.90-0.97 on cod muscle.  0.93 is a
    central value.

    Worth knowing: that paper attributes the scattering to the muscle FIBRES
    themselves, 50-200 um across, rather than to sub-micron organelles -- and
    cites Rogers et al. (2014), the same Whittle-Matern reference this project
    uses.  If the dominant scatterers really are that large, the correlation
    length fitted in `optical_properties.calibrate` should come out large too,
    which it does (about 1.6 um)."""

    calib_wavelength_um: float = 0.80
    """Wavelength at which the two targets above are quoted."""

    band_um: tuple[float, float] = (0.60, 1.10)
    """The full band over which the calibrated medium is checked, and the band
    the spectra are recorded on.

    CHOSEN FROM A MEASUREMENT
    ------------------------
    The resonance survey (`find_resonance.py`) measured the mode directly in
    water:

        resonance          860.4 nm
        FWHM               148.9 nm   ->  Q = 5.8
        half-max crossings 790 and 938 nm
        peak-to-floor      2659x

    A band must contain both half-maximum crossings with margin, or the
    linewidth, which every figure of merit divides by, cannot be measured.
    A 500-900 nm band, for example, would miss the 938 nm crossing.

    600-1100 nm gives about 160 nm of margin beyond each crossing -- room for
    the tissue-versus-water shift, for seed-to-seed scatter, and for the
    bound-versus-unbound shift the study exists to measure.

    THE OTHER CONSTRAINTS IT SATISFIES
    ----------------------------------
      * Below 600 nm gold's interband absorption is severe and Q falls
        further.  Starting at 600 stays clear of it.
      * The bulk cell is band_mean / (n_max * min_steps_per_wavelength) and
        must stay under `TissueConfig.l_min_um` (30 nm) or the solver cannot
        resolve what the tissue model contains.  At 600-1100 it is 27.4 nm.
      * Vraalstad's cod measurements run to 1700 nm, so the whole band has
        published tissue optical properties behind it.
      * The 970 nm water overtone lies inside the band, but mu_a there peaks
        at 0.08 /mm -- a 12.5 mm absorption length against a 5 um box.
        Irrelevant to a near-field measurement; it would matter for the
        photon budget at depth, where `photon_budget.py` already handles it.
      * It is cheaper than a 500-900 nm band: 17.5 M cells against 24.2 M,
        because longer wavelengths permit coarser bulk cells."""

    fit_bounds_delta_n_sq: tuple[float, float] = (1e-6, 1e-2)
    fit_bounds_l_c_um: tuple[float, float] = (0.02, 5.0)
    """Search ranges for the two fitted parameters.  Deliberately generous —
    if the optimiser runs to a bound, the target pair is not reachable at the
    fixed `m` and you should say so rather than quietly accept the edge."""


# ==========================================================================
#  3. The sensor
# ==========================================================================
@dataclass(frozen=True)
class BowtieConfig:
    """
    A gold bowtie antenna: two triangles tip to tip, with a small gap.

    PLAIN LANGUAGE
    --------------
    Light hitting a metal nanostructure makes the electrons slosh.  At the
    right colour they slosh resonantly, and the electric field at the sharp
    tips becomes enormously stronger than the light that created it.  That
    intense spot is the "hot spot", and it is where a bound cadmium complex
    would produce the biggest signal.

    A bowtie squeezes the field into the gap between two triangular tips.  It
    gives the strongest confinement of the common designs, which is exactly
    why it is also the design where the *steric* question bites hardest: the
    field may be strongest in a crevice too narrow for the analyte to enter.
    """

    side_nm: float = 120.0
    """Length of each triangle's base, in nanometres."""

    gap_nm: float = 20.0
    """Tip-to-tip separation.  The hot spot lives here.  Smaller gap = stronger
    field = worse steric accessibility.  That trade-off is the point."""

    thickness_nm: float = 40.0
    """Metal film thickness (the bowtie's height in z)."""

    tip_radius_nm: float = 5.0
    """Real fabricated tips are rounded, never mathematically sharp.  A perfect
    point produces a field singularity that never converges with mesh
    refinement, so simulating one is a way of generating a number that means
    nothing.  Always keep this finite."""

    apex_angle_deg: float = 60.0
    """Opening angle at each tip."""

    substrate_index: float = 1.46
    """Fused silica.  Set to None in `sensor.py` for a free-standing structure."""


@dataclass(frozen=True)
class SurfaceStackConfig:
    """
    The chemistry sitting between the gold and the tissue.

    PLAIN LANGUAGE
    --------------
    Between the metal and the fish there are three layers that most simulation
    papers leave out, and all three reduce the signal:

    1. The *recognition layer* — the aptamer, chitosan, or protein that
       actually grabs cadmium.  It has thickness and its own refractive index,
       and that index changes slightly when cadmium binds.  That change is the
       signal.

    2. The *fouling layer* — everything else in the fish that sticks to the
       sensor without being asked.  Unavoidable in any real tissue contact.
       It pushes the analyte further from the metal, where the field is weaker.

    3. The *steric exclusion mask* — not a material, a rule.  A cadmium-ligand
       complex is a finite-sized object; it physically cannot reach into a
       crevice narrower than itself.  Field maxima inside such crevices look
       great in a plot and contribute nothing to the measurement.
    """

    recognition_thickness_nm: float = 3.0
    recognition_n_unbound: float = 1.4530
    recognition_n_bound: float = 1.4543
    """Refractive index of the recognition layer without and with cadmium.

    These are DERIVED rather than free parameters.  Call

        from recognition_layer import estimate_index_change, apply_to_config
        surf = apply_to_config(cfg.surface, estimate_index_change(cfg.surface))

    which computes them from the de Feijter relation (de Feijter, Benjamins &
    Veer, *Biopolymers* 17:1759, 1978) using published refractive index
    increments: DNA 0.17 mL/g, protein 0.185 mL/g, chitosan 0.16-0.18 mL/g.

    The values sitting here are that calculation's mass-only result at the
    default site density, kept as a sensible starting point.

    THE CAVEAT THAT MUST TRAVEL WITH THIS NUMBER: the mass-only difference is a
    LOWER BOUND.  A cadmium ion is 112 Da against an aptamer's ~9500 Da, so
    binding adds almost no mass -- but it makes the aptamer FOLD, and the
    conformational densification moves far more optical weight than the ion
    does.  Published work (J. Phys. Chem. C 2018, doi:10.1021/acs.jpcc.8b07298;
    Analyst 2022, doi:10.1039/D2AN00824F) shows that term dominates and can
    take EITHER SIGN depending on spacer length, immobilisation and packing.

    So: use the mass-only bound for a defensible pessimistic result, and sweep
    the conformational term. Do not quote a single value as if it were known."""

    fouling_thickness_nm: float = 4.0
    fouling_n: float = 1.45
    """A nonspecific protein adlayer.  Mandatory for any tissue-contact
    scenario.  Set thickness to 0 to model the (unrealistic) clean case and
    quantify how much it costs you."""

    analyte_hydrodynamic_radius_nm: float = 0.6
    """Effective radius of the hydrated cadmium-ligand complex.  Any region of
    the near field whose local aperture is smaller than this is excluded from
    the sensing budget by `sensor.steric_accessibility_mask`."""

    steric_accessible_fraction: float = 0.49
    """Fraction of the nominal binding sites the hydrated analyte can actually
    reach, from `sensor.steric_accessibility_mask` applied to the COATED
    geometry at the analyte radius above.  The bare metal returns 0.846; the
    coatings are what close the narrow apertures, so the coated number is the
    one that governs.

    `recognition_layer.estimate_index_change` uses it, so "full coverage"
    means every site the analyte can reach rather than every site that
    exists.  Setting it to 1.0 counts every site.

    The polymer itself is NOT scaled by this: every grafted aptamer is
    present whether or not cadmium can reach it.  Only the bound cadmium mass
    and the conformational response scale, because only occupied sites fold.
    """

    binding_site_density_per_nm2: float = 0.04
    """Capture sites per square nanometre of recognition surface.
    0.04 /nm^2 = 4e12 /cm^2.

    SOURCE
    ------
    This sets how many cadmium ions a patch can hold, and therefore BOTH the
    maximum achievable signal AND the Poisson counting noise.  An optimistic
    value flatters the design twice over.

    Steel, Herne & Tarlov (*Anal. Chem.* 70:4670, 1998) varied ssDNA density
    over (1-10)e12 /cm^2 by chronocoulometry and found that hybridization
    efficiency has a MAXIMUM: it is 100% only below about **4e12 /cm^2**,
    which is 2500 A^2 per molecule -- the footprint of a 25-mer duplex.  Pack
    tighter and the duplex simply does not fit.

    Herne & Tarlov (*JACS* 119:8916, 1997) reach the same conclusion from the
    other direction: their most densely packed pure-ssDNA sample hybridized
    NOT AT ALL, blocked sterically and electrostatically.  Their MCH-diluted
    monolayers span 2.9e10 to 5.7e12 /cm^2.

    Pons et al. (*Analyst* 147:4197, 2022) used 4 pmol/cm^2 = 2.4e12 /cm^2.

    4e12 /cm^2 = **0.04 /nm^2** is therefore the defensible default: the
    highest density at which every site still works.

    INDEPENDENT CHECK: running the de Feijter relation backwards
    (`recognition_layer.max_plausible_site_density`) gives a purely physical
    ceiling near 0.17 /nm^2 -- above the measured maximum, as it should be.
    The physics bound and the experiments agree.

    This is the steric-accessibility argument of `sensor.py`, confirmed
    experimentally on a different system.

    To set it for YOUR layer: measure the layer index by ellipsometry or an
    SPR angle shift and invert with
    `recognition_layer.site_density_from_layer_index`."""


# ==========================================================================
#  4. The FDTD run itself
# ==========================================================================
@dataclass(frozen=True)
class FDTDConfig:
    """
    Numerical settings for Tidy3D.

    PLAIN LANGUAGE
    --------------
    FDTD chops space into little cubes and time into little ticks, then lets
    Maxwell's equations play out.  Two competing pressures set the cube size:
    the hot spot needs cubes of about a nanometre, and the tissue domain needs
    to be several microns across.  Using nanometre cubes everywhere would need
    more cubes than exist in any computer, so we refine only where it matters.

    Tidy3D supports this: `GridSpec.auto` with mesh-override structures gives
    a fine grid in the gap and a coarse one in the tissue (a solver with a
    single uniform grid, such as Meep, cannot).
    """

    domain_um: tuple[float, float, float] = (5.0, 5.0, 2.5)
    """Overall simulation box, in microns.  Must be large compared with the
    tissue correlation length `l_c_um` or your random medium is really just one
    blob and the ensemble statistics are meaningless.  `convergence.py`
    verifies this."""

    dl_gap_nm: float = 2.0
    """Grid step inside the bowtie gap.  This is the expensive number.

    2 nm is the production default for the ENSEMBLE, and the reasoning is
    worth stating because it is what makes the campaign affordable.

    The ensemble measures a seed-to-seed VARIANCE.  A systematic error from a
    coarse mesh shifts every realisation the same way and cancels in the
    spread -- it is common-mode.  So the ensemble does not need the mesh that
    a single absolute resonance wavelength would need.

    What you do need is one mesh-convergence series (stage 3c, four runs in
    water at 4, 2, 1 and 0.5 nm) to quantify the systematic offset, plus the
    paired bound/unbound run at fine mesh.  Six fine-mesh runs instead of
    forty-seven: the same information for a fraction of the cost.

    Halving this also nearly doubles the number of time steps, because the
    Courant condition ties the time step to the smallest cell.  That is why it
    costs more than the cell count alone suggests."""

    dl_metal_nm: float = 4.0
    """Grid step in the rest of the metal.

    Kept at twice `dl_gap_nm` so refining the gap refines its neighbourhood
    too.  If this is left FINER than the gap step it silently overrides it, and
    a mesh-convergence series can end up testing the same mesh twice -- see
    `convergence.print_gap_mesh_study`, which detects exactly that."""

    min_steps_per_wavelength: int = 20
    """Grid resolution in the bulk tissue, as steps per wavelength in the
    medium.

    A CONSTRAINT THAT COUPLES THIS TO THE TISSUE MODEL
    --------------------------------------------------
    This cannot be lowered freely to save money.  The tissue's index spectrum
    is deliberately cut off at `TissueConfig.l_min_um` (30 nm), and stage 3 of
    the convergence study exists to check that the answer does not depend on the
    mesh.  If the FDTD bulk cell is COARSER than l_min, the solver averages
    away structure between l_min and the cell size -- re-truncating the
    spectrum at its own grid scale and quietly undoing that whole argument.

    At the 500-900 nm band in n = 1.35 muscle:

        min_steps_per_wvl = 12  ->  40 nm cells  ->  VIOLATES a 30 nm cutoff
        min_steps_per_wvl = 16  ->  30 nm cells  ->  exactly at the limit
        min_steps_per_wvl = 20  ->  24 nm cells  ->  safe

    `simulation.check_mesh_vs_cutoff` enforces this and warns if you break it.
    To go coarser, raise l_min first -- and justify that on tissue structure,
    not on cost."""

    tissue_voxel_nm: float = 10.0
    """Voxel size of the imported random-index map.  Deliberately separate from
    the FDTD grid: the tissue map is physical data, the FDTD grid is numerics.
    Sweeping this is how you prove your noise floor is not a mesh artefact."""

    taper_width_um: float = 0.4
    """Width of the buffer in which the index fluctuations fade smoothly to the
    uniform background before reaching the absorbing boundary.

    Absorbing boundaries assume the medium does not change along the direction
    they absorb.  Running a random medium straight into one produces spurious
    reflections and, in long runs, slow field *growth*.  Tapering is the fix.
    Do not set this to zero."""

    source: Literal["planewave", "tfsf"] = "planewave"
    """Which illumination to use.

    `planewave` is the robust default.  `tfsf` separates incident from
    scattered light and is the right tool for a clean scattering
    cross-section, but it assumes a uniform grid in the two directions
    transverse to its injection axis — and our mesh overrides deliberately
    make the grid non-uniform there.  Tidy3D will warn you.  Use `tfsf` only
    for cross-section work on a uniform grid; for finding a resonance and its
    linewidth, `planewave` plus the hot-spot probe is cleaner."""

    absorber_layers: int = 20
    """Thickness of the absorbing boundary, in layers.

    A COST LEVER THAT IS EASY TO MISS
    ---------------------------------
    Absorbing layers sit OUTSIDE the domain you asked for, so they enlarge the
    simulation.  At 40 layers of ~24 nm the box grows by about 1 um on every
    side: a 5 um domain becomes 6.9 um of grid, which is 2.6x the cells, spent
    entirely on physics you do not care about.

    20 layers roughly halves that.  It reflects a little more, but two things
    make that acceptable here: the tissue fluctuations are already tapered to a
    uniform background before the boundary (`tissue.generate_tissue`), so the
    absorber sees the translation-invariant medium it was designed for; and the
    observable is a near field in the gap, far from the walls.

    Check rather than assume -- `simulation.check_decay` shows late-time energy
    growth if the boundary is inadequate."""

    boundary: Literal["pml", "absorber"] = "absorber"
    """Boundary treatment: `absorber` (adiabatic lossy layer) or `pml`.

    ALL VALUES IN THE PAPER USE `pml`.  `boundary_test.py` shows that the
    absorber does not converge under a depth sweep (the linewidth changes with
    domain depth) while PML does.  The default here remains `absorber` so the
    reference absorber ensemble in results/ can be reproduced; pass
    `--boundary pml` to the scripts, or set this field, for new runs."""

    run_time_ps: float = 0.35
    """How long to let the fields run, in picoseconds.

    DERIVED FROM THE PHYSICS
    ------------------------
    One might expect random media to ring down slowly and need several ps.
    Two facts rule that out for THIS system:

      * The scattering mean free path is ~200 um and the box is 5 um.  There
        is no cavity and no multiple scattering to trap light -- a photon
        crosses the box in about 22 fs and leaves.
      * The plasmon itself is lossy.  Its amplitude decay time is 2Q/omega,
        which at Q = 10 and 750 nm is 6.8 fs.

    `simulation.required_run_time_ps` computes the real requirement: time for
    the source pulse to finish (about 12 fs for this band) plus time for the
    resonance to decay to the shutoff threshold.  Even at Q = 50 that totals
    0.56 ps.

    THIS IS THE SINGLE BIGGEST COST LEVER: cost is linear in run time."""

    expected_q: float = 12.0
    """Expected resonance quality factor, used to derive the run time.

    A gold bowtie in the visible typically lands between 5 and 20.  If a run
    comes back with a much higher Q, `simulation.check_decay` reports that the
    field had not decayed -- the check and the derivation share this number, so
    they cannot drift apart."""

    shutoff: float = 1e-7
    """Stop early once the field energy has decayed to this fraction of its
    peak.  Lower than Tidy3D's 1e-5 default so the decaying tail of the
    resonance is fully captured."""

    freq_points: int = 301
    """Number of wavelength samples in the recorded spectrum.  Needs to be
    dense enough to resolve the resonance linewidth to a small fraction of the
    resonance shifts you care about."""

    symmetry: tuple[int, int, int] = (0, 0, 0)
    """Symmetry planes.  A bowtie in a *uniform* medium has symmetry you could
    exploit for a big speed-up — but a *random* medium has none.  Keep this at
    (0,0,0) for tissue runs; you may use it for the water reference runs, and
    `simulation.py` will warn if you try to use it with a random medium."""

    # -- the mesh-convergence series (stage 3c) ---------------------------
    mesh_study_dl_nm: tuple[float, ...] = (4.0, 2.0, 1.0)
    """Gap cell sizes for the mesh-convergence series, in nanometres.

    WHY 0.5 nm IS NOT IN THIS LIST BY DEFAULT
    -----------------------------------------
    Cost in FDTD scales as (cells) x (time steps), and BOTH grow when you
    halve a cell — the Courant condition ties the time step to the smallest
    cell in the whole domain.  Measured on this exact geometry, relative to
    the 2 nm ensemble mesh:

        4.0 nm   13.9 M cells,   70,969 steps   ->  0.33x
        2.0 nm   21.4 M cells,  138,464 steps   ->  1.00x
        1.0 nm   41.5 M cells,  268,987 steps   ->  3.77x
        0.5 nm  112.3 M cells,  530,820 steps   -> 20.11x

    That single 0.5 nm point costs more than the other three combined and
    more than half of a forty-seed ensemble.  Run 4/2/1 first, look at how
    much the peak has stopped moving, and pay for 0.5 nm only if the trend
    has not flattened.  Appending 0.5 to this tuple is all it takes."""

    mesh_study_domain_um: tuple[float, float, float] | None = (3.0, 3.0, 2.0)
    """Simulation box for the mesh-convergence series only.  `None` reuses
    the production `domain_um`.

    The mesh study runs in WATER.  The production box is 5 x 5 x 2.5 um
    because the *tissue* needs to hold enough independent correlation
    volumes to make the ensemble statistics meaningful — a requirement that
    simply does not apply to a homogeneous water reference, which only needs
    room for the antenna, its near field and the absorber.  Shrinking it to
    3 x 3 x 2 um cuts the series cost by about a third at every mesh.

    Two caveats:

      * The absolute resonance wavelength in the small box may differ
        slightly from the large one.  That is fine — the mesh study measures
        a DIFFERENCE between meshes, and the box is identical across the
        series.
      * `convergence.gap_mesh_study` prints the box it used, so the shrink
        can never happen silently."""


# ==========================================================================
#  5. The ensemble
# ==========================================================================
@dataclass(frozen=True)
class EnsembleConfig:
    """
    How many random tissues to generate, and how to read the spread.

    PLAIN LANGUAGE
    --------------
    Every fish is different, and every patch of one fish is different.  So we
    generate many statistically identical but individually different tissues,
    run each, and look at how much the answer moves.  That spread — with *no
    cadmium present at all* — is the noise floor imposed by the tissue itself.

    Two warnings live in the defaults below.
    """

    n_seeds: int = 40
    """Number of independent tissue realisations.

    Estimating a *spread* is much harder than estimating an average.  The
    relative error on an estimated standard deviation is about
    1/sqrt(2(N-1)): 14% at N=25, 10% at N=50, 7% at N=100.  Since the noise
    floor is proportional to that spread, N=25 gives you a headline number
    with a 14% error bar.  Say so, or run more seeds."""

    bootstrap_resamples: int = 5000
    """Resamples used to put a confidence interval on the spread.  Cheap —
    it is pure post-processing on numbers you already have."""

    base_seed: int = 20260825
    """Master seed.  Every realisation's seed is derived from this, so the
    whole ensemble is exactly reproducible from one integer."""

    regress_out_mean_index: bool = True
    """Split the spread into two parts before quoting a noise floor.

    Each finite box has its own average index, which drifts from seed to seed
    purely because the box is finite.  A resonance shifts with average index
    for entirely ordinary reasons.  If you do not subtract that, your
    "heterogeneity noise floor" is partly just a statement about the box size
    you chose.  Leave this True; `ensemble.py` reports both numbers."""


# ==========================================================================
#  6. The regulatory conversion
# ==========================================================================
@dataclass(frozen=True)
class RegulatoryConfig:
    """
    Turning a wobble in nanometres into a number in mg/kg.

    PLAIN LANGUAGE
    --------------
    The end product has to be comparable with a legal limit, which is written
    in milligrams of cadmium per kilogram of fish.  Getting there needs a chain
    of conversions, and each link is an assumption worth stating out loud.
    """

    eu_limit_mg_per_kg: float = 0.05
    """Commission Regulation (EU) 2023/915, cadmium in fish muscle."""

    tissue_density_kg_per_m3: float = 1050.0
    """Used to convert mg/kg into ions per cubic micron."""

    langmuir_K_per_M: float = 2.90e7
    """Binding affinity of the recognition layer, in 1/molar.

    SOURCE: Wu, Zhan, Wang & Zhou, *Analyst* 139:1550-1561 (2014),
    doi:10.1039/C3AN02117C, report Kd = 34.5 nM for the DNA aptamer they
    selected against Cd(II) -- the most widely reused Cd(II) aptamer, listed
    as CAO-1 in the Gao et al. review (*Biosensors* 13:612, 2023).
    K = 1/Kd = 2.90e7 /M.

    BUT NOTE THE SPREAD: the same review lists another Cd(II) aptamer (CAO-3)
    at Kd = 81.39 uM, about 2400x weaker.  Published aptamers for this target
    span more than three orders of magnitude in affinity.

    Note carefully: because the floor is inverted through a Langmuir isotherm,
    the resulting concentration scales as 1/K.  The claim that the noise floor
    is "independent of recognition chemistry" is therefore not true as usually
    stated.  What IS chemistry-independent is the minimum detectable *surface
    coverage*, and the inventory-limited bound below.  `noise_floor.py`
    reports all three separately."""

    free_fraction: float = 0.84
    """Fraction of total cadmium available to the sensor.

    A major assumption of the whole calculation.  Reference
    methods (ICP-MS) measure TOTAL cadmium after acid digestion; a chelating or
    aptamer layer sees only the free ion, and in muscle most cadmium is bound
    to metallothionein and other cysteine-rich ligands.

    MEASURED FOR FISH MUSCLE
    ------------------------
    He, Ke & Wang (*J. Agric. Food Chem.* 58:3517, 2010,
    doi:10.1021/jf100227n) measured in-vitro digestion bioaccessibility of Cd
    in RAW muscle of two farmed marine fish, three size classes each:

        seabass        93.2 +/- 2.9%  (small)     84.8 +/- 1.9%  (large)
        red seabream   77.1 +/- 4.9%  (small)     89.9 +/- 1.9%  (medium)
                       73.7 +/- 2.8%  (large)

    Range 73.7-93.2%, mean about 84%, which is this default.  Cadmium was the
    MOST bioaccessible of the six elements they studied.  Maulvault et al.
    (*Food Chem. Toxicol.* 49:2808, 2011) find 95.3% in raw crab brown meat.

    A mollusc INFOGEST study reports below 1% (Milea et al., *J. Xenobiot.*
    15:92, 2025), but molluscs are not fish muscle.

    TWO CAVEATS THAT STILL MATTER
    -----------------------------
    1. COOKING LOWERS IT, sometimes sharply: frying dropped Cd bioaccessibility
       to 36.2% in large seabass.  If the sample is cooked, use that.

    2. BIOACCESSIBILITY IS NOT THE SAME QUANTITY AS FREE IONIC Cd2+.  It
       measures what a digestion protocol releases into solution.  That is the
       right proxy for a workflow WITH a digestion or release step.  For a
       sensor pressed against intact tissue, what matters is the free ionic
       fraction, which is not this number and is probably much smaller.

    So: 0.84 is right for a digestion-based workflow and cited.  For direct
    tissue contact it is an upper bound.  `noise_floor.free_fraction_sweep`
    still reports the requirement across the whole range.

    FOR DIRECT CONTACT, USE `free_fraction_direct_contact` BELOW.  It is
    between three and nine orders of magnitude smaller, and the two numbers
    are not substitutes for each other in any regime."""

    free_fraction_direct_contact: float = 1.0e-3
    """Free IONIC Cd(II) fraction, for a sensor pressed against intact tissue
    with no digestion or release step.

    THIS HAS NEVER BEEN MEASURED
    ----------------------------
    Not in fish muscle, not in any vertebrate muscle, not by DGT, ion-
    selective electrode, Donnan membrane, voltammetry or ultrafiltration.
    The value here is a DERIVED BOUND.

    WHAT IS KNOWN, AND WHAT EACH PIECE CONTRIBUTES
    ----------------------------------------------
    1. In fish LIVER and GONAD cytosol, essentially all cadmium is
       ligand-bound.  Urien, Jacob, Couture & Campbell, *Environments*
       5(9):102 (2018), doi:10.3390/environments5090102, separated white
       sucker liver cytosol by SEC-ICP-MS: all hepatic cytosolic Cd eluted in
       the 10-2 kDa pool, the major peak co-eluting with a metallothionein
       standard, with NO free Cd fraction detected.  That is a detection-limit
       statement, not a quantified free fraction -- and it is liver, not
       muscle.  Do not import it.

    2. In MUSCLE, metallothionein is at or below detection.  Kovarova, Kizek,
       Adam et al., *Sensors* 9(6):4789-4803 (2009), doi:10.3390/s90604789,
       found MT non-detectable in carp muscle in every experimental group
       while muscle Cd rose dose-dependently from 0.05 to 81.18 ug/kg.  So the
       liver argument does not carry over: MT is probably NOT the dominant Cd
       ligand in muscle.

    3. Which leaves glutathione as the realistic ligand.  Watly, Laczkowski,
       Padjasek & Krezel, *Inorg. Chem.* 60(7):4657-4675 (2021),
       doi:10.1021/acs.inorgchem.0c03639, measured Cd-glutathione by
       potentiometry: log beta(CdL) = 9.00(1), log beta(CdL2) = 15.05(2), and
       an apparent conditional constant at pH 7.4 of log K = 5.93.

    THE ARITHMETIC, WHICH IS OURS AND NOT THE LITERATURE'S
    -------------------------------------------------------
        K(pH 7.4) = 10^5.93 = 8.5e5 /M
        at a nominal 1 mM cytosolic GSH:
            free fraction = 1/(1 + K[GSH]) = 1.2e-3       <- this default
        including CdL2 formation, lower still.
        if even 1 uM of MT-like cysteine-rich ligand is present
        (log K ~ 14.8, Quinn & Wilcox, *Metallomics* 16:mfae041 (2024),
        doi:10.1093/mtomcs/mfae041):
            free fraction ~ 1e-9

    So: of order 1e-3 or smaller, plausibly spanning 1e-9 to 1e-3.  Label the
    calculation as a construction from two cited constants, not as a measured
    value.

    ONE GAP THAT REMAINS.  The GSH concentration in fish WHITE MUSCLE is the
    missing ingredient (see Wu et al., *Comp. Biochem. Physiol. C* 2002, PMID
    11912048).  1 mM is a nominal cytosolic figure, not a fish-muscle
    measurement.

    AND A SECOND, INDEPENDENT REASON THE DIGESTION NUMBER IS WRONG HERE.
    A sensor pressed against intact tissue sees INTERSTITIAL fluid at the
    contact plane, not cytosol.  Every speciation number above is cytosolic.
    Nothing in the literature resolves this, and no amount of modelling can.

    WHAT THIS MEANS FOR THE DEVICE.  Run `noise_floor.free_fraction_sweep`
    across both regimes.  If the sensor only clears the EU limit at 0.84, its
    claim is conditional on a digestion or release step."""

    free_fraction_direct_contact_range: tuple[float, float] = (1e-9, 1e-3)
    """The plausible span for the direct-contact regime: GSH-only at the top,
    MT-present at the bottom.  Six orders of magnitude, because that is the
    actual state of knowledge."""

    measurement_time_s: float = 60.0
    """How long the sensor is held against the tissue.

    This is what sets the size of the region the sensor can draw cadmium from:
    ions diffuse a distance of roughly sqrt(6 D t) in time t, so a longer
    contact time sweeps a larger volume and lowers the inventory-limited
    floor.  It is a genuine design parameter and should be reported.

    Sixty seconds is a plausible screening measurement.  Note that this makes
    the inventory floor a real, non-circular bound: the reachable volume comes
    from diffusion physics, not from the concentration you are trying to
    measure."""

    diffusion_coeff_um2_per_s: float = 500.0
    """Hindered diffusion coefficient of Cd2+ in tissue, um^2/s
    (5e-10 m^2/s).  Used only to report an equilibration timescale for the
    depletion volume, so you can check the measurement is not transport-limited."""


# ==========================================================================
#  6b. The money
# ==========================================================================
@dataclass(frozen=True)
class BudgetConfig:
    """
    A hard ceiling on what the campaign may spend, and the arithmetic needed
    to compare against it.

    HOW THE COST IS ESTIMATED
    -------------------------
    Runs do not all cost the same: the mesh-convergence series runs at finer
    meshes than the ensemble, and a finer mesh costs more in two ways at once
    (more cells, and more time steps, because the Courant condition ties the
    time step to the smallest cell anywhere in the box).  Multiplying Tidy3D's
    per-simulation estimate by the number of runs would understate the total
    by roughly 45%.  `simulation.estimate_campaign_cost` therefore weights
    each run by its (cells x steps), and `--submit` refuses to start when the
    weighted total exceeds `flexcredits`.
    """

    flexcredits: float = 50.0
    """Maximum FlexCredits the whole campaign may cost.  `run_all.py --submit`
    aborts above this unless `--over-budget` is passed explicitly."""

    contingency: float = 1.15
    """Multiplier applied to the projected total before comparing against the
    ceiling.  Tidy3D's per-simulation figure is a worst case for the run time,
    but the WEIGHTS are a model, and a model that is 15% optimistic when you
    have already committed forty runs is an expensive kind of optimistic."""


# ==========================================================================
#  6c. The one number that converts an index change into a wavelength
# ==========================================================================
@dataclass(frozen=True)
class TransductionConfig:
    """
    How much the resonance moves per unit change in refractive index.

    WHY THIS IS ITS OWN DATACLASS
    -----------------------------
    These values are used by several stages (the stage 3b box-wobble
    conversion, the stage 5b shift estimate and the synthetic ensemble), so
    they live in one place and cannot drift apart.  They are measured by the
    paired bound/unbound protocol in `paired_run.py`.

    NOTE: the values below come from the paired run with the ADIABATIC
    ABSORBER boundary.  The PML values used in the paper (S_bulk 194.1
    nm/RIU, f = 0.62 at 3 nm) are stored in `redesign.MEASURED` and
    `redesign.MEASURED_LAYER_SERIES`, which the figures and
    `studies/audit_values.py` use.
    """

    bulk_sensitivity_nm_per_riu: float = 181.9
    """Resonance shift per refractive index unit, for the bulk medium.

    MEASURED (absorber boundary).  `paired_run` stepped the aqueous
    background from n = 1.333 to 1.338 and measured the resonance move by
    909 pm, giving 181.9 nm/RIU.  Gold bowties of this size are typically
    quoted at 200-400 nm/RIU; this mode sits a little below because it leans
    into the n = 1.46 substrate rather than into the water.

    The shift was measured by whole-line least squares
    (`observables.relative_shift_pm`), stable to 3.3% across signal
    thresholds.  Differencing two peak fits is not reliable for shifts this
    small (see `observables.relative_shift_pm`)."""

    mode_overlap: float = 0.5674
    """Fraction of the mode's sensing weight inside the recognition layer.

    MEASURED (absorber boundary), from the stored paired simulations.

        applied layer index step            0.0013   (1.4530 -> 1.4543)
        measured layer shift                134.2 pm
        bulk shift for a 0.005 step         909.5 pm  ->  181.9 nm/RIU
        overlap = 134.2 / (181.9 * 0.0013 * 1000)  =  0.5674

    The denominator uses the index step the simulation actually applied
    (0.0013), not the de Feijter full-coverage estimate (0.0116).

    Note that f is a layer-to-water response ratio rather than a bounded
    fraction; see `layer_series.py` for the four-thickness series that gives
    the sensing decay length (12-21 nm)."""

    measured: bool = True
    """True when both values above are measured by `paired_run.py` rather
    than prior estimates."""


# ==========================================================================
#  7. Everything, bundled
# ==========================================================================
@dataclass(frozen=True)
class StudyConfig:
    """The complete description of one study.  Print it, save it, cite it."""

    tissue: TissueConfig = field(default_factory=TissueConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    bowtie: BowtieConfig = field(default_factory=BowtieConfig)
    surface: SurfaceStackConfig = field(default_factory=SurfaceStackConfig)
    fdtd: FDTDConfig = field(default_factory=FDTDConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    regulatory: RegulatoryConfig = field(default_factory=RegulatoryConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    transduction: TransductionConfig = field(
        default_factory=TransductionConfig)

    # -- convenience ------------------------------------------------------
    @property
    def wavelengths_um(self):
        """The wavelength grid the spectra are recorded on."""
        import numpy as np
        lo, hi = self.calibration.band_um
        return np.linspace(lo, hi, self.fdtd.freq_points)

    @property
    def freqs_hz(self):
        """The same grid expressed as frequencies, which is what Tidy3D wants."""
        return C_LIGHT_UM_S / self.wavelengths_um

    def save(self, path: str) -> None:
        """Write the whole configuration to JSON, for reproducibility."""
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    def summary(self) -> str:
        """A human-readable dump, suitable for pasting into a lab notebook."""
        t, c, f, e, r = (
            self.tissue, self.calibration, self.fdtd,
            self.ensemble, self.regulatory,
        )
        lines = [
            "STUDY CONFIGURATION",
            "=" * 60,
            "Tissue (continuous random medium)",
            f"  baseline index n0            {t.n0}",
            f"  index variance (dn)^2        {t.delta_n_sq:.3e}  "
            f"(RMS dn = {t.delta_n_sq ** 0.5:.4f})",
            f"  correlation length           {t.l_c_um * 1000:.0f} nm",
            f"  shape parameter m            {t.m}   "
            f"({'mass fractal, D=' + format(2 * t.m, '.2f') if t.m < 1.5 else 'finite variance'})",
            f"  inner cutoff                 {t.l_min_um * 1000:.1f} nm",
            f"  anisotropy (x,y,z)           {t.anisotropy_xyz}",
            "",
            "Calibration targets",
            f"  mu_s at {c.calib_wavelength_um * 1000:.0f} nm         "
            f"{c.target_mu_s_per_um:.3e} /um  "
            f"({c.target_mu_s_per_um * 1000:.1f} /mm)",
            f"  anisotropy g                 {c.target_g}",
            f"  scattering mean free path    "
            f"{1.0 / c.target_mu_s_per_um:.1f} um",
            "",
            "FDTD",
            f"  domain                       {f.domain_um} um",
            f"  gap grid step                {f.dl_gap_nm} nm",
            f"  tissue voxel                 {f.tissue_voxel_nm} nm",
            f"  taper before boundary        {f.taper_width_um * 1000:.0f} nm",
            f"  boundary                     {f.boundary}",
            "",
            "Ensemble",
            f"  seeds                        {e.n_seeds}  "
            f"(rel. error on sigma ~ {100 / (2 * (e.n_seeds - 1)) ** 0.5:.0f}%)",
            f"  regress out mean index       {e.regress_out_mean_index}",
            "",
            "Regulatory",
            f"  EU limit                     {r.eu_limit_mg_per_kg} mg/kg",
            f"  assumed free Cd fraction     {r.free_fraction}"
            + ("   <-- ASSUMES A RELEASE STEP" if r.free_fraction == 1.0 else ""),
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print(StudyConfig().summary())
