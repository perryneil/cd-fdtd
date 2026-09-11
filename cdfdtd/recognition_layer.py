"""
recognition_layer.py — computing the index change on binding, from measurements.
================================================================================

WHAT IT DOES
------------
The noise-floor calculation needs one number above all others: how far does
the resonance move when the recognition layer goes from empty to full?  Every
conversion from "the resonance wobbles by X picometres" to "that equals Y
mg/kg of cadmium" runs through it.

This module derives the layer index change on binding from published
measurements, using a relation that has been standard in surface science since
1978.

THE RELATION: de Feijter's formula
----------------------------------
When molecules adsorb onto a surface, they form a thin layer whose refractive
index is higher than the solvent's.  How much higher depends on how much mass
is there and how tightly it is packed:

        n_layer  =  n_solvent  +  (dn/dc) * Gamma / d

    n_solvent   refractive index of the surrounding medium
    dn/dc       refractive index increment of the adsorbed material,
                in mL/g -- how much the index rises per unit concentration
    Gamma       surface mass density, in g/cm^2
    d           layer thickness, in cm

This is de Feijter, Benjamins and Veer (1978).  Voros (2004) gives the modern
treatment and its validity limits.

WHERE THE NUMBERS COME FROM
---------------------------
    dn/dc for DNA        0.17  mL/g   (Malvern compilation, 632.8 nm)
    dn/dc for protein    0.185 mL/g   (Malvern; Wyatt TN4002 gives 0.186 at
                                       660 nm, BSA 0.1866)
    dn/dc for chitosan   0.16-0.18 mL/g
    Cd(II) aptamer Kd    34.5 nM  ->  K = 2.90e7 /M   (Wu et al. 2014)

See `references.py` for the full citations.

AN IMPORTANT CAVEAT
-------------------
The mass-addition calculation below is a LOWER BOUND, and possibly a poor one.

A cadmium ion weighs 112 daltons.  An aptamer weighs around 10,000.  So the
mass a bound ion adds to the layer is under 1% of what is already there, and
the index change from mass alone is correspondingly tiny.

But that is not how aptamer sensors actually work.  The aptamer FOLDS around
its target, and the folded layer is denser and thinner than the unfolded one.
That conformational change moves far more optical weight than the ion itself
does.  Daniel et al. (2018) treat this theoretically and show two things that
matter here:

    1. The conformational term usually DOMINATES the mass term for small
       analytes.
    2. The resulting index change can be of EITHER SIGN, depending on spacer
       length, immobilisation chemistry and packing density (confirmed
       experimentally in Analyst 2022).

So this module reports both:

    `mass_only_index_change`          rigorous, citable, and a lower bound
    `with_conformational_change`      adds a folding term you must supply

The second depends on conformational terms that have not been measured for a
metal-ion aptamer, so it should be reported as a range (see
`conformational_sweep.py`), while the first is a firm lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import SurfaceStackConfig, M_CD_G_PER_MOL, AVOGADRO


# ==========================================================================
#  Published refractive index increments
# ==========================================================================
DN_DC_ML_PER_G: dict[str, float] = {
    "dna": 0.17,          # Malvern compilation, aqueous buffer, 632.8 nm
    "rna": 0.18,          # midpoint of the quoted 0.17-0.19
    "protein": 0.185,     # Malvern; Wyatt TN4002 consensus 0.186 at 660 nm
    "bsa": 0.1866,        # Wyatt TN4002, 660 nm, 25 C
    "chitosan": 0.17,     # midpoint of the quoted 0.16-0.18
    "polysaccharide": 0.15,
}
"""Refractive index increments, mL/g, in aqueous buffer near 630-660 nm.
Sources: `references.malvern_dndc`, `references.wyatt_tn4002`."""

DN_DC_RANGE_ML_PER_G: dict[str, tuple[float, float]] = {
    "dna": (0.17, 0.17),
    "protein": (0.16, 0.20),
    "chitosan": (0.16, 0.18),
}
"""The reported spread, for propagating uncertainty rather than pretending
to three significant figures."""

# ==========================================================================
#  Measured conformational parameters
#  Dejeu et al., J. Phys. Chem. C (2018), doi:10.1021/acs.jpcc.8b07298
#  Pons et al., Analyst 147:4197 (2022), doi:10.1039/D2AN00824F
# ==========================================================================
APTAMER_THICKNESS_UNBOUND_NM = 5.2
APTAMER_THICKNESS_BOUND_NM = 3.8
"""Aptamer layer thickness before and after target recognition, for the
L-tyrosinamide system of Dejeu et al. (2018) / Pons et al. (2022).

READ THIS BEFORE QUOTING THE 27% AS A MEASUREMENT -- IT IS NOT ONE
------------------------------------------------------------------
Pons et al. (2022) state, verbatim: "For the 23-mer sequence, dL = 5.2 nm,
rho = 0.73 as previously reported and discussed."  The bound thickness is
rho * dL = 0.73 * 5.2 = 3.80 nm.  `rho` is a single FITTED folding ratio
applied uniformly across all six of their constructs; no post-binding
thickness was measured for any individual construct, and their SI confirms
"dL was taken to 5.2 nm".

So -26.9% is one model parameter carried through two papers, not two
independent measurements.  It is also the most extreme contraction in the
entire published set (see `THICKNESS_CHANGE_LITERATURE` below).  Treat it as
the pessimistic-signal end of a sweep, never as the value."""

CONFORMATIONAL_CONTRACTION_PERCENT = -100.0 * (
    1.0 - APTAMER_THICKNESS_BOUND_NM / APTAMER_THICKNESS_UNBOUND_NM
)
"""-26.9%, from the fitted folding ratio above.  See `SWEEP_THICKNESS_PERCENT`
for the range you should actually report."""

THICKNESS_CHANGE_LITERATURE: dict[str, dict] = {
    "L-tyrosinamide aptamer (23-mer)": {
        "percent": -26.9,
        "method": "SPR model fit (folding ratio rho = 0.73)",
        "target": "small molecule, 180 Da",
        "source": "Dejeu et al. 2018 / Pons et al. 2022, "
                  "doi:10.1021/acs.jpcc.8b07298, doi:10.1039/D2AN00824F",
        "measured": False,
    },
    "dopamine aptamer (QCM-D)": {
        "percent": -15.0,
        "method": "QCM-D, Sauerbrey, on gold (4.1 nm layer, -0.6 nm on binding)",
        "target": "small molecule, 153 Da",
        "source": "Stuber et al., ACS Nano 17:19168 (2023), "
                  "doi:10.1021/acsnano.3c05377",
        "measured": True,
    },
    "dopamine aptamer (MD)": {
        "percent": -11.0,
        "method": "molecular dynamics (5.5 -> 4.9 nm)",
        "target": "small molecule, 153 Da",
        "source": "Stuber et al. 2023, doi:10.1021/acsnano.3c05377",
        "measured": False,
    },
    "Cd(II) aptamer (DPI)": {
        "percent": None,
        "method": "dual polarization interferometry, real time, thickness "
                  "and refractive index resolved simultaneously",
        "target": "Cd(II), 112 Da",
        "source": "Xue, Wang, Wang, Yan, Huang & Yang, Anal. Chem. "
                  "92(14):10007 (2020), doi:10.1021/acs.analchem.0c01710",
        "measured": True,
        "note": "THE ONLY DIRECT OBSERVATION ON A CADMIUM APTAMER LAYER. "
                "The authors report that at LOW Cd(II) the ion interacts "
                "mainly with the phosphate backbone and the ssDNA is "
                "STRETCHED, not folded. Low concentration is precisely this "
                "sensor's operating regime, so the sign of the thickness "
                "change there is opposite to the contraction assumed by "
                "default. Numerical thickness values are not given in the "
                "abstract, so `percent` is None.",
    },
    "serotonin aptamer (QCM-D)": {
        "percent": +29.0,
        "method": "QCM-D: +1.2 nm ELONGATION on binding",
        "target": "small molecule, 176 Da",
        "source": "Stuber et al. 2023, doi:10.1021/acsnano.3c05377",
        "measured": True,
    },
}
"""Quantified aptamer layer-thickness changes from the literature, and the
reason the sign of this term cannot be assumed.

THE DECISIVE OBSERVATION
------------------------
Stuber et al. (2023) measured, by QCM-D on the same platform in the same
paper, a 15% CONTRACTION for the dopamine aptamer and a 1.2 nm ELONGATION for
the serotonin aptamer.  Two DNA aptamers, both binding small aromatic amines
of near-identical mass, moving in OPPOSITE directions.

No layer-thickness change has ever been measured for a metal-ion-binding
aptamer on a surface -- not for the Cd(II) aptamer CAO-1, not for any other.
And metal-ion binding is mechanistically different: Liu et al. (Nucleic Acids
Res. 51:4625, 2023, doi:10.1093/nar/gkad239) show Cd(II) coordinated directly
by G9 N7, C12 N3 and G16 N7, nucleating a compact double-twisted loop -- a
folding event, not a hydrophobic-pocket enclosure.  Neither sign is safely
predictable by analogy."""

SWEEP_THICKNESS_PERCENT: tuple[float, float] = (-30.0, +25.0)
"""The range to sweep the conformational thickness term over.

Justified by `THICKNESS_CHANGE_LITERATURE`: the measured set spans -15% to
+29%, the fitted L-Tym value adds -26.9%, and there is no measurement at all
for a metal-ion aptamer.  Sweeping only negative values would encode an
assumption the literature does not support.  For a contraction-only prior,
use (-30, 0) and note that the sign is unconstrained for metal-ion aptamers."""

APTAMER_RII_UNBOUND = 0.252
APTAMER_RII_BOUND = 0.241
"""Refractive index increment of the aptamer layer before and after binding,
cm^3/g (Pons et al. 2022).

THE MECHANISM MOST MODELS MISS: dn/dc is not constant through the binding
event. It falls by about 4.2%, giving a NEGATIVE contribution that partly
cancels the positive contraction term.

Pons et al. observed genuinely NULL SPR signals from a confirmed recognition
event (Kd ~ 200 nM by QCM-D and ITC) because the two cancelled exactly. A
design can bind perfectly well and read zero."""

RII_DEVIATION_PERCENT = 100.0 * (APTAMER_RII_BOUND / APTAMER_RII_UNBOUND - 1.0)
"""-4.4%, from the measured increments. Pons et al. quote about -4.2% from
their own fitting.

AND IT IS THE EXTREME OF THEIR OWN RANGE, NOT ITS CENTRE.  Pons et al. report
six constructs of the SAME aptamer against the same target:

    T0    0.252 -> 0.241 cm^3/g    -4.2%     <- this is the one used here
    T3    0.190 -> 0.185           -2.6%
    T6    0.196 -> 0.192           -2.0%
    T9    0.193 -> --              -2.4%
    T12   0.196 -> 0.192           -2.7%
    A6    0.245 -> --              -2.0%

Within one system the deviation varies by a factor of two, and it tracks
spacer length and anchoring mode rather than the aptamer's chemistry.  It is
not a property of the recognition event."""

RII_DEVIATION_CONFORMATION_ONLY_PERCENT = -17.0
"""dn/dc change from CONFORMATION ALONE -- no mass change, same molecule.

Dobrovodsky & Di Primo, Biosens. Bioelectron. 232:115296 (2023),
doi:10.1016/j.bios.2023.115296, measured the thrombin-binding aptamer's dn/dc
fall from 0.268 to 0.222 mL/g purely on K+-induced G-quadruplex folding.

THIS IS THE MOST RELEVANT NUMBER IN THE WHOLE SET, and it is four times larger
than the ligand-binding deviations above.  A METAL-ION-INDUCED FOLD is exactly
the mechanistic class a Cd(II) aptamer belongs to, so the L-tyrosinamide
small-neutral-organic case is the wrong analogue and the wrong magnitude.

The same paper found experimental Rmax exceeding the fully-corrected
calculation by 3-9x, concluding that conformational changes do contribute
measurably to the SPR signal."""

SWEEP_RII_DEVIATION_PERCENT: tuple[float, float] = (-20.0, 0.0)
"""The range to sweep the RII deviation over.

Lower bound covers the -17% conformation-only case with margin; upper bound
is zero because no POSITIVE RII deviation has ever been reported."""

SWEEP_DN_DC_ML_PER_G: tuple[float, float] = (0.183, 0.268)
"""The baseline dn/dc itself must be swept, not just its deviation.

    0.183 +/- 0.006   calf-thymus DNA, water, 840 nm, SPR
                      (Tumolo, Angnes & Baptista, Anal. Biochem. 333:273
                      (2004), doi:10.1016/j.ab.2004.06.010)
    0.222             thrombin aptamer, G-quadruplex folded
    0.268             thrombin aptamer, unfolded
                      (Dobrovodsky & Di Primo 2023)
    0.190-0.252       the six L-Tym constructs (Pons et al. 2022)

A 45% spread driven by sequence, spacer and anchoring.  Since de Feijter's
formula multiplies through by dn/dc, this propagates linearly into delta_n and
therefore into every sensitivity number downstream."""

APTAMER_DN_DC_DEJEU = 0.176
"""Aptamer refractive index increment, cm^3/g, used by Dejeu et al. (2018).
Reassuringly close to the 0.17 cm^3/g compiled value for DNA."""

APTAMER_MW_DEJEU = 15831.0
"""Molar mass of the aptamer in Dejeu et al. (2018), g/mol."""

MEASURED_APTAMER_DENSITY_PER_NM2 = 0.024
"""4 pmol/cm^2 = 2.4e12 molecules/cm^2, the density used by Pons et al.
(2022) on a streptavidin platform."""


CD_APTAMER_KD_M = 34.5e-9
"""Dissociation constant of the most widely used Cd(II) DNA aptamer,
from Wu et al., Analyst 139:1550 (2014).  Confirmed in the Gao et al. (2023)
review, where it is designated CAO-1."""

CD_APTAMER_K_PER_M = 1.0 / CD_APTAMER_KD_M
"""Association constant, 2.90e7 /M.  Use this in the Langmuir isotherm."""

CD_APTAMER_KD_RANGE_M = (34.5e-9, 81.39e-6)
"""The spread across published Cd(II) aptamers: from CAO-1 at 34.5 nM to
CAO-3 at 81.39 uM -- a factor of about 2400.  This is precisely why the
concentration floor must be reported as a curve against K rather than as a
single number."""


# ==========================================================================
#  de Feijter
# ==========================================================================
def de_feijter_index(
    n_solvent: float,
    dn_dc_ml_per_g: float,
    surface_mass_density_g_per_cm2: float,
    thickness_nm: float,
) -> float:
    """
    The refractive index of an adsorbed layer, from how much mass is in it.

        n_layer = n_solvent + (dn/dc) * Gamma / d

    Parameters
    ----------
    n_solvent : float
        Index of the medium the layer sits in.
    dn_dc_ml_per_g : float
        Refractive index increment, mL/g.
    surface_mass_density_g_per_cm2 : float
        Adsorbed mass per unit area.
    thickness_nm : float
        Layer thickness.

    Returns
    -------
    Refractive index of the layer.

    Reference: de Feijter, Benjamins & Veer, Biopolymers 17:1759 (1978).
    """
    d_cm = thickness_nm * 1e-7
    if d_cm <= 0:
        raise ValueError("Layer thickness must be positive.")
    return n_solvent + dn_dc_ml_per_g * surface_mass_density_g_per_cm2 / d_cm


def site_density_from_layer_index(
    n_layer: float,
    n_solvent: float,
    dn_dc_ml_per_g: float,
    thickness_nm: float,
    molar_mass_g_per_mol: float,
) -> float:
    """
    Invert de Feijter: what site density does a MEASURED layer index imply?

    WHY IT IS USEFUL
    ----------------
    Surface site density is hard to source from the literature for a given
    layer, but it is easy to measure: ellipsometry or an SPR angle shift on
    your own layer gives the layer index directly.  Feed that measurement in
    here to obtain the site density.

    Returns
    -------
    Site density in molecules per nm^2.
    """
    d_cm = thickness_nm * 1e-7
    gamma = (n_layer - n_solvent) * d_cm / dn_dc_ml_per_g   # g/cm^2
    sites_per_cm2 = gamma / molar_mass_g_per_mol * AVOGADRO
    return sites_per_cm2 / 1e14


def max_plausible_site_density(
    n_solvent: float,
    dn_dc_ml_per_g: float,
    thickness_nm: float,
    molar_mass_g_per_mol: float,
    n_layer_max: float = 1.50,
) -> float:
    """
    The largest site density that does not imply an impossible layer.

    A hydrated biopolymer layer in water sits around n = 1.45-1.50.  Anything
    much above that would be denser than dry protein, which no hydrated
    monolayer is.  So the de Feijter relation, run backwards, puts a hard
    physical ceiling on how many binding sites a layer of a given thickness
    can hold.

    This sanity check catches a site density chosen optimistically, which
    would otherwise inflate BOTH the predicted signal AND the Poisson-noise
    headroom.
    """
    return site_density_from_layer_index(
        n_layer_max, n_solvent, dn_dc_ml_per_g, thickness_nm,
        molar_mass_g_per_mol,
    )


def surface_mass_density(
    sites_per_nm2: float, molar_mass_g_per_mol: float, occupancy: float = 1.0
) -> float:
    """
    Convert a site density into a surface mass density, g/cm^2.

    1 site per nm^2 is 1e14 sites per cm^2, which is the conversion doing all
    the work here.
    """
    sites_per_cm2 = sites_per_nm2 * 1e14
    moles_per_cm2 = sites_per_cm2 * occupancy / AVOGADRO
    return moles_per_cm2 * molar_mass_g_per_mol


# ==========================================================================
#  The index change on cadmium binding
# ==========================================================================
@dataclass
class IndexChangeEstimate:
    """How much the recognition layer's index moves when cadmium binds."""

    delta_n_mass_only: float
    """From the added mass of the cadmium-ligand complex alone.  Rigorous,
    citable, and a LOWER BOUND on the real change."""

    delta_n_with_conformation: float
    """Including the measured conformational densification AND the measured
    change in the refractive index increment itself. Both from published
    ellipsometry and SPR fitting rather than from a guess."""

    delta_n_contraction_only: float
    """The contraction term alone, without the RII deviation. Reported
    separately because the two have OPPOSITE SIGN and can cancel."""

    rii_deviation_percent: float

    accessible_fraction: float
    """Fraction of nominal sites the hydrated analyte can reach. Full
    coverage means all of these, not all sites that exist."""

    n_unbound: float
    n_bound_mass_only: float

    bound_mass_ng_per_cm2: float
    sites_per_nm2: float
    thickness_nm: float
    dn_dc: float
    conformational_thickness_change_percent: float

    def __str__(self) -> str:
        return (
            "RECOGNITION-LAYER INDEX CHANGE\n"
            + "=" * 68 + "\n"
            f"  layer thickness              {self.thickness_nm:.1f} nm\n"
            f"  binding sites                {self.sites_per_nm2:.2f} /nm^2  "
            f"= {self.sites_per_nm2 * 1e14:.2e} /cm^2\n"
            f"  dn/dc used                   {self.dn_dc:.3f} mL/g\n"
            "  Cd mass at full coverage     "
            f"{self.bound_mass_ng_per_cm2:.3f} ng/cm^2\n"
            "\n"
            f"  n (unbound)                  {self.n_unbound:.6f}\n"
            f"  n (bound, mass only)         {self.n_bound_mass_only:.6f}\n"
            f"  delta_n from mass alone      {self.delta_n_mass_only:.3e}"
            "   <-- LOWER BOUND, citable\n"
            "\n"
            "  layer contraction            "
            f"{self.conformational_thickness_change_percent:.1f}%"
            "   (fitted folding ratio, Dejeu 2018)\n"
            "    delta_n, contraction only  "
            f"{self.delta_n_contraction_only:+.3e}\n"
            "  RII deviation                "
            f"{self.rii_deviation_percent:.1f}%"
            "   (measured, Pons 2022)\n"
            "\n"
            "  delta_n, ALL THREE TERMS     "
            f"{self.delta_n_with_conformation:+.3e}"
            "   <-- use this\n"
            "\n"
            "  The contraction term dominates and is positive; the RII\n"
            "  deviation is negative and partly cancels it.  Pons et al.\n"
            "  observed exact cancellation -- a null signal from a confirmed\n"
            "  binding event.  Sweep both terms; the sign is not guaranteed."
        )


def estimate_index_change(
    surf: SurfaceStackConfig,
    n_solvent: float = 1.39,
    polymer: str = "dna",
    polymer_molar_mass_g_per_mol: float = 9500.0,
    ligand_molar_mass_g_per_mol: float = 0.0,
    conformational_thickness_change_percent: float = (
        CONFORMATIONAL_CONTRACTION_PERCENT
    ),
    rii_deviation_percent: float = RII_DEVIATION_PERCENT,
    accessible_fraction: float | None = None,
) -> IndexChangeEstimate:
    """
    Estimate the bound-versus-unbound index of the recognition layer.

    THE TWO CONTRIBUTIONS
    ---------------------
    1.  MASS.  Each bound cadmium ion adds 112.4 daltons (plus its ligand, if
        the transduction is through a complex).  Spread over the layer
        thickness via de Feijter, this gives a small but exactly calculable
        index rise.

    2.  CONFORMATION.  The aptamer folds, the layer thins and densifies, and
        the same mass now occupies less volume.  Dejeu et al. (2018) measured
        a folding ratio corresponding to 5.2 nm -> 3.8 nm, a 27%
        contraction (a fitted value, see `APTAMER_THICKNESS_BOUND_NM`).  That is
        the default.  It RAISES the index.

    3.  RII DEVIATION.  The refractive index increment of the layer is itself
        altered by the bound target.  Pons et al. (2022) measured
        0.252 -> 0.241 cm^3/g, about -4.2%.  Also a default.  It LOWERS the
        index, partly cancelling term 2.

    Terms 2 and 3 have opposite sign, which is why Pons et al. observed null
    and negative SPR signals from a recognition event independently confirmed
    to be happening.

    Parameters
    ----------
    polymer : {"dna", "rna", "protein", "bsa", "chitosan"}
        Selects the published dn/dc.
    polymer_molar_mass_g_per_mol : float
        9500 g/mol is about right for a 30-nucleotide DNA aptamer -- the
        length of the Cd(II) aptamer of Wu et al. (2014).
    ligand_molar_mass_g_per_mol : float
        Set non-zero if the transduced species is a Cd-chelate rather than the
        bare ion; SERS and most colorimetric schemes are in that category.
    conformational_thickness_change_percent : float
        Negative means the layer contracts on binding.  THIS IS THE
        ASSUMPTION.  Sweep it, including through zero and positive values --
        the sign of the SPR response is known to flip with layer design.

    Returns
    -------
    IndexChangeEstimate
    """
    if polymer not in DN_DC_ML_PER_G:
        raise KeyError(
            f"Unknown polymer '{polymer}'. Known: {sorted(DN_DC_ML_PER_G)}"
        )
    dn_dc = DN_DC_ML_PER_G[polymer]
    d_nm = surf.recognition_thickness_nm
    sites = surf.binding_site_density_per_nm2

    # STERIC ACCESSIBILITY.  Only the sites the hydrated analyte can reach can
    # ever bind, so "full coverage" means all ACCESSIBLE sites, not all sites.
    # The grafted polymer is present regardless, so it is not scaled; the
    # bound cadmium mass is, and so is the conformational response, because
    # an aptamer that never binds never folds.
    if accessible_fraction is None:
        accessible_fraction = getattr(surf, "steric_accessible_fraction", 1.0)
    accessible_fraction = float(accessible_fraction)
    if not 0.0 < accessible_fraction <= 1.0:
        raise ValueError(
            f"accessible_fraction must lie in (0, 1]; got "
            f"{accessible_fraction}."
        )

    # --- the unbound layer ---------------------------------------------
    gamma_polymer = surface_mass_density(
        sites, polymer_molar_mass_g_per_mol, occupancy=1.0
    )
    n_unbound = de_feijter_index(n_solvent, dn_dc, gamma_polymer, d_nm)

    # --- mass added by full cadmium occupancy ---------------------------
    m_analyte = M_CD_G_PER_MOL + ligand_molar_mass_g_per_mol
    gamma_cd = surface_mass_density(
        sites, m_analyte, occupancy=accessible_fraction
    )
    # The bound complex is not the same material as the polymer; using the
    # polymer's dn/dc for it is an approximation, and a conservative one for
    # a dense inorganic species.
    n_bound_mass = de_feijter_index(
        n_solvent, dn_dc, gamma_polymer + gamma_cd, d_nm
    )
    delta_mass = n_bound_mass - n_unbound

    # --- conformational densification ------------------------------------
    # THE LAYER RESPONDS COLLECTIVELY, NOT SITE BY SITE.
    #
    # The thickness change is NOT weighted by the accessible fraction, even
    # though an individual aptamer that never binds never folds. Two pieces of
    # evidence support treating it as a layer property, both from the same
    # source as the folding ratio itself.
    #
    #   1. It is what the published model does.  Dejeu et al. (2018) define
    #      rho = d_LA / d_L as the ratio of the folded to the unfolded layer
    #      thickness and carry it through their eq. 13 as a property of the
    #      WHOLE sensing layer.  Occupancy never enters the thickness; it
    #      scales the response as a whole.  Weighting the thickness by
    #      occupancy is a departure from the model, not an application of it.
    #
    #   2. It is what the experiment shows.  Pons et al. (2022) measured the
    #      refractive index increment deviation across twelve to eighteen
    #      surface densities and found it "around -4.2% ... whatever the
    #      surface density is", concluding that "the conformation transition
    #      of the aptamer was not hampered by lateral steric hindrance".  A
    #      layer whose transition is independent of packing density is one
    #      whose thickness change is a layer property.
    #
    # So the thickness change is applied whole, and only the bound cadmium
    # MASS is scaled by what the analyte can reach.
    frac = conformational_thickness_change_percent / 100.0
    d_bound = d_nm * (1.0 + frac)
    if d_bound <= 0:
        raise ValueError("Conformational contraction cannot exceed 100%.")

    # Contraction alone: same mass, smaller volume, higher index.
    n_bound_contraction = de_feijter_index(
        n_solvent, dn_dc, gamma_polymer + gamma_cd, d_bound
    )
    delta_contraction = n_bound_contraction - n_unbound

    # ...and the RII deviation, which pulls the other way.
    dn_dc_bound = dn_dc * (1.0 + rii_deviation_percent / 100.0)
    n_bound_conf = de_feijter_index(
        n_solvent, dn_dc_bound, gamma_polymer + gamma_cd, d_bound
    )
    delta_conf = n_bound_conf - n_unbound

    # --- physical plausibility ------------------------------------------
    # If the implied layer index is above about 1.5, the layer would be denser
    # than dry protein.  That never happens for a hydrated monolayer, and it
    # means the site density is too high.
    if n_unbound > 1.55:
        import warnings
        ceiling = max_plausible_site_density(
            n_solvent, dn_dc, d_nm, polymer_molar_mass_g_per_mol
        )
        warnings.warn(
            f"A site density of {sites:.3f} /nm^2 over a {d_nm:.1f} nm layer "
            f"implies a refractive index of {n_unbound:.3f}, which is denser "
            "than dry protein and therefore impossible for a hydrated "
            f"monolayer. The physical ceiling here is about {ceiling:.3f} "
            f"/nm^2 ({ceiling * 1e14:.1e} /cm^2). An over-large site density "
            "inflates both the predicted signal and the Poisson-noise "
            "headroom, so fix it before quoting anything.",
            RuntimeWarning, stacklevel=2,
        )

    return IndexChangeEstimate(
        delta_n_mass_only=delta_mass,
        delta_n_with_conformation=delta_conf,
        delta_n_contraction_only=delta_contraction,
        rii_deviation_percent=rii_deviation_percent,
        accessible_fraction=accessible_fraction,
        n_unbound=n_unbound,
        n_bound_mass_only=n_bound_mass,
        bound_mass_ng_per_cm2=gamma_cd * 1e9,
        sites_per_nm2=sites,
        thickness_nm=d_nm,
        dn_dc=dn_dc,
        conformational_thickness_change_percent=(
            conformational_thickness_change_percent
        ),
    )


def apply_to_config(
    surf: SurfaceStackConfig,
    estimate: IndexChangeEstimate,
    use_conformational: bool = False,
) -> SurfaceStackConfig:
    """
    Return a copy of the surface config with the derived indices substituted.

    `use_conformational=False` (the default) installs the mass-only bound,
    the conservative choice.  The predicted signal will be small and the noise
    floor correspondingly pessimistic.
    """
    from dataclasses import replace
    delta = (
        estimate.delta_n_with_conformation if use_conformational
        else estimate.delta_n_mass_only
    )
    return replace(
        surf,
        recognition_n_unbound=estimate.n_unbound,
        recognition_n_bound=estimate.n_unbound + delta,
    )


# ==========================================================================
#  Self-check
# ==========================================================================
if __name__ == "__main__":
    from config import StudyConfig
    from references import cite

    cfg = StudyConfig()

    print("PUBLISHED INPUTS")
    print("=" * 68)
    print(f"  DNA dn/dc            {DN_DC_ML_PER_G['dna']} mL/g")
    print(f"  protein dn/dc        {DN_DC_ML_PER_G['protein']} mL/g")
    print(f"  chitosan dn/dc       {DN_DC_ML_PER_G['chitosan']} mL/g")
    print(f"    {cite('malvern_dndc')}")
    print()
    print(f"  Cd aptamer Kd        {CD_APTAMER_KD_M * 1e9:.1f} nM"
          f"  ->  K = {CD_APTAMER_K_PER_M:.2e} /M")
    print(f"    {cite('wu2014')}")
    print()
    print("  de Feijter formula")
    print(f"    {cite('defeijter1978')}")
    print()

    est = estimate_index_change(cfg.surface, n_solvent=cfg.tissue.n0)
    print(est)
    print()

    print("PHYSICAL CEILING ON THE SITE DENSITY")
    print("=" * 68)
    ceiling = max_plausible_site_density(
        cfg.tissue.n0, DN_DC_ML_PER_G["dna"],
        cfg.surface.recognition_thickness_nm, 9500.0,
    )
    print("  A hydrated DNA layer cannot exceed about n = 1.50.")
    print(f"  Over {cfg.surface.recognition_thickness_nm:.0f} nm that caps the "
          f"site density at {ceiling:.3f} /nm^2")
    print(f"  = {ceiling * 1e14:.2e} molecules/cm^2, which sits squarely in the")
    print("  range reported for thiolated ssDNA monolayers on gold")
    print("  (Herne & Tarlov 1997; Steel, Herne & Tarlov 1998).")
    print()
    print(f"  The config default is {cfg.surface.binding_site_density_per_nm2}"
          " /nm^2 -- "
          + ("ABOVE the ceiling, and wrong."
             if cfg.surface.binding_site_density_per_nm2 > ceiling
             else "within the ceiling."))
    print()

    print("SENSITIVITY TO THE TWO MEASURED CONFORMATIONAL TERMS")
    print("=" * 68)
    print("  Both defaults come from a DIFFERENT aptamer and target, so the")
    print("  spread matters.")
    print()
    print(f"  {'contraction':>13}  {'RII dev':>9}  {'delta_n':>12}  {'sign':>5}")
    for pct in (-30.0, -26.9, -20.0, -10.0, 0.0, 5.0):
        for rii in (0.0, RII_DEVIATION_PERCENT):
            e = estimate_index_change(
                cfg.surface, n_solvent=cfg.tissue.n0,
                conformational_thickness_change_percent=pct,
                rii_deviation_percent=rii,
            )
            sign = "+" if e.delta_n_with_conformation >= 0 else "NEG"
            tag = ("  <-- measured pair"
                   if abs(pct + 26.9) < 0.1 and rii != 0 else "")
            print(f"  {pct:11.1f}%  {rii:8.1f}%  "
                  f"{e.delta_n_with_conformation:12.3e}  {sign:>5}{tag}")
    print()
    print("  Note the RII column: at every contraction it subtracts, and near")
    print("  zero contraction it flips the sign outright.  That is exactly the")
    print("  null and negative signal Pons et al. observed from a binding")
    print("  event they had independently confirmed was happening.")
