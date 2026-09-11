"""
references.py — where every number in this project came from.
=============================================================

WHAT IT CONTAINS
----------------
A simulation is only as trustworthy as its inputs, so this module holds a
machine-readable bibliography: every entry pairs a citation with the specific
quantity it supplies and the value taken from it.

    from references import REFERENCES, cite, provenance_table
    print(cite("wu2014"))            # one formatted citation
    print(provenance_table())        # every sourced value, with its source

Anything NOT in this file is a modelling assumption, not a measurement, and
`unsourced_parameters()` lists those explicitly so they cannot hide.

RANGES
------
Several values below are ranges rather than single numbers, because the
literature does not agree on a single number.  These are reported as ranges
and swept, not averaged into a false precision.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    """One paper, and what this project takes from it."""

    key: str
    authors: str
    title: str
    venue: str
    year: int
    doi: str = ""
    url: str = ""
    supplies: tuple[str, ...] = ()
    """Short descriptions of the quantities taken from this source."""

    note: str = ""

    def formatted(self) -> str:
        bits = [f"{self.authors}, \"{self.title},\" {self.venue}, {self.year}."]
        if self.doi:
            bits.append(f"https://doi.org/{self.doi}")
        elif self.url:
            bits.append(self.url)
        return " ".join(bits)


# ==========================================================================
#  The bibliography
# ==========================================================================
REFERENCES: dict[str, Reference] = {}


def _add(ref: Reference) -> None:
    REFERENCES[ref.key] = ref


# --- Recognition chemistry: binding affinity -----------------------------
_add(Reference(
    key="wu2014",
    authors="Y. Wu, S. Zhan, L. Wang, and P. Zhou",
    title="Selection of a DNA aptamer for cadmium detection based on cationic "
          "polymer mediated aggregation of gold nanoparticles",
    venue="Analyst, vol. 139, no. 6, pp. 1550-1561",
    year=2014,
    doi="10.1039/C3AN02117C",
    supplies=(
        "Cd(II) aptamer dissociation constant Kd = 34.5 nM "
        "-> association constant K = 2.90e7 /M",
        "aqueous Cd(II) detection limit 4.6 nM for the parent assay",
        "aptamer selected from a 30-nucleotide random library "
        "(T- and G-rich sequence)",
    ),
    note="This is the 'Cd-4' aptamer, referred to as CAO-1 in the review "
         "literature and the most widely reused Cd(II) aptamer.",
))

_add(Reference(
    key="gao2023",
    authors="Z. Gao, Y. Wang, H. Wang, X. Li, Y. Xu, and J. Qiu",
    title="Recent aptamer-based biosensors for Cd2+ detection",
    venue="Biosensors, vol. 13, no. 6, art. 612",
    year=2023,
    doi="10.3390/bios13060612",
    supplies=(
        "confirms Kd = 34.5 nM for the most-used Cd(II) aptamer (CAO-1)",
        "reported Cd(II) aptasensor detection limits span 5 pM to 4.6 nM",
        "an alternative aptamer (CAO-3) has Kd = 81.39 uM, i.e. ~2000x weaker "
        "-- the spread that makes a C_floor-versus-K curve necessary",
    ),
))

# --- Recognition chemistry: refractive index increments ------------------
_add(Reference(
    key="defeijter1978",
    authors="J. A. de Feijter, J. Benjamins, and F. A. Veer",
    title="Ellipsometry as a tool to study the adsorption behavior of "
          "synthetic and biopolymers at the air-water interface",
    venue="Biopolymers, vol. 17, no. 7, pp. 1759-1772",
    year=1978,
    doi="10.1002/bip.1978.360170711",
    supplies=(
        "the de Feijter formula relating adsorbed surface mass density to "
        "layer refractive index: n_layer = n_solvent + (dn/dc) * Gamma / d",
    ),
    note="The original source for the formula used in "
         "`recognition_layer.de_feijter_index`.",
))

_add(Reference(
    key="voros2004",
    authors="J. Voros",
    title="The density and refractive index of adsorbing protein layers",
    venue="Biophysical Journal, vol. 87, no. 1, pp. 553-561",
    year=2004,
    doi="10.1529/biophysj.103.030072",
    supplies=(
        "the standard treatment of adsorbed-layer density and refractive "
        "index, and the validity limits of the de Feijter relation",
    ),
))

_add(Reference(
    key="wyatt_tn4002",
    authors="Wyatt Technology",
    title="The refractive index increment of proteins (Technical Note TN4002)",
    venue="Wyatt Technology technical note",
    year=2019,
    url="https://www.wyatt.com/files/tech-notes/instruments/optilab/"
        "tn4002a-protein-refractive-index-increment.pdf",
    supplies=(
        "protein dn/dc consensus value 0.186 mL/g at 660 nm in PBS",
        "BSA dn/dc = 0.1866 mL/g at 660 nm, 25 C",
        "human proteome average dn/dc = 0.1877 mL/g at 660 nm",
    ),
))

_add(Reference(
    key="malvern_dndc",
    authors="Malvern Panalytical",
    title="Refractive index increment dn/dc values",
    venue="Malvern Panalytical knowledge centre",
    year=2023,
    url="https://www.malvernpanalytical.com/en/learn/knowledge-center/"
        "insights/refractive-index-increment-dndc-values",
    supplies=(
        "DNA dn/dc = 0.17 mL/g (aqueous buffer, 632.8 nm, 25 C)",
        "protein dn/dc = 0.185 mL/g, range 0.16-0.20",
        "chitosan dn/dc = 0.16-0.18 mL/g",
        "general polysaccharide dn/dc = 0.15 mL/g",
    ),
    note="A compilation rather than a primary measurement.  Adequate for the "
         "order-of-magnitude role dn/dc plays here, but cite the primary "
         "source for whichever polymer you actually use.",
))

# --- Recognition chemistry: surface density ------------------------------
_add(Reference(
    key="herne1997",
    authors="T. M. Herne and M. J. Tarlov",
    title="Characterization of DNA probes immobilized on gold surfaces",
    venue="Journal of the American Chemical Society, vol. 119, no. 38, "
          "pp. 8916-8920",
    year=1997,
    doi="10.1021/ja9719586",
    supplies=(
        "pure HS-ssDNA on bare gold: 3.1 (+/-0.03) e12 molecules/cm^2 "
        "(32P radiolabelling)",
        "HS-ssDNA/MCH mixed monolayers are tunable over 2.9e10 to 5.7e12 "
        "molecules/cm^2 by varying DNA exposure time (Table 1)",
        "maximum coverage requires KH2PO4 above 0.4 M; coverage grows 5-fold "
        "from 2.7e-4 M to 1.0 M buffer (electrostatic screening)",
        "THE CRITICAL RESULT: the sample with the HIGHEST ssDNA coverage "
        "showed NO measurable hybridization at all -- the strands were too "
        "tightly packed for the complement to reach them, sterically and "
        "electrostatically",
    ),
    note="Direct experimental support for this project's steric-exclusion "
         "argument: maximum packing is NOT maximum useful binding sites. "
         "Densest is worst.",
))

_add(Reference(
    key="steel1998",
    authors="A. B. Steel, T. M. Herne, and M. J. Tarlov",
    title="Electrochemical quantitation of DNA immobilized on gold",
    venue="Analytical Chemistry, vol. 70, no. 22, pp. 4670-4677",
    year=1998,
    doi="10.1021/ac980037q",
    supplies=(
        "ssDNA surface densities precisely varied over (1-10) e12 "
        "molecules/cm^2 using mixed monolayers, by chronocoulometry",
        "hybridization efficiency exhibits a MAXIMUM with increasing surface "
        "density -- it is 100% only BELOW about 4e12 molecules/cm^2",
        "4e12 molecules/cm^2 = 2500 A^2 per molecule, which is the footprint "
        "of a 25-mer duplex -- i.e. the useful ceiling is set by the duplex "
        "simply not fitting",
        "method detection limit 1e11 molecules/cm^2",
    ),
    note="4e12 /cm^2 = 0.04 sites/nm^2 is the value used as the config "
         "default: the highest density at which every site still works.",
))

_add(Reference(
    key="dejeu2018",
    authors="J. Dejeu, H. Bonnet, N. Spinelli, E. Defrancq, "
            "L. Coche-Guerente, A. Van Der Heyden, and P. Labbe",
    title="Impact of conformational transition on SPR signals: theoretical "
          "treatment and application to small analytes/aptamer recognition",
    venue="Journal of Physical Chemistry C",
    year=2018,
    doi="10.1021/acs.jpcc.8b07298",
    supplies=(
        "CONFORMATIONAL CONTRACTION: on target recognition the aptamer "
        "layer thickness falls from 5.2 nm to 3.8 nm -- a 27% decrease -- "
        "from a fitted folding ratio (rho = 0.73), not a separately measured "
        "bound thickness",
        "aptamer refractive index increment 0.176 cm^3/g",
        "small-analyte (L-tyrosinamide, MW 180.2) refractive index "
        "increment 0.219 cm^3/g",
        "aptamer molar mass 15831 g/mol; 560 RU immobilised",
        "confirms that the conformational term exceeds the mass-weighted "
        "(Wilson-formula) response, and that responses may be positive or "
        "negative depending on whether the hydrodynamic radius grows or "
        "shrinks",
    ),
    note="Uses de Feijter's relationship throughout, which is the same "
         "approach `recognition_layer.py` takes.",
))

_add(Reference(
    key="pons2022",
    authors="M. Pons, M. Perenon, H. Bonnet, E. Gillon, C. Vallee, "
            "L. Coche-Guerente, E. Defrancq, N. Spinelli, "
            "A. Van der Heyden, and J. Dejeu",
    title="Conformational transition in SPR experiments: impact of spacer "
          "length, immobilization mode and aptamer density on signal sign "
          "and amplitude",
    venue="Analyst, vol. 147, no. 19, pp. 4197-4207",
    year=2022,
    doi="10.1039/D2AN00824F",
    supplies=(
        "THE REFRACTIVE INDEX INCREMENT ITSELF CHANGES ON BINDING: the "
        "aptamer RII falls from 0.252 to 0.241 cm^3/g on recognition, a "
        "deviation of about -4.2%",
        "aptamer surface density about 4 pmol/cm^2 = 2.4e12 molecules/cm^2 "
        "= 0.024 sites/nm^2 on a streptavidin platform",
        "aptamer layer thickness 5.2 nm for a 23-mer, packing density "
        "rho = 0.73",
        "NEGATIVE and NULL SPR signals were observed for a genuine, "
        "confirmed recognition event (Kd ~ 200 nM by QCM-D and ITC): a "
        "positive contribution from folding and a negative one from the RII "
        "deviation can cancel exactly",
        "adding a spacer between the binding site and the immobilisation "
        "chemistry produced a NULL signal despite effective recognition",
    ),
    note="The most consequential paper of the set for this project. It means "
         "delta_n has THREE terms of differing sign -- added mass, layer "
         "contraction, and RII deviation -- and a design can land on zero "
         "signal while binding perfectly well. `recognition_layer.py` "
         "models all three.",
))

# --- The tissue: optical properties --------------------------------------
_add(Reference(
    key="vraalstad2025",
    authors="V. Vraalstad, A. Postelmans, M. O'Farrell, J. Tschudi, "
            "J. P. Wold, and W. Saeys",
    title="Understanding the impact of drying on the optical properties of "
          "cured muscle tissue: a case study of dried salt-cured cod",
    venue="Food and Bioprocess Technology, vol. 18, no. 11, pp. 9392-9408",
    year=2025,
    doi="10.1007/s11947-025-03985-5",
    supplies=(
        "MEASURED refractive index of cod muscle: 1.36 to 1.38 at 589 nm "
        "(Abbe refractometer) for water contents 55.4% down to 45.4%; the "
        "index rises as water content falls",
        "MEASURED scattering coefficient mu_s = 4 to 6 /mm over 500-1700 nm",
        "MEASURED anisotropy factor g = 0.90 to 0.97 -- strongly forward "
        "scattering",
        "mu_a baseline 0 to 0.08 /mm, with water OH overtones at 750, 970 "
        "and 1450 nm and a protein CH overtone near 1200 nm",
        "mu_s' follows a power law a * lambda^-b with b between 1 and 2",
        "the scattering centres are identified as the muscle FIBRES "
        "themselves, 50-200 um across, not sub-micron organelles",
    ),
    note="The closest published measurement to the fish muscle this project "
         "models.  NOTE THE EXTRAPOLATION: fresh fish muscle is ~75% water, "
         "beyond the measured 45-55% range, so the fresh-tissue index is "
         "BELOW the measured band.  Extrapolating the reported trend gives "
         "roughly 1.34-1.35, which is what `TissueConfig.n0` uses; this is "
         "an extrapolation.",
))

_add(Reference(
    key="jacques2013",
    authors="S. L. Jacques",
    title="Optical properties of biological tissues: a review",
    venue="Physics in Medicine and Biology, vol. 58, no. 11, pp. R37-R61",
    year=2013,
    doi="10.1088/0031-9155/58/11/R37",
    supplies=(
        "tabulated reduced scattering coefficients for soft tissue including "
        "muscle across the visible and near infrared, used as the "
        "calibration target when fish-specific data are unavailable",
    ),
))

_add(Reference(
    key="rogers2014",
    authors="J. D. Rogers, A. J. Radosevich, J. Yi, and V. Backman",
    title="Modeling light scattering in tissue as continuous random media "
          "using a versatile refractive index correlation function",
    venue="IEEE Journal of Selected Topics in Quantum Electronics, vol. 20, "
          "no. 2, pp. 173-186",
    year=2014,
    doi="10.1109/JSTQE.2013.2280999",
    supplies=(
        "the Whittle-Matern refractive-index correlation family and its "
        "power spectral density",
        "the Born-approximation route from the power spectrum to mu_s and g, "
        "implemented in `optical_properties.scattering_from_spectrum`",
    ),
))

# --- Cadmium speciation ---------------------------------------------------
_add(Reference(
    key="he2010",
    authors="M. He, C.-H. Ke, and W.-X. Wang",
    title="Effects of cooking and subcellular distribution on the "
          "bioaccessibility of trace elements in two marine fish species",
    venue="Journal of Agricultural and Food Chemistry, vol. 58, no. 6, "
          "pp. 3517-3523",
    year=2010,
    doi="10.1021/jf100227n",
    supplies=(
        "RAW FISH MUSCLE Cd bioaccessibility (in-vitro digestion, Table 3): "
        "seabass 93.2 +/- 2.9% (small) and 84.8 +/- 1.9% (large); red "
        "seabream 77.1 +/- 4.9% (small), 89.9 +/- 1.9% (medium), "
        "73.7 +/- 2.8% (large) -- range 73.7-93.2%, mean about 84%",
        "cadmium was the MOST bioaccessible of the six elements studied "
        "(pattern Fe < Se < Zn < As ~ Cu ~ Cd)",
        "cooking reduces it: frying dropped Cd to 36.2% in large seabass and "
        "52.2% in large red seabream",
        "across all six elements in raw muscle, bioaccessibility was 45-93%",
        "previously reported Cd bioaccessibility in shellfish: 20-84%",
    ),
    note="In FISH MUSCLE cadmium is highly bioaccessible, so "
         "`free_fraction` for a digestion-based workflow is ~0.84, far above "
         "the <1% reported for some mollusc matrices under INFOGEST.",
))

_add(Reference(
    key="maulvault2011",
    authors="A. L. Maulvault, R. Machado, C. Afonso, H. M. Lourenco, "
            "M. L. Nunes, I. Coelho, T. Langerholc, and A. Marques",
    title="Bioaccessibility of Hg, Cd and As in cooked black scabbard fish "
          "and edible crab",
    venue="Food and Chemical Toxicology, vol. 49, no. 11, pp. 2808-2815",
    year=2011,
    doi="10.1016/j.fct.2011.07.059",
    supplies=(
        "Cd bioaccessibility at the small-intestine step of crab brown meat: "
        "95.3% uncooked, 83.9% steamed, 83.7% boiled",
        "staged release: 5.8-13.1% of total Cd already accessible in the "
        "mouth step, 45.9-72.0% by the stomach step",
        "compiled Cd bioaccessibility for other seafood (after Amiard et al. "
        "2008): green mussel 36%, scallop 20%, clam 48%, oyster 68%, "
        "neogastropod 72%",
    ),
    note="NOTE: the cadmium data here are for CRAB BROWN MEAT; the fish "
         "(black scabbard) part of this paper concerns mercury. Use `he2010` "
         "for fish muscle Cd. Included because it brackets the range and "
         "shows the staged release through digestion.",
))

_add(Reference(
    key="milea2025",
    authors="S. Milea, I. Simionov, N. Lazar, C. Iticescu, M. Timofti, "
            "P. Georgescu, and C. Faggio",
    title="Comprehensive risk assessment of metals and minerals in seafood "
          "using bioaccessibility correction",
    venue="Journal of Xenobiotics, vol. 15, no. 3, art. 92",
    year=2025,
    doi="10.3390/jox15030092",
    supplies=(
        "cadmium bioaccessibility BELOW 1% in several seafood matrices "
        "(mussels, cephalopod tentacles) under the INFOGEST 2.0 in-vitro "
        "digestion protocol",
    ),
    note="The low end of the bioaccessibility range, and the reason "
         "`free_fraction` must be swept rather than fixed.  These are "
         "molluscs, not fish muscle; do not transfer the number directly.",
))

_add(Reference(
    key="eu2023915",
    authors="European Commission",
    title="Commission Regulation (EU) 2023/915 on maximum levels for certain "
          "contaminants in food (recasting Regulation (EC) No 1881/2006)",
    venue="Official Journal of the European Union, L 119, p. 103",
    year=2023,
    supplies=(
        "the regulatory maximum level for cadmium in fish muscle: "
        "0.050 mg/kg wet weight",
    ),
))


# ==========================================================================
#  Values that are NOT sourced
# ==========================================================================
UNSOURCED: dict[str, str] = {
    "SurfaceStackConfig.analyte_hydrodynamic_radius_nm":
        "Hydrated radius of the Cd-ligand complex.  Depends on the ligand, "
        "which is not fixed in this study.  Sweep it.",

    "TissueConfig.l_min_um":
        "Inner cutoff of the index correlation.  A modelling choice about the "
        "smallest scale at which muscle has optically relevant index "
        "STRUCTURE.  Not measured here.  Its effect is bounded by the "
        "convergence study rather than by a citation.",

    "TissueConfig.m":
        "Whittle-Matern shape parameter.  Should come from low-coherence "
        "enhanced backscattering measurements of the actual tissue.  Held "
        "fixed and swept in `optical_properties.sensitivity_to_m`.",

    "BowtieConfig (all geometry)":
        "A design under test, not a measurement.  No citation applies.",

}


RESOLVED_BY_READING = {
    "SurfaceStackConfig.binding_site_density_per_nm2":
        "Steel et al. 1998 give (1-10)e12 /cm^2 achievable, with "
        "100% hybridization efficiency only below 4e12 /cm^2; Herne & Tarlov "
        "1997 give 2.9e10-5.7e12 /cm^2 for MCH mixed monolayers and show "
        "that the densest layer hybridizes not at all; Pons et al. 2022 "
        "used 4 pmol/cm^2 = 2.4e12 /cm^2. Default 0.04 /nm^2 "
        "(4e12 /cm^2). The independent de Feijter ceiling of ~0.17 /nm^2 "
        "sits comfortably above the measured maximum, so the two agree.",

    "RegulatoryConfig.free_fraction":
        "Set for a digestion-based workflow. He, Ke & Wang 2010 measure "
        "73.7-93.2% Cd bioaccessibility in RAW fish muscle across two "
        "species and three size classes, with cadmium the most bioaccessible "
        "of six elements. Default 0.84 (the mean). Cooking lowers it "
        "to as little as 36%. NOT resolved for a direct-contact sensor "
        "on intact tissue, where the free ionic fraction is what matters "
        "rather than what a digestion protocol releases.",

    "recognition-layer conformational term":
        "Defaults from Dejeu et al. 2018 (5.2 -> 3.8 nm, -27%, a fitted "
        "folding ratio) and Pons et al. 2022 (RII 0.252 -> 0.241 cm^3/g, "
        "-4.2%), both in `recognition_layer.py`. Their transferability to a "
        "Cd(II) aptamer is treated as a range; see below.",

    "calibration targets mu_s and g":
        "Set from cod muscle. Vraalstad et al. 2025 measure mu_s = "
        "4-6 /mm and g = 0.90-0.97 over 500-1700 nm. Defaults 5 /mm and "
        "0.93.",
}


# ==========================================================================
#  Convenience
# ==========================================================================
def cite(key: str) -> str:
    """One formatted citation."""
    if key not in REFERENCES:
        raise KeyError(f"No reference '{key}'. Known: {sorted(REFERENCES)}")
    return REFERENCES[key].formatted()


def bibliography(sort: bool = True) -> str:
    """Every reference, formatted, ready to paste into a manuscript."""
    keys = sorted(REFERENCES) if sort else list(REFERENCES)
    lines = []
    for i, k in enumerate(keys, 1):
        lines.append(f"[{i}] ({k}) {REFERENCES[k].formatted()}")
    return "\n\n".join(lines)


def provenance_table() -> str:
    """Every sourced value and where it came from."""
    out = ["SOURCED VALUES", "=" * 74]
    for key in sorted(REFERENCES):
        ref = REFERENCES[key]
        if not ref.supplies:
            continue
        out.append(f"\n{key} -- {ref.venue}, {ref.year}")
        if ref.doi:
            out.append(f"  doi:{ref.doi}")
        for s in ref.supplies:
            out.append(f"    * {s}")
        if ref.note:
            out.append(f"    NOTE: {ref.note}")
    return "\n".join(out)


def unsourced_parameters() -> str:
    """Everything that is an assumption rather than a measurement."""
    out = [
        "UNSOURCED PARAMETERS -- assumptions, not measurements",
        "=" * 74,
        "These must be swept, or measured, before any result is quoted.",
    ]
    for name, why in UNSOURCED.items():
        out.append(f"\n  {name}")
        for line in _wrap(why, 68):
            out.append(f"      {line}")
    out += [
        "",
        "",
        "SET FROM THE LITERATURE",
        "=" * 74,
    ]
    for name, how in RESOLVED_BY_READING.items():
        out.append(f"\n  {name}")
        for line in _wrap(how, 68):
            out.append(f"      {line}")
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
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


if __name__ == "__main__":
    print(provenance_table())
    print()
    print(unsourced_parameters())
    print()
    print("BIBLIOGRAPHY")
    print("=" * 74)
    print(bibliography())


# ==========================================================================
#  Tissue optics, anisotropy, conformational terms and cadmium speciation
# ==========================================================================
_add(Reference(
    key="vanbeers2018",
    authors="B. Van Beers, S. Kokawa, B. Aernouts, N. Watte, S. De Smet, "
            "W. Saeys",
    title="Evolution of the bulk optical properties of bovine muscles during "
          "wet aging",
    venue="Meat Science 136:50-58",
    year=2018,
    doi="10.1016/j.meatsci.2017.10.010",
    supplies=(
        "mu_s = 158-185 /cm and g = 0.927-0.957 at 800 nm for bovine skeletal "
        "muscle, by the same DIS/IAD technique and laboratory as Vraalstad's "
        "cod measurement -- the cross-species check in "
        "`optical_properties.check_mu_s_prime`",
        "mu_s' = 4-9 /cm at 721 nm, the quantity the two species agree on",
    ),
    note="Contradicts the cod mu_s by about 3x while corroborating g. The "
         "likely reason is that the cod samples were dried salt-cured at "
         "45-56% water. NOTE THE DOI: ...10.009 is a different paper about "
         "nitrite in Chinese meat products; this one is ...10.010, pp. 50-58 "
         "(several repositories quote 60-68, which is wrong).",
))

_add(Reference(
    key="bolin1989",
    authors="F. P. Bolin, L. E. Preuss, R. C. Taylor, R. J. Ference",
    title="Refractive index of some mammalian tissues using a fiber optic "
          "cladding method",
    venue="Applied Optics 28(12):2297-2303",
    year=1989,
    doi="10.1364/AO.28.002297",
    supplies=(
        "n = 1.38-1.41 at 632.8 nm across mammalian tissues including bovine "
        "muscle, species not a significant factor -- the corroboration that "
        "n0 = 1.35 for high-water raw fillet sits sensibly below",
    ),
))

_add(Reference(
    key="marquez1998",
    authors="G. Marquez, L. V. Wang, S.-P. Lin, J. A. Schwartz, S. L. Thomsen",
    title="Anisotropy in the absorption and scattering spectra of chicken "
          "breast tissue",
    venue="Applied Optics 37(4):798-804",
    year=1998,
    doi="10.1364/AO.37.000798",
    supplies=(
        "mu_s' differs by up to 10% between probe orientations parallel and "
        "perpendicular to the muscle fibres (mu_a by up to 50%) -- the ONLY "
        "numeric muscle anisotropy ratio in the literature, and what "
        "`anisotropy.aspect_from_measured_ratio` inverts to constrain the "
        "correlation-function aspect ratio",
    ),
    note="Directional effective coefficients from a diffusion fit, not tensor "
         "components. No paper reports mu_s parallel and perpendicular as a "
         "tensor for muscle of any species.",
))

_add(Reference(
    key="shuaib2010",
    authors="A. Shuaib, G. Yao",
    title="Equi-intensity distribution of optical reflectance in a fibrous "
          "turbid medium",
    venue="Applied Optics 49(5):838-844",
    year=2010,
    doi="10.1364/AO.49.000838",
    supplies=(
        "the method that maps a measured reflectance ellipse axis-ratio onto "
        "a mu_s' ratio in fibrous tissue",
    ),
    note="Cite this for the method rather than Kienle et al. 2004, whose "
         "experiment is porcine ARTERY at 633 nm, not muscle.",
))

_add(Reference(
    key="lichtenegger2022",
    authors="A. Lichtenegger et al.",
    title="Longitudinal investigation of a zebrafish model of Alzheimer's "
          "disease using polarization-sensitive optical coherence tomography",
    venue="Journal of Biomedical Optics 27(1):016001",
    year=2022,
    doi="10.1117/1.JBO.27.1.016001",
    supplies=(
        "birefringence of zebrafish skeletal muscle in vivo: "
        "dn = 1.6e-3 +/- 0.6e-4 (1 month) and 1.8e-3 +/- 0.3e-4 (2 months), "
        "Jones-matrix PS-OCT at 1310 nm -- sized against the model's RMS "
        "index fluctuation in `anisotropy.birefringence_context`",
    ),
    note="Measured at 1310 nm, transferred to the 500-900 nm band. Form "
         "birefringence from aligned sub-wavelength structure is only weakly "
         "dispersive, so the transfer is reasonable -- but it is a transfer.",
))

_add(Reference(
    key="stuber2023",
    authors="A. Stuber, A. Douaki, J. Hengsteler, D. Buckingham, "
            "D. Momotenko, D. Garoli, N. Nakatsuka",
    title="Aptamer conformational dynamics modulate neurotransmitter sensing "
          "in nanopores",
    venue="ACS Nano 17(19):19168-19179",
    year=2023,
    doi="10.1021/acsnano.3c05377",
    supplies=(
        "QCM-D layer thickness change on binding: -15% for a dopamine "
        "aptamer (4.1 nm layer, -0.6 nm) and +1.2 nm ELONGATION for a "
        "serotonin aptamer -- two DNA aptamers binding small aromatic amines "
        "of near-identical mass, moving in OPPOSITE directions",
        "MD corroboration for dopamine: 5.5 -> 4.9 nm",
    ),
    note="The single most important reference for "
         "`recognition_layer.SWEEP_THICKNESS_PERCENT`. It is why the sweep "
         "must cross zero: the SIGN of the conformational term is not "
         "predictable by analogy, and no metal-ion aptamer has been measured "
         "on a surface at all.",
))

_add(Reference(
    key="dobrovodsky2023",
    authors="D. Dobrovodsky, C. Di Primo",
    title="Do conformational changes contribute to the surface plasmon "
          "resonance signal?",
    venue="Biosensors and Bioelectronics 232:115296",
    year=2023,
    doi="10.1016/j.bios.2023.115296",
    supplies=(
        "dn/dc of the thrombin-binding aptamer falling from 0.268 to "
        "0.222 mL/g on K+-induced G-quadruplex FOLDING alone -- a 17% change "
        "with no mass change, and the right mechanistic analogue for a "
        "metal-ion-induced fold",
        "the SPR-derived surface value 0.164 mL/g for the same aptamer",
    ),
    note="Values from the arXiv preprint (2304.05063). They also found "
         "experimental Rmax exceeding the fully-corrected calculation by "
         "3-9x.",
))

_add(Reference(
    key="tumolo2004",
    authors="T. Tumolo, L. Angnes, M. S. Baptista",
    title="Determination of the refractive index increment (dn/dc) of "
          "molecule and macromolecule solutions by surface plasmon resonance",
    venue="Analytical Biochemistry 333(2):273-279",
    year=2004,
    doi="10.1016/j.ab.2004.06.010",
    supplies=(
        "dn/dc = 0.183 +/- 0.006 cm^3/g for calf-thymus DNA in water at "
        "840 nm (BSA 0.190 +/- 0.002 for comparison) -- the lower end of "
        "`recognition_layer.SWEEP_DN_DC_ML_PER_G`",
    ),
))

_add(Reference(
    key="liu2023cdaptamer",
    authors="H. Liu, Y. Gao, J. Mathivanan et al.",
    title="Crystal structures and identification of novel Cd2+-specific DNA "
          "aptamer",
    venue="Nucleic Acids Research 51(9):4625-4636",
    year=2023,
    doi="10.1093/nar/gkad239",
    supplies=(
        "the structural basis for treating Cd(II) recognition as a FOLDING "
        "event: Cd2+ coordinated by G9 N7, C12 N3 and G16 N7 plus two waters "
        "at 2.4 A, nucleating a double-twisted CBL loop",
        "Kd = 0.340 +/- 0.017 uM for the 25-nt DNA1 aptamer",
    ),
    note="A different aptamer from CAO-1, and solution-phase only. No "
         "surface-immobilised conformational measurement exists for ANY "
         "Cd(II) aptamer.",
))

_add(Reference(
    key="watly2021",
    authors="J. Watly, M. Laczkowski, M. Padjasek, A. Krezel",
    title="Phytochelatins as a dynamic system for Cd(II) buffering from the "
          "micro- to femtomolar range",
    venue="Inorganic Chemistry 60(7):4657-4675",
    year=2021,
    doi="10.1021/acs.inorgchem.0c03639",
    supplies=(
        "Cd-glutathione stability constants: log beta(CdL) = 9.00(1), "
        "log beta(CdL2) = 15.05(2), and an apparent conditional constant at "
        "pH 7.4 of log K = 5.93 -- the basis of the derived free-ion bound "
        "in `RegulatoryConfig.free_fraction_direct_contact`",
    ),
))

_add(Reference(
    key="quinn2024",
    authors="C. F. Quinn, D. E. Wilcox",
    title="Thermodynamic origin of the affinity, selectivity, and domain "
          "specificity of metallothionein for essential and toxic metal ions",
    venue="Metallomics 16(10):mfae041",
    year=2024,
    doi="10.1093/mtomcs/mfae041",
    supplies=(
        "Cd(II) binding to the metallothionein alpha domain, "
        "K = 6.5e14 /M (log K ~ 14.8) by ITC at pH 7.4 -- the lower end of "
        "the free-ion bound, if cysteine-rich ligands are present",
    ),
    note="Alpha-domain constant, as reported by ITC.",
))

_add(Reference(
    key="urien2018",
    authors="N. Urien, S. Jacob, P. Couture, P. G. C. Campbell",
    title="Cytosolic distribution of metals (Cd, Cu) and metalloids (As, Se) "
          "in livers and gonads of field-collected fish",
    venue="Environments 5(9):102",
    year=2018,
    doi="10.3390/environments5090102",
    supplies=(
        "SEC-ICP-MS speciation in white sucker LIVER cytosol: all hepatic "
        "cytosolic Cd in the 10-2 kDa pool, major peak co-eluting with a "
        "metallothionein standard, no free Cd detected",
    ),
    note="Liver and gonad, NOT muscle. A detection-limit statement, not a "
         "quantified free fraction. Does not transfer to muscle -- see "
         "kovarova2009.",
))

_add(Reference(
    key="kovarova2009",
    authors="J. Kovarova, R. Kizek, V. Adam et al.",
    title="Effect of cadmium chloride on metallothionein levels in carp",
    venue="Sensors 9(6):4789-4803",
    year=2009,
    doi="10.3390/s90604789",
    supplies=(
        "metallothionein NON-DETECTABLE in carp muscle in every experimental "
        "group, while muscle Cd rose dose-dependently from 0.05 to "
        "81.18 ug/kg -- the reason the liver speciation result cannot be "
        "imported into muscle",
    ),
    note="Treat 'non-detectable' as an upper bound and quote it as such; the "
         "paper does not state the detection limit in the accessible text.",
))


UNSOURCED["TissueConfig.anisotropy_xyz"] = (
    "The aspect ratio of the index CORRELATION FUNCTION in muscle has never "
    "been measured, in any species, by anyone. The Whittle-Matern and "
    "continuous-random-media literature is formulated and fitted "
    "isotropically. State it as a morphometry-derived modelling assumption "
    "(myofibrils 1-2 um across, sarcomeres 2-2.5 um along the fibre) "
    "CONSTRAINED by the Marquez mu_s' ratio -- see "
    "`anisotropy.aspect_from_measured_ratio` -- and never as measured."
)

UNSOURCED["RegulatoryConfig.free_fraction_direct_contact"] = (
    "Free ionic Cd(II) in fish muscle has NEVER been measured -- not by DGT, "
    "ion-selective electrode, Donnan membrane technique, voltammetry or "
    "ultrafiltration. The default of 1e-3 is DERIVED from Cd-glutathione "
    "stability constants (watly2021) at a nominal 1 mM cytosolic GSH, with "
    "the 1e-9 lower bound allowing for cysteine-rich ligands (quinn2024). "
    "Two ingredients are themselves unsourced: the GSH concentration in fish "
    "WHITE MUSCLE (see Wu et al., Comp. Biochem. Physiol. C, 2002, PMID "
    "11912048), and the fact that a sensor on intact tissue meets "
    "INTERSTITIAL fluid rather than the cytosol every speciation measurement "
    "refers to. The number is a construction, not a measurement."
)

UNSOURCED["CalibrationConfig.target_mu_s_per_um"] = (
    "CONTESTED rather than unsourced. Cod gives 4-6 /mm (vraalstad2025); "
    "bovine skeletal muscle by the same technique in the same laboratory "
    "gives 15.8-18.5 /mm (vanbeers2018). No measurement of RAW fish muscle "
    "mu_s exists in any species. The two datasets agree far better in mu_s', "
    "so quote that as well -- `optical_properties.check_mu_s_prime`."
)

UNSOURCED["TissueConfig.n0"] = (
    "DERIVED, not measured. No measurement of raw fish muscle refractive "
    "index exists in any species; the one fish paper (vraalstad2025) is "
    "dried salt-cured cod at 45-56% water. 1.35 comes from two-component "
    "volume mixing anchored on that value and extrapolated to 80% water; the "
    "full arithmetic is in the `TissueConfig.n0` docstring. Published "
    "dispersion for fish muscle is Segelstein's WATER "
    "dispersion rigidly shifted, not a measurement."
)

RESOLVED_BY_READING["mode overlap factor"] = (
    "Measured, not assumed. `paired_run.py` measures it from three "
    "simulations: unbound layer, bound layer, and a background-index step "
    "that calibrates the bulk sensitivity in the same mesh with the same "
    "fitter, so systematics cancel in the ratio. It is a layer-to-water "
    "response ratio and can exceed 1 (see `layer_series.py`)."
)

RESOLVED_BY_READING["conformational transferability"] = (
    "Treated as a RANGE, not a value. Two points about the borrowed "
    "numbers: the 5.2 -> 3.8 nm contraction is one fitted folding ratio "
    "(rho = 0.73) applied uniformly across six constructs, not two measured "
    "thicknesses; and -4.4% is the extreme of Pons et al.'s own -2.0 to "
    "-4.2% range. stuber2023 shows the SIGN is not predictable (dopamine "
    "-15%, serotonin +29%, same method, same paper), and dobrovodsky2023 "
    "gives -17% dn/dc from folding alone, which is the right mechanistic "
    "class for a metal ion. Swept as a 2-D surface in "
    "`conformational_sweep.py`, because the two terms trade off and can null."
)
