"""
anisotropy.py — fibre-aligned (anisotropic) tissue and a polarisation-sensitive antenna.
=========================================================================================

    python anisotropy.py        # offline, a few seconds

WHY IT MATTERS
--------------
The rest of this package treats the tissue as a statistically isotropic
random medium: blobs of index with no preferred direction.  Fish muscle is not
that.  Myofibrils are aligned bundles roughly 1-2 um across and far longer,
stacked into sarcomeres of 2-2.5 um along the fibre.  The index correlation
function of such a medium is stretched along the fibre axis.

Two consequences:

  1. `mu_s` and `g` become direction-dependent.  Light travelling along the
     fibres does not scatter the same way as light travelling across them.

  2. A BOWTIE IS POLARISATION-SENSITIVE.  Its resonance is driven by the field
     along its long axis.  In an anisotropic medium the scattering seen by
     that polarisation differs from the orthogonal one, so the answer depends
     on how the sensor is rotated on the fillet.  An isotropic model cannot
     represent that dependence at all.

WHAT THE LITERATURE CONSTRAINS, AND WHAT IT DOES NOT
----------------------------------------------------

  MEASURED.  Marquez, Wang, Lin, Schwartz & Thomsen, *Applied Optics* 37(4)
  798-804 (1998), doi:10.1364/AO.37.000798, measured chicken breast by
  oblique-incidence reflectometry at 400-800 nm with the probe aligned at 0
  and 90 degrees to the fibres.  `mu_s'` differed by UP TO 10% between the two
  orientations -- a ratio of about 1.1.  (`mu_a` differed by up to 50%, which
  is a separate and larger effect, consistently lower along the fibres.)

  MEASURED.  Shuaib & Yao, *Applied Optics* 49(5) 838-844 (2010),
  doi:10.1364/AO.49.000838, is the paper that maps a measured reflectance
  ellipse axis-ratio onto a `mu_s'` ratio, and shows the diffusion-equation
  prediction only holds for small fibre diameters.  Cite this one for the
  method, not Kienle 2004 -- Kienle's experiment is porcine ARTERY, not
  muscle.

  MEASURED, and in a fish.  Lichtenegger et al., *J. Biomed. Opt.* 27(1)
  016001 (2022), doi:10.1117/1.JBO.27.1.016001, measured zebrafish skeletal
  muscle in vivo by Jones-matrix PS-OCT: birefringence dn = 1.6e-3 +/- 0.6e-4
  (1 month) and 1.8e-3 +/- 0.3e-4 (2 months), at 1310 nm.

  NOT MEASURED, IN ANY SPECIES, BY ANYONE.  The anisotropy of the index
  CORRELATION FUNCTION itself: correlation lengths along versus across the
  fibre, structure-function anisotropy, or a correlation-function aspect
  ratio.  The whole continuous-random-media / Whittle-Matern literature is
  formulated and fitted isotropically.  There is no measurement to put into
  `TissueConfig.anisotropy_xyz`.

  NOT MEASURED EITHER.  `mu_s` parallel and perpendicular as tensor
  components, for muscle of any species.  Marquez's numbers are directional
  effective coefficients from a diffusion fit, not a tensor.

WHAT THIS MODULE DOES
---------------------
It runs the inference backwards.  The aspect ratio is unmeasured, but its
CONSEQUENCE -- the `mu_s'` ratio between the two probe orientations -- was
measured by Marquez.  So:

    `fibre_axis_study`          for a given aspect ratio, compute mu_s and g
                                for each propagation direction and each
                                polarisation, exactly, from the anisotropic
                                Born integral.

    `aspect_from_measured_ratio`  invert that: which aspect ratio reproduces
                                the measured mu_s' ratio of about 1.1?

The result is that the morphology is strongly anisotropic, but the OPTICAL
anisotropy it produces at the relevant wavelength is small, and the aspect
ratio is constrained by a measurement rather than chosen freely.  It remains a
modelling assumption, not a measured value.

THE MATHEMATICS
---------------
`tissue.generate_tissue` implements anisotropy by stretching the spatial
frequency axes before evaluating the spectrum:

    Phi_aniso(q)  =  det(A) * Phi_iso(|A q|),     A = diag(ax, ay, az)

so a stretch factor `az > 1` widens the correlation length along z by that
factor.  The `det(A)` prefactor is what keeps the total variance equal to
`delta_n_sq`: without it, stretching the medium would quietly change how much
index fluctuation it contains, and every comparison between aspect ratios
would be confounded by an amplitude change.

(A note on `tissue.generate_tissue`: it rescales each realisation to the
band-limited variance computed ISOTROPICALLY.  This is intentional: an
anisotropic realisation then carries the same total variance as its isotropic
counterpart, so a comparison between them isolates SHAPE.  The anisotropic
band-limited variance, if needed, is the same integral with the stretched
spectrum.)

The Born scattering integral then loses its azimuthal symmetry and has to be
done in two dimensions:

    mu_s  =  Int  2 pi k^4 Phi_aniso(q(s_hat)) (1 - |e.s_hat|^2)  dOmega
    g     =  <s_hat . k_in_hat>,  weighted by the same integrand

with q(s_hat) = k (s_hat - k_in_hat).  The isotropic code in
`optical_properties.scattering_from_spectrum` is the special case where the
azimuthal integral can be done analytically; this module reproduces that
result to about four digits when the aspect ratio is 1, which is the test in
`_self_test`.
"""

from __future__ import annotations

import numpy as np

from config import StudyConfig, TissueConfig
from optical_properties import ScatteringProperties
from tissue import _psd_normalisation, _spectrum_shape


# ==========================================================================
#  The anisotropic spectrum
# ==========================================================================
def anisotropic_psd(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray,
    delta_n_sq: float, l_c: float, m: float, l_min: float,
    aspect_xyz: tuple[float, float, float],
) -> np.ndarray:
    """
    The index power spectrum of a fibre-aligned medium.

    `aspect_xyz` are stretch factors on the CORRELATION LENGTH: (1, 1, 3)
    means blobs three times longer along z than across it, which is the
    convention `TissueConfig.anisotropy_xyz` and `tissue.generate_tissue`
    already use.

    The `det(A)` factor preserves the total variance, so that changing the
    aspect ratio changes the medium's shape and nothing else.
    """
    ax, ay, az = (float(v) for v in aspect_xyz)
    q_eff = np.sqrt((ax * qx) ** 2 + (ay * qy) ** 2 + (az * qz) ** 2)
    amplitude = _psd_normalisation(delta_n_sq, l_c, m, l_min)
    return (ax * ay * az) * amplitude * _spectrum_shape(q_eff, l_c, m, l_min)


# ==========================================================================
#  Direction-resolved scattering
# ==========================================================================
def scattering_anisotropic(
    wavelength_um: float,
    n0: float,
    delta_n_sq: float,
    l_c_um: float,
    m: float,
    l_min_um: float,
    aspect_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    k_in_hat: tuple[float, float, float] = (0.0, 0.0, 1.0),
    pol_hat: tuple[float, float, float] | None = None,
    n_theta: int = 361,
    n_phi: int = 181,
) -> ScatteringProperties:
    """
    `mu_s` and `g` for one propagation direction and one polarisation.

    Parameters
    ----------
    k_in_hat : unit vector
        Direction the light travels.
    pol_hat : unit vector, optional
        Electric field direction.  Must be perpendicular to `k_in_hat`; if
        omitted, an arbitrary perpendicular is chosen, which is only
        meaningful when the medium is isotropic about `k_in_hat`.

    Notes
    -----
    Angles here are measured in the LAB frame from `k_in_hat`, so `g` is the
    usual forward-scattering asymmetry regardless of how the fibres lie.
    """
    k = 2.0 * np.pi * n0 / wavelength_um

    kin = np.asarray(k_in_hat, dtype=float)
    kin /= np.linalg.norm(kin)

    # Build an orthonormal frame (e1, e2, kin).
    seed_vec = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed_vec, kin)) > 0.9:
        seed_vec = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(kin, seed_vec)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(kin, e1)

    if pol_hat is None:
        pol = e1
    else:
        pol = np.asarray(pol_hat, dtype=float)
        pol = pol - np.dot(pol, kin) * kin      # project out any longitudinal part
        norm = np.linalg.norm(pol)
        if norm < 1e-12:
            raise ValueError("pol_hat is parallel to k_in_hat")
        pol /= norm

    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    # Scattering direction in the lab frame.
    s = (np.sin(TH) * np.cos(PH))[..., None] * e1 \
        + (np.sin(TH) * np.sin(PH))[..., None] * e2 \
        + np.cos(TH)[..., None] * kin

    qvec = k * (s - kin)
    phi_of_q = anisotropic_psd(
        qvec[..., 0], qvec[..., 1], qvec[..., 2],
        delta_n_sq, l_c_um, m, l_min_um, aspect_xyz,
    )

    # Dipole radiation pattern: no power radiated along the driving field.
    pol_factor = 1.0 - (s @ pol) ** 2

    integrand = 2.0 * np.pi * k ** 4 * phi_of_q * pol_factor * np.sin(TH)

    dphi = 2.0 * np.pi / n_phi
    mu_s = float(np.trapezoid(integrand.sum(axis=1) * dphi, theta))
    if mu_s <= 0:
        return ScatteringProperties(wavelength_um, 0.0, 0.0)

    g_num = float(np.trapezoid((integrand * np.cos(TH)).sum(axis=1) * dphi,
                               theta))
    return ScatteringProperties(wavelength_um, mu_s, g_num / mu_s)


# ==========================================================================
#  The study
# ==========================================================================
#: Fibres are taken to run along z.  These are the three geometries that
#: matter for a sensor pressed flat against a fillet whose fibres lie in the
#: contact plane.
_GEOMETRIES = {
    "across fibres, E || fibres": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "across fibres, E _|_ fibres": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "along fibres": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
}


def fibre_axis_study(
    cfg: StudyConfig,
    aspects: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0),
    wavelength_um: float | None = None,
) -> list[dict]:
    """
    How much optical anisotropy does a given morphological aspect ratio buy?

    Returns one row per aspect ratio, each carrying `mu_s` and `g` for the
    three geometries above, the reduced coefficients, and the `mu_s'` ratio
    that Marquez et al. measured to be about 1.1 in chicken breast.

    Entirely offline and free.  Runs in a second or two.
    """
    t = cfg.tissue
    wl = wavelength_um if wavelength_um is not None \
        else cfg.calibration.calib_wavelength_um

    rows = []
    for a in aspects:
        aspect = (1.0, 1.0, float(a))
        entry = {"aspect": float(a), "wavelength_um": wl, "geom": {}}
        for name, (kin, pol) in _GEOMETRIES.items():
            sp = scattering_anisotropic(
                wl, t.n0, t.delta_n_sq, t.l_c_um, t.m, t.l_min_um,
                aspect_xyz=aspect, k_in_hat=kin, pol_hat=pol,
            )
            entry["geom"][name] = {
                "mu_s": sp.mu_s_per_um,
                "g": sp.g,
                "mu_s_prime": sp.mu_s_reduced_per_um,
            }

        across_par = entry["geom"]["across fibres, E || fibres"]["mu_s_prime"]
        across_perp = entry["geom"]["across fibres, E _|_ fibres"]["mu_s_prime"]
        along = entry["geom"]["along fibres"]["mu_s_prime"]

        # Marquez rotated the PROBE, i.e. changed the propagation direction in
        # the tissue plane.  The closest model analogue is along-fibre versus
        # across-fibre propagation.
        entry["mu_s_prime_ratio"] = (
            max(along, across_perp) / min(along, across_perp)
            if min(along, across_perp) > 0 else float("nan")
        )
        # What a bowtie actually cares about: rotating the ANTENNA, which
        # changes the polarisation at fixed propagation direction.
        entry["polarisation_ratio"] = (
            max(across_par, across_perp) / min(across_par, across_perp)
            if min(across_par, across_perp) > 0 else float("nan")
        )
        rows.append(entry)
    return rows


#: Marquez et al. 1998, chicken breast, oblique-incidence reflectometry:
#: mu_s' differed by up to 10% between probe orientations.
MARQUEZ_MU_S_PRIME_RATIO = 1.10
MARQUEZ_CITATION = (
    "Marquez, Wang, Lin, Schwartz & Thomsen, Appl. Opt. 37(4):798-804 (1998), "
    "doi:10.1364/AO.37.000798 (chicken breast; mu_s' differs by up to 10% "
    "between probe orientations, mu_a by up to 50%)"
)


def aspect_from_measured_ratio(
    cfg: StudyConfig,
    target_ratio: float = MARQUEZ_MU_S_PRIME_RATIO,
    bracket: tuple[float, float] = (1.0, 20.0),
    tol: float = 1e-3,
) -> dict:
    """
    Turn the one measured anisotropy number into a constraint on the one
    unmeasured modelling parameter.

    The correlation-function aspect ratio has never been measured in muscle.
    Its consequence -- the `mu_s'` ratio between propagation along and across
    the fibres -- has been, by Marquez et al.  This finds the aspect ratio at
    which the model reproduces that measurement.

    WHY THIS SCANS INSTEAD OF BISECTING
    -----------------------------------
    Bisection would assume the mu_s' ratio is monotonic in the aspect ratio,
    and the printed table shows it is not: the mu_s' ratio is NOT MONOTONIC in
    the aspect ratio.  At the calibrated tissue it runs 1.0000, 1.0041, 1.0039,
    1.0069, 1.0372 for aspects 1, 1.5, 2, 3, 5 -- it dips between 1.5 and 2.
    Bisection assumes monotonicity, and on a function like this it can
    converge to a point that is not a root, or miss a root entirely, while
    reporting success.

    So this scans a dense grid, finds every sign change, and interpolates.  It
    returns ALL roots, because more than one aspect ratio reproducing the same
    measurement is itself the answer: the measurement does not determine the
    parameter.

    WHY THE FUNCTION IS SO FLAT
    ---------------------------
    Stretching the correlation function along the fibre raises mu_s steeply
    (5.0 -> 19.4 /mm from aspect 1 to 5) but ALSO drives g towards 1
    (0.930 -> 0.985).  Since mu_s' = mu_s (1 - g), the two effects very nearly
    cancel: elongated blobs scatter more light, but they scatter it more
    forwards.  The reduced coefficient is therefore a WEAK constraint on the
    aspect ratio, and the fitted value should be reported with that caveat
    rather than as a tight determination.

    Returns the root(s), whether the target is reachable at all, and how
    sensitive the ratio is to the aspect ratio near the solution.
    """
    import numpy as _np

    def ratio_at(a: float) -> float:
        return fibre_axis_study(cfg, aspects=(a,))[0]["mu_s_prime_ratio"]

    lo, hi = bracket
    grid = _np.linspace(lo, hi, 40)
    vals = _np.array([ratio_at(float(a)) for a in grid])

    diffs = _np.diff(vals)
    monotonic = bool(_np.all(diffs >= -1e-6) or _np.all(diffs <= 1e-6))

    f = vals - target_ratio
    roots = []
    for i in range(len(grid) - 1):
        if f[i] == 0.0:
            roots.append(float(grid[i]))
        elif f[i] * f[i + 1] < 0:
            t = -f[i] / (f[i + 1] - f[i])
            roots.append(float(grid[i] + t * (grid[i + 1] - grid[i])))

    if not roots:
        edge = lo if abs(vals[0] - target_ratio) < abs(vals[-1] - target_ratio) \
            else hi
        return {
            "aspect": edge,
            "achieved_ratio": float(ratio_at(edge)),
            "target_ratio": target_ratio,
            "roots": [],
            "monotonic": monotonic,
            "hit_bound": True,
            "message": (
                f"The measured ratio {target_ratio:.2f} is outside what this "
                f"correlation model can produce over aspect {lo:g}-{hi:g} "
                f"(it spans {vals.min():.3f}-{vals.max():.3f}). The "
                f"measurement does not constrain the aspect ratio here -- "
                f"report that, do not quote the edge."
            ),
        }

    a = roots[0]
    # How hard is this root pinned?  d(ratio)/d(aspect) near the solution.
    da = max(0.02 * a, 0.05)
    slope = (ratio_at(a + da) - ratio_at(max(a - da, 1.0))) / (
        (a + da) - max(a - da, 1.0))
    # A 0.02 uncertainty on a ratio measured as "up to 10%" is generous.
    span = abs(0.02 / slope) if abs(slope) > 1e-9 else float("inf")

    msg = (f"An aspect ratio of {a:.2f} reproduces the Marquez mu_s' ratio "
           f"of {target_ratio:.2f}.")
    if len(roots) > 1:
        msg += (f" BUT SO DO {len(roots) - 1} OTHER VALUES "
                f"({', '.join(f'{r:.2f}' for r in roots[1:])}) -- the "
                f"measurement does not pick one.")
    if not monotonic:
        msg += (" The ratio is non-monotonic in the aspect ratio, so this was "
                "found by scanning rather than by bisection.")
    if span > 1.0:
        msg += (f" It is a LOOSE constraint: a plausible +/-0.02 on the "
                f"measured ratio moves the fitted aspect ratio by about "
                f"+/-{span:.1f}. Quote it as an order of magnitude, not a "
                f"determination.")

    return {
        "aspect": a,
        "achieved_ratio": float(ratio_at(a)),
        "target_ratio": target_ratio,
        "roots": roots,
        "monotonic": monotonic,
        "sensitivity_aspect_per_0p02": span,
        "hit_bound": False,
        "message": msg,
    }


# ==========================================================================
#  Birefringence, which this model cannot represent
# ==========================================================================
#: Lichtenegger et al., J. Biomed. Opt. 27(1):016001 (2022),
#: doi:10.1117/1.JBO.27.1.016001 -- zebrafish skeletal muscle in vivo,
#: Jones-matrix PS-OCT at 1310 nm.
ZEBRAFISH_BIREFRINGENCE = 1.7e-3
ZEBRAFISH_BIREFRINGENCE_RANGE = (1.6e-3, 1.8e-3)
BIREFRINGENCE_CITATION = (
    "Lichtenegger et al., J. Biomed. Opt. 27(1):016001 (2022), "
    "doi:10.1117/1.JBO.27.1.016001 (zebrafish skeletal muscle in vivo, "
    "PS-OCT at 1310 nm: dn = 1.6e-3 +/- 0.6e-4 at 1 month, "
    "1.8e-3 +/- 0.3e-4 at 2 months)"
)


def birefringence_context(cfg: StudyConfig) -> dict:
    """
    Size the one anisotropy effect this package does not model: birefringence.

    `tissue.to_tidy3d_medium` builds a SCALAR custom medium: one index per
    voxel, the same for every polarisation.  Real muscle is birefringent, and
    unlike the correlation-function aspect ratio, this one HAS been measured
    in a fish.

    The question worth answering is whether it matters here, and the way to
    answer it is to compare the birefringence against the index contrast the
    model does carry.  If dn_biref is a small fraction of the RMS index
    fluctuation, the scalar model is a defensible approximation.  If it is
    comparable, a scalar medium is the wrong model and no amount of meshing
    will fix it.

    Note the wavelength mismatch: the measurement is at 1310 nm and the study
    band is 500-900 nm.  Form birefringence from aligned sub-wavelength
    structure is only weakly dispersive, so transferring it is reasonable, but
    it is a transfer and should be labelled as one.
    """
    rms_dn = float(cfg.tissue.delta_n_sq) ** 0.5
    dn_b = ZEBRAFISH_BIREFRINGENCE
    ratio = dn_b / rms_dn if rms_dn > 0 else float("inf")

    if ratio < 0.10:
        verdict = (
            f"Birefringence is {100 * ratio:.1f}% of the RMS index "
            f"fluctuation the model carries.  A scalar medium is a "
            f"defensible approximation at that level."
        )
    elif ratio < 0.35:
        verdict = (
            f"Birefringence is {100 * ratio:.1f}% of the RMS index "
            f"fluctuation.  Not negligible.  A bowtie is polarisation-driven, "
            f"so the polarisation sensitivity from `fibre_axis_study` should "
            f"be reported alongside the result, and the scalar medium treated "
            f"as a limitation."
        )
    else:
        verdict = (
            f"Birefringence is {100 * ratio:.1f}% of the RMS index "
            f"fluctuation -- comparable to the contrast being modelled.  A "
            f"SCALAR custom medium is the wrong model for this tissue.  "
            f"Either use an anisotropic medium or restrict the claim to a "
            f"fixed, stated antenna orientation relative to the fibres."
        )

    return {
        "birefringence": dn_b,
        "birefringence_range": ZEBRAFISH_BIREFRINGENCE_RANGE,
        "rms_index_fluctuation": rms_dn,
        "ratio": ratio,
        "verdict": verdict,
        "citation": BIREFRINGENCE_CITATION,
        "measured_at_nm": 1310.0,
    }


# ==========================================================================
#  Reporting
# ==========================================================================
def print_anisotropy_study(cfg: StudyConfig, rows: list[dict] | None = None
                           ) -> None:
    """The whole anisotropy picture, in one table and three verdicts."""
    if rows is None:
        rows = fibre_axis_study(cfg)

    wl = rows[0]["wavelength_um"] * 1000
    print(f"  Direction-resolved scattering at {wl:.0f} nm, fibres along z.")
    print("  'aspect' is the correlation length along the fibre divided by")
    print("  the one across it.  aspect = 1 is the isotropic baseline.")
    print()
    print(f"  {'aspect':>7}  {'across,E||':>21}  {'across,E_|_':>21}  "
          f"{'along fibres':>21}")
    print(f"  {'':>7}  {'mu_s/mm':>9} {'g':>11}  {'mu_s/mm':>9} {'g':>11}  "
          f"{'mu_s/mm':>9} {'g':>11}")
    for r in rows:
        cells = []
        for name in ("across fibres, E || fibres",
                     "across fibres, E _|_ fibres",
                     "along fibres"):
            gm = r["geom"][name]
            cells.append(f"{gm['mu_s'] * 1000:9.3f} {gm['g']:11.4f}")
        print(f"  {r['aspect']:7.2f}  " + "  ".join(cells))

    print()
    print(f"  {'aspect':>7}  {'mu_s-prime ratio':>18}  "
          f"{'polarisation ratio':>19}")
    print("           (along vs across)      (antenna rotated)")
    for r in rows:
        print(f"  {r['aspect']:7.2f}  {r['mu_s_prime_ratio']:18.4f}  "
              f"{r['polarisation_ratio']:19.4f}")

    print()
    fit = aspect_from_measured_ratio(cfg)
    for line in _wrap(fit["message"], 68):
        print(f"  {line}")
    if fit["hit_bound"]:
        print("  *** The measurement does not constrain the model here. ***")
    print(f"  Constraint from: {MARQUEZ_CITATION[:60]}...")

    # The number that actually matters to a bowtie, evaluated at the aspect
    # ratio the measurement pins rather than at an arbitrary one.
    if not fit["hit_bound"]:
        at_fit = fibre_axis_study(cfg, aspects=(fit["aspect"],))[0]
        pr = at_fit["polarisation_ratio"]
        print()
        print(f"  AT THAT ASPECT RATIO, rotating the antenna 90 degrees on")
        print(f"  the fillet changes mu_s' by a factor of {pr:.2f}.")
        if pr > 1.25:
            print("  The sensor's reading depends on its orientation relative")
            print("  to the muscle fibres, which an isotropic model cannot")
            print("  represent. Fix and state the orientation, or report both.")
        else:
            print("  Small enough to report as a bound rather than model.")

    print()
    bi = birefringence_context(cfg)
    print(f"  Birefringence not modelled: dn = {bi['birefringence']:.2e} "
          f"against an RMS index fluctuation of "
          f"{bi['rms_index_fluctuation']:.2e}")
    for line in _wrap(bi["verdict"], 68):
        print(f"    {line}")

    print()
    print("  SUMMARY")
    print("  The correlation-function aspect ratio has never been measured in")
    print("  muscle, in any species.  It is a morphometry-derived modelling")
    print("  assumption constrained by the Marquez mu_s' ratio. The")
    print("  polarisation ratio above shows how much the answer depends on how")
    print("  the antenna is rotated on the fillet.")


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
#  Self-test: at aspect 1, this must reproduce the isotropic module
# ==========================================================================
def _self_test(cfg: StudyConfig | None = None) -> dict:
    """
    The 2-D quadrature here and the 1-D quadrature in `optical_properties`
    are different numerical schemes for the same integral.  At aspect ratio 1
    they must agree, and if they ever stop agreeing one of them is wrong.
    """
    from optical_properties import scattering_from_spectrum

    cfg = cfg or StudyConfig()
    t = cfg.tissue
    wl = cfg.calibration.calib_wavelength_um

    iso = scattering_from_spectrum(wl, t.n0, t.delta_n_sq, t.l_c_um, t.m,
                                   t.l_min_um)
    ani = scattering_anisotropic(wl, t.n0, t.delta_n_sq, t.l_c_um, t.m,
                                 t.l_min_um, aspect_xyz=(1.0, 1.0, 1.0))

    d_mu = abs(ani.mu_s_per_um - iso.mu_s_per_um) / max(iso.mu_s_per_um, 1e-30)
    d_g = abs(ani.g - iso.g)
    return {
        "mu_s_iso": iso.mu_s_per_um,
        "mu_s_aniso_at_1": ani.mu_s_per_um,
        "g_iso": iso.g,
        "g_aniso_at_1": ani.g,
        "rel_error_mu_s": d_mu,
        "abs_error_g": d_g,
        "ok": bool(d_mu < 5e-3 and d_g < 5e-3),
    }


if __name__ == "__main__":
    cfg = StudyConfig()
    st = _self_test(cfg)
    print("Self-test against the isotropic module at aspect = 1:")
    print(f"  mu_s   {st['mu_s_iso']:.6e} vs {st['mu_s_aniso_at_1']:.6e}  "
          f"(rel. error {st['rel_error_mu_s']:.2e})")
    print(f"  g      {st['g_iso']:.6f} vs {st['g_aniso_at_1']:.6f}  "
          f"(abs. error {st['abs_error_g']:.2e})")
    print(f"  {'PASS' if st['ok'] else 'FAIL -- one of the two is wrong'}")
    print()
    print_anisotropy_study(cfg)
