"""
tissue.py — building a synthetic piece of fish muscle.
======================================================

WHAT THIS MODULE DOES, IN ONE SENTENCE
--------------------------------------
It manufactures a three-dimensional map of refractive index that has the same
statistical texture as fish muscle, and hands it to Tidy3D as a material.

THE IDEA, WITHOUT MATHEMATICS
-----------------------------
Imagine a photograph of clouds.  You cannot predict the brightness of any
single pixel, but you can describe the picture statistically: clouds have a
typical blob size, and a typical contrast.  Given those two numbers you can
*generate* a new cloud picture that nobody could distinguish from a real one,
even though it is entirely made up.

Tissue is the same.  We do not know where every mitochondrion sits, and we do
not need to.  We need the blob size and the contrast.  Then we generate a
tissue that is statistically right, run light through it, and repeat with a
different random draw to see how much the answer wobbles.

The recipe for generating such a field is called *spectral synthesis*:

    1.  Fill the box with pure random noise (no structure at all).
    2.  Take its Fourier transform — this decomposes the noise into waves of
        every possible size.
    3.  Turn the volume up on the wave sizes that tissue has a lot of, and
        turn it down on the sizes tissue does not have.  The "volume knob
        setting versus wave size" curve is the power spectral density.
    4.  Transform back.  You now have noise that is structured like tissue.

Step 3 is where all the physics lives.

THE ONE TRAP EVERYBODY FALLS INTO
---------------------------------
The tissue spectrum used here (Whittle-Matern, shape parameter m) has a
nasty property in the regime that actually describes tissue, m < 1.5: it
contains infinite total power.  Structure keeps appearing as you look at
finer and finer scales, without limit.

Real tissue obviously does not do this.  Below a few nanometres there is no
"structure", just molecules.  So there is a smallest scale, and it is a
*physical* property of the tissue.

If you forget to say what that smallest scale is, the computer picks one for
you: your voxel size.  And then every result you compute changes when you
change the mesh, which means none of them mean anything.

This module therefore requires an explicit inner cutoff (`l_min_um` in the
config), applies it as a smooth spectral roll-off, and *reports* how much of
the nominal variance actually survives onto the grid.  `convergence.py` uses
those reports to prove the answer is mesh-independent.

WHAT COMES OUT
--------------
`generate_tissue()` returns a `TissueRealization`, which carries:

    n            the 3-D refractive index array
    x, y, z      the coordinate axes, in microns
    diagnostics  a dictionary of the quantities needed to judge the field:
                 realised variance, requested variance, fraction of variance
                 captured by the grid, fraction of voxels clipped, mean index

`to_tidy3d_medium()` converts it into a `td.CustomMedium` you can drop into a
simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import TissueConfig, FDTDConfig


# ==========================================================================
#  The Whittle-Matern spectrum
# ==========================================================================
def whittle_matern_psd(
    k: np.ndarray,
    delta_n_sq: float,
    l_c: float,
    m: float,
    l_min: float,
) -> np.ndarray:
    """
    The "volume knob" curve: how much index fluctuation lives at each spatial
    frequency.

    PLAIN LANGUAGE
    --------------
    `k` is a spatial frequency: large k means fine detail, small k means broad
    blobs.  This function says how much of the tissue's index variation sits at
    each level of detail.  It is large at small k (tissue has plenty of coarse
    structure) and falls away at large k, with the steepness of the fall
    controlled by `m`.

    THE FORMULA
    -----------
        Phi(k)  =  A * (1 + k^2 l_c^2)^(-m) * exp(-(k / k_cut)^2)

    with k_cut = 2*pi / l_min, and A chosen so that

        integral of Phi over all of 3-D k-space  =  delta_n_sq

    WHY THE EXPONENTIAL FACTOR IS PART OF THE PHYSICS, NOT A FUDGE
    --------------------------------------------------------------
    Without it, the integral above diverges whenever m <= 3/2 — which is
    precisely the mass-fractal regime that describes real tissue.  The
    divergence is the mathematics saying "structure keeps appearing forever as
    you zoom in".  Tissue does not do that: below a few nanometres there are
    molecules, not structure.  `l_min` is where structure stops.

    Imposing it explicitly has a large practical payoff.  If you leave it out,
    the computer silently substitutes your voxel size as the cutoff, and then
    every result changes when you change the mesh.  With `l_min` stated, the
    variance is a well-defined physical quantity and the mesh is free to be a
    numerical choice again.  `convergence.py` verifies exactly this.

    Parameters
    ----------
    k : array
        Spatial-frequency magnitude, radians per micron.
    delta_n_sq : float
        Total variance of the index fluctuations (the normalisation target).
    l_c : float
        Correlation length, microns.
    m : float
        Shape parameter.  m < 1.5 is a mass fractal of dimension 2m.
    l_min : float
        Inner cutoff, microns.  Structure smaller than this does not exist.

    Returns
    -------
    array, same shape as `k`, in units of index^2 * micron^3.
    """
    amplitude = _psd_normalisation(delta_n_sq, l_c, m, l_min)
    return amplitude * _spectrum_shape(k, l_c, m, l_min)


def _spectrum_shape(
    k: np.ndarray, l_c: float, m: float, l_min: float
) -> np.ndarray:
    """The bare spectral shape, before normalisation: Matern roll-off times
    the physical inner cutoff."""
    k_cut = 2.0 * np.pi / l_min
    return (1.0 + (k * l_c) ** 2) ** (-m) * np.exp(-((k / k_cut) ** 2))


def _psd_normalisation(
    delta_n_sq: float, l_c: float, m: float, l_min: float,
    n_points: int = 6000,
) -> float:
    """
    The constant A that makes the spectrum integrate to `delta_n_sq`.

    Computed numerically rather than in closed form.  A closed form exists only
    for m > 3/2 (it involves Gamma(m - 3/2), which is where the divergence
    shows up); with the inner cutoff in place the integral is finite for every
    m, but no longer elementary.  Numerical integration is exact enough,
    transparent, and works in every regime — which is worth more here than
    elegance.
    """
    k_hi = 40.0 * np.pi / l_min          # far past the cutoff; integrand ~ 0
    k_lo = 1e-6 / max(l_c, 1e-9)
    k = np.logspace(np.log10(k_lo), np.log10(k_hi), n_points)
    integrand = 4.0 * np.pi * k ** 2 * _spectrum_shape(k, l_c, m, l_min)
    total = float(np.trapezoid(integrand, k))
    if total <= 0:
        raise ValueError("Degenerate Whittle-Matern spectrum; check l_c, m, l_min.")
    return delta_n_sq / total


def band_limited_variance(
    delta_n_sq: float,
    l_c: float,
    m: float,
    l_min: float,
    k_min: float,
    k_max: float,
    n_points: int = 6000,
) -> float:
    """
    How much of the index variance actually fits between two spatial
    frequencies.

    PLAIN LANGUAGE
    --------------
    A simulation box of finite size cannot hold blobs bigger than the box, and
    a finite mesh cannot represent blobs smaller than a voxel.  The variance
    that ends up on your grid is therefore always less than the variance you
    asked for.  This says how much less.

    WHY YOU SHOULD CARE
    -------------------
    In the fractal regime this is not a small correction.  Reporting it turns
    a hidden mesh dependence into a stated, quantified modelling choice — and
    it is the number `convergence.py` watches to prove the final noise floor is
    physics rather than an artefact of your grid.

    Parameters
    ----------
    k_min : float
        Lowest represented spatial frequency, typically 2*pi / box_size.
    k_max : float
        Highest, typically the Nyquist frequency pi / voxel.

    Returns
    -------
    float — variance carried by that band, in index^2.  Always <= delta_n_sq.
    """
    if k_max <= k_min:
        return 0.0
    k = np.logspace(np.log10(max(k_min, 1e-12)), np.log10(k_max), n_points)
    psd = whittle_matern_psd(k, delta_n_sq, l_c, m, l_min)
    integrand = 4.0 * np.pi * k ** 2 * psd
    return float(np.trapezoid(integrand, k))


# ==========================================================================
#  The generated tissue
# ==========================================================================
@dataclass
class TissueRealization:
    """One randomly generated piece of synthetic muscle."""

    n: np.ndarray
    """3-D array of refractive index, indexed [ix, iy, iz]."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    """Coordinate axes in microns (cell centres)."""

    seed: int
    """The integer that produced it.  Re-running with the same seed and the
    same config reproduces this array bit for bit."""

    diagnostics: dict[str, Any]
    """Every quantity needed to check or reproduce the medium.  Print it."""

    @property
    def mean_index(self) -> float:
        """The volume-averaged index of THIS realisation.

        This drifts from seed to seed simply because the box is finite, and a
        resonance moves with it for completely ordinary reasons.  `ensemble.py`
        regresses this out so that what is left is genuine near-field speckle
        rather than a statement about the box size you happened to pick."""
        return float(self.n.mean())

    def mean_index_in_region(
        self, xlim: tuple[float, float],
        ylim: tuple[float, float],
        zlim: tuple[float, float],
    ) -> float:
        """Average index inside a sub-box — normally the interrogation volume.

        More relevant than the whole-box mean, because the resonance responds
        to the index the mode actually samples."""
        ix = (self.x >= xlim[0]) & (self.x <= xlim[1])
        iy = (self.y >= ylim[0]) & (self.y <= ylim[1])
        iz = (self.z >= zlim[0]) & (self.z <= zlim[1])
        if not (ix.any() and iy.any() and iz.any()):
            return float("nan")
        return float(self.n[np.ix_(ix, iy, iz)].mean())

    def summary(self) -> str:
        d = self.diagnostics
        return (
            f"Tissue realization (seed {self.seed})\n"
            f"  grid                 {self.n.shape}  "
            f"at {d['voxel_nm']:.1f} nm voxels\n"
            f"  mean index           {self.mean_index:.5f}  "
            f"(target n0 = {d['n0']:.5f})\n"
            f"  realised variance    {d['realised_variance']:.4e}  "
            f"(RMS dn = {d['realised_variance'] ** 0.5:.5f})\n"
            f"  nominal variance     {d['nominal_variance']:.4e}\n"
            f"  variance on grid     {100 * d['variance_capture_fraction']:.1f}%"
            f"  <- mesh-dependent if this is far from 100%\n"
            f"  voxels clipped       {100 * d['clipped_fraction']:.4f}%\n"
            f"  k range represented  {d['k_min']:.2f} to {d['k_max']:.1f} rad/um\n"
            f"  blobs across box     {d['blobs_across_box']:.1f}  "
            f"<- want >> 1 for meaningful statistics"
        )


# ==========================================================================
#  The generator
# ==========================================================================
def _selftest_rfft_equivalence(n=(24, 20, 16), seed=7) -> float:
    """Prove the real-transform route reproduces the full-complex one.

    The optimisation in `generate_tissue` replaces

        ifftn(fftn(white) * sqrt(psd_full)).real

    with

        irfftn(rfftn(white) * sqrt(psd_half))

    These agree exactly because `white` is real, so its spectrum is Hermitian,
    and the PSD depends on k only through k^2, so it is even in every axis and
    preserves that symmetry.  The negative-kz half therefore carries no
    information.  This function checks it numerically rather than asserting
    it, and returns the maximum absolute difference.

    Run it after any change to the spectral synthesis:

        python -c "import tissue; print(tissue._selftest_rfft_equivalence())"
    """
    gx, gy, gz = n
    dv = 0.01
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((gx, gy, gz))

    def psd_of(kmag):
        # any smooth even function of |k| will do for the equivalence test
        out = np.exp(-(kmag * 0.05) ** 2) / (1.0 + kmag ** 2)
        out.flat[0] = 0.0
        return out

    kxf = 2 * np.pi * np.fft.fftfreq(gx, d=dv)
    kyf = 2 * np.pi * np.fft.fftfreq(gy, d=dv)
    kzf = 2 * np.pi * np.fft.fftfreq(gz, d=dv)
    KX, KY, KZ = np.meshgrid(kxf, kyf, kzf, indexing="ij")
    full = np.fft.ifftn(
        np.fft.fftn(white) * np.sqrt(psd_of(np.sqrt(KX**2 + KY**2 + KZ**2)))
    ).real

    kzr = 2 * np.pi * np.fft.rfftfreq(gz, d=dv)
    kmag_half = np.sqrt((kxf**2)[:, None, None] + (kyf**2)[None, :, None]
                        + (kzr**2)[None, None, :])
    half = np.fft.irfftn(
        np.fft.rfftn(white) * np.sqrt(psd_of(kmag_half)),
        s=(gx, gy, gz), axes=(0, 1, 2))

    return float(np.max(np.abs(full - half)))


def generate_tissue(
    tissue_cfg: TissueConfig,
    fdtd_cfg: FDTDConfig,
    seed: int,
    domain_um: tuple[float, float, float] | None = None,
    voxel_nm: float | None = None,
    apply_taper: bool = True,
    oversample: float = 1.0,
) -> TissueRealization:
    """
    Generate one random realisation of fish muscle.

    THE FIVE STEPS
    --------------
    1.  Build the coordinate grid and the matching spatial-frequency grid.
    2.  Evaluate the Whittle-Matern spectrum on that frequency grid, and apply
        the physical inner cutoff so fine structure stops where tissue stops
        having structure.
    3.  Filter white noise through it.
    4.  Rescale to the variance that this band can actually carry, and record
        how much of the nominal variance that is.
    5.  Taper the fluctuations to zero near the domain edges, so the absorbing
        boundary sees a uniform medium, then clip and add the baseline index.

    Parameters
    ----------
    seed : int
        Anything reproducible.  `ensemble.py` derives one per realisation from
        a single master seed.
    domain_um, voxel_nm : optional
        Override the config, for the convergence studies.
    apply_taper : bool
        Set False only when generating a bulk slab for the analytic
        calibration cross-check, where there is no absorbing boundary to
        protect.

    Returns
    -------
    TissueRealization
    """
    domain = domain_um if domain_um is not None else fdtd_cfg.domain_um
    dv = (voxel_nm if voxel_nm is not None else fdtd_cfg.tissue_voxel_nm) * 1e-3

    # ---- step 1: grids ------------------------------------------------
    # The grid we will KEEP.
    nx, ny, nz = (max(8, int(round(L / dv))) for L in domain)
    x = (np.arange(nx) - (nx - 1) / 2) * dv
    y = (np.arange(ny) - (ny - 1) / 2) * dv
    z = (np.arange(nz) - (nz - 1) / 2) * dv

    # The grid we will GENERATE on, which may be larger -- see `oversample`.
    over = max(1.0, float(oversample))
    gx, gy, gz = (max(8, int(round(n * over))) for n in (nx, ny, nz))

    # Spatial frequencies in radians per micron.  Anisotropy is applied here,
    # by stretching the correlation length differently along each axis: a
    # longer correlation length along z means the spectrum falls off sooner in
    # kz, which means blobs elongated along z.
    # MEMORY.  A full meshgrid plus a complex FFT would need about 96 bytes
    # per voxel (roughly 12 GB for a 5 x 5 x 5 um domain at 10 nm voxels).
    # Two choices keep the peak near 32 bytes per voxel without changing
    # the result:
    #
    #   1. Broadcast the three 1-D frequency axes into k_mag directly, rather
    #      than building three full meshgrid arrays.
    #
    #   2. Use the REAL transform pair. `white` is real, so its spectrum is
    #      Hermitian and the negative-kz half is redundant. rfftn stores only
    #      kz >= 0, which halves both the spectrum and the PSD evaluated
    #      against it. The result is identical to the full-complex route to
    #      floating-point precision, which `_selftest_rfft_equivalence`
    #      checks.
    ax, ay, az = tissue_cfg.anisotropy_xyz
    kx = (2 * np.pi * np.fft.fftfreq(gx, d=dv) * ax).astype(np.float64)
    ky = (2 * np.pi * np.fft.fftfreq(gy, d=dv) * ay).astype(np.float64)
    # rfftn reduces the LAST axis, so kz runs over the non-negative half only.
    kz = (2 * np.pi * np.fft.rfftfreq(gz, d=dv) * az).astype(np.float64)

    k_mag = np.sqrt(
        (kx ** 2)[:, None, None]
        + (ky ** 2)[None, :, None]
        + (kz ** 2)[None, None, :]
    )

    # ---- step 2: the spectrum, with a PHYSICAL inner cutoff -----------
    # The inner cutoff already lives inside `whittle_matern_psd`: it is what
    # makes the total variance finite in the fractal regime, and it is applied
    # as a smooth Gaussian roll-off rather than a hard truncation, because a
    # hard edge in frequency space produces ringing in real space (the Gibbs
    # phenomenon).
    psd = whittle_matern_psd(
        k_mag, tissue_cfg.delta_n_sq, tissue_cfg.l_c_um, tissue_cfg.m,
        tissue_cfg.l_min_um,
    )

    # The k = 0 component is the mean.  We set the mean explicitly later, so
    # remove it here to keep the fluctuation field exactly zero-mean.
    psd.flat[0] = 0.0

    # ---- step 3: filter white noise -----------------------------------
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((gx, gy, gz))
    # sqrt in place on the PSD, then filter through the real transform pair.
    # `white` is released before the inverse transform allocates, which is
    # what keeps the peak down rather than merely the total.
    np.sqrt(psd, out=psd)
    spectrum = np.fft.rfftn(white)
    del white
    spectrum *= psd
    del psd
    filtered = np.fft.irfftn(spectrum, s=(gx, gy, gz), axes=(0, 1, 2))
    del spectrum

    # ---- step 4: rescale to the band-limited variance ------------------
    # The field is NOT forced to the nominal variance delta_n_sq, because this
    # grid cannot hold that much in the fractal regime. It is given the
    # variance the represented band can carry, and the ratio is reported.
    k_box_min = 2 * np.pi / max(domain)
    k_upper = np.pi / dv                       # Nyquist of the tissue voxel grid
    var_band = band_limited_variance(
        tissue_cfg.delta_n_sq, tissue_cfg.l_c_um, tissue_cfg.m,
        tissue_cfg.l_min_um, k_box_min, k_upper,
    )
    var_nominal = tissue_cfg.delta_n_sq

    # Normalise on the FULL generated field, then crop.  Normalising after
    # cropping would erase precisely the mean drift we may want to keep.
    filtered = filtered - filtered.mean()
    std_now = filtered.std()
    if std_now > 0:
        filtered *= np.sqrt(var_band) / std_now

    if over > 1.0:
        i0, j0, k0 = (gx - nx) // 2, (gy - ny) // 2, (gz - nz) // 2
        filtered = filtered[i0:i0 + nx, j0:j0 + ny, k0:k0 + nz]

    # ---- step 5: taper, clip, add baseline ----------------------------
    if apply_taper and fdtd_cfg.taper_width_um > 0:
        filtered *= _edge_taper(x, y, z, fdtd_cfg.taper_width_um, domain)

    n_field = tissue_cfg.n0 + filtered
    n_clipped = np.clip(n_field, tissue_cfg.n_clip_min, tissue_cfg.n_clip_max)
    clipped_fraction = float((n_clipped != n_field).mean())
    n_field = n_clipped

    diagnostics = {
        "n0": tissue_cfg.n0,
        "voxel_nm": dv * 1e3,
        "domain_um": domain,
        "realised_variance": float(filtered.var()),
        "band_limited_variance": var_band,
        "nominal_variance": var_nominal,
        "variance_capture_fraction": (
            var_band / var_nominal if var_nominal > 0 else float("nan")
        ),
        "clipped_fraction": clipped_fraction,
        "k_min": k_box_min,
        "k_max": k_upper,
        "blobs_across_box": min(domain) / tissue_cfg.l_c_um,
        "m": tissue_cfg.m,
        "l_c_um": tissue_cfg.l_c_um,
        "l_min_um": tissue_cfg.l_min_um,
        "tapered": apply_taper,
        "oversample": over,
        "mean_drift": float(filtered.mean()),
    }

    return TissueRealization(
        n=n_field, x=x, y=y, z=z, seed=seed, diagnostics=diagnostics
    )


def _edge_taper(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    width_um: float, domain: tuple[float, float, float],
) -> np.ndarray:
    """
    A window that is 1 in the middle of the box and falls smoothly to 0 near
    every face.

    WHY THIS IS NOT OPTIONAL
    ------------------------
    Absorbing boundaries (PML or a lossy absorber) are derived on the
    assumption that the medium does not vary along the direction being
    absorbed.  A random medium violates that everywhere.  The result is
    spurious reflections and — in long runs, which random media require —
    fields that slowly *grow* instead of decaying, quietly destroying the run.

    Tapering the fluctuations to zero over a buffer layer means the boundary
    sees a uniform medium, which is exactly what it was designed for.  The cost
    is that the outer `width_um` of the box is not real tissue, so make the box
    correspondingly larger.
    """
    def axis_taper(coords: np.ndarray, length: float) -> np.ndarray:
        edge = length / 2.0
        d = edge - np.abs(coords)                # distance to the nearest face
        t = np.clip(d / width_um, 0.0, 1.0)
        return 0.5 * (1.0 - np.cos(np.pi * t))   # raised cosine, C1-continuous

    tx = axis_taper(x, domain[0])
    ty = axis_taper(y, domain[1])
    tz = axis_taper(z, domain[2])
    return tx[:, None, None] * ty[None, :, None] * tz[None, None, :]


# ==========================================================================
#  Handing the tissue to Tidy3D
# ==========================================================================
def to_tidy3d_medium(realization: TissueRealization, interp: str = "linear",
                     dtype=np.float32):
    """
    Wrap the index array as a Tidy3D material.

    PLAIN LANGUAGE
    --------------
    Tidy3D normally builds a scene from shapes (a box, a cylinder, a polygon)
    each made of a uniform material.  A random medium is not a shape — it is a
    number at every point.  `CustomMedium` is the object for that: you hand it
    an array of permittivity values on a grid, and it interpolates onto
    whatever FDTD mesh the solver ends up using.

    Note the conversion: permittivity is the *square* of refractive index.

    Parameters
    ----------
    interp : {"linear", "nearest"}
        How to interpolate between your voxels and the FDTD cells.  "linear" is
        smoother and generally what you want for a continuous random medium;
        "nearest" is faster and reproduces the voxelisation exactly, which is
        occasionally what you want for a convergence test.
    """
    import tidy3d as td

    # SINGLE PRECISION IS SUFFICIENT HERE, AND IT HALVES THE UPLOAD.
    #
    # The permittivity array travels to the solver and comes back inside every
    # stored .hdf5.  At the production box that is 62.5 million values; at the
    # larger box used for the finite-box study it is 125 million, which is
    # 1.0 GB per simulation in float64 and 0.5 GB in float32.  On a slow link
    # that difference is hours across an ensemble.
    #
    # The precision cost is nothing.  Round-tripping this field through
    # float32 perturbs the refractive index by 1.3e-8 rms against a tissue
    # contrast of 1.0e-2, a relative error of 1.2e-6.  It is also a thousand
    # times smaller than the 1.3e-3 index step the paired protocol applies,
    # so it cannot reach any quantity this study reports.
    #
    # Pass dtype=np.float64 if double precision is required.
    eps = td.SpatialDataArray(
        (realization.n ** 2).astype(dtype),
        coords={
            "x": realization.x,
            "y": realization.y,
            "z": realization.z,
        },
    )
    return td.CustomMedium.from_eps_raw(eps, interp_method=interp)


def water_medium(n: float = 1.333):
    """A uniform medium standing in for the aqueous reference case.

    Every tissue quantity is reported as a ratio against this, because that
    ratio is what converts a published water-calibrated figure of merit into a
    tissue one."""
    import tidy3d as td
    return td.Medium(permittivity=n ** 2)


# ==========================================================================
#  Self-check
# ==========================================================================
if __name__ == "__main__":
    from config import StudyConfig

    cfg = StudyConfig()
    print(cfg.summary())
    print()

    # A quick, small realisation so this runs in a second.
    real = generate_tissue(
        cfg.tissue, cfg.fdtd, seed=1,
        domain_um=(2.0, 2.0, 2.0), voxel_nm=20.0,
    )
    print(real.summary())
    print()

    # Verify the spectrum normalisation for a case where it is well defined.
    print("Normalisation check - the spectrum must integrate to delta_n_sq")
    print("in every regime, fractal (m < 1.5) included:")
    for m in (1.15, 1.35, 2.0, 2.5):
        target = 1.0e-3
        got = band_limited_variance(target, 0.3, m, 0.005, 1e-6, 1e6)
        print(f"  m = {m:<5}  asked {target:.4e}  got {got:.4e}  "
              f"ratio {got / target:.5f}")
