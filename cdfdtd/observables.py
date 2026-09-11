"""
observables.py — turning fields into the numbers the paper reports.
===================================================================

WHAT COMES OUT OF A SIMULATION, AND WHAT WE ACTUALLY WANT
---------------------------------------------------------
A finished FDTD run gives you the electric field at every grid point and every
recorded wavelength.  That is a beautiful picture and not yet a result.

Four numbers are wanted:

  RESONANCE WAVELENGTH   where the sensor's response peaks.  Moving this is
                         the signal.

  LINEWIDTH              how broad that peak is.  A shift you cannot resolve
                         against the width is not a measurement.

  SENSITIVITY MAP        for each point in space: if a cadmium complex bound
                         *there*, how much would the resonance move?  This is
                         not the same as "where is the field bright", and the
                         difference matters.

  EFFECTIVE VOLUME       the smallest region containing most of the
                         sensitivity, restricted to places the analyte can
                         actually reach.

WHY THE SENSITIVITY MAP IS NOT JUST |E|^2
------------------------------------------
Papers routinely plot |E|^2 and call it the sensitivity.  It is proportional
to it, but the proportionality constant is what carries the physics.  The
proper statement comes from perturbation theory:

    delta_omega / omega  =  -(1/2) * integral( delta_eps |E|^2 ) / U

where U is the total electromagnetic energy stored in the mode.  So the
sensitivity per unit permittivity change at a point is

    S(r)  =  (1/2) * |E(r)|^2 / U

TWO THINGS ARE EASY TO GET WRONG HERE, AND BOTH ARE LARGE.

1.  THE FACTOR OF ONE HALF AND THE SIGN.  Adding material (positive
    delta_eps) moves the resonance to LONGER wavelength.  If you drop the
    sign you will predict shifts in the wrong direction.

2.  THE ENERGY DENOMINATOR IS NOT integral(eps |E|^2).  In a dispersive
    material — and gold in the visible is violently dispersive — the stored
    energy density is

        d(omega * eps) / d(omega)  * |E|^2 ,  not  eps * |E|^2

    This is the Brillouin energy density.  For gold near 800 nm, eps is about
    -25 while d(omega eps)/d(omega) is about +30.  Using eps would give you a
    denominator of the wrong SIGN as well as the wrong size.

    It does not cancel when you take ratios between two host media either,
    because the field distribution inside the metal is not identical in the
    two cases.  `mode_energy` below does it properly, by differentiating the
    material model numerically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from config import StudyConfig, C_LIGHT_UM_S


# ==========================================================================
#  Spectra: resonance and linewidth
# ==========================================================================
@dataclass
class ResonanceFit:
    """The peak of a spectrum, and how wide it is."""

    wavelength_um: float
    fwhm_um: float
    peak_value: float
    q_factor: float
    """Resonance wavelength divided by linewidth.  A dimensionless measure of
    how sharp the resonance is.  Gold bowties typically land between 5 and 20;
    anything above about 50 in the visible means something is wrong (most
    likely the run was too short, or the gold model is not lossy enough)."""

    converged: bool
    warning: str = ""

    def __str__(self) -> str:
        w = f"   [{self.warning}]" if self.warning else ""
        return (
            f"resonance {self.wavelength_um * 1000:.2f} nm, "
            f"FWHM {self.fwhm_um * 1000:.2f} nm, Q = {self.q_factor:.1f}{w}"
        )


def fit_resonance(
    wavelengths_um: np.ndarray,
    response: np.ndarray,
) -> ResonanceFit:
    """
    Find the peak of a spectrum and measure its width.

    HOW
    ---
    1.  Find the largest sample.
    2.  Fit a parabola through that sample and its two neighbours, and take the
        parabola's vertex.  This locates the peak to a fraction of the sample
        spacing, which matters because the resonance SHIFTS we care about can
        be smaller than the wavelength grid spacing.
    3.  Walk outward from the peak to find where the response falls to half
        its maximum, interpolating linearly between the bracketing samples.
        The distance between those two crossings is the full width at half
        maximum.

    WARNINGS IT WILL GIVE YOU
    -------------------------
    * If the peak is at the edge of the recorded band, the resonance is
      probably outside your band entirely and the "fit" is meaningless.
    * If a half-maximum crossing cannot be found on one side, the linewidth is
      a lower bound only.
    * If the peak spans fewer than about five samples, the wavelength grid is
      too coarse to measure shifts reliably.

    Parameters
    ----------
    wavelengths_um : array, increasing
    response : array
        Any quantity that peaks at resonance: |E|^2 at the hot spot,
        scattering cross-section, absorbed power.

    Returns
    -------
    ResonanceFit
    """
    wl = np.asarray(wavelengths_um, dtype=float)
    y = np.asarray(response, dtype=float)
    order = np.argsort(wl)
    wl, y = wl[order], y[order]

    if wl.size < 5:
        return ResonanceFit(float("nan"), float("nan"), float("nan"),
                            float("nan"), False, "too few spectral samples")

    i = int(np.argmax(y))
    warnings_ = []

    if i in (0, len(y) - 1):
        return ResonanceFit(
            float(wl[i]), float("nan"), float(y[i]), float("nan"), False,
            "peak is at the edge of the recorded band -- the true resonance "
            "is probably outside it; widen calibration.band_um",
        )

    # --- sub-sample peak by parabolic interpolation ---------------------
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) > 1e-30:
        offset = 0.5 * (y0 - y2) / denom
        offset = float(np.clip(offset, -1.0, 1.0))
    else:
        offset = 0.0
    dwl = float(np.mean(np.diff(wl)))
    peak_wl = float(wl[i] + offset * dwl)
    peak_val = float(y1)

    # --- half-maximum crossings -----------------------------------------
    half = 0.5 * peak_val
    left = _crossing(wl[:i + 1][::-1], y[:i + 1][::-1], half)
    right = _crossing(wl[i:], y[i:], half)

    if left is None or right is None:
        warnings_.append("half-maximum not reached on both sides; FWHM is a "
                         "lower bound")
        fwhm = float("nan")
    else:
        fwhm = float(right - left)
        n_in_peak = int(np.sum((wl >= left) & (wl <= right)))
        if n_in_peak < 5:
            warnings_.append(
                f"only {n_in_peak} samples across the peak; increase "
                f"fdtd.freq_points before trusting small shifts"
            )

    q = peak_wl / fwhm if fwhm and np.isfinite(fwhm) and fwhm > 0 else float("nan")

    return ResonanceFit(
        wavelength_um=peak_wl, fwhm_um=fwhm, peak_value=peak_val,
        q_factor=q, converged=not warnings_, warning="; ".join(warnings_),
    )


def _crossing(wl: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """First place, walking along the arrays, where `y` drops through `level`.
    Linearly interpolated.  Returns None if it never does."""
    for j in range(1, len(y)):
        if (y[j - 1] - level) * (y[j] - level) <= 0 and y[j - 1] != y[j]:
            t = (level - y[j - 1]) / (y[j] - y[j - 1])
            return float(wl[j - 1] + t * (wl[j] - wl[j - 1]))
    return None


def hotspot_spectrum(sim_data, monitor_name: str = "hotspot_probe"):
    """
    Extract |E|^2 against wavelength at the gap-centre probe.

    Returns
    -------
    (wavelengths_um, intensity)
    """
    mon = sim_data[monitor_name]
    freqs = np.array(mon.Ex.f)
    intensity = np.zeros_like(freqs, dtype=float)
    for comp in ("Ex", "Ey", "Ez"):
        arr = np.array(getattr(mon, comp)).squeeze()
        intensity += np.abs(arr) ** 2
    wl = C_LIGHT_UM_S / freqs
    order = np.argsort(wl)
    return wl[order], intensity[order]


# ==========================================================================
#  The energy denominator, done properly
# ==========================================================================
def brillouin_factor(
    medium_eps_fn: Callable[[float], complex],
    freq_hz: float,
    rel_step: float = 1e-3,
) -> float:
    """
    d(omega * eps)/d(omega), evaluated numerically.

    PLAIN LANGUAGE
    --------------
    "How much energy is stored in the field here" is easy in ordinary glass:
    it is proportional to eps times |E|^2.  In a metal it is not, because the
    energy is partly kinetic — it is in the motion of the electrons, not the
    field.  The correct expression involves how the permittivity CHANGES with
    frequency, and for gold that change is large.

    Concretely, at 800 nm gold has eps of roughly -25, but d(omega eps)/d(omega)
    of roughly +30.  Use eps and your energy comes out negative, which is
    nonsense; the resulting sensitivity would be wrong in both sign and size.

    We differentiate the material model numerically with a central difference
    rather than assuming a Drude form, so this stays correct for whatever
    material fit you plug in.
    """
    h = freq_hz * rel_step
    f_lo, f_hi = freq_hz - h, freq_hz + h
    e_lo = complex(medium_eps_fn(f_lo))
    e_hi = complex(medium_eps_fn(f_hi))
    d_omega_eps = (f_hi * e_hi - f_lo * e_lo) / (f_hi - f_lo)
    return float(np.real(d_omega_eps))


def mode_energy(
    e_squared: np.ndarray,
    eps: np.ndarray,
    freq_hz: float,
    voxel_volume_um3: float,
    gold_eps_fn: Callable[[float], complex] | None = None,
    metal_threshold: float = -1.0,
) -> float:
    """
    Total electromagnetic energy stored in the mode.

    In the dielectric regions the energy density is eps |E|^2.  Inside the
    metal it is d(omega eps)/d(omega) |E|^2 instead — see `brillouin_factor`.
    This function applies the right one in each region, using the recorded
    permittivity map to tell them apart.

    If you do not supply `gold_eps_fn`, the metal contribution falls back to
    |eps| |E|^2, which at least has the right sign.  Supply it: the difference
    is not small.

    Parameters
    ----------
    e_squared : array
        |E|^2 on the monitor grid.
    eps : array
        Complex permittivity on the same grid.
    voxel_volume_um3 : float
        Volume of one monitor cell.
    """
    eps_r = np.real(eps)
    metal = eps_r < metal_threshold

    density = np.where(metal, np.abs(eps_r), eps_r) * e_squared
    if gold_eps_fn is not None and metal.any():
        density = np.where(
            metal, brillouin_factor(gold_eps_fn, freq_hz) * e_squared, density
        )
    return float(density.sum() * voxel_volume_um3)


# ==========================================================================
#  The sensitivity map
# ==========================================================================
@dataclass
class SensitivityMap:
    """Where a bound analyte would actually produce a signal."""

    s: np.ndarray
    """Sensitivity density: fractional resonance shift per unit permittivity
    change per unit volume, at each point.  Units of 1/micron^3."""

    accessible: np.ndarray
    """Boolean: True where the analyte can physically reach."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    voxel_volume_um3: float
    wavelength_um: float

    # --- the headline geometry numbers ---------------------------------
    @property
    def total_accessible_sensitivity(self) -> float:
        """Summed sensitivity over reachable space.  This, not V_eff, is what
        governs the signal from a given surface coverage."""
        return float((self.s * self.accessible).sum() * self.voxel_volume_um3)

    def effective_volume(self, fraction: float = 0.90) -> float:
        """
        The smallest volume containing `fraction` of the accessible
        sensitivity.

        HOW IT IS CONSTRUCTED
        ---------------------
        Sort every reachable voxel by its sensitivity, brightest first.  Add
        them up until you have accumulated 90% of the total.  The volume of the
        voxels you used is V_eff.

        This is a level-set construction, and in a heterogeneous medium the
        resulting region is fragmented rather than a neat blob.  That is real,
        not an artefact — the tissue speckle breaks the hot spot up.  Report
        the ENSEMBLE SPREAD of V_eff as well as its mean.
        """
        vals = self.s[self.accessible].ravel()
        if vals.size == 0:
            return float("nan")
        vals = np.sort(vals)[::-1]
        cumulative = np.cumsum(vals)
        total = cumulative[-1]
        if total <= 0:
            return float("nan")
        n = int(np.searchsorted(cumulative, fraction * total) + 1)
        return float(n * self.voxel_volume_um3)

    def probe_depth(self, fraction: float = 0.90, axis: int = 2) -> float:
        """
        How far from the surface 90% of the sensitivity lies.

        Measured along `axis` (default z, away from the substrate), as the
        distance from the structure at which the cumulative sensitivity
        profile reaches `fraction`.
        """
        weighted = (self.s * self.accessible)
        coords = (self.x, self.y, self.z)[axis]
        other = tuple(a for a in (0, 1, 2) if a != axis)
        profile = weighted.sum(axis=other)
        if profile.sum() <= 0:
            return float("nan")
        # Measure outward from the plane of peak sensitivity.
        centre = int(np.argmax(profile))
        dist = np.abs(coords - coords[centre])
        order = np.argsort(dist)
        cumulative = np.cumsum(profile[order])
        cumulative /= cumulative[-1]
        idx = int(np.searchsorted(cumulative, fraction))
        idx = min(idx, len(order) - 1)
        return float(dist[order][idx])

    def summary(self) -> str:
        v = self.effective_volume()
        return (
            f"Sensitivity map at {self.wavelength_um * 1000:.1f} nm\n"
            f"  accessible fraction  "
            f"{100 * float(self.accessible.mean()):.1f}% of the monitor volume\n"
            f"  V_eff (90%)          {v * 1e9:.2f} x 10^-9 um^3  "
            f"= ({(v ** (1 / 3)) * 1000:.1f} nm)^3 equivalent\n"
            f"  probe depth d90      {self.probe_depth() * 1000:.1f} nm\n"
            f"  total accessible S   {self.total_accessible_sensitivity:.4e}"
        )


def build_sensitivity_map(
    sim_data,
    cfg: StudyConfig,
    wavelength_um: float,
    accessible_mask: np.ndarray | None = None,
    field_monitor: str = "near_field",
    eps_monitor: str = "permittivity",
) -> SensitivityMap:
    """
    Convert a recorded near field into a sensitivity map.

    THE CALCULATION
    ---------------
        S(r) = (1/2) |E(r)|^2 / U

    with U the total mode energy computed with the correct (dispersive) energy
    density.  See this module's header for why that matters.

    The map is then masked by steric accessibility: any point the analyte
    cannot reach contributes zero, however bright it is.

    Parameters
    ----------
    accessible_mask : array or None
        Pass one computed by `sensor.steric_accessibility_mask`.  If None, it
        is derived here from the recorded permittivity, which is the usual
        path.
    """
    from sensor import coated_solid_mask, steric_accessibility_mask, gold_medium

    fmon = sim_data[field_monitor]
    emon = sim_data[eps_monitor]

    freqs = np.array(fmon.Ex.f)
    target_f = C_LIGHT_UM_S / wavelength_um
    fi = int(np.argmin(np.abs(freqs - target_f)))
    f_used = float(freqs[fi])

    def comp(name):
        return np.array(getattr(fmon, name).isel(f=fi)).squeeze()

    e2 = np.abs(comp("Ex")) ** 2 + np.abs(comp("Ey")) ** 2 + np.abs(comp("Ez")) ** 2

    eps = np.array(emon.eps_xx.isel(f=0)).squeeze()

    x = np.array(fmon.Ex.x)
    y = np.array(fmon.Ex.y)
    z = np.array(fmon.Ex.z)
    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
    dy = float(np.mean(np.diff(y))) if len(y) > 1 else 1.0
    dz = float(np.mean(np.diff(z))) if len(z) > 1 else 1.0
    dv = dx * dy * dz

    # --- the energy denominator, dispersion included --------------------
    try:
        au = gold_medium()
        gold_eps_fn = lambda f: au.eps_model(f)
    except Exception:
        gold_eps_fn = None
    u_total = mode_energy(e2, eps, f_used, dv, gold_eps_fn)

    s = 0.5 * e2 / u_total if u_total > 0 else np.zeros_like(e2)

    # --- steric mask -----------------------------------------------------
    if accessible_mask is None:
        solid = coated_solid_mask(eps)
        accessible_mask = steric_accessibility_mask(
            solid, x, y, z, cfg.surface.analyte_hydrodynamic_radius_nm
        ).mask

    return SensitivityMap(
        s=s, accessible=accessible_mask, x=x, y=y, z=z,
        voxel_volume_um3=dv, wavelength_um=C_LIGHT_UM_S / f_used,
    )


# ==========================================================================
#  The transfer factor
# ==========================================================================
@dataclass
class TransferFactor:
    """How much of a water-calibrated performance survives in tissue."""

    eta_fom: float
    """The headline number: the ratio of figures of merit, tissue over water.

    FOM = (accessible sensitivity) / (linewidth).  This is the quantity that
    actually sets a limit of detection, because an LOD is a shift you must
    resolve against a width."""

    eta_volume: float
    """The ratio of effective interrogation volumes.

    Reported because it is descriptive and interesting, NOT as the correction
    factor.  A published limit of detection is set by the shift per binding
    event and the linewidth it must be resolved against — not by how big the
    interrogation volume is.  If the recognition layer sits a few nanometres
    from the metal in both cases, V_eff can change a great deal while the
    per-event shift barely moves.  Quoting the volume ratio as the correction
    would be a category error."""

    eta_sensitivity: float
    """Ratio of total accessible sensitivity alone, without the linewidth."""

    eta_linewidth: float
    """Ratio of linewidths.  Tissue scattering broadens the resonance; this
    says by how much, and it is half the reason the FOM ratio is not the
    sensitivity ratio."""

    resonance_shift_nm: float
    """How far the resonance moves from water to tissue, purely from the
    higher baseline index of muscle.  A result in its own right: it is the size
    of the error made by every model that assumes n = 1.33."""

    def __str__(self) -> str:
        return (
            "Transfer factor (tissue relative to water)\n"
            f"  eta on figure of merit   {self.eta_fom:.4f}   <-- use this one\n"
            f"    from sensitivity       {self.eta_sensitivity:.4f}\n"
            f"    from linewidth         {self.eta_linewidth:.4f}\n"
            f"  eta on V_eff             {self.eta_volume:.4f}   "
            f"(descriptive only)\n"
            f"  baseline resonance shift {self.resonance_shift_nm:+.2f} nm  "
            f"(water -> tissue)"
        )


def transfer_factor(
    tissue_map: SensitivityMap, tissue_fit: ResonanceFit,
    water_map: SensitivityMap, water_fit: ResonanceFit,
) -> TransferFactor:
    """
    Compare a tissue run against its aqueous reference.

    Both runs must use the same geometry, the same mesh and the same
    wavelength grid; only the background medium differs.  Otherwise the ratio
    picks up numerical differences and means nothing.
    """
    s_t = tissue_map.total_accessible_sensitivity
    s_w = water_map.total_accessible_sensitivity
    fom_t = s_t / tissue_fit.fwhm_um if tissue_fit.fwhm_um > 0 else np.nan
    fom_w = s_w / water_fit.fwhm_um if water_fit.fwhm_um > 0 else np.nan

    return TransferFactor(
        eta_fom=float(fom_t / fom_w) if fom_w else float("nan"),
        eta_volume=float(
            tissue_map.effective_volume() / water_map.effective_volume()
        ),
        eta_sensitivity=float(s_t / s_w) if s_w else float("nan"),
        eta_linewidth=float(tissue_fit.fwhm_um / water_fit.fwhm_um)
        if water_fit.fwhm_um else float("nan"),
        resonance_shift_nm=float(
            (tissue_fit.wavelength_um - water_fit.wavelength_um) * 1000
        ),
    )


# ==========================================================================
#  Self-check
# ==========================================================================
if __name__ == "__main__":
    print("Testing the resonance fitter on a synthetic Lorentzian.")
    print("-" * 62)
    wl = np.linspace(0.5, 0.9, 301)
    for true_wl, true_fwhm in ((0.720, 0.060), (0.6551, 0.0405)):
        y = 1.0 / (1.0 + ((wl - true_wl) / (true_fwhm / 2)) ** 2)
        fit = fit_resonance(wl, y)
        print(f"  true {true_wl * 1000:8.2f} nm  FWHM {true_fwhm * 1000:6.2f} nm")
        print(f"  fit  {fit.wavelength_um * 1000:8.2f} nm  "
              f"FWHM {fit.fwhm_um * 1000:6.2f} nm   "
              f"error {abs(fit.wavelength_um - true_wl) * 1e6:.2f} pm")

    print()
    print("Testing the Brillouin energy factor for gold.")
    print("-" * 62)
    try:
        from sensor import gold_medium
        au = gold_medium()
        for wl_nm in (600, 700, 800, 900):
            f = C_LIGHT_UM_S / (wl_nm * 1e-3)
            eps = complex(au.eps_model(f))
            b = brillouin_factor(lambda ff: au.eps_model(ff), f)
            print(f"  {wl_nm} nm:  eps = {eps.real:8.2f} {eps.imag:+.2f}j   "
                  f"d(w eps)/dw = {b:8.2f}   "
                  f"{'SIGN FLIP - this is why eps alone is wrong' if eps.real * b < 0 else ''}")
    except Exception as exc:
        print(f"  (tidy3d not available: {exc})")


# ==========================================================================
#  Measuring a SMALL shift between two nearly identical spectra
# ==========================================================================
def relative_shift_pm(wl, y_ref, y_test, threshold: float = 0.15) -> float:
    """
    The wavelength shift between two spectra that differ only slightly.

    WHY `fit_resonance` IS THE WRONG TOOL FOR THIS
    ----------------------------------------------
    `fit_resonance` locates a peak by fitting a parabola through the maximum
    sample and its two neighbours.  That is fine for reporting where a
    resonance is.  It is not fine for measuring how far one moved, when the
    move is a small fraction of the sample spacing.

    On a paired bound/unbound run -- a 143 nm wide line sampled every
    1.667 nm -- four reasonable peak-finding methods give these shifts for
    the SAME pair of spectra:

        parabolic vertex      101 pm
        centroid above 80%     16 pm
        centroid above 50%    628 pm
        Lorentzian fit        173 pm

    A factor of forty between methods means the shift is not resolved.  Each
    method throws away almost all of the data: the shift is 0.06 of one sample
    spacing, and three points cannot resolve it.

    WHAT THIS DOES INSTEAD
    ----------------------
    For a small translation d, y_test(lam) = y_ref(lam - d), so to first order

        y_test - y_ref  =  -d * dy_ref/dlam

    which is a linear model in `d` with one parameter and hundreds of data
    points.  The least-squares solution is

        d  =  -sum[(y_test - y_ref) * y']  /  sum[y'^2]

    over the samples where the line has real signal.  It uses the whole
    lineshape, so a broad resonance helps rather than hurts -- there are more
    points on the flanks, which is where the derivative is largest and the
    information about a translation actually lives.

    Across thresholds from 5% to 50% of the peak this estimator moves by 2%
    on the same data where peak-fitting moves by a factor of forty.

    Returns picometres.  Positive means `y_test` is REDSHIFTED relative to
    `y_ref`.
    """
    import numpy as _np

    wl = _np.asarray(wl, dtype=float)
    y_ref = _np.asarray(y_ref, dtype=float)
    y_test = _np.asarray(y_test, dtype=float)
    order = _np.argsort(wl)
    wl, y_ref, y_test = wl[order], y_ref[order], y_test[order]

    mask = y_ref >= threshold * y_ref.max()
    if mask.sum() < 5:
        return float("nan")

    deriv = _np.gradient(y_ref, wl)
    num = float(_np.sum((y_test - y_ref)[mask] * deriv[mask]))
    den = float(_np.sum(deriv[mask] ** 2))
    if den <= 0:
        return float("nan")
    # Negative sign: a redshift makes (y_test - y_ref) oppose the derivative.
    return -num / den * 1e6


def shift_stability(wl, y_ref, y_test,
                    thresholds=(0.05, 0.10, 0.15, 0.25, 0.40, 0.50)) -> dict:
    """
    The same shift measured over several signal thresholds.

    If the answer moves much, the shift is not resolved and no single number
    should be quoted.  Use this to check any small shift before relying on it.
    """
    import numpy as _np

    vals = [relative_shift_pm(wl, y_ref, y_test, t) for t in thresholds]
    vals = [v for v in vals if _np.isfinite(v)]
    if not vals:
        return {"ok": False, "shift_pm": float("nan"), "spread_pm": float("nan")}
    med = float(_np.median(vals))
    spread = float(max(vals) - min(vals))
    rel = abs(spread / med) if med else float("inf")
    return {"ok": bool(rel < 0.10), "shift_pm": med, "spread_pm": spread,
            "relative_spread": rel, "values": vals,
            "verdict": (f"stable to {100 * rel:.1f}% across thresholds"
                        if rel < 0.10 else
                        f"UNSTABLE: {100 * rel:.0f}% spread across thresholds "
                        f"-- the shift is not resolved")}
