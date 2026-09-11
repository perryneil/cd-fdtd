"""
photon_budget.py — how many photons actually reach the antenna.
===============================================================

    python photon_budget.py        # offline, instant

WHAT IT DOES
------------
Estimates how many photons reach the antenna through the tissue above it, and
the resonance-fitting precision that photon count allows. If the tissue
extinguished the illumination, no near-field enhancement would matter, so this
bound is checked independently of the FDTD calculation.

A full Monte Carlo transport calculation is not needed for this. Monte Carlo
gives a spatially resolved fluence map in a heterogeneous geometry. What is
needed here is a scalar: the number of photons delivered to a point at depth z,
and the shot noise on that number. That has closed-form answers in two
regimes, and the two regimes bracket the truth.

    BALLISTIC (Beer-Lambert).  Photons that have not scattered at all:
        T_ball(z) = exp(-(mu_s + mu_a) z)
    This is a hard LOWER bound on delivered photons and the only component
    that arrives with its wavefront intact.

    DIFFUSE (diffusion approximation).  All photons, scattered or not, once
    many scattering events have occurred:
        phi(z) ~ exp(-z / delta),   delta = 1/sqrt(3 mu_a (mu_s' + mu_a))
    This is the practical upper bound, valid for z >> transport mean free
    path, and it is enormously larger than ballistic in forward-scattering
    tissue.

WHY THE SEPARATION IS CLEAN -- AND WHERE IT IS NOT
--------------------------------------------------
`simulation.build_source` argues the transport and near-field problems
separate: transport sets how many photons arrive and with what angular
spread, the FDTD problem sets what one photon does once it reaches the
antenna.  That is right as far as it goes, and it is why a scalar budget
suffices.

The part that does NOT separate, and which this module reports: a bowtie is
driven by the field component along its axis, and diffuse
light arrives from all directions.  A fully diffuse illumination drives the
antenna less efficiently than a collimated beam aligned with it, by a factor
that depends on the angular distribution.  The ratio is reported as
`polarisation_efficiency` with its assumption stated, and it multiplies the
final signal.  Together with `anisotropy.py`, this shows how much the answer
depends on orientation.

Everything here is offline, free, and instant.
"""

from __future__ import annotations

import numpy as np

from config import StudyConfig

H_PLANCK = 6.62607015e-34      # J s
C_LIGHT_M_PER_S = 2.99792458e8


# ==========================================================================
#  Transport lengths
# ==========================================================================
def transport_lengths(cfg: StudyConfig,
                      mu_s_per_um: float | None = None,
                      g: float | None = None,
                      mu_a_per_um: float | None = None) -> dict:
    """
    The four lengths that decide everything about light delivery.

        scattering mfp     1 / mu_s            how far before it scatters
        transport mfp      1 / (mu_s' + mu_a)  how far before it forgets
                                               which way it was going
        absorption length  1 / mu_a            how far before it is gone
        penetration depth  1/sqrt(3 mu_a (mu_s' + mu_a))
                                               the diffusion e-folding length

    Defaults come from the calibration targets, not from a realisation, so
    this can be run before any tissue is generated.
    """
    mu_s = mu_s_per_um if mu_s_per_um is not None \
        else cfg.calibration.target_mu_s_per_um
    gg = g if g is not None else cfg.calibration.target_g
    mu_a = mu_a_per_um if mu_a_per_um is not None else cfg.tissue.mu_a_per_um

    mu_s_prime = mu_s * (1.0 - gg)
    mu_t = mu_s + mu_a
    mu_eff = np.sqrt(3.0 * mu_a * (mu_s_prime + mu_a))

    return {
        "mu_s_per_um": mu_s,
        "g": gg,
        "mu_a_per_um": mu_a,
        "mu_s_prime_per_um": mu_s_prime,
        "scattering_mfp_um": 1.0 / mu_s if mu_s > 0 else float("inf"),
        "transport_mfp_um": (1.0 / (mu_s_prime + mu_a)
                             if (mu_s_prime + mu_a) > 0 else float("inf")),
        "absorption_length_um": 1.0 / mu_a if mu_a > 0 else float("inf"),
        "penetration_depth_um": 1.0 / mu_eff if mu_eff > 0 else float("inf"),
        "mu_eff_per_um": float(mu_eff),
        "albedo": mu_s / mu_t if mu_t > 0 else float("nan"),
    }


def delivered_fraction(depth_um: float, lengths: dict) -> dict:
    """
    The fraction of incident photons reaching `depth_um`, both ways.

    `ballistic` is exact and pessimistic.  `diffuse` uses the diffusion
    e-folding and is optimistic; it is also only meaningful once the depth
    exceeds a transport mean free path, and `diffusion_valid` says whether it
    does.  Both are reported; the gap between them is the uncertainty on the
    photon budget, which a Monte Carlo run would narrow.
    """
    z = float(depth_um)
    mu_t = lengths["mu_s_per_um"] + lengths["mu_a_per_um"]
    ballistic = float(np.exp(-mu_t * z))
    diffuse = float(np.exp(-z / lengths["penetration_depth_um"]))
    return {
        "depth_um": z,
        "ballistic": ballistic,
        "diffuse": diffuse,
        "ratio": diffuse / ballistic if ballistic > 0 else float("inf"),
        "diffusion_valid": bool(z > lengths["transport_mfp_um"]),
        "transport_mfp_um": lengths["transport_mfp_um"],
    }


# ==========================================================================
#  Counting photons
# ==========================================================================
def photon_budget(
    cfg: StudyConfig,
    depth_um: float = 100.0,
    power_mw: float = 1.0,
    collection_efficiency: float = 0.01,
    polarisation_efficiency: float = 0.5,
    wavelength_um: float | None = None,
    integration_time_s: float | None = None,
) -> dict:
    """
    Photons per measurement, and the resonance-fitting precision they allow.

    THE CHAIN
    ---------
        incident photons  =  P * t / (h c / lambda)
        x  delivered fraction at depth        (transport)
        x  collection efficiency              (optics, stated not derived)
        x  polarisation efficiency            (see module docstring)
        =  detected photons N

    A Lorentzian resonance of full width `FWHM` fitted from N photons can be
    located to about

        sigma_lambda  ~  FWHM / sqrt(N)

    which is the standard shot-noise-limited centroid precision.  That number
    is the thing to compare against the resonance shift the sensor is trying
    to detect.  If sigma_lambda exceeds the shift, the device is
    photon-starved and the answer is more light or more time, not a better
    antenna.

    Parameters
    ----------
    collection_efficiency : float
        Fraction of photons leaving the antenna region that reach the
        detector.  A stated instrument parameter, not a derived one; 1% is a
        plausible fibre-coupled value and the number is reported so a reader
        can substitute their own.
    polarisation_efficiency : float
        Fraction of the arriving field that drives the bowtie's long axis.
        0.5 is the unpolarised/diffuse limit; 1.0 is a collimated beam
        aligned with the antenna.  This is a real factor, not a fudge, and it
        is the reason the diffuse and collimated cases differ even at equal
        photon count.
    """
    lengths = transport_lengths(cfg)
    wl = wavelength_um if wavelength_um is not None \
        else cfg.calibration.calib_wavelength_um
    t_int = integration_time_s if integration_time_s is not None \
        else cfg.regulatory.measurement_time_s

    e_photon = H_PLANCK * C_LIGHT_M_PER_S / (wl * 1e-6)     # joules
    incident = (power_mw * 1e-3) * t_int / e_photon

    dv = delivered_fraction(depth_um, lengths)

    out = {"lengths": lengths, "delivery": dv, "incident_photons": incident,
           "photon_energy_J": e_photon, "wavelength_um": wl,
           "integration_time_s": t_int, "power_mw": power_mw,
           "collection_efficiency": collection_efficiency,
           "polarisation_efficiency": polarisation_efficiency}

    for mode in ("ballistic", "diffuse"):
        n = (incident * dv[mode] * collection_efficiency
             * polarisation_efficiency)
        out[mode] = {
            "detected_photons": n,
            "shot_noise_relative": 1.0 / np.sqrt(n) if n > 0 else float("inf"),
        }
    return out


def resonance_precision(budget: dict, fwhm_nm: float,
                        mode: str = "diffuse") -> dict:
    """
    Shot-noise-limited precision on the resonance CENTROID, in picometres.

    Compare this against the shift the sensor must detect.  It is the
    instrument-side floor, independent of the FDTD calculation, so it bounds
    the answer from a direction the simulation cannot.
    """
    n = budget[mode]["detected_photons"]
    if n <= 0:
        return {"sigma_pm": float("inf"), "detected_photons": 0.0,
                "mode": mode}
    sigma_nm = fwhm_nm / np.sqrt(n)
    return {
        "mode": mode,
        "detected_photons": n,
        "fwhm_nm": fwhm_nm,
        "sigma_pm": sigma_nm * 1000.0,
    }


def required_photons(fwhm_nm: float, target_shift_pm: float) -> float:
    """
    Inverting the same relation: how many photons are needed to resolve a
    given shift at unit signal-to-noise?

        N = (FWHM / shift)^2

    Useful as a design target that does not depend on any of the modelling
    assumptions -- it is pure counting statistics.
    """
    if target_shift_pm <= 0:
        return float("inf")
    return float((fwhm_nm * 1000.0 / target_shift_pm) ** 2)


def starvation_depth(cfg: StudyConfig, fwhm_nm: float,
                     target_shift_pm: float,
                     z_max_um: float = 5.0e4, n: int = 4000) -> dict:
    """
    The depth at which the photon budget stops being sufficient.

    A single scalar from the same closed forms, and the practical design
    number: it tells you whether the sensor can be placed where it needs to
    be.
    """
    need = required_photons(fwhm_nm, target_shift_pm)
    lengths = transport_lengths(cfg)
    b0 = photon_budget(cfg, depth_um=0.0)
    n0 = (b0["incident_photons"] * b0["collection_efficiency"]
          * b0["polarisation_efficiency"])

    if n0 <= need:
        return {"ballistic_depth_um": 0.0, "diffuse_depth_um": 0.0,
                "required_photons": need,
                "note": "Photon-starved even at zero depth."}

    mu_t = lengths["mu_s_per_um"] + lengths["mu_a_per_um"]
    z_ball = float(np.log(n0 / need) / mu_t) if mu_t > 0 else float("inf")
    z_diff = float(np.log(n0 / need) * lengths["penetration_depth_um"])
    return {
        "ballistic_depth_um": min(z_ball, z_max_um),
        "diffuse_depth_um": min(z_diff, z_max_um),
        "required_photons": need,
        "photons_at_zero_depth": n0,
    }


# ==========================================================================
#  Reporting
# ==========================================================================
def print_photon_budget(cfg: StudyConfig, depth_um: float = 100.0,
                        fwhm_nm: float = 40.0,
                        target_shift_pm: float = 35.0) -> dict:
    """Print the analytic photon budget and the depth at which it runs out."""
    b = photon_budget(cfg, depth_um=depth_um)
    L = b["lengths"]
    d = b["delivery"]

    print("  Transport lengths at the calibration wavelength:")
    print(f"    scattering mean free path   {L['scattering_mfp_um']:10.1f} um")
    print(f"    transport mean free path    {L['transport_mfp_um']:10.1f} um")
    print(f"    absorption length           {L['absorption_length_um']:10.1f} um")
    print(f"    diffusion penetration depth {L['penetration_depth_um']:10.1f} um")
    print(f"    single-scattering albedo    {L['albedo']:10.4f}")
    print()
    print(f"  Delivery to a depth of {d['depth_um']:.0f} um:")
    print(f"    ballistic (unscattered)     {d['ballistic']:10.3e}")
    print(f"    diffuse (all photons)       {d['diffuse']:10.3e}")
    print(f"    ratio                       {d['ratio']:10.3e}")
    if not d["diffusion_valid"]:
        print(f"    NOTE: the depth is under one transport mean free path "
              f"({d['transport_mfp_um']:.1f} um),")
        print("    so the diffusion figure is not yet valid there. In that")
        print("    regime the ballistic number is the physical one.")
    print()
    print(f"  Photon budget: {b['power_mw']:.1f} mW for "
          f"{b['integration_time_s']:.0f} s at "
          f"{b['wavelength_um'] * 1000:.0f} nm,")
    print(f"  collection {100 * b['collection_efficiency']:.1f}%, "
          f"polarisation {100 * b['polarisation_efficiency']:.0f}%:")
    print(f"    incident                    {b['incident_photons']:10.3e}")
    print(f"    detected (ballistic)        "
          f"{b['ballistic']['detected_photons']:10.3e}")
    print(f"    detected (diffuse)          "
          f"{b['diffuse']['detected_photons']:10.3e}")
    print()

    need = required_photons(fwhm_nm, target_shift_pm)
    print(f"  To resolve a {target_shift_pm:.0f} pm shift on a "
          f"{fwhm_nm:.0f} nm linewidth at SNR 1 needs")
    print(f"  {need:.3e} detected photons (pure counting statistics -- no")
    print("  modelling assumption enters this number at all).")
    print()
    for mode in ("ballistic", "diffuse"):
        p = resonance_precision(b, fwhm_nm, mode)
        ok = p["sigma_pm"] <= target_shift_pm
        margin = target_shift_pm / p["sigma_pm"] if p["sigma_pm"] > 0 else 0.0
        print(f"    {mode:>10}: centroid precision {p['sigma_pm']:9.3e} pm  "
              f"{'OK, ' + format(margin, '.1e') + 'x margin' if ok else '<-- PHOTON-STARVED'}")
    print()

    # The design number: at what depth does the budget actually run out?
    zc = starvation_depth(cfg, fwhm_nm, target_shift_pm)
    print(f"  The budget runs out at a depth of "
          f"{zc['ballistic_depth_um']:.0f} um (ballistic) / "
          f"{zc['diffuse_depth_um']:.0f} um (diffuse).")
    print("  Shallower than that, there are more than enough photons and the")
    print("  transport problem is genuinely irrelevant to the sensitivity")
    print("  claim. Deeper than that, the measurement is photon-starved and")
    print("  no antenna design rescues it.")
    print()
    print("  If both rows are photon-starved, the fix is more light or longer")
    print("  integration, not a better antenna. This bound is independent of")
    print("  the FDTD calculation.")
    print()
    print("  WHY NO MONTE CARLO. The two rows above bracket the truth. Monte")
    print("  Carlo would narrow the bracket; it would not change which side")
    print("  of the threshold the answer falls on unless the bracket spans")
    print("  it.")
    return b


if __name__ == "__main__":
    print_photon_budget(StudyConfig())
