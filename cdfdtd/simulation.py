"""
simulation.py — putting the sensor, the tissue and the light into one box.
==========================================================================

WHAT FDTD IS, IN ONE PARAGRAPH
------------------------------
Finite-Difference Time-Domain is the most literal possible way to solve
Maxwell's equations.  Chop space into a grid of tiny cubes.  Chop time into
tiny ticks.  Store the electric field on the cube edges and the magnetic field
on the cube faces.  Then repeatedly apply the two rules that Maxwell wrote
down: a changing magnetic field creates a curling electric field, and vice
versa.  Run it forward and you have simulated light.  There is no
approximation in the method itself, only in how finely you chop.

THE CENTRAL DIFFICULTY OF THIS PARTICULAR PROBLEM
-------------------------------------------------
The bowtie gap needs cubes about half a nanometre across.  The tissue region
needs to be several microns across.  Filling several microns with half-
nanometre cubes would need something like 10^12 cubes, which no computer has.

So the grid must be *non-uniform*: fine where the physics is fine, coarse
where it is smooth.

Solvers with a single uniform Cartesian grid (Meep, for example) cannot do
this.  Tidy3D can: `GridSpec.auto` accepts mesh-override structures that force
a specified step size inside a chosen box.  This module uses them.

WHAT THIS MODULE BUILDS
-----------------------
Three kinds of simulation, all from the same pieces:

  `build_sensor_simulation`   the main event: bowtie + coatings + tissue (or
                              water), illuminated, with monitors that record
                              the spectrum, the near field, and the local
                              permittivity.

  `build_slab_simulation`     a sensor-free slab of tissue, used to spot-check
                              the analytic scattering calibration in
                              `optical_properties.py`.

  `estimate_cost`             asks Tidy3D what a simulation would cost before
                              you spend anything on it.

NOTHING HERE TALKS TO THE CLOUD unless you explicitly ask it to.  Building and
validating a simulation is free and happens on your machine; only `run` and
`estimate_cost` reach out.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np

from config import StudyConfig, NM, C_LIGHT_UM_S
from sensor import build_bowtie, build_surface_stack
from tissue import TissueRealization, to_tidy3d_medium, water_medium


# ==========================================================================
#  Monitor names — one place, so post-processing never guesses
# ==========================================================================
FLUX_MONITOR = "scattered_flux"
FIELD_MONITOR = "near_field"
EPS_MONITOR = "permittivity"
FIELD_TIME_MONITOR = "decay_probe"
HOTSPOT_MONITOR = "hotspot_probe"


# ==========================================================================
#  Grid
# ==========================================================================
def build_grid_spec(cfg: StudyConfig):
    """
    A non-uniform mesh: very fine in the gap, fine in the metal, coarse in the
    tissue.

    HOW TIDY3D DOES THIS
    --------------------
    `GridSpec.auto` normally chooses cell sizes from the local refractive index
    and the wavelength — high index means shorter wavelength means smaller
    cells.  That is sensible everywhere except at a plasmonic hot spot, where
    the field varies far faster than the wavelength suggests.

    `MeshOverrideStructure` lets you override that in a chosen box: "inside
    this region, use cells of exactly this size".  It is not a material and
    does not appear in the physics; it only instructs the mesher.

    THE THREE REGIONS
    -----------------
      gap box     dl_gap_nm (0.5 nm by default).  Expensive, unavoidable.
      metal box   dl_metal_nm (2 nm).  Fields inside gold decay over a ~25 nm
                  skin depth, which needs resolving but not as finely as the
                  gap.
      everything  min_steps_per_wavelength, chosen by Tidy3D from the local
      else        index.

    A word of warning: `dl_min` is left at 0 so Tidy3D does not silently clamp
    your gap resolution.  That means a careless override can produce an
    enormous simulation.  Always run `estimate_cost` first.
    """
    import tidy3d as td

    b = cfg.bowtie
    gap = b.gap_nm * NM
    thick = b.thickness_nm * NM
    side = b.side_nm * NM
    coat = (cfg.surface.recognition_thickness_nm
            + cfg.surface.fouling_thickness_nm) * NM

    # THE GAP THE MESH MUST RESOLVE IS THE PHYSICAL ONE.
    #
    # `gap_nm` positions the SHARP apices; the tip fillet then pulls each
    # rounded tip back by r/sin(apex/2) - r, so the physical tip-to-tip
    # distance is gap_nm + 2 * setback.  At the as-built 5 nm fillet and 60
    # degree apex that is gap_nm + 10 nm.  A geometry with gap_nm = 0 still
    # has a physical gap, so the override is keyed on the physical gap, which
    # is strictly positive whenever the fillet is.  Keying it on gap_nm would
    # leave that gap on the coarse bulk mesh.
    half_apex = np.radians(b.apex_angle_deg) / 2.0
    setback = b.tip_radius_nm * NM * (1.0 / np.sin(half_apex) - 1.0)
    physical_gap = gap + 2.0 * setback

    # Fine box around the gap, generous enough to include the coatings and a
    # little of the surrounding medium where the field is still strong.
    gap_box = td.Box(
        center=(0, 0, 0),
        size=(physical_gap + 6 * coat + 20 * NM,
              side * 0.8,
              thick + 6 * coat + 20 * NM),
    )
    metal_box = td.Box(
        center=(0, 0, 0),
        size=(2 * side + gap, 1.4 * side, thick + 4 * coat),
    )

    dl_gap = cfg.fdtd.dl_gap_nm * NM
    dl_metal = cfg.fdtd.dl_metal_nm * NM

    # A DOMAIN-WIDE override fixing the bulk cell size.
    #
    # WHY THIS IS NOT OPTIONAL, AND WHAT GOES WRONG WITHOUT IT
    # --------------------------------------------------------
    # `GridSpec.auto` picks the bulk cell from the LOCAL REFRACTIVE INDEX:
    # higher index means shorter wavelength in the medium means smaller cells.
    # The aqueous reference sits in water (n = 1.333); the tissue runs sit in a
    # custom medium whose index reaches `n_clip_max`.  So the mesher gives the
    # two runs DIFFERENT GRIDS -- for example 335 x 318 x 201 for water against
    # 335 x 318 x 203 for tissue.
    #
    # That would silently break the comparison between runs.
    # `observables.transfer_factor` divides a tissue figure of merit by its
    # aqueous counterpart and states, correctly, that the two runs must share
    # a mesh or "the ratio picks up numerical differences and means nothing".
    # A 1% difference in z sampling can move a resonance by more than the
    # 35 pm speckle residual the study is trying to resolve, and it would move
    # it in a way that looks like tissue physics.
    #
    # Fixing the bulk cell explicitly removes the medium's influence on the
    # grid, so every run in the campaign is on the same mesh by construction
    # rather than by luck.  It also makes the bulk cell a stated number that
    # `check_mesh_vs_cutoff` can test against `TissueConfig.l_min_um`, instead
    # of something the mesher chose implicitly.
    bulk_dl = bulk_cell_um(cfg)
    domain = cfg.fdtd.domain_um
    bulk_box = td.Box(center=(0, 0, 0),
                      size=(domain[0] * 2, domain[1] * 2, domain[2] * 2))

    overrides = [
        td.MeshOverrideStructure(
            geometry=bulk_box, dl=(bulk_dl,) * 3, name="mesh_bulk",
        ),
        td.MeshOverrideStructure(
            geometry=metal_box, dl=(dl_metal,) * 3, name="mesh_metal",
        ),
        td.MeshOverrideStructure(
            geometry=gap_box, dl=(dl_gap,) * 3, name="mesh_gap",
        ),
    ]

    return td.GridSpec.auto(
        wavelength=float(np.mean(cfg.calibration.band_um)),
        min_steps_per_wvl=cfg.fdtd.min_steps_per_wavelength,
        override_structures=overrides,
        dl_min=0.0,
    )


def bulk_cell_um(cfg: StudyConfig) -> float:
    """
    The bulk cell size, fixed explicitly rather than left to the mesher.

    Derived the way `GridSpec.auto` already derived it, but from a FIXED index
    rather than the local one, so that water runs and tissue runs land on the
    same grid:

        dl = lambda_mean / (n_max * min_steps_per_wavelength)

    Two deliberate choices:

    `n_max` is `TissueConfig.n_clip_max`, the ceiling the tissue generator
    clips to -- not the mean index and not water's.  It is the only value that
    is identical for every run in the campaign AND fine enough for all of
    them.  Being slightly conservative in the bulk is cheap, because the bulk
    is not where the cells are.

    `lambda_mean` rather than `lambda_min` reproduces Tidy3D's own convention
    and therefore leaves the cost unchanged from the auto grid.  The
    consequence: at the BLUE end of the band the medium wavelength is shorter
    than the mean by band_mean/band_lo, so the effective steps-per-wavelength
    there is that factor lower than `min_steps_per_wavelength` nominally
    promises.  If a result depends on the short-wavelength end, raise
    `min_steps_per_wavelength` and pay for it rather than assuming the nominal
    figure applies across the whole band.
    """
    lam = float(np.mean(cfg.calibration.band_um))
    n_max = max(cfg.tissue.n_clip_max, cfg.tissue.n0, 1.333)
    return float(lam / (n_max * cfg.fdtd.min_steps_per_wavelength))


def grids_match(sim_a, sim_b, tol: float = 1e-9) -> dict:
    """
    Are two simulations actually on the same mesh?

    `transfer_factor` and every paired comparison in this study assume they
    are, so this check makes the assumption explicit.
    """
    import numpy as _np

    out = {"shape_a": None, "shape_b": None, "same_shape": False,
           "max_boundary_diff_um": float("inf"), "ok": False}
    try:
        ba = [_np.asarray(sim_a.grid.boundaries.to_list[i]) for i in range(3)]
        bb = [_np.asarray(sim_b.grid.boundaries.to_list[i]) for i in range(3)]
    except Exception as exc:                       # pragma: no cover
        out["error"] = str(exc)
        return out

    out["shape_a"] = tuple(len(v) - 1 for v in ba)
    out["shape_b"] = tuple(len(v) - 1 for v in bb)
    out["same_shape"] = out["shape_a"] == out["shape_b"]
    if out["same_shape"]:
        out["max_boundary_diff_um"] = float(max(
            _np.abs(a - b).max() for a, b in zip(ba, bb)
        ))
        out["ok"] = out["max_boundary_diff_um"] <= tol
    out["verdict"] = (
        "OK - both runs are on an identical mesh, so a ratio between them "
        "reflects the medium and not the grid."
        if out["ok"] else
        f"MESH MISMATCH: {out['shape_a']} against {out['shape_b']}. Any "
        f"transfer factor or paired difference computed from these two runs "
        f"mixes a physics change with a grid change, and the grid change can "
        f"easily be larger. Fix the mesh before quoting a ratio."
    )
    return out


# ==========================================================================
#  Boundaries
# ==========================================================================
def build_boundary_spec(cfg: StudyConfig):
    """
    How the simulation box ends.

    PLAIN LANGUAGE
    --------------
    A simulation box has walls, but the physical problem does not.  We want
    light that reaches the edge to leave and never come back, as if the box
    continued forever.  The standard trick is an absorbing layer: a region
    where the equations are modified so waves are swallowed without reflecting.

    TWO FLAVOURS
    ------------
    A *Perfectly Matched Layer* (PML) is mathematically reflectionless, thin
    and efficient, but its derivation assumes the medium does not change along
    the direction being absorbed.  A random tissue would violate that at every
    point, which can cause spurious reflections or slow field growth.

    An *Absorber* is a plain lossy layer that does not depend on the medium,
    but it reflects more.

    `tissue.generate_tissue` tapers the index fluctuations smoothly to the
    uniform background before they reach the boundary, so either boundary sees
    a uniform medium.  With that taper in place, `boundary_test.py` shows that
    PML converges under a depth sweep while the absorber does not (the
    linewidth changes with domain depth).  ALL VALUES IN THE PAPER USE PML.
    `config.FDTDConfig.boundary` defaults to "absorber" only so that the
    reference absorber ensemble in results/ can be reproduced; pass `--boundary pml` for
    new runs.
    """
    import tidy3d as td

    if cfg.fdtd.boundary == "pml":
        return td.BoundarySpec.all_sides(boundary=td.PML())
    return td.BoundarySpec.all_sides(
        boundary=td.Absorber(num_layers=cfg.fdtd.absorber_layers)
    )


# ==========================================================================
#  Source
# ==========================================================================
def build_source(cfg: StudyConfig, pol_along_axis: bool = True):
    """
    The light that illuminates the sensor.

    POLARISATION MATTERS ENORMOUSLY FOR A BOWTIE
    --------------------------------------------
    A bowtie only produces its hot spot when the electric field points ALONG
    the bowtie axis, driving charge towards the tips.  Polarised across the
    axis the same structure is nearly inert.  `pol_along_axis=True` gives the
    useful case; running the other one is a good check that the resonance you
    found is the mode you think it is.

    TWO SOURCE TYPES, AND WHY THE DEFAULT IS THE SIMPLER ONE
    ---------------------------------------------------------
    A *plane wave* is exactly what it sounds like: uniform illumination
    crossing the whole box.  Simple, robust, indifferent to how the grid is
    laid out.

    A *total-field / scattered-field* (TFSF) source injects a plane wave only
    inside a chosen box, so that outside it you see purely scattered light.
    That separation is what gives a clean scattering cross-section without a
    separate reference run.  But TFSF assumes a uniform grid in the two
    directions transverse to its injection axis — and our mesh overrides
    deliberately make the grid strongly non-uniform there, to resolve the gap.
    The mismatch leaks incident field into the scattered region.

    Since the observables here are the resonance wavelength and its linewidth,
    both of which come straight out of the hot-spot probe, the plane wave is
    the better default.  Switch `cfg.fdtd.source` to "tfsf" only if you need a
    cross-section AND you have turned the mesh overrides off.

    A NOTE ON THE ILLUMINATION ANGLE
    --------------------------------
    One could take the angular spectrum from a Monte Carlo transport
    calculation and build an ensemble of sources from it.  That matters for
    the photon budget, but the perturbative sensitivity map S(r) is a property
    of the resonant *mode* and is largely insensitive to how the mode is
    excited.  The angular spectrum sets how much signal you get, not where the
    sensor is sensitive, which is why a single plane wave is used here.
    """
    import tidy3d as td

    lo, hi = cfg.calibration.band_um
    f_lo, f_hi = C_LIGHT_UM_S / hi, C_LIGHT_UM_S / lo
    f0 = 0.5 * (f_lo + f_hi)
    fwidth = (f_hi - f_lo) / 2.0
    pulse = td.GaussianPulse(freq0=f0, fwidth=fwidth)
    pol = 0.0 if pol_along_axis else np.pi / 2

    dx, dy, dz = cfg.fdtd.domain_um

    # --- how far the source must sit from the absorbing boundary ---------
    # The source offset must NOT be tied to `taper_width_um` (a tissue
    # setting). Too little clearance, for example 400 nm at 900 nm in fused
    # silica (0.65 of a wavelength in the medium), couples the source to the
    # boundary and the spectrum shows Fabry-Perot fringes.
    #
    # A source needs clearance measured in WAVELENGTHS IN THE MEDIUM IT SITS
    # IN, not in nanometres, and the binding case is the reddest wavelength in
    # the band.  Three quarters of a wavelength is the usual working minimum;
    # below about half, the source couples to the boundary.
    #
    # Note this scales with the band: widening to the near infrared makes the
    # requirement larger, so a fixed offset in nanometres is not enough.
    n_src = cfg.bowtie.substrate_index or cfg.tissue.n0
    lam_max_in_medium = hi / n_src
    clearance = max(cfg.fdtd.taper_width_um, 0.75 * lam_max_in_medium)

    # Never push the source past the middle of the box; if the domain is too
    # short to give the clearance, take half the half-height and let
    # `source_clearance_check` say so rather than silently placing it badly.
    clearance = min(clearance, 0.45 * dz)
    inset = 2 * clearance

    if cfg.fdtd.source == "tfsf":
        return td.TFSF(
            center=(0, 0, 0),
            size=(dx - inset, dy - inset, dz - inset),
            source_time=pulse,
            direction="+",
            injection_axis=2,
            pol_angle=pol,
            name="tfsf",
        )

    return td.PlaneWave(
        center=(0, 0, -dz / 2 + clearance),
        size=(td.inf, td.inf, 0),
        source_time=pulse,
        direction="+",
        pol_angle=pol,
        name="plane_wave",
    )


def metal_present_check(cfg: StudyConfig, sim=None) -> dict:
    """
    Is there actually any gold in this simulation?

    WHY THIS CHECK IS NEEDED
    ------------------------
    The recognition and antifouling shells are enlarged copies of the bowtie
    polygons.  If they were listed after the metal, Tidy3D's later-wins
    overlap rule would make them overwrite it, and the simulation would model
    dielectric triangles in water.  Nothing downstream would complain: the
    solver runs, a spectrum comes back, and a fitter finds a peak that is only
    a Fabry-Perot fringe.  This check makes that impossible to miss.

    The test samples the permittivity along the bowtie axis at the calibration
    wavelength and asks whether Re(eps) goes negative anywhere -- the defining
    signature of a metal, and something no dielectric in this model can fake.
    """
    import numpy as _np

    sim = sim if sim is not None else build_sensor_simulation(cfg, None)
    freq = C_LIGHT_UM_S / cfg.calibration.calib_wavelength_um

    b = cfg.bowtie
    half_apex = _np.radians(b.apex_angle_deg) / 2.0
    height = (b.side_nm * NM) / (2.0 * _np.tan(half_apex))
    # A line through the metal: from just outside one triangle's base, along
    # the axis, to the other.  Must cross gold twice if gold exists.
    x_end = (b.gap_nm * NM) / 2.0 + height
    xs = _np.linspace(-x_end * 0.95, x_end * 0.95, 241)

    try:
        import tidy3d as td
        eps = sim.epsilon(
            box=td.Box(center=(0, 0, 0),
                       size=(2 * x_end * 0.95, 0, 0)),
            coord_key="centers", freq=freq,
        )
        vals = _np.asarray(eps).ravel()
    except Exception as exc:                       # pragma: no cover
        return {"ok": False, "error": str(exc),
                "verdict": f"could not sample the permittivity: {exc}"}

    re = _np.real(vals)
    n_metal = int((re < -1.0).sum())
    frac = n_metal / max(len(re), 1)

    ok = n_metal > 0
    if ok:
        verdict = (
            f"OK - gold is present: {100 * frac:.0f}% of samples along the "
            f"bowtie axis have Re(eps) < -1 (minimum {re.min():.1f} at "
            f"{cfg.calibration.calib_wavelength_um * 1000:.0f} nm)."
        )
    else:
        verdict = (
            f"NO METAL IN THE SIMULATION. Re(eps) along the bowtie axis spans "
            f"{re.min():.2f} to {re.max():.2f} -- all dielectric. The most "
            f"likely cause is STRUCTURE ORDER: the recognition and "
            f"antifouling shells are enlarged copies of the bowtie polygons, "
            f"so if they are listed AFTER the gold they overwrite it "
            f"entirely. The metal must come last. Nothing downstream of this "
            f"is a plasmonic result."
        )
    return {"ok": bool(ok), "fraction_metal": frac,
            "eps_min": float(re.min()), "eps_max": float(re.max()),
            "n_samples": int(len(re)), "verdict": verdict}


def source_clearance_check(cfg: StudyConfig) -> dict:
    """
    Is the source far enough from the absorbing boundary?

    Measured in wavelengths in the medium the source sits in, at the REDDEST
    wavelength in the band -- the binding case, and the one a fixed offset in
    nanometres always gets wrong when the band moves.

    Below about half a wavelength the source couples to the absorber and the
    recorded spectrum picks up Fabry-Perot fringes that look like structure.
    """
    lo, hi = cfg.calibration.band_um
    dx, dy, dz = cfg.fdtd.domain_um
    n_src = cfg.bowtie.substrate_index or cfg.tissue.n0
    lam_max = hi / n_src
    want = 0.75 * lam_max
    have = min(max(cfg.fdtd.taper_width_um, want), 0.45 * dz)
    ratio = have / lam_max

    if ratio >= 0.7:
        verdict = (f"OK - the source sits {have * 1000:.0f} nm from the "
                   f"boundary, {ratio:.2f} wavelengths at the reddest point "
                   f"in the band.")
    elif ratio >= 0.5:
        verdict = (f"MARGINAL - {have * 1000:.0f} nm is only {ratio:.2f} "
                   f"wavelengths. Expect some boundary coupling. Grow "
                   f"FDTDConfig.domain_um in z.")
    else:
        verdict = (f"TOO CLOSE - {have * 1000:.0f} nm is {ratio:.2f} "
                   f"wavelengths at {hi * 1000:.0f} nm in a medium of index "
                   f"{n_src:.2f}. The source will couple to the absorber and "
                   f"the spectrum will carry fringes. The domain is too short "
                   f"in z for this band: it needs at least "
                   f"{(want / 0.45) * 1000:.0f} nm.")

    return {"clearance_um": have, "wanted_um": want,
            "wavelengths": ratio, "lam_max_in_medium_um": lam_max,
            "domain_z_um": dz, "ok": bool(ratio >= 0.7),
            "min_domain_z_um": want / 0.45, "verdict": verdict}


# ==========================================================================
#  Monitors
# ==========================================================================
def build_monitors(cfg: StudyConfig, light: bool = False):
    """
    What we record.

    FOUR MONITORS, FOUR JOBS
    ------------------------
    1. FLUX — the spectrum.  A closed box around the bowtie; the net power
       flowing out of it, wavelength by wavelength.  The peak of this curve is
       the resonance, and its width is the linewidth.  Both are needed: the
       figure of merit is the shift divided by the width, and a sensor with a
       big shift and a huge width is worse than one with a modest shift and a
       narrow line.

    2. NEAR FIELD — the electric field on a fine 3-D grid around the gap, at
       the resonance wavelength.  This is what `observables.py` turns into the
       sensitivity map.

    3. PERMITTIVITY — what material is at each grid point, recorded on the same
       grid as the near field.  Needed for two things: the energy-density
       normalisation of the sensitivity, and identifying the metal so the
       steric mask knows what to dilate.  Recording it rather than re-deriving
       it guarantees it matches what the solver actually meshed.

    4. FIELD-TIME PROBE — the field at one point as a function of time.  This
       is the decay check.  If the run
       stops while the field is still ringing, the resonance appears
       artificially broad, and you would report a linewidth that is really a
       statement about your run time.  `check_decay` inspects this.
    """
    import tidy3d as td

    freqs = list(cfg.freqs_hz)
    b = cfg.bowtie
    side = b.side_nm * NM
    thick = b.thickness_nm * NM
    gap = b.gap_nm * NM
    coat = (cfg.surface.recognition_thickness_nm
            + cfg.surface.fouling_thickness_nm) * NM

    # --- 0. the hot-spot probe: the cheapest and most important monitor -
    # A single point at the centre of the gap, recorded at EVERY wavelength.
    # The peak of |E|^2 against wavelength is the resonance; the width of that
    # peak is the linewidth.  Both headline numbers come from this one tiny
    # monitor, which stores a few kilobytes.
    hotspot = td.FieldMonitor(
        center=(0, 0, 0), size=(0, 0, 0), freqs=freqs,
        name=HOTSPOT_MONITOR, colocate=True,
    )

    # --- 1. spectrum ---------------------------------------------------
    flux_box_size = (2.2 * side + gap, 1.8 * side, thick + 12 * coat + 60 * NM)
    flux = td.FluxMonitor(
        center=(0, 0, 0), size=flux_box_size, freqs=freqs[::10],
        name=FLUX_MONITOR,
    )

    # --- 2 & 3. near field and permittivity, on the same grid -----------
    # Big enough to contain the accessible hot spot and the field decay away
    # from it; small enough that the stored array stays manageable.
    # Only a handful of wavelengths: the sensitivity map is evaluated AT the
    # resonance, and a 3-D field array at 300 wavelengths would be tens of
    # gigabytes per run - times the number of seeds in the ensemble.
    nf_size = (
        gap + 8 * coat + 40 * NM,
        min(side, 80 * NM),
        thick + 8 * coat + 40 * NM,
    )
    n_nf = max(1, len(freqs) // 5)
    field = td.FieldMonitor(
        center=(0, 0, 0), size=nf_size, freqs=freqs[::n_nf],
        name=FIELD_MONITOR, colocate=True,
    )
    eps = td.PermittivityMonitor(
        center=(0, 0, 0), size=nf_size,
        freqs=[freqs[len(freqs) // 2]], name=EPS_MONITOR,
    )

    # --- 4. decay probe -------------------------------------------------
    probe = td.FieldTimeMonitor(
        center=(0, 0, 0), size=(0, 0, 0),
        fields=["Ex"], name=FIELD_TIME_MONITOR, interval=20,
    )

    if light:
        # ENSEMBLE RUNS DO NOT NEED THE FIELD MAPS.
        #
        # The ensemble measures one number per seed -- the resonance shift --
        # plus the local mean index used as the regressor.  Those come from
        # the hot-spot probe and from the tissue realisation itself.  The
        # near-field and permittivity maps exist to build the sensitivity
        # map, which is a property of the MODE and is wanted once, not once
        # per realisation.
        #
        # They are also most of the monitor data: tens of gigabytes across
        # an ensemble for maps that are near-copies of each other.  Dropping
        # them leaves the decay probe and the flux, which are small and are
        # the two things that say whether the run is trustworthy.
        return [hotspot, flux, probe]
    return [hotspot, flux, field, eps, probe]


# ==========================================================================
#  Assembling a simulation
# ==========================================================================
def build_sensor_simulation(
    cfg: StudyConfig,
    realization: TissueRealization | None,
    cd_bound: bool = False,
    pol_along_axis: bool = True,
    light_monitors: bool = False,
    background_n: float | None = None,
):
    """
    The main simulation: bowtie in tissue (or in water).

    Parameters
    ----------
    realization : TissueRealization or None
        The random tissue to embed the sensor in.  Pass None for the AQUEOUS
        REFERENCE run — that is the baseline every tissue result is quoted
        against, and it is what makes the transfer factor meaningful.
    cd_bound : bool
        Whether the recognition layer is in its cadmium-bound state.  Run both
        and difference the resonances to get the shift per unit coverage.
    pol_along_axis : bool
        Drive the bowtie mode (True) or the inert cross-polarisation (False).

    Returns
    -------
    td.Simulation — validated but not submitted anywhere.
    """
    import tidy3d as td

    # `background_n` overrides the aqueous background index.  Without it,
    # `realization=None` always gives water at n = 1.333 regardless of
    # `cfg.tissue.n0`.  `paired_run.build_pair` uses it for the bulk step.
    background = (
        to_tidy3d_medium(realization) if realization is not None
        else water_medium(background_n if background_n is not None else 1.333)
    )

    # ---- STRUCTURE ORDER IS LOAD-BEARING -------------------------------
    # Tidy3D resolves overlaps by LATER STRUCTURE WINS.  The recognition and
    # antifouling shells are ENLARGED COPIES of the bowtie polygons, so they
    # contain the metal's footprint entirely and their slab bounds extend past
    # it in z.  Listed after the gold, they would not coat it -- they would
    # DELETE it, leaving no plasmon and no resonance.  `metal_present_check`
    # verifies the result.
    #
    # Correct order, outermost first:
    #     substrate  ->  fouling  ->  recognition  ->  GOLD
    _bowtie = build_bowtie(cfg.bowtie)
    _substrate = [s for s in _bowtie if getattr(s, "name", "") == "substrate"]
    _metal = [s for s in _bowtie if getattr(s, "name", "") != "substrate"]

    structures = [
        *_substrate,
        *build_surface_stack(cfg.bowtie, cfg.surface, cd_bound=cd_bound),
        *_metal,
    ]

    symmetry = cfg.fdtd.symmetry
    if realization is not None and any(s != 0 for s in symmetry):
        warnings.warn(
            "Symmetry planes were requested with a RANDOM medium.  A random "
            "medium has no symmetry, so this would force an unphysical mirror "
            "image of the tissue and corrupt the ensemble statistics.  "
            "Symmetry has been disabled for this run.",
            RuntimeWarning, stacklevel=2,
        )
        symmetry = (0, 0, 0)

    sim = td.Simulation(
        center=(0, 0, 0),
        size=cfg.fdtd.domain_um,
        medium=background,
        structures=structures,
        sources=[build_source(cfg, pol_along_axis)],
        monitors=build_monitors(cfg, light=light_monitors),
        grid_spec=build_grid_spec(cfg),
        boundary_spec=build_boundary_spec(cfg),
        run_time=cfg.fdtd.run_time_ps * 1e-12,
        shutoff=cfg.fdtd.shutoff,
        symmetry=symmetry,
    )
    return sim


def build_slab_simulation(
    cfg: StudyConfig,
    realization: TissueRealization,
    slab_thickness_um: float = 2.0,
):
    """
    A sensor-free slab of tissue, for spot-checking the scattering calibration.

    READ THIS BEFORE TRUSTING THE RESULT
    ------------------------------------
    The scattering mean free path in muscle is 50-200 microns; this slab is a
    couple of microns thick.  So it contains a small fraction of one scattering
    event, and the scattered power is a small difference between large numbers.
    Edge diffraction from the finite slab and absorption of the near-field halo
    by the boundary both contaminate it.

    On top of that, a single realisation does not have a phase function — it
    has speckle.  Fitting a smooth curve to one realisation's angular pattern
    gives a number, but not the one you wanted.

    This is why the *calibration* in `optical_properties.py` is analytic, and
    this simulation is only ever a cross-check.  Use it to confirm the Born
    approximation has not broken down, by comparing an ENSEMBLE-AVERAGED far
    field against the analytic phase function.  Do not use one run of this to
    fit anything.
    """
    import tidy3d as td

    lo, hi = cfg.calibration.band_um
    f0 = C_LIGHT_UM_S / (0.5 * (lo + hi))
    fwidth = f0 * 0.3

    slab = td.Structure(
        geometry=td.Box(center=(0, 0, 0),
                        size=(td.inf, td.inf, slab_thickness_um)),
        medium=to_tidy3d_medium(realization),
        name="tissue_slab",
    )

    src = td.PlaneWave(
        center=(0, 0, -cfg.fdtd.domain_um[2] / 2 + 0.3),
        size=(td.inf, td.inf, 0),
        source_time=td.GaussianPulse(freq0=f0, fwidth=fwidth),
        direction="+",
        pol_angle=0,
        name="plane_wave",
    )

    far = td.FieldProjectionAngleMonitor(
        center=(0, 0, slab_thickness_um / 2 + 0.2),
        size=(td.inf, td.inf, 0),
        freqs=[f0],
        name="far_field",
        phi=[0.0, np.pi / 2],
        theta=list(np.linspace(0.0, np.pi / 2, 91)),
        far_field_approx=True,
        proj_distance=1e6,
    )

    return td.Simulation(
        center=(0, 0, 0),
        size=cfg.fdtd.domain_um,
        medium=td.Medium(permittivity=cfg.tissue.n0 ** 2),
        structures=[slab],
        sources=[src],
        monitors=[far],
        grid_spec=td.GridSpec.auto(
            wavelength=0.5 * (lo + hi),
            min_steps_per_wvl=cfg.fdtd.min_steps_per_wavelength,
        ),
        boundary_spec=build_boundary_spec(cfg),
        run_time=cfg.fdtd.run_time_ps * 1e-12,
        shutoff=cfg.fdtd.shutoff,
    )


# ==========================================================================
#  How long must the run actually be?
# ==========================================================================
def required_run_time_ps(cfg: StudyConfig, expected_q: float | None = None,
                         safety: float = 1.5) -> dict:
    """
    Derive the run time from physics.

    WHY
    ---
    Run time is the single largest cost lever in the whole project -- FDTD
    cost is linear in it -- and it is easy to overestimate by an order of
    magnitude on the belief that random media ring down slowly.

    TWO CONTRIBUTIONS, BOTH SHORT
    -----------------------------
    1. THE SOURCE has to finish.  A Gaussian pulse of spectral width `fwidth`
       lasts roughly 2 x offset / (2 pi fwidth); with Tidy3D's default offset
       of 5 and a 500-900 nm band that is about 12 fs.

    2. THE RESONANCE has to decay.  A mode of quality factor Q has an
       amplitude decay time tau = 2Q / omega.  Reaching the shutoff threshold
       takes tau * ln(1/shutoff).  At Q = 20 and 750 nm that is 220 fs.

    WHAT ABOUT THE RANDOM MEDIUM?
    -----------------------------
    It does not trap light here, and the numbers say so plainly: the
    scattering mean free path is around 200 um while the box is 5 um, so a
    photon crosses and leaves in about 22 fs without scattering once.  Slow
    ringdown would need a cavity, or multiple scattering, and this system has
    neither.  If you ever move to a strongly scattering regime -- a much
    larger box, or far higher index contrast -- revisit this.

    Returns
    -------
    dict with the breakdown and a recommended run time in picoseconds.
    """
    import numpy as _np
    q = expected_q if expected_q is not None else cfg.fdtd.expected_q

    lo, hi = cfg.calibration.band_um
    f_lo, f_hi = C_LIGHT_UM_S / hi, C_LIGHT_UM_S / lo
    f0 = 0.5 * (f_lo + f_hi)
    fwidth = (f_hi - f_lo) / 2.0

    t_source = 2 * 5.0 / (2 * _np.pi * fwidth)          # offset=5 default
    tau = 2.0 * q / (2 * _np.pi * f0)
    t_decay = tau * _np.log(1.0 / max(cfg.fdtd.shutoff, 1e-12))
    total = safety * (t_source + t_decay)

    transit = max(cfg.fdtd.domain_um) * cfg.tissue.n0 / C_LIGHT_UM_S

    return {
        "expected_q": q,
        "source_fs": t_source * 1e15,
        "mode_tau_fs": tau * 1e15,
        "decay_fs": t_decay * 1e15,
        "box_transit_fs": transit * 1e15,
        "recommended_ps": total * 1e12,
        "configured_ps": cfg.fdtd.run_time_ps,
        "wasteful": bool(cfg.fdtd.run_time_ps > 3 * total * 1e12),
        "too_short": bool(cfg.fdtd.run_time_ps < total * 1e12),
        "max_q_supported": float(
            cfg.fdtd.run_time_ps * 1e-12 / safety - t_source
        ) / (2.0 / (2 * _np.pi * f0) * _np.log(1.0 / max(cfg.fdtd.shutoff, 1e-12)))
        if cfg.fdtd.run_time_ps * 1e-12 / safety > t_source else 0.0,
    }


def check_mesh_vs_cutoff(cfg: StudyConfig, sim=None) -> dict:
    """
    Is the FDTD bulk cell fine enough for the tissue model to mean anything?

    THE COUPLING THIS CATCHES
    -------------------------
    The tissue's index spectrum is cut off at `TissueConfig.l_min_um`, and the
    convergence study checks that the synthesised contrast does not depend on the
    voxel size.  That proof is void if the FDTD grid is coarser than the
    cutoff: the solver then averages away everything between l_min and its own
    cell, re-truncating the spectrum at a scale nobody chose.

    It is a tempting mistake, because coarsening the bulk mesh is the second
    biggest cost lever after run time, and nothing else complains.
    """
    import numpy as _np
    if sim is None:
        sim = build_sensor_simulation(cfg, realization=None)
    steps = _np.concatenate([
        _np.diff(_np.array(sim.grid.boundaries.x)),
        _np.diff(_np.array(sim.grid.boundaries.y)),
        _np.diff(_np.array(sim.grid.boundaries.z)),
    ])
    bulk_nm = float(_np.percentile(steps, 90) * 1e3)
    l_min_nm = cfg.tissue.l_min_um * 1e3
    ok = bulk_nm <= l_min_nm * 1.05
    return {
        "bulk_cell_nm": bulk_nm,
        "l_min_nm": l_min_nm,
        "ok": bool(ok),
        "verdict": (
            f"OK - bulk cells are {bulk_nm:.1f} nm against a {l_min_nm:.0f} nm "
            f"tissue cutoff, so the solver resolves everything the tissue "
            f"model contains."
            if ok else
            f"VIOLATION - bulk cells are {bulk_nm:.1f} nm but the tissue "
            f"spectrum runs down to {l_min_nm:.0f} nm. The solver is averaging "
            f"away structure between those two scales, which re-truncates the "
            f"spectrum at the grid and undoes the stage 3 convergence "
            f"argument. Either raise min_steps_per_wavelength, or raise "
            f"l_min_um to {bulk_nm:.0f} nm and justify it on tissue structure."
        ),
    }


# ==========================================================================
#  Reporting, cost, and running
# ==========================================================================
@dataclass
class SimulationReport:
    """What you should look at before spending anything."""

    n_cells: int
    grid_shape: tuple[int, int, int]
    min_step_nm: float
    run_time_ps: float
    n_monitors: int
    monitor_data_gb: float

    def __str__(self) -> str:
        return (
            "Simulation report\n"
            f"  grid                 {self.grid_shape[0]} x "
            f"{self.grid_shape[1]} x {self.grid_shape[2]}\n"
            f"  total cells          {self.n_cells:,}\n"
            f"  smallest cell        {self.min_step_nm:.3f} nm\n"
            f"  run time             {self.run_time_ps:.2f} ps\n"
            f"  monitors             {self.n_monitors}\n"
            f"  estimated data       {self.monitor_data_gb:.3f} GB"
        )


def report(sim) -> SimulationReport:
    """Summarise a built simulation without contacting the cloud."""
    grid = sim.grid
    shape = tuple(len(c) for c in (grid.boundaries.x, grid.boundaries.y,
                                   grid.boundaries.z))
    steps = np.concatenate([
        np.diff(np.array(grid.boundaries.x)),
        np.diff(np.array(grid.boundaries.y)),
        np.diff(np.array(grid.boundaries.z)),
    ])
    try:
        data_gb = float(sim.monitors_data_size and
                        sum(sim.monitors_data_size.values()) / 1e9)
    except Exception:
        data_gb = float("nan")

    return SimulationReport(
        n_cells=int(np.prod([max(1, s - 1) for s in shape])),
        grid_shape=tuple(max(1, s - 1) for s in shape),
        min_step_nm=float(steps.min() * 1e3),
        run_time_ps=float(sim.run_time * 1e12),
        n_monitors=len(sim.monitors),
        monitor_data_gb=data_gb,
    )


def estimate_cost(sim, task_name: str = "cost_estimate") -> float | None:
    """
    Ask Tidy3D what this simulation would cost, in FlexCredits, before running.

    This DOES contact the cloud (it has to upload the simulation to be
    costed), but it does not start a solve and does not charge you.  Run it
    before any ensemble.  An ensemble is `n_seeds` times one simulation, and
    finding out afterwards is an expensive way to learn.
    """
    from tidy3d import web
    try:
        task_id = web.upload(sim, task_name=task_name, verbose=False)
        cost = web.estimate_cost(task_id, verbose=False)
        return float(cost) if cost is not None else None
    except Exception as exc:            # offline, no API key, etc.
        warnings.warn(f"Could not estimate cost: {exc}", RuntimeWarning)
        return None


def simulation_work(cfg: StudyConfig, dl_gap_nm: float = None,
                    domain_um=None, scale_metal: bool = True) -> dict:
    """
    How much numerical work one simulation is, in (cells x time steps).

    This is the quantity FDTD cost is very nearly proportional to, and it is
    a reliable way to compare a 0.5 nm run against a 2 nm one before running
    either.  Built offline; nothing is uploaded.
    """
    from dataclasses import replace as _replace

    kwargs = {}
    if dl_gap_nm is not None:
        kwargs["dl_gap_nm"] = dl_gap_nm
        if scale_metal:
            kwargs["dl_metal_nm"] = dl_gap_nm * 2.0
    if domain_um is not None:
        kwargs["domain_um"] = tuple(domain_um)

    c = _replace(cfg, fdtd=_replace(cfg.fdtd, **kwargs)) if kwargs else cfg
    sim = build_sensor_simulation(c, realization=None)
    cells = int(sim.num_cells)
    steps = int(sim.num_time_steps)
    return {
        "dl_gap_nm": dl_gap_nm if dl_gap_nm is not None else cfg.fdtd.dl_gap_nm,
        "domain_um": tuple(c.fdtd.domain_um),
        "cells": cells,
        "time_steps": steps,
        "work": float(cells) * float(steps),
    }


def estimate_campaign_cost(cfg: StudyConfig, credits_per_sim: float) -> dict:
    """
    The whole bill, not one simulation -- and not one simulation's price
    multiplied by the run count.

    WHY NOT A SIMPLE MULTIPLICATION
    -------------------------------
    Multiplying one simulation's price by the run count is correct only
    when every run costs the same, and several of these deliberately do not: the
    mesh-convergence series runs at FINER meshes than the ensemble.  Halving a
    cell roughly doubles the cell count in each of three dimensions' worth of
    refined region AND roughly doubles the number of time steps, because the
    Courant condition ties the time step to the smallest cell anywhere in the
    domain.  Measured on this geometry, a 0.5 nm run is about 20x a 2 nm run.

    A flat multiplication would understate the total by something like 45%.

    WHAT IT DOES
    ------------
    Every item is built offline, its (cells x time steps) measured, and
    Tidy3D's live per-simulation estimate scaled by the ratio to the ensemble
    simulation it was quoted for.  The runs are:

        1  aqueous reference                 (ensemble mesh)
        N  tissue ensemble realisations      (ensemble mesh)
        2  paired bound/unbound              (finest mesh in the series --
                                              this pair measures the mode
                                              overlap factor and needs it)
        k  mesh convergence, in water        (cfg.fdtd.mesh_study_dl_nm,
                                              in cfg.fdtd.mesh_study_domain_um)

    `credits_per_sim` must be the figure Tidy3D quoted for the ENSEMBLE-mesh
    simulation, which is what `run_all.py` asks it for.

    Tidy3D's own estimate is a worst case based on the full run time; runs
    that hit the field-decay shutoff early cost less, and `web.real_cost()`
    reports what you were actually charged.  The weights here are a model,
    which is why `BudgetConfig.contingency` exists.
    """
    n = cfg.ensemble.n_seeds
    base = simulation_work(cfg)                       # ensemble mesh, prod box
    unit = credits_per_sim / base["work"]             # credits per unit work

    mesh_dl = tuple(cfg.fdtd.mesh_study_dl_nm)
    mesh_dom = cfg.fdtd.mesh_study_domain_um or cfg.fdtd.domain_um

    lines = [
        {"label": "aqueous reference", "n": 1, "work": base["work"],
         "note": f"{cfg.fdtd.dl_gap_nm:.1f} nm gap"},
        {"label": f"tissue ensemble ({n} seeds)", "n": n, "work": base["work"],
         "note": f"{cfg.fdtd.dl_gap_nm:.1f} nm gap"},
    ]

    # The paired bound/unbound runs resolve a sub-nanometre layer, so they
    # run at the finest mesh in the convergence series, not the ensemble one.
    # Three, not two: the unbound run doubles as the low point of the
    # bulk-sensitivity pair that turns a shift into a mode overlap.  See
    # `paired_run.build_pair`.
    fine = min(mesh_dl)
    w_pair = simulation_work(cfg, dl_gap_nm=fine)["work"]
    lines.append({"label": "paired bound/unbound/bulk", "n": 3, "work": w_pair,
                  "note": f"{fine:.1f} nm gap, production box"})

    for dl in mesh_dl:
        w = simulation_work(cfg, dl_gap_nm=dl, domain_um=mesh_dom)["work"]
        lines.append({
            "label": f"mesh convergence @ {dl:.1f} nm", "n": 1, "work": w,
            "note": f"water, {mesh_dom[0]:g}x{mesh_dom[1]:g}x{mesh_dom[2]:g} um",
        })

    for ln in lines:
        ln["each"] = unit * ln["work"]
        ln["cost"] = ln["n"] * ln["each"]
        ln["rel"] = ln["work"] / base["work"]

    total = sum(ln["cost"] for ln in lines)
    ceiling = cfg.budget.flexcredits
    projected = total * cfg.budget.contingency

    return {
        "per_sim": credits_per_sim,
        "lines": lines,
        "items": {ln["label"]: ln["n"] for ln in lines},   # back-compat
        "n_runs": sum(ln["n"] for ln in lines),
        "total": total,
        "contingency": cfg.budget.contingency,
        "projected": projected,
        "ceiling": ceiling,
        "over_budget": bool(projected > ceiling),
        "headroom": ceiling - projected,
        "base_cells": base["cells"],
        "base_steps": base["time_steps"],
    }


def check_decay(sim_data, threshold: float = 0.01) -> dict:
    """
    Did the fields actually die down before the run ended?

    WHY THIS IS NOT OPTIONAL HERE
    -----------------------------
    A resonance's linewidth is set by how quickly its energy leaks away.  If
    the simulation stops while the field is still ringing, the recorded
    spectrum is the Fourier transform of a truncated signal, which is broader
    than the true one.  You would then report a linewidth that is really a
    statement about your `run_time`, and — because the figure of merit divides
    by the linewidth — a sensitivity that is too pessimistic.

    Check every run, not a representative one.

    Returns a dict with the final-to-peak field ratio and a verdict.
    """
    try:
        probe = sim_data[FIELD_TIME_MONITOR]
        ex = np.abs(np.array(probe.Ex).squeeze())
    except Exception as exc:
        return {"ok": False, "reason": f"no decay probe: {exc}"}

    if ex.size == 0:
        return {"ok": False, "reason": "empty decay probe"}

    peak = float(ex.max())
    tail = float(np.mean(ex[-max(1, ex.size // 50):]))
    ratio = tail / peak if peak > 0 else float("inf")
    return {
        "ok": bool(ratio < threshold),
        "peak_field": peak,
        "tail_field": tail,
        "tail_over_peak": ratio,
        "verdict": (
            "Converged - the field decayed before the run ended."
            if ratio < threshold else
            f"NOT CONVERGED - the field was still at {100 * ratio:.1f}% of "
            f"its peak when the run stopped.  The linewidth is "
            f"truncation-broadened.  Increase run_time_ps and re-run."
        ),
    }


def require_tidy3d() -> None:
    """Fail with an explanation, not a bare ImportError.

    Every script here that builds or runs a simulation needs tidy3d, and the
    usual reason it is missing is a shell that never activated the project
    virtual environment: `python` then resolves to the system interpreter,
    which has numpy but not tidy3d, and the traceback points at an import
    fifteen frames down rather than at the shell.
    """
    try:
        import tidy3d  # noqa: F401
        return
    except ImportError:
        pass

    import sys

    exe = sys.executable
    here = os.path.dirname(os.path.abspath(__file__))
    venv_win = os.path.join(os.path.dirname(here), ".venv-win",
                            "Scripts", "python.exe")
    venv_nix = os.path.join(os.path.dirname(here), ".venv", "bin", "python")
    lines = [
        "tidy3d is not importable, so nothing here can build or run a",
        "simulation.",
        "",
        f"  interpreter in use   {exe}",
    ]
    for cand in (venv_win, venv_nix):
        if os.path.isfile(cand):
            lines += [
                f"  project environment  {cand}",
                "",
                "That looks like a shell where the project environment was",
                "never activated. Either activate it, or call the interpreter",
                "directly, which is immune to activation:",
                "",
                f"    {cand} {os.path.basename(sys.argv[0] or 'script.py')} "
                f"{' '.join(sys.argv[1:])}".rstrip(),
            ]
            break
    else:
        lines += [
            "",
            "No project environment was found next to the package. Install",
            "the dependencies with:",
            "",
            "    python -m pip install -r requirements.txt",
        ]
    raise SystemExit("\n".join("  " + ln if ln else "" for ln in lines))


def hdf5_looks_complete(path: str, min_bytes: int = 1_000_000) -> bool:
    """Is this file a complete result, or an interrupted download?

    A resume path that trusts `os.path.isfile` would load a truncated
    download and corrupt the result.  This check is cheap.
    """
    if not os.path.isfile(path) or os.path.getsize(path) < min_bytes:
        return False
    try:
        import h5py
    except ImportError:
        # FAIL OPEN, NOT CLOSED.  If the structural check cannot run, a file
        # of plausible size is accepted rather than declared broken -- calling
        # a good result broken would resubmit a simulation that has already
        # run, which is the more expensive mistake.
        return True
    try:
        with h5py.File(path, "r") as fh:
            return len(fh.keys()) > 0
    except Exception:
        return False


def load_task(task_id: str, path: str, verbose: bool = True):
    """Download the result of a task that ALREADY RAN (no new charge).

    The solver finishing and the results reaching this machine are two
    different events, and the second can fail on its own.  When it does, the
    result exists on the server but not locally; this downloads it.
    """
    from tidy3d import web

    web.monitor(task_id, verbose=verbose)
    return web.load(task_id, path=path, replace_existing=True, verbose=verbose)


def run(sim, task_name: str, path: str = "data/sim.hdf5", verbose: bool = True,
        download_attempts: int = 4):
    """
    Actually submit the simulation.  This costs FlexCredits.

    Deliberately a thin wrapper with an explicit name, so that every place in
    this project that spends money is easy to find with a text search.

    AN INTERRUPTED DOWNLOAD MUST NOT REQUIRE A NEW RUN.
    ---------------------------------------------------
    `web.run` uploads, waits and downloads as a unit, and if the download
    fails the exception carries no task id.  Using `web.Job` instead keeps the
    task id in hand, so a failed transfer is retried
    against the SAME task, and a permanent failure names the id and the
    command that recovers it.  The simulation is submitted once either way.
    """
    import time

    from tidy3d import web

    job = web.Job(simulation=sim, task_name=task_name, verbose=verbose)
    task_id = None
    try:
        data = job.run(path=path)
        task_id = getattr(job, "task_id", None)
        if hdf5_looks_complete(path):
            return data
        raise RuntimeError(f"{path} is short or unreadable after download")
    except Exception as exc:
        task_id = task_id or getattr(job, "task_id", None)
        if task_id is None:
            raise
        print(f"\n  Transfer failed for task {task_id}: {exc}")
        print("  The solver ran and the credits are spent. Retrying the "
              "DOWNLOAD only;")
        print("  nothing is resubmitted.")
        for attempt in range(1, download_attempts + 1):
            wait = min(60, 5 * 2 ** (attempt - 1))
            print(f"  attempt {attempt}/{download_attempts} in {wait}s ...",
                  flush=True)
            time.sleep(wait)
            try:
                data = load_task(task_id, path=path, verbose=verbose)
                if hdf5_looks_complete(path):
                    print("  recovered.")
                    return data
                print("  downloaded file still looks incomplete.")
            except Exception as exc2:
                print(f"  still failing: {exc2}")
        raise RuntimeError(
            f"The simulation '{task_name}' COMPLETED and was billed, but the "
            f"result could not be downloaded after {download_attempts} "
            f"attempts.\n"
            f"  task id: {task_id}\n"
            f"  Do NOT resubmit. When the network is back, recover it with:\n"
            f"      python fetch_task.py {task_id} {path}\n"
            f"  then re-run whatever you were running; it will find the file "
            f"and skip that simulation."
        ) from exc


# ==========================================================================
#  Self-check
# ==========================================================================
if __name__ == "__main__":
    from tissue import generate_tissue

    cfg = StudyConfig()

    print("Building the aqueous reference simulation (no tissue)...")
    sim_water = build_sensor_simulation(cfg, realization=None)
    print(report(sim_water))
    print()

    print("Building a tissue simulation...")
    real = generate_tissue(cfg.tissue, cfg.fdtd, seed=1)
    print(real.summary())
    print()
    sim_tissue = build_sensor_simulation(cfg, realization=real)
    print(report(sim_tissue))
    print()
    print("Both simulations validated locally.  Nothing was uploaded.")
    print("Run `estimate_cost(sim)` before committing to an ensemble.")
