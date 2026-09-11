"""
sensor.py — the gold bowtie and the chemistry stuck to it.
==========================================================

WHAT A BOWTIE IS AND WHY WE USE ONE
-----------------------------------
Two gold triangles, tip to tip, with a small gap between them.  Seen from
above it looks like a bow tie.

When light of the right colour hits gold at this size, the electrons in the
metal slosh back and forth in step with the light.  This is a *plasmon
resonance*.  At the sharp tips, the sloshing charge piles up, and the electric
field in the gap between the tips becomes far stronger than the field of the
light that created it — often by a factor of tens or hundreds in amplitude.

That tiny, intensely bright region is the "hot spot".  If a cadmium complex
binds there, it shifts the resonance colour by a measurable amount.  If it
binds a hundred nanometres away, it shifts it by essentially nothing.  So the
sensor's real sensing volume is not "the surface" — it is the small region
where the field is strong AND the analyte can physically reach.

That second condition is the one this module takes seriously.

THE FOUR LAYERS
---------------
Working outward from the metal:

  1. GOLD.  Modelled with measured optical constants (Johnson & Christy 1972),
     fitted across the whole simulated band.  Not a two-term Drude model — a
     Drude fit is accurate in the infrared and quietly wrong in the visible,
     where interband transitions matter and where the resonance actually sits.

  2. RECOGNITION LAYER.  The aptamer, chitosan, polyaniline or protein that
     grabs cadmium.  A thin conformal shell.  Its refractive index changes
     slightly when cadmium binds; that change is the entire signal.

  3. FOULING LAYER.  Everything else in the fish that sticks to the sensor
     uninvited.  Unavoidable on tissue contact.  It does no sensing and it
     pushes the analyte further out into weaker field.  Most published models
     omit it, which flatters the predicted performance.

  4. STERIC EXCLUSION MASK.  Not a material — a rule.  A hydrated
     cadmium-ligand complex is roughly a nanometre across.  It cannot squeeze
     into a crevice narrower than itself, no matter how bright that crevice is.
     Any field maximum inside such a crevice is unreachable, and counting it
     towards your sensitivity is simply an error.

     A bowtie makes this vivid: shrinking the gap makes the hot spot brighter,
     and past a point makes it *less* useful, because the brightest part is now
     inaccessible.  There is an optimum gap, and it is not the smallest one.
     Finding it is a result worth reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from config import BowtieConfig, SurfaceStackConfig, NM


# ==========================================================================
#  Materials
# ==========================================================================
def gold_medium(variant: str = "JohnsonChristy1972"):
    """
    Gold, from measured optical constants.

    WHY NOT A DRUDE MODEL
    ---------------------
    The Drude model treats metal electrons as a free gas.  It works well in the
    infrared.  In the visible it fails, because gold's d-band electrons can be
    promoted across an interband gap around 2.4 eV (roughly 520 nm), producing
    absorption that a free-electron model knows nothing about.  Since plasmonic
    Cd sensors are usually designed in exactly that region, a Drude fit gives
    plausible-looking, wrong answers.

    Tidy3D ships multi-coefficient fits to the published measurements, valid
    across the whole band.  Use them.

    Available variants include 'JohnsonChristy1972' (the classic evaporated-film
    data most papers cite), 'Olmon2012evaporated', 'Olmon2012crystal', and
    'RakicLorentzDrude1998'.  If your fabricated films are single-crystal, the
    Olmon crystal data is a better match and will predict narrower resonances.
    """
    import tidy3d as td
    return td.material_library["Au"][variant]


def dielectric(n: float):
    """A simple lossless material of refractive index `n`."""
    import tidy3d as td
    return td.Medium(permittivity=n ** 2)


# ==========================================================================
#  Geometry
# ==========================================================================
def _rounded_triangle_vertices(
    side_um: float,
    apex_angle_deg: float,
    tip_radius_um: float,
    tip_x_um: float,
    pointing: int,
    n_round: int = 12,
) -> list[tuple[float, float]]:
    """
    Vertices of one triangle of the bowtie, with a rounded tip.

    WHY THE TIP MUST BE ROUNDED
    ---------------------------
    A mathematically sharp point produces an electric field that goes to
    infinity.  In a simulation that shows up as a field value that keeps rising
    every time you refine the mesh, and never settles.  You can always produce
    an impressive enhancement factor by refining a sharp tip a bit more; the
    number means nothing.

    Real tips are rounded by fabrication — 3 to 10 nm is typical for
    electron-beam lithography.  Putting that radius in the model makes the
    answer converge, and makes it correspond to something you could build.

    Parameters
    ----------
    pointing : {+1, -1}
        +1 means the tip points in the +x direction; -1 the other way.
    """
    half_apex = np.radians(apex_angle_deg) / 2.0
    height = side_um / (2.0 * np.tan(half_apex))

    # Untranslated triangle with its tip at the origin pointing along +x.
    base_x = height
    base_half = side_um / 2.0

    # Replace the sharp tip with a circular arc of radius `tip_radius_um`.
    # The arc centre sits back from the tip along the axis by r / sin(half_apex).
    arc_centre_x = tip_radius_um / np.sin(half_apex)
    arc_span = np.pi - 2 * half_apex
    angles = np.linspace(np.pi - arc_span / 2, np.pi + arc_span / 2, n_round)
    arc = [
        (arc_centre_x + tip_radius_um * np.cos(a),
         tip_radius_um * np.sin(a))
        for a in angles
    ]

    # Winding matters: the vertex list must trace the outline once without
    # crossing itself, or the solver rejects it.  The arc above runs from the
    # +y side of the tip round to the -y side, so the base corners must follow
    # in the order -y then +y to close the loop cleanly.
    verts = [*arc, (base_x, -base_half), (base_x, base_half)]
    return [(pointing * vx + tip_x_um, vy) for vx, vy in verts]


def build_bowtie(cfg: BowtieConfig, include_substrate: bool = True) -> list:
    """
    Assemble the bowtie as a list of Tidy3D structures.

    The gap is centred on the origin, the bowtie axis runs along x, the metal
    film lies in the x-y plane and has thickness along z.

    Returns
    -------
    list of `td.Structure` — substrate first (if requested), then the two
    triangles.
    """
    import tidy3d as td

    side = cfg.side_nm * NM
    gap = cfg.gap_nm * NM
    thick = cfg.thickness_nm * NM
    tip_r = cfg.tip_radius_nm * NM

    au = gold_medium()
    structures = []

    if include_substrate and cfg.substrate_index is not None:
        # A half-space below the metal.  Large enough to reach the boundary.
        structures.append(
            td.Structure(
                geometry=td.Box(
                    center=(0, 0, -thick / 2 - 5.0),
                    size=(td.inf, td.inf, 10.0),
                ),
                medium=dielectric(cfg.substrate_index),
                name="substrate",
            )
        )

    for sign, label in ((+1, "left"), (-1, "right")):
        # sign = +1 -> tip points +x, so the triangle body sits at negative x.
        verts = _rounded_triangle_vertices(
            side, cfg.apex_angle_deg, tip_r,
            tip_x_um=-sign * gap / 2.0,
            pointing=-sign,
        )
        structures.append(
            td.Structure(
                geometry=td.PolySlab(
                    vertices=verts,
                    axis=2,
                    slab_bounds=(-thick / 2, thick / 2),
                ),
                medium=au,
                name=f"bowtie_{label}",
            )
        )

    return structures


def build_surface_stack(
    bowtie_cfg: BowtieConfig,
    surf_cfg: SurfaceStackConfig,
    cd_bound: bool,
) -> list:
    """
    The conformal recognition and fouling layers around the metal.

    HOW THE SHELLS ARE BUILT
    ------------------------
    The shells are enlarged copies of the bowtie polygons, stacked outward,
    relying on Tidy3D's structure ordering (later structures override earlier
    ones) so the metal punches back through the middle.

    Growing a triangle by an exact normal offset `pad` is NOT the same as
    adding `pad` to its dimensions. Naively adding `2*pad` to the side length
    and shifting the tip by `pad` cancels almost exactly along the axis of a
    60-degree bowtie, leaving the shell nearly absent in the gap. The exact
    offset below avoids that, and `coating.measure_shell_thickness` checks
    the built thickness against a true signed distance function.

    THE CORRECT OFFSET
    ------------------
    For a triangle of apex half-angle `a`, offsetting every edge outward by
    `pad`:

        the apex slides outward along the axis by   pad / sin(a)
        the base edge slides back by                pad
        so the height grows by                      pad * (1 + 1/sin(a))
        and the base length grows by                2*pad*(tan(a) + 1/cos(a))
        the tip arc radius grows by                 pad

    With the tip arc handled that way the arc CENTRE does not move, which is
    exactly the condition for a true offset of a rounded corner.  The result
    is exact everywhere except the two outer base corners, where a real offset
    would round the corner with radius `pad` and this construction leaves it
    sharp.  Those corners carry a negligible share of the mode energy;
    `coating.py` quantifies that too.

    A SEPARATE THING WORTH KNOWING ABOUT `gap_nm`
    ---------------------------------------------
    `BowtieConfig.gap_nm` positions the two SHARP apices.  Rounding the tip
    pulls the metal back from that apex by `tip_radius * (1/sin(a) - 1)`, so
    the physical tip-to-tip gap is larger than `gap_nm` -- at the defaults, 30
    nm of real gap for a nominal 20.  `coating.effective_gap_nm` reports the
    true value; report it rather than `gap_nm`.

    Parameters
    ----------
    cd_bound : bool
        Which state of the recognition layer to build.  Running the same
        geometry in both states and differencing the resonance wavelength is
        how the "shift per unit surface coverage" coefficient is measured.
    """
    import tidy3d as td

    side = bowtie_cfg.side_nm * NM
    gap = bowtie_cfg.gap_nm * NM
    thick = bowtie_cfg.thickness_nm * NM
    tip_r = bowtie_cfg.tip_radius_nm * NM

    t_rec = surf_cfg.recognition_thickness_nm * NM
    t_foul = surf_cfg.fouling_thickness_nm * NM
    n_rec = (
        surf_cfg.recognition_n_bound if cd_bound
        else surf_cfg.recognition_n_unbound
    )

    layers = []
    # Build outermost first, so the inner layers (added later) override it.
    for pad, n_layer, name in (
        (t_rec + t_foul, surf_cfg.fouling_n, "fouling"),
        (t_rec, n_rec, "recognition"),
    ):
        if pad <= 0:
            continue
        half_apex = np.radians(bowtie_cfg.apex_angle_deg) / 2.0
        grown_side = side + 2.0 * pad * (
            np.tan(half_apex) + 1.0 / np.cos(half_apex)
        )
        apex_shift = pad / np.sin(half_apex)

        for sign, label in ((+1, "left"), (-1, "right")):
            verts = _rounded_triangle_vertices(
                grown_side, bowtie_cfg.apex_angle_deg, tip_r + pad,
                tip_x_um=-sign * (gap / 2.0 - apex_shift),
                pointing=-sign,
            )
            layers.append(
                td.Structure(
                    geometry=td.PolySlab(
                        vertices=verts,
                        axis=2,
                        slab_bounds=(-thick / 2 - pad, thick / 2 + pad),
                    ),
                    medium=dielectric(n_layer),
                    name=f"{name}_{label}",
                )
            )
    return layers


# ==========================================================================
#  Steric accessibility
# ==========================================================================
@dataclass
class AccessibilityMap:
    """Where the analyte can and cannot physically go."""

    mask: np.ndarray
    """Boolean array on the field-monitor grid.  True = the centre of a
    cadmium-ligand complex could sit here."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    probe_radius_nm: float

    @property
    def accessible_fraction(self) -> float:
        return float(self.mask.mean())

    def __str__(self) -> str:
        return (
            f"Accessibility map ({self.mask.shape}), probe radius "
            f"{self.probe_radius_nm:.2f} nm:  "
            f"{100 * self.accessible_fraction:.1f}% of the volume reachable"
        )


def steric_accessibility_mask(
    solid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    probe_radius_nm: float,
) -> AccessibilityMap:
    """
    Work out where a finite-sized analyte molecule can actually reach.

    THE ALGORITHM, IN PLAIN LANGUAGE
    --------------------------------
    Imagine rolling a ball of the analyte's size over the structure.  Anywhere
    the ball can reach without overlapping the solid is accessible.  Anywhere
    it cannot — a crevice narrower than the ball, a pore with too small a mouth
    — is not, even if there is technically empty space in there.

    This is the same construction crystallographers use for the "solvent
    accessible surface", and in image processing it is called a *morphological
    opening*.  Concretely:

        1. Dilate the solid by the probe radius.  Anything within one radius of
           the metal is now forbidden, because the ball's centre cannot go
           there without the ball overlapping metal.
        2. What remains is where the ball's centre may sit.
        3. Keep only the part of that space connected to the outside world.  A
           sealed internal void is geometrically empty but chemically
           unreachable, and this step removes it.

    WHY THIS MATTERS SO MUCH FOR A BOWTIE
    -------------------------------------
    The brightest field in a bowtie is at the very centre of the gap, right
    against the tips.  Squeeze the gap and the field rises steeply — and the
    accessible fraction of that hot spot falls just as steeply.  Past a certain
    gap width you are making the sensor brighter and *worse*.  Only a
    calculation like this one can find where that crossover sits.

    Parameters
    ----------
    solid : boolean array
        True where the structure (metal plus its coating layers) is.
    x, y, z : arrays
        Coordinate axes in microns.  Must be uniformly spaced.
    probe_radius_nm : float
        Hydrodynamic radius of the analyte complex.

    Returns
    -------
    AccessibilityMap
    """
    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
    dy = float(np.mean(np.diff(y))) if len(y) > 1 else 1.0
    dz = float(np.mean(np.diff(z))) if len(z) > 1 else 1.0
    spacing = np.array([dx, dy, dz])

    r_um = probe_radius_nm * NM
    radii_exact = r_um / spacing
    radii = np.maximum(1, np.round(radii_exact).astype(int))

    # If the probe is smaller than a voxel, the rounding above inflates it to
    # one full voxel and the mask over-excludes.  Warn rather than silently
    # producing a pessimistic accessible fraction: the monitor grid must
    # resolve the analyte, not just the field.
    if np.any(radii_exact < 1.0):
        import warnings
        worst = float(np.min(radii_exact))
        warnings.warn(
            f"Steric probe radius is only {worst:.2f} voxels across on this "
            f"grid, so it has been rounded up to 1 voxel and the accessible "
            f"fraction is an underestimate. Refine the field monitor to at "
            f"least {probe_radius_nm:.2f} nm cells for a quantitative mask.",
            RuntimeWarning, stacklevel=2,
        )

    # --- step 1: dilate the solid by the probe radius -------------------
    ball = _ellipsoid_structuring_element(radii)
    forbidden = ndimage.binary_dilation(solid, structure=ball)

    # --- step 2: what is left is where the probe centre may sit ---------
    free = ~forbidden

    # --- step 3: keep only the part connected to the outside ------------
    labels, n_labels = ndimage.label(free)
    if n_labels == 0:
        return AccessibilityMap(free, x, y, z, probe_radius_nm)

    # Any label touching a face of the box is "outside-connected".
    boundary_labels = set()
    for sl in (
        np.s_[0, :, :], np.s_[-1, :, :],
        np.s_[:, 0, :], np.s_[:, -1, :],
        np.s_[:, :, 0], np.s_[:, :, -1],
    ):
        boundary_labels.update(np.unique(labels[sl]).tolist())
    boundary_labels.discard(0)

    reachable = np.isin(labels, list(boundary_labels))
    return AccessibilityMap(reachable, x, y, z, probe_radius_nm)


def _ellipsoid_structuring_element(radii: np.ndarray) -> np.ndarray:
    """A ball, expressed in voxels, allowing for anisotropic voxel spacing."""
    rx, ry, rz = (int(r) for r in radii)
    gx, gy, gz = np.ogrid[-rx:rx + 1, -ry:ry + 1, -rz:rz + 1]
    return ((gx / rx) ** 2 + (gy / ry) ** 2 + (gz / rz) ** 2) <= 1.0


def solid_mask_from_permittivity(
    eps: np.ndarray, threshold_real: float = -1.0
) -> np.ndarray:
    """
    Identify the metal from a permittivity map recorded during the simulation.

    Gold below its plasma frequency has a large NEGATIVE real permittivity —
    around -25 at 800 nm.  Every dielectric in the scene (tissue, water,
    recognition layer, substrate) has a positive one, between about 1.7 and
    2.2.  So a threshold anywhere below zero separates them cleanly, and -1 is
    a safe, unfussy choice.

    Using the *recorded* permittivity rather than re-deriving the geometry
    guarantees the mask matches the structure the solver actually meshed,
    including any staircasing or subpixel smoothing.
    """
    return np.real(eps) < threshold_real


def coated_solid_mask(
    eps: np.ndarray,
    metal_threshold: float = -1.0,
    coating_eps_range: tuple[float, float] = (2.0, 2.25),
) -> np.ndarray:
    """
    Metal plus its coating layers — the full obstacle the analyte must get past.

    The coating range corresponds to refractive indices of about 1.41 to 1.50,
    which covers the recognition and fouling layers.  Adjust if you change
    those indices, and be aware that tissue at n = 1.39 (eps = 1.93) sits just
    below the window, deliberately: tissue is not an obstacle, it is the
    medium the analyte arrives through.
    """
    metal = solid_mask_from_permittivity(eps, metal_threshold)
    lo, hi = coating_eps_range
    coating = (np.real(eps) >= lo) & (np.real(eps) <= hi)
    return metal | coating


# ==========================================================================
#  Self-check
# ==========================================================================

# ==========================================================================
#  Steric accessibility of the REAL geometry, coatings included
# ==========================================================================
def gap_accessibility(
    cfg,
    gap_nm: float | None = None,
    include_coatings: bool = True,
    n_grid: int = 201,
    half_span_um: float = 0.10,
) -> dict:
    """
    Can the analyte actually reach the hot spot of the ACTUAL bowtie?

    WHY THE REAL GEOMETRY IS NEEDED
    -------------------------------
    A simplified geometry (two half-spaces separated by `gap_nm`, without
    triangles, rounded tips or coatings) gives misleading accessibility, for
    two reasons that push in the same direction:

      * The recognition and antifouling layers are 3 + 4 = 7 nm on EVERY
        surface, so they consume 14 nm of any gap.  This is the dominant
        steric effect in the design.

      * `gap_nm` positions the SHARP apices.  Rounding the tip pulls the metal
        back by `r(1/sin a - 1)`, so the physical gap is wider than the
        configured one -- 30 nm for a nominal 20 (see
        `coating.effective_gap_nm`).

    With both included, the bare metal geometry is accessible at essentially
    every gap (93-99%); it is the COATING STACK that excludes the analyte,
    sharply once the free channel approaches the analyte's diameter.  Coating
    thickness is therefore a design variable for accessibility.

    THE HOT-SPOT REGION, DEFINED EXPLICITLY
    ---------------------------------------
    Reported fractions are of the volume

        |x| <= physical_gap/2,  |y| <= tip_radius,  |z| <= thickness/2

    i.e. the volume between the two rounded tips, laterally no wider than the
    tip radius.  That is a geometric proxy for where the field is, and it is
    stated rather than assumed: a looser y-window would inflate the fraction
    by including open space beyond the triangle edges.

    Returns a dict with the physical gap, the free channel width, whether the
    analyte fits at all, and the reachable fraction with and without coatings.
    """
    from dataclasses import replace as _replace
    from coating import bowtie_sdf_2d, effective_gap_nm

    b = cfg.bowtie if gap_nm is None else _replace(cfg.bowtie,
                                                   gap_nm=float(gap_nm))
    s = cfg.surface
    t_coat = (s.recognition_thickness_nm + s.fouling_thickness_nm) * NM
    probe_nm = s.analyte_hydrodynamic_radius_nm

    eg = effective_gap_nm(b)
    gap_true_nm = eg["effective_gap_nm"]
    free_nm = gap_true_nm - 2.0 * t_coat * 1000
    fits = free_nm > 2.0 * probe_nm

    ax = np.linspace(-half_span_um, half_span_um, n_grid)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")

    d2 = bowtie_sdf_2d(b, X[:, :, 0], Y[:, :, 0])
    dz = np.abs(ax) - b.thickness_nm * NM / 2.0
    D2, DZ = d2[:, :, None], dz[None, None, :]
    d3 = (np.sqrt(np.maximum(D2, 0.0) ** 2 + np.maximum(DZ, 0.0) ** 2)
          + np.minimum(np.maximum(D2, DZ), 0.0))

    region = ((np.abs(X) <= gap_true_nm * NM / 2.0)
              & (np.abs(Y) <= b.tip_radius_nm * NM)
              & (np.abs(Z) < b.thickness_nm * NM / 2.0))

    out = {}
    for label, t in (("bare", 0.0), ("coated", t_coat)):
        amap = steric_accessibility_mask(d3 <= t, ax, ax, ax, probe_nm)
        vals = amap.mask[region]
        out[label] = float(vals.mean()) if vals.size else 0.0

    return {
        "configured_gap_nm": float(b.gap_nm),
        "physical_gap_nm": gap_true_nm,
        "coating_nm_per_side": t_coat * 1000,
        "free_channel_nm": free_nm,
        "analyte_diameter_nm": 2.0 * probe_nm,
        "analyte_fits": bool(fits),
        "reachable_bare": out["bare"],
        "reachable_coated": out["coated"] if include_coatings else None,
        "exclusion_factor": (out["bare"] / out["coated"]
                             if out["coated"] > 0 else float("inf")),
        "region_note": (
            f"|x| <= {gap_true_nm / 2:.1f} nm, "
            f"|y| <= {b.tip_radius_nm:.1f} nm, "
            f"|z| <= {b.thickness_nm / 2:.1f} nm"
        ),
    }


if __name__ == "__main__":
    from config import StudyConfig

    cfg = StudyConfig()
    print("Bowtie geometry")
    print(f"  triangle side   {cfg.bowtie.side_nm} nm")
    print(f"  gap             {cfg.bowtie.gap_nm} nm")
    print(f"  tip radius      {cfg.bowtie.tip_radius_nm} nm")
    print()

    verts = _rounded_triangle_vertices(
        cfg.bowtie.side_nm * NM, cfg.bowtie.apex_angle_deg,
        cfg.bowtie.tip_radius_nm * NM,
        tip_x_um=-cfg.bowtie.gap_nm * NM / 2, pointing=-1,
    )
    xs = [v[0] * 1000 for v in verts]
    ys = [v[1] * 1000 for v in verts]
    print(f"  one triangle spans x {min(xs):.1f} to {max(xs):.1f} nm, "
          f"y {min(ys):.1f} to {max(ys):.1f} nm")
    print(f"  ({len(verts)} vertices, including the rounded tip arc)")

    # --- demonstrate the steric mask on a synthetic bowtie-like solid ----
    print()
    print("Steric accessibility demonstration")
    print("-" * 60)
    n = 81
    ax = np.linspace(-0.1, 0.1, n)          # +/- 100 nm, 2.5 nm voxels
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")

    # Two blocks separated by a gap, standing in for the tips.
    half_gap = cfg.bowtie.gap_nm * NM / 2
    solid = (np.abs(X) > half_gap) & (np.abs(Y) < 0.05) & (np.abs(Z) < 0.02)

    for gap_nm in (5, 10, 20, 40):
        hg = gap_nm * NM / 2
        s = (np.abs(X) > hg) & (np.abs(Y) < 0.05) & (np.abs(Z) < 0.02)
        amap = steric_accessibility_mask(
            s, ax, ax, ax, cfg.surface.analyte_hydrodynamic_radius_nm
        )
        # Fraction of the GAP volume that is reachable.
        gap_region = (np.abs(X) <= hg) & (np.abs(Y) < 0.05) & (np.abs(Z) < 0.02)
        in_gap = amap.mask[gap_region]
        frac = float(in_gap.mean()) if in_gap.size else 0.0
        print(f"  gap {gap_nm:3d} nm -> {100 * frac:5.1f}% of the hot-spot "
              f"volume is reachable by the analyte")
    print()
    print("  This is the trade-off: a tighter gap gives a brighter hot spot")
    print("  that the analyte increasingly cannot get into.")
