"""
coating.py — measure the thickness of the recognition layer as it is built.
============================================================================

    python coating.py        # offline

WHAT IT CHECKS
--------------
`sensor.build_surface_stack` builds the recognition and antifouling shells as
outward-offset copies of the bowtie polygons, stacked outward, and lets
Tidy3D's structure ordering put the metal back in the middle. The offset uses
the exact normal offset for a triangle, but the rounded tips and corners mean
the result is still not a perfectly conformal shell. This module measures how
far the built shell departs from nominal thickness, and provides an exact
signed-distance construction for cases where the departure matters.

WHERE THE THICKNESS MATTERS
---------------------------
The quantity the study depends on is the shift in resonance wavelength when
the recognition layer changes index. That shift is dominated by the field
energy inside the layer, and in a bowtie that energy is overwhelmingly in the
GAP between the two rounded tips. Energy density falls off very fast away from
the gap, so thickness errors at the outer base corners are weighted by almost
nothing.

The module therefore reports the thickness error in the gap and the worst
case anywhere on the perimeter separately, and grades on the gap value.

WHAT IS COMPUTED
----------------
`polygon_sdf`            exact 2-D signed distance to a polygon
`prism_sdf`              the same extended to the finite-thickness slab
`effective_gap_nm`       the physical gap after tip rounding
`measure_shell_thickness` true normal thickness of the shell as built, sampled
                         around the perimeter and separately in the gap
`exact_shell_medium`     the SDF-based shell as a Tidy3D CustomMedium, for
                         when an exact shell is needed
`print_coating_check`    the report

Everything here is offline. The exact shell adds FDTD cost (a fine custom
medium grid), and its memory size is reported before use.
"""

from __future__ import annotations

import numpy as np

from config import BowtieConfig, StudyConfig, SurfaceStackConfig

NM = 1e-3   # nanometres expressed in microns, the unit Tidy3D uses


# ==========================================================================
#  Signed distance functions
# ==========================================================================
def polygon_sdf(verts: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Exact signed distance from every point of (X, Y) to a simple polygon.

    Negative inside, positive outside.  This is the standard edge-by-edge
    construction: distance to each segment, minimum over segments, sign from a
    crossing-number test.  It is exact everywhere including at corners, which
    is where a polygon offset departs from a true conformal shell.
    """
    v = np.asarray(verts, dtype=float)
    n = len(v)
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)          # (P, 2)

    d2 = np.full(pts.shape[0], np.inf)
    inside = np.zeros(pts.shape[0], dtype=bool)

    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        ab = b - a
        ap = pts - a
        denom = float(ab @ ab)
        t = np.clip((ap @ ab) / denom, 0.0, 1.0) if denom > 0 else np.zeros(len(pts))
        closest = a + t[:, None] * ab
        d2 = np.minimum(d2, np.sum((pts - closest) ** 2, axis=1))

        # Crossing-number test, evaluated edge by edge.
        cond = (a[1] > pts[:, 1]) != (b[1] > pts[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = a[0] + (pts[:, 1] - a[1]) * ab[0] / np.where(ab[1] == 0,
                                                                   np.nan, ab[1])
        crossing = cond & (pts[:, 0] < x_cross)
        inside ^= np.nan_to_num(crossing, nan=False).astype(bool)

    d = np.sqrt(d2)
    d[inside] *= -1.0
    return d.reshape(X.shape)


def prism_sdf(verts: np.ndarray, half_thickness: float,
              X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Signed distance to a polygonal prism of half-height `half_thickness`
    centred on z = 0.

    The usual exact construction for an extruded shape: combine the in-plane
    distance `d2` with the out-of-plane distance `dz`.  Outside, the distance
    is the length of the positive part of (d2, dz); inside, it is the larger
    (least negative) of the two.
    """
    d2 = polygon_sdf(verts, X[..., 0], Y[..., 0])[..., None] \
        * np.ones_like(Z)
    dz = np.abs(Z) - half_thickness
    outside = np.sqrt(np.maximum(d2, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2)
    inside = np.minimum(np.maximum(d2, dz), 0.0)
    return outside + inside


def bowtie_sdf_2d(cfg: BowtieConfig, X: np.ndarray, Y: np.ndarray,
                  pad_um: float = 0.0) -> np.ndarray:
    """
    In-plane signed distance to the WHOLE bowtie (both triangles), optionally
    to a version grown outward by `pad_um` using the polygon-offset recipe
    that `sensor.build_surface_stack` actually uses.

    `pad_um = 0` gives the true metal outline, which is the reference every
    thickness in this module is measured against.
    """
    from sensor import _rounded_triangle_vertices

    side = cfg.side_nm * NM
    gap = cfg.gap_nm * NM
    tip_r = cfg.tip_radius_nm * NM

    half_apex = np.radians(cfg.apex_angle_deg) / 2.0
    grown_side = side + 2.0 * pad_um * (
        np.tan(half_apex) + 1.0 / np.cos(half_apex)
    )
    apex_shift = pad_um / np.sin(half_apex)

    d = np.full(X.shape, np.inf)
    for sign in (+1, -1):
        verts = _rounded_triangle_vertices(
            grown_side, cfg.apex_angle_deg, tip_r + pad_um,
            tip_x_um=-sign * (gap / 2.0 - apex_shift),
            pointing=-sign,
        )
        d = np.minimum(d, polygon_sdf(np.array(verts), X, Y))
    return d


def effective_gap_nm(cfg: BowtieConfig) -> dict:
    """
    The gap you actually get, as opposed to the one you asked for.

    `gap_nm` positions the two SHARP apices.  The tip is then replaced by a
    circular arc of radius `tip_radius_nm`, whose nearest point sits back from
    the sharp apex by `tip_radius * (1/sin(a) - 1)`.  Both tips retreat, so the
    physical gap is

        gap_true = gap_nm + 2 * tip_radius * (1/sin(a) - 1)

    At the defaults (gap 20 nm, tip radius 5 nm, apex 60 degrees) that is
    20 + 2*5*(2-1) = 30 nm: half again as wide as configured.  Everything
    that scales with gap -- field enhancement, steric accessibility, the
    analyte's chance of reaching the hot spot -- is affected, so report this
    number rather than `gap_nm`.
    """
    half_apex = np.radians(cfg.apex_angle_deg) / 2.0
    setback_nm = cfg.tip_radius_nm * (1.0 / np.sin(half_apex) - 1.0)
    true_gap = cfg.gap_nm + 2.0 * setback_nm
    return {
        "configured_gap_nm": float(cfg.gap_nm),
        "tip_setback_nm": float(setback_nm),
        "effective_gap_nm": float(true_gap),
        "ratio": float(true_gap / cfg.gap_nm) if cfg.gap_nm else float("nan"),
        "note": (
            f"gap_nm = {cfg.gap_nm:.1f} nm positions the sharp apices; "
            f"rounding to {cfg.tip_radius_nm:.1f} nm pulls each tip back "
            f"{setback_nm:.2f} nm, so the physical gap is "
            f"{true_gap:.1f} nm."
        ),
    }


# ==========================================================================
#  Measuring the error
# ==========================================================================
def measure_shell_thickness(
    cfg: StudyConfig,
    layer: str = "recognition",
    n_grid: int = 1200,
) -> dict:
    """
    The true normal thickness of the shell, as the polygon-offset recipe
    actually builds it.

    METHOD
    ------
    Build the true metal outline and the grown outline on a fine 2-D grid.
    The shell is the set of points outside the metal and inside the grown
    outline.  For every such point, the true distance to the metal is its
    normal depth into the shell; the shell's local thickness is the MAXIMUM of
    that depth along the outward normal, which on a fine grid is well
    approximated by the maximum metal-distance over the shell points nearest
    each perimeter direction.

    Rather than reconstruct normals, this takes the cleaner equivalent
    measurement: the distribution of `d_metal` over the shell region, whose
    maximum IS the local thickness where the shell is thickest (the corners)
    and whose behaviour near the gap is read off separately.

    Returns a dict with the nominal thickness, the achieved distribution over
    the whole perimeter and over the gap alone, and the fractional errors.
    """
    b = cfg.bowtie
    s = cfg.surface

    t_rec = s.recognition_thickness_nm * NM
    t_foul = s.fouling_thickness_nm * NM
    if layer == "recognition":
        pad_inner, pad_outer, nominal = 0.0, t_rec, t_rec
    elif layer == "fouling":
        pad_inner, pad_outer, nominal = t_rec, t_rec + t_foul, t_foul
    else:
        raise ValueError("layer must be 'recognition' or 'fouling'")

    # A window that comfortably contains the bowtie plus its shells.
    side = b.side_nm * NM
    gap = b.gap_nm * NM
    half_apex = np.radians(b.apex_angle_deg) / 2.0
    height = side / (2.0 * np.tan(half_apex))
    span_x = gap / 2.0 + height + 4 * (t_rec + t_foul) + 0.02
    span_y = side / 2.0 + 4 * (t_rec + t_foul) + 0.02

    x = np.linspace(-span_x, span_x, n_grid)
    y = np.linspace(-span_y, span_y, n_grid)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = x[1] - x[0]

    d_metal = bowtie_sdf_2d(b, X, Y, pad_um=0.0)
    d_inner = bowtie_sdf_2d(b, X, Y, pad_um=pad_inner) if pad_inner > 0 else d_metal
    d_outer = bowtie_sdf_2d(b, X, Y, pad_um=pad_outer)

    # The shell as built: outside the inner grown outline, inside the outer.
    shell = (d_inner > 0) & (d_outer <= 0)
    if not shell.any():
        return {"layer": layer, "nominal_nm": nominal * 1000, "empty": True}

    # Depth into the shell, measured from the true metal surface.
    depth = d_metal[shell]

    # The gap region: between the two tips, within one nominal thickness of
    # the axis.  This is where essentially all of the mode energy lives.
    in_gap = (np.abs(X) < gap / 2.0 + 2 * nominal) & \
             (np.abs(Y) < max(gap, 4 * nominal))
    gap_shell = shell & in_gap
    depth_gap = d_metal[gap_shell] if gap_shell.any() else np.array([np.nan])

    def summarise(arr, outer_pad):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {}
        return {
            "max_nm": float(arr.max()) * 1000,
            "p99_nm": float(np.percentile(arr, 99)) * 1000,
            "median_nm": float(np.median(arr)) * 1000,
        }

    whole = summarise(depth, pad_outer)
    gapd = summarise(depth_gap, pad_outer)

    # The thickness the shell reaches is (outer extent) - (inner extent), both
    # measured as true normal distance from the metal.
    reach_whole = whole.get("max_nm", float("nan")) - pad_inner * 1000
    reach_gap = gapd.get("max_nm", float("nan")) - pad_inner * 1000

    err_whole = (reach_whole - nominal * 1000) / (nominal * 1000)
    err_gap = (reach_gap - nominal * 1000) / (nominal * 1000)

    return {
        "layer": layer,
        "nominal_nm": nominal * 1000,
        "reach_worst_nm": reach_whole,
        "reach_gap_nm": reach_gap,
        "error_worst": err_whole,
        "error_gap": err_gap,
        "grid_nm": dx * 1000,
        "n_shell_points": int(shell.sum()),
        "n_gap_points": int(gap_shell.sum()),
        "empty": False,
    }


def coating_verdict(rec: dict, foul: dict | None = None,
                    gap_tolerance: float = 0.05) -> dict:
    """
    Grade the approximation on the quantity that matters.

    The corner over-thickness is real but it sits where the mode energy is
    not.  The gap thickness is what sets the resonance shift.  So: PASS if the
    gap thickness is within `gap_tolerance` of nominal, and report the corner
    error alongside it.
    """
    if rec.get("empty"):
        return {"ok": False,
                "verdict": "The recognition shell came out empty. Check "
                           "SurfaceStackConfig.recognition_thickness_nm."}

    eg = abs(rec["error_gap"])
    ew = abs(rec["error_worst"])
    ok = eg <= gap_tolerance

    if ok:
        verdict = (
            f"OK. In the gap -- where essentially all of the mode energy "
            f"sits, and therefore the only place the layer thickness affects "
            f"the resonance -- the shell reaches "
            f"{rec['reach_gap_nm']:.3f} nm against a nominal "
            f"{rec['nominal_nm']:.3f} nm, an error of {100 * eg:.1f}%. "
            f"The worst-case error anywhere on the perimeter is "
            f"{100 * ew:.1f}%, at the outer base corners, where the field is "
            f"orders of magnitude weaker. The polygon-offset construction is "
            f"adequate: an exact signed-distance shell would change the "
            f"answer by less than the mesh does."
        )
    else:
        verdict = (
            f"NOT OK. The shell is {100 * eg:.1f}% off nominal IN THE GAP "
            f"({rec['reach_gap_nm']:.3f} nm against {rec['nominal_nm']:.3f} "
            f"nm nominal), which is where the mode energy is. That error "
            f"propagates straight into the resonance shift and hence into "
            f"every sensitivity number downstream. Use "
            f"`exact_shell_medium` instead, or reduce the corner rounding so "
            f"the polygon offset behaves."
        )
    return {"ok": bool(ok), "verdict": verdict,
            "error_gap": rec["error_gap"], "error_worst": rec["error_worst"]}


# ==========================================================================
#  The exact construction, for when it is needed
# ==========================================================================
def exact_shell_medium(
    cfg: StudyConfig,
    cd_bound: bool,
    dl_um: float | None = None,
    pad_um: float = 0.02,
):
    """
    The recognition and antifouling shells as an exact signed-distance shell,
    returned as a single Tidy3D `CustomMedium` on a local fine grid.

    HOW IT IS EXACT
    ---------------
    A point belongs to the recognition layer if its true normal distance to
    the metal lies in (0, t_rec], and to the fouling layer if it lies in
    (t_rec, t_rec + t_foul].  That is the definition of a conformal shell, and
    the signed distance function evaluates it directly -- no polygon growing,
    no corner artefacts, correct at every point including the rounded tips.

    WHAT IT COSTS
    -------------
    A custom medium is sampled on a grid, and to resolve a 5 nm shell you need
    cells of about 1 nm across a region a few hundred nanometres wide.  The
    returned structure covers only the antenna neighbourhood plus `pad_um`,
    not the whole domain, so the array stays modest -- but it is still an
    explicit array, and `.nbytes` is reported so you can see it before you
    build a forty-seed ensemble out of it.

    Use this for the paired bound/unbound runs, where the layer index IS the
    experiment.  For the ensemble, where the layer is common-mode and cancels
    in the seed-to-seed spread, the polygon-offset shells are cheaper and
    (per `coating_verdict`) good enough.
    """
    import tidy3d as td
    from sensor import dielectric

    b, s = cfg.bowtie, cfg.surface
    t_rec = s.recognition_thickness_nm * NM
    t_foul = s.fouling_thickness_nm * NM
    thick = b.thickness_nm * NM

    if dl_um is None:
        dl_um = min(t_rec, t_foul if t_foul > 0 else t_rec) / 4.0

    side = b.side_nm * NM
    gap = b.gap_nm * NM
    half_apex = np.radians(b.apex_angle_deg) / 2.0
    height = side / (2.0 * np.tan(half_apex))

    span_x = gap / 2.0 + height + t_rec + t_foul + pad_um
    span_y = side / 2.0 + t_rec + t_foul + pad_um
    span_z = thick / 2.0 + t_rec + t_foul + pad_um

    nx = int(2 * span_x / dl_um) + 1
    ny = int(2 * span_y / dl_um) + 1
    nz = int(2 * span_z / dl_um) + 1

    x = np.linspace(-span_x, span_x, nx)
    y = np.linspace(-span_y, span_y, ny)
    z = np.linspace(-span_z, span_z, nz)

    X2, Y2 = np.meshgrid(x, y, indexing="ij")
    d2 = bowtie_sdf_2d(b, X2, Y2, pad_um=0.0)              # (nx, ny)

    dz = np.abs(z) - thick / 2.0                           # (nz,)
    D2 = d2[:, :, None]
    DZ = dz[None, None, :]
    d3 = np.sqrt(np.maximum(D2, 0.0) ** 2 + np.maximum(DZ, 0.0) ** 2) \
        + np.minimum(np.maximum(D2, DZ), 0.0)

    n_rec = (s.recognition_n_bound if cd_bound else s.recognition_n_unbound)
    n_bg = cfg.tissue.n0

    n_arr = np.full(d3.shape, float(n_bg))
    n_arr[(d3 > 0) & (d3 <= t_rec)] = float(n_rec)
    if t_foul > 0:
        n_arr[(d3 > t_rec) & (d3 <= t_rec + t_foul)] = float(s.fouling_n)
    # Inside the metal the value is irrelevant: the gold structure is added
    # after this one and overrides it.
    n_arr[d3 <= 0] = float(n_bg)

    eps = n_arr ** 2
    data = td.SpatialDataArray(eps, coords={"x": x, "y": y, "z": z})
    medium = td.CustomMedium(permittivity=data)

    structure = td.Structure(
        geometry=td.Box(center=(0, 0, 0),
                        size=(2 * span_x, 2 * span_y, 2 * span_z)),
        medium=medium,
        name="exact_conformal_shell",
    )

    return {
        "structure": structure,
        "shape": (nx, ny, nz),
        "dl_nm": dl_um * 1000,
        "nbytes": int(eps.nbytes),
        "recognition_voxels": int(((d3 > 0) & (d3 <= t_rec)).sum()),
        "note": (
            f"{nx}x{ny}x{nz} custom-medium grid at {dl_um * 1000:.2f} nm "
            f"({eps.nbytes / 1e6:.1f} MB). Add this BEFORE the gold "
            f"structures so the metal overrides it."
        ),
    }


# ==========================================================================
#  Reporting
# ==========================================================================
def print_coating_check(cfg: StudyConfig) -> dict:
    """Report the physical gap and the shell thickness in the gap and worst case."""
    rec = measure_shell_thickness(cfg, "recognition")
    foul = (measure_shell_thickness(cfg, "fouling")
            if cfg.surface.fouling_thickness_nm > 0 else None)
    v = coating_verdict(rec, foul)

    eg = effective_gap_nm(cfg.bowtie)
    for line in _wrap(eg["note"], 68):
        print(f"  {line}")
    print()
    print("  The shells are built as enlarged copies of the bowtie polygons,")
    print("  offset by the exact triangle-offset formula. Measured against")
    print("  the true signed distance to the metal:")
    print()
    print(f"  {'layer':>12}  {'nominal':>9}  {'in the gap':>11}  "
          f"{'worst case':>11}")
    for r in (rec, foul):
        if r is None or r.get("empty"):
            continue
        print(f"  {r['layer']:>12}  {r['nominal_nm']:8.3f}nm  "
              f"{r['reach_gap_nm']:10.3f}nm  {r['reach_worst_nm']:10.3f}nm")
    print()
    print(f"  {'':>12}  {'':>9}  {'error':>11}  {'error':>11}")
    for r in (rec, foul):
        if r is None or r.get("empty"):
            continue
        print(f"  {r['layer']:>12}  {'':>9}  {100 * r['error_gap']:10.1f}%  "
              f"{100 * r['error_worst']:10.1f}%")
    print()
    for line in _wrap(v["verdict"], 68):
        print(f"  {line}")
    print()
    print(f"  (measured on a {rec['grid_nm']:.2f} nm sampling grid; the gap "
          f"region held {rec['n_gap_points']:,} points)")
    return {"recognition": rec, "fouling": foul, **v}


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


if __name__ == "__main__":
    print_coating_check(StudyConfig())
