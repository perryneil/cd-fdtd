"""
conformational_sweep.py — sweep the two conformational terms of the layer index change.
=====================================================================================

    python conformational_sweep.py        # offline, runs in seconds

WHAT IT DOES
------------
The index change of the recognition layer on binding is built from three
terms.  One of them -- the added mass of the cadmium ion -- is exactly
calculable and tiny.  The other two are large, they have OPPOSITE SIGNS, and
both were measured on a completely different chemical system:

    contraction     the layer thins and densifies       raises the index
    RII deviation   dn/dc falls on binding              lowers the index

Both come from an L-TYROSINAMIDE aptamer (Dejeu et al. 2018, Pons et al.
2022).  Cadmium is not L-tyrosinamide.  The magnitudes are published; their
transferability is not, and cannot be, because nobody has measured either
quantity for a metal-ion-binding aptamer on a surface.

This module sweeps both terms together as a 2-D SURFACE rather than as two
1-D sweeps, for the reason given below.

WHY A SURFACE AND NOT TWO SWEEPS
--------------------------------
The central experimental result of Pons et al. (2022) is that these two terms
TRADE OFF AGAINST EACH OTHER, and that the trade-off can produce a null or
even NEGATIVE SPR signal from a recognition event they independently confirmed
by QCM-D and ITC.  A layer can bind its target perfectly well and read zero.

Two independent 1-D sweeps cannot show that, because the null is a CURVE in
the (contraction, RII deviation) plane, not a point on either axis.  A design
that lands on that curve gives no signal at any concentration.  `null_contour`
finds the curve; `distance_to_null` says how far the nominal design sits from
it.

THE RANGES, AND WHERE THEY COME FROM
------------------------------------
    thickness change    -30% to +25%   `recognition_layer.SWEEP_THICKNESS_PERCENT`
    RII deviation       -20% to  0%    `recognition_layer.SWEEP_RII_DEVIATION_PERCENT`
    baseline dn/dc      0.183-0.268    `recognition_layer.SWEEP_DN_DC_ML_PER_G`

The first range crosses zero, which is the point.  Stuber et al. (2023)
measured, by the same method in the same paper, a 15% CONTRACTION for a
dopamine aptamer and a 1.2 nm ELONGATION for a serotonin aptamer -- two DNA
aptamers binding small aromatic amines of near-identical mass, moving in
opposite directions.  The sign of this term is not known for a Cd(II)
aptamer, so the sweep must include both signs.

The second range is set by the one measurement in the right mechanistic
class: Dobrovodsky & Di Primo (2023) saw dn/dc fall 17% on K+-induced
G-quadruplex FOLDING, with no mass change at all.  A Cd(II) aptamer folds
around its ion (Liu et al. 2023 have the crystal structure), so that case,
not the L-tyrosinamide one, is the analogue.

Everything here is offline, free, and runs in seconds.
"""

from __future__ import annotations

import numpy as np

from config import StudyConfig
from recognition_layer import (
    SWEEP_DN_DC_ML_PER_G,
    SWEEP_RII_DEVIATION_PERCENT,
    SWEEP_THICKNESS_PERCENT,
    THICKNESS_CHANGE_LITERATURE,
    CONFORMATIONAL_CONTRACTION_PERCENT,
    RII_DEVIATION_PERCENT,
    DN_DC_ML_PER_G,
    estimate_index_change,
)


# ==========================================================================
#  The surface
# ==========================================================================
def conformational_surface(
    cfg: StudyConfig,
    n_thick: int = 41,
    n_rii: int = 41,
    thickness_range: tuple[float, float] = SWEEP_THICKNESS_PERCENT,
    rii_range: tuple[float, float] = SWEEP_RII_DEVIATION_PERCENT,
    dn_dc: float | None = None,
    polymer: str = "dna",
) -> dict:
    """
    delta_n over the (thickness change, RII deviation) plane.

    Returns the grid, the surface, the nominal point, and the location of the
    null contour where the two terms cancel exactly.
    """
    import recognition_layer as rl

    thick = np.linspace(*thickness_range, n_thick)
    rii = np.linspace(*rii_range, n_rii)

    # Optionally override the baseline dn/dc for this evaluation.
    original = DN_DC_ML_PER_G[polymer]
    if dn_dc is not None:
        DN_DC_ML_PER_G[polymer] = float(dn_dc)

    try:
        surface = np.zeros((n_thick, n_rii))
        for i, t in enumerate(thick):
            for j, r in enumerate(rii):
                est = estimate_index_change(
                    cfg.surface,
                    n_solvent=cfg.tissue.n0,
                    polymer=polymer,
                    conformational_thickness_change_percent=float(t),
                    rii_deviation_percent=float(r),
                )
                surface[i, j] = est.delta_n_with_conformation

        nominal = estimate_index_change(
            cfg.surface,
            n_solvent=cfg.tissue.n0,
            polymer=polymer,
            conformational_thickness_change_percent=(
                CONFORMATIONAL_CONTRACTION_PERCENT
            ),
            rii_deviation_percent=RII_DEVIATION_PERCENT,
        )
    finally:
        DN_DC_ML_PER_G[polymer] = original

    return {
        "thickness_percent": thick,
        "rii_percent": rii,
        "delta_n": surface,
        "dn_dc": dn_dc if dn_dc is not None else original,
        "nominal_thickness_percent": CONFORMATIONAL_CONTRACTION_PERCENT,
        "nominal_rii_percent": RII_DEVIATION_PERCENT,
        "nominal_delta_n": nominal.delta_n_with_conformation,
        "delta_n_min": float(surface.min()),
        "delta_n_max": float(surface.max()),
        "crosses_zero": bool(surface.min() < 0 < surface.max()),
        "fraction_negative": float((surface < 0).mean()),
        "span_ratio": (
            float(surface.max() / surface.min())
            if surface.min() != 0 else float("inf")
        ),
    }


def null_contour(surf: dict) -> list[tuple[float, float]]:
    """
    The curve in the (thickness, RII) plane where delta_n = 0 exactly.

    Found by linear interpolation along each RII column.  An empty list means
    the design never nulls anywhere in the swept region, which is the good
    outcome and worth stating as such.
    """
    thick = surf["thickness_percent"]
    rii = surf["rii_percent"]
    d = surf["delta_n"]

    pts = []
    for j, r in enumerate(rii):
        col = d[:, j]
        sign_change = np.where(np.sign(col[:-1]) * np.sign(col[1:]) < 0)[0]
        for i in sign_change:
            t0, t1 = thick[i], thick[i + 1]
            v0, v1 = col[i], col[i + 1]
            t_star = t0 + (t1 - t0) * (0.0 - v0) / (v1 - v0)
            pts.append((float(t_star), float(r)))
    return pts


def distance_to_null(surf: dict) -> dict:
    """
    How close is the nominal design to reading zero?

    Reported in percentage points of the thickness term, because that is the
    axis with real physical uncertainty -- the sign is not even known.
    """
    pts = null_contour(surf)
    if not pts:
        return {
            "nulls": False,
            "margin_percentage_points": float("inf"),
            "verdict": (
                "The design does not null anywhere in the swept region. "
                "delta_n keeps the same sign across the whole plausible "
                "range of both conformational terms, so the transduction "
                "direction is robust even though the magnitudes are not."
            ),
        }

    t_nom = surf["nominal_thickness_percent"]
    r_nom = surf["nominal_rii_percent"]
    best = min(pts, key=lambda p: abs(p[0] - t_nom) + abs(p[1] - r_nom))
    margin = abs(best[0] - t_nom)

    if margin < 5.0:
        verdict = (
            f"The nominal design sits {margin:.1f} percentage points from a "
            f"null (delta_n = 0 at a thickness change of {best[0]:+.1f}% "
            f"when the RII deviation is {best[1]:+.1f}%). That is inside the "
            f"uncertainty on a term whose SIGN is unknown for metal-ion "
            f"aptamers. The predicted signal is not safe: report the surface, "
            f"not the point."
        )
    elif margin < 15.0:
        verdict = (
            f"A null lies {margin:.1f} percentage points away in the "
            f"thickness term (at {best[0]:+.1f}% with RII {best[1]:+.1f}%). "
            f"Within the swept range but not adjacent to the nominal design. "
            f"Report the contour and the margin."
        )
    else:
        verdict = (
            f"The nearest null is {margin:.1f} percentage points away in the "
            f"thickness term. Comfortable, but the surface still spans a "
            f"large range in magnitude -- quote the range, not one number."
        )
    return {
        "nulls": True,
        "nearest_null": best,
        "margin_percentage_points": margin,
        "verdict": verdict,
    }


def dn_dc_sensitivity(cfg: StudyConfig,
                      dn_dc_range: tuple[float, float] = SWEEP_DN_DC_ML_PER_G,
                      n: int = 9) -> list[dict]:
    """
    The third swept quantity: the BASELINE dn/dc, which de Feijter's formula
    multiplies through and which therefore scales delta_n almost linearly.

    The literature spread is 0.183 to 0.268 cm^3/g -- a factor of 1.46 -- and
    it is driven by sequence, spacer and anchoring, not by the recognition
    event, so the dn/dc used should always be stated alongside delta_n.
    """
    rows = []
    for v in np.linspace(*dn_dc_range, n):
        s = conformational_surface(cfg, n_thick=3, n_rii=3, dn_dc=float(v))
        rows.append({
            "dn_dc": float(v),
            "delta_n_nominal": s["nominal_delta_n"],
        })
    base = rows[len(rows) // 2]["delta_n_nominal"]
    for r in rows:
        r["relative"] = (r["delta_n_nominal"] / base) if base else float("nan")
    return rows


# ==========================================================================
#  Reporting
# ==========================================================================
def print_conformational_sweep(cfg: StudyConfig) -> dict:
    """The whole transferability question, answered with a surface."""
    print("  The two conformational terms were measured on an L-TYROSINAMIDE")
    print("  aptamer. Cadmium is not L-tyrosinamide. Every quantified layer-")
    print("  thickness change in the literature, and what it was measured on:")
    print()
    print(f"  {'system':>34}  {'change':>8}  {'measured?':>9}")
    for name, e in THICKNESS_CHANGE_LITERATURE.items():
        # Entries without a numerical value (percent is None) are listed with
        # "n.r." (not reported) so the qualitative observation still shows.
        change = (f"{e['percent']:+7.1f}%" if e["percent"] is not None
                  else f"{'n.r.':>8}")
        print(f"  {name:>34}  {change}  "
              f"{'yes' if e['measured'] else 'fitted':>9}")
    print()
    print("  Two DNA aptamers binding small aromatic amines of near-identical")
    print("  mass, measured by the same method in the same paper, move in")
    print("  OPPOSITE directions. The only observation on a Cd(II) aptamer")
    print("  layer (DPI) reports stretching at low Cd(II) without a")
    print("  numerical value. So the sweep must cross zero.")
    print()

    surf = conformational_surface(cfg)
    print(f"  delta_n over thickness {surf['thickness_percent'][0]:+.0f}% to "
          f"{surf['thickness_percent'][-1]:+.0f}% and RII "
          f"{surf['rii_percent'][0]:+.0f}% to {surf['rii_percent'][-1]:+.0f}%:")
    print()
    print(f"    nominal (as configured)   {surf['nominal_delta_n']:+.4e}")
    print(f"    minimum over the surface  {surf['delta_n_min']:+.4e}")
    print(f"    maximum over the surface  {surf['delta_n_max']:+.4e}")
    print(f"    fraction of the surface with the opposite sign  "
          f"{100 * surf['fraction_negative']:.1f}%")
    print("    (that fraction is an AREA on a uniform grid, not a")
    print("     probability -- it weights the corners of the range as")
    print("     heavily as its centre. Read it as 'the opposite sign is")
    print("     a large part of the plausible region', nothing more.)")
    print()

    # A coarse text rendering of the surface, so its shape and sign structure
    # are visible without plotting 1681 numbers.
    _print_surface(surf)

    print()
    dist = distance_to_null(surf)
    for line in _wrap(dist["verdict"], 68):
        print(f"  {line}")

    print()
    print("  Baseline dn/dc, which multiplies straight through:")
    print(f"    {'dn/dc':>8}  {'delta_n':>12}  {'relative':>9}")
    for r in dn_dc_sensitivity(cfg):
        print(f"    {r['dn_dc']:8.3f}  {r['delta_n_nominal']:12.4e}  "
              f"{r['relative']:8.2f}x")
    print()
    print("  Report delta_n as a RANGE over this surface, since neither term")
    print("  has been measured for a metal-ion aptamer.")

    return {"surface": surf, "null": dist}


def _print_surface(surf: dict, n_rows: int = 12, n_cols: int = 7) -> None:
    """A small ASCII contour table -- enough to see the sign structure."""
    thick = surf["thickness_percent"]
    rii = surf["rii_percent"]
    d = surf["delta_n"]

    ri = np.linspace(0, len(rii) - 1, n_cols).astype(int)
    ti = np.linspace(0, len(thick) - 1, n_rows).astype(int)

    print("        RII deviation ->")
    header = "  thick   " + "".join(f"{rii[j]:+8.1f}%" for j in ri)
    print(header)
    for i in ti:
        cells = "".join(f"{d[i, j]:+9.2e}"[:9] for j in ri)
        print(f"  {thick[i]:+6.1f}% {cells}")
    print("  (delta_n; a sign change anywhere in this table is a null "
          "contour)")


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
    print_conformational_sweep(StudyConfig())
