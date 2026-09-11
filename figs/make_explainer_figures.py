#!/usr/bin/env python3
"""
make_explainer_figures.py -- explanatory figures for readers outside plasmonics.

    python figs/make_explainer_figures.py

Runs offline; no solver is called. It is also run by make_figures.py.

    fig_device.pdf    the sensor in plan view and the surface layer stack,
                      on a real nanometre scale
    fig_concept.pdf   how a resonance becomes a measurement, where the noise
                      comes from, and why averaging independent sites helps
    fig_summary.pdf   the headline numbers on one page, each marked as
                      measured, measured with an assumption, or uncertain

The labels on fig_summary are written out in fig_summary() below. Update them
if the stored measurements in cdfdtd/redesign.py change.
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight",
})

GOLD = "#C8A22A"
LAYER = "#4E79A7"
BRUSH = "#9DB8D2"
WATER = "#EAF1F7"
INK = "0.15"


def _find_root():
    for c in (os.path.dirname(HERE), os.path.join(os.path.dirname(HERE),
                                                  "cdfdtd"), os.getcwd()):
        if os.path.isfile(os.path.join(c, "redesign.py")):
            return c
    return None


# ==========================================================================
def fig_device():
    """What the sensor is, in plan and in cross-section, to scale."""
    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)

    # ---- (a) plan view: two triangles tip to tip ----------------------
    a = fig.add_subplot(gs[0])
    side, gap = 120.0, 30.0
    h = side * math.sqrt(3) / 2

    def tri(xtip, sign):
        return Polygon([[xtip, 0], [xtip + sign * h, side / 2],
                        [xtip + sign * h, -side / 2]],
                       closed=True, facecolor=GOLD, edgecolor=INK, lw=0.9)
    a.add_patch(tri(-gap / 2, -1))
    a.add_patch(tri(gap / 2, +1))

    a.annotate("", xy=(-gap / 2, 0), xytext=(gap / 2, 0),
               arrowprops=dict(arrowstyle="<->", lw=0.9, color="crimson"))
    a.text(0, 14, f"gap\n{gap:.0f} nm", ha="center", va="bottom",
           fontsize=7.5, color="crimson")
    a.annotate("", xy=(gap / 2, -side / 2 - 14),
               xytext=(gap / 2 + h, -side / 2 - 14),
               arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK))
    a.text(gap / 2 + h / 2, -side / 2 - 20, f"{side:.0f} nm",
           ha="center", va="top", fontsize=7.5)
    a.text(0, side / 2 + 20, "gold, 40 nm thick", ha="center", fontsize=8)
    a.set_xlim(-h - 34, h + 34); a.set_ylim(-side / 2 - 52, side / 2 + 40)
    a.set_aspect("equal"); a.axis("off")

    # ---- (b) the layer stack at the metal surface, magnified ----------
    #
    # Drawn on a real distance axis rather than to scale against the 40 nm
    # metal: at true relative scale a 3 nm layer beside a 120 nm triangle would
    # be invisible.
    b = fig.add_subplot(gs[1])
    b.add_patch(Rectangle((-6, 0), 6, 1, facecolor=GOLD, edgecolor=INK, lw=0.9))
    b.text(-3, 0.5, "gold", ha="center", va="center", fontsize=8.5)
    b.add_patch(Rectangle((0, 0), 3, 1, facecolor=LAYER, edgecolor=INK, lw=0.7))
    b.add_patch(Rectangle((3, 0), 4, 1, facecolor=BRUSH, edgecolor=INK, lw=0.7))
    water = Rectangle((7, 0), 25, 1, facecolor=WATER, edgecolor=INK, lw=0.7)
    b.add_patch(water)

    # Speckle stands for the random index texture of the tissue.  Ellipses,
    # not circles: the two axes are on wildly different data scales here, so a
    # circle of radius r in data units is not round and overflows the box.
    # Clipped to the water rectangle so none of it escapes into the labels.
    from matplotlib.patches import Ellipse
    rng = np.random.default_rng(5)
    for _ in range(130):
        e = Ellipse((rng.uniform(7.0, 32.0), rng.uniform(0.0, 1.0)),
                    width=rng.uniform(0.8, 2.2), height=rng.uniform(0.10, 0.26),
                    facecolor="#C3D5E4", edgecolor="none", alpha=0.85, zorder=2)
        e.set_clip_path(water)
        b.add_patch(e)

    # Labels for the two nanometre-thin layers go ABOVE, on leader lines and
    # at two different heights, because at this magnification the layers are
    # narrower than their own names.
    b.annotate("recognition layer, 3 nm", xy=(1.5, 1.03), xytext=(-6.3, 2.02),
               ha="left", va="center", fontsize=7.4, color=LAYER,
               arrowprops=dict(arrowstyle="-", lw=0.7, color=LAYER,
                               shrinkA=1, shrinkB=1))
    b.annotate("antifouling brush, 4 nm", xy=(5.0, 1.03), xytext=(6.6, 1.62),
               ha="left", va="center", fontsize=7.4, color="#4F7095",
               arrowprops=dict(arrowstyle="-", lw=0.7, color="#4F7095",
                               shrinkA=1, shrinkB=1))
    b.text(21.0, 1.10, "fish muscle: water with random index texture",
           ha="center", va="bottom", fontsize=7.6, color="#37546B")
    b.text(19.5, 0.5, "n = 1.35 on average", ha="center", va="center",
           fontsize=7.4, color="#2B4256", zorder=4)

    # distance axis
    b.plot([0, 0], [-0.62, 1.10], color=INK, lw=0.6, ls=":")
    for xt in (0, 5, 10, 15, 20, 25, 30):
        b.plot([xt, xt], [-0.07, 0.0], color=INK, lw=0.7)
        b.text(xt, -0.14, f"{xt}", ha="center", va="top", fontsize=7)
    b.text(15, -0.38, "distance from the gold surface (nm)", ha="center",
           va="top", fontsize=7.6)

    # the sensing depth, which is a RESULT and the point of the panel
    b.add_patch(Rectangle((0, -1.02), 12, 0.30, facecolor="crimson",
                          alpha=0.16, edgecolor="none"))
    b.add_patch(Rectangle((12, -1.02), 9, 0.30, facecolor="crimson",
                          alpha=0.07, edgecolor="none"))
    b.annotate("", xy=(0, -0.87), xytext=(21, -0.87),
               arrowprops=dict(arrowstyle="<->", lw=0.9, color="crimson"))
    b.text(10.5, -1.12, "the antenna senses this far out: 12-21 nm\n"
                        "(well past its own coatings, into the tissue)",
           ha="center", va="top", fontsize=7.4, color="crimson")

    b.set_xlim(-6.5, 32.5); b.set_ylim(-1.85, 2.25)
    b.axis("off")

    # Panel titles as figure text: panel (a) has a fixed aspect ratio, so its
    # axes box shrinks and an axes title would sit lower than panel (b)'s.
    fig.text(0.055, 0.965, "(a) One antenna, seen from above", fontsize=9)
    fig.text(0.505, 0.965, "(b) The surface, magnified", fontsize=9)

    fig.savefig(os.path.join(HERE, "fig_device.pdf"))
    plt.close(fig)


# ==========================================================================
def fig_concept(sigma_pm=485.4, signal_pm=1446.0, fwhm_nm=162.8,
                centre_nm=849.4):
    """How a resonance becomes a measurement, and where the noise comes from."""
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    wl = np.linspace(centre_nm - 260, centre_nm + 260, 900)

    def lor(c, w=fwhm_nm):
        return 1.0 / (1.0 + ((wl - c) / (w / 2)) ** 2)

    # ---- (1) the resonance and what a shift means ---------------------
    a = ax[0]
    a.plot(wl, lor(centre_nm), color=INK, lw=1.3)
    a.plot(wl, lor(centre_nm + 30), color="crimson", lw=1.3, ls="--")
    a.annotate("", xy=(centre_nm + 30, 1.06), xytext=(centre_nm, 1.06),
               arrowprops=dict(arrowstyle="->", lw=1.0, color="crimson"))
    a.text(centre_nm + 15, 1.12, "peak moves", ha="center", fontsize=7.2,
           color="crimson")
    a.set_ylim(0, 1.3); a.set_yticks([])
    a.set_xlabel("wavelength (nm)")
    a.set_title("1. The antenna has a peak.\nAnything near it that changes\n"
                "the local optics moves the peak.", fontsize=7.8, loc="left")

    # ---- (2) the two things that move it, to the same scale -----------
    b = ax[1]
    b.barh([1], [signal_pm], color="0.35", edgecolor=INK, lw=0.8, height=0.5)
    b.barh([0], [sigma_pm], color="0.78", edgecolor=INK, lw=0.8, height=0.5)
    b.errorbar(sigma_pm, 0, xerr=[[sigma_pm - 343], [583 - sigma_pm]],
               fmt="none", ecolor=INK, capsize=3, lw=0.9)
    b.axvline(3 * sigma_pm, color="crimson", ls="--", lw=1.0)
    b.text(3 * sigma_pm - 55, 1.72, "3x noise:\nthe bar to clear", ha="right",
           va="center", fontsize=7.0, color="crimson")
    # Values inside / beyond the bars, kept clear of the 3x line and of the
    # error bar, so nothing in this panel sits on top of anything else.
    b.text(signal_pm - 45, 1, f"{signal_pm:.0f} pm", va="center", ha="right",
           fontsize=7.5, color="w")
    b.text(660, 0, f"{sigma_pm:.0f} pm", va="center", ha="left", fontsize=7.5)
    b.set_yticks([0, 1])
    b.set_yticklabels(["noise:\ntissue texture", "signal:\ncadmium bound"],
                      fontsize=7.5)
    b.set_xlim(0, 1900); b.set_ylim(-0.55, 2.15)
    b.set_xlabel("how far the peak moves (pm)")
    b.set_title("2. The signal is bigger than the\nnoise, but not 3x bigger.\n"
                "One antenna is not enough.", fontsize=7.8, loc="left")

    # ---- (3) averaging over independent sites -------------------------
    c = ax[2]
    N = np.arange(1, 41)
    c.plot(N, 3 * sigma_pm / np.sqrt(N), color=INK, lw=1.3)
    c.axhline(signal_pm, color="0.35", lw=1.2, ls="-")
    c.text(40, signal_pm + 60, "signal", ha="right", fontsize=7.2, color="0.3")
    need = 17
    c.plot([need], [3 * sigma_pm / math.sqrt(need)], "o", ms=5,
           color="crimson", zorder=5)
    c.annotate(f"{need} sites", xy=(need, 3 * sigma_pm / math.sqrt(need)),
               xytext=(need + 5, 3 * sigma_pm / math.sqrt(need) + 420),
               fontsize=7.5, color="crimson",
               arrowprops=dict(arrowstyle="->", lw=0.8, color="crimson"))
    c.set_xlabel("number of independent antennas")
    c.set_ylabel("noise bar to clear (pm)")
    c.set_ylim(0, 1700)
    c.set_title("3. Noise averages down as $\\sqrt{N}$.\nSpread the antennas "
                "far enough\napart and the signal wins.", fontsize=7.8,
                loc="left")

    for x in ax:
        x.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_concept.pdf"))
    plt.close(fig)


# ==========================================================================
def fig_summary():
    """Every headline number on one page, each marked with its status."""
    rows = [
        ("What the tissue does to the reading", "482 pm",
         "measured, 34 tissue draws", 1),
        ("What full cadmium coverage would do", "1446 pm",
         "measured on tissue x assumed chemistry", 2),
        ("Signal divided by noise, one antenna", "2.98x",
         "needs 3x, so one antenna fails", 1),
        ("Antennas needed, spaced 15 um apart", "17",
         "a 60 x 60 um patch", 1),
        ("Detection limit at that array", "22 nM",
         "free cadmium, at the assumed affinity", 2),
        ("How deep the antenna senses", "12-21 nm",
         "range, not a single value", 3),
        ("Chemistry assumption, worst case", "-1092/+1606 pm",
         "the largest uncertainty in the paper", 3),
    ]
    status_colour = {1: "#2E7D32", 2: "#F9A825", 3: "#C62828"}
    status_text = {1: "measured", 2: "measured x assumption",
                   3: "uncertain"}

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axis("off")
    y = len(rows)
    ax.text(0.0, y + 0.75, "The whole paper in seven numbers", fontsize=11,
            fontweight="bold")
    for i, (what, val, note, st) in enumerate(rows):
        yy = y - i - 0.4
        ax.add_patch(FancyBboxPatch((0.0, yy - 0.34), 7.45, 0.72,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor="0.965", edgecolor="0.85",
                                    lw=0.7))
        ax.add_patch(Rectangle((0.0, yy - 0.34), 0.055, 0.72,
                               facecolor=status_colour[st], edgecolor="none"))
        ax.text(0.16, yy, what, va="center", fontsize=8.4)
        ax.text(4.78, yy, val, va="center", fontsize=9.2, fontweight="bold",
                ha="right")
        ax.text(4.94, yy, note, va="center", fontsize=7.2, color="0.35")
    for j, st in enumerate((1, 2, 3)):
        ax.add_patch(Rectangle((0.0 + j * 2.4, -0.55), 0.09, 0.2,
                               facecolor=status_colour[st], edgecolor="none"))
        ax.text(0.18 + j * 2.4, -0.45, status_text[st], fontsize=7.4,
                va="center")
    ax.set_xlim(-0.05, 7.5); ax.set_ylim(-0.8, y + 1.15)
    fig.savefig(os.path.join(HERE, "fig_summary.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_device();   print("  fig_device.pdf")
    fig_concept();  print("  fig_concept.pdf")
    fig_summary();  print("  fig_summary.pdf")
    print("\nExplanatory figures written to", HERE)
