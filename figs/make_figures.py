"""
make_figures.py -- regenerate every figure in the paper from the saved results.

    python figs/make_figures.py                      # PML ensemble (default)
    python figs/make_figures.py --results PATH.json  # another ensemble

Reads cdfdtd/data/ensemble_pml/seed_results.json and the measured values in
cdfdtd/redesign.py, and writes the PDFs into figs/. It also runs
make_explainer_figures.py. Nothing here calls Tidy3D, so it runs offline.
"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_package_root():
    """Locate the cdfdtd package directory, wherever figs/ has been placed.

    Works with figs/ beside the package (this repository's layout) or inside
    it, and from any working directory.
    """
    candidates = [
        os.path.dirname(HERE),                              # figs/ inside pkg
        os.path.join(os.path.dirname(HERE), "cdfdtd"),      # figs/ beside pkg
        os.getcwd(),
        os.path.join(os.getcwd(), "cdfdtd"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "redesign.py")):
            return c
    raise SystemExit(
        "Could not find the cdfdtd package (looked for redesign.py in:\n  "
        + "\n  ".join(candidates)
        + "\nRun this from the repository, or move figs/ beside cdfdtd/.")


ROOT = _find_package_root()
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight",
})


#  WHICH ENSEMBLE THE FIGURES ARE DRAWN FROM
#  -----------------------------------------
#  data/ensemble_pml/ holds the 34 tissue realizations run with a perfectly
#  matched layer (PML) boundary. These are the values used in the paper.
#  results/ holds the same realizations run with an adiabatic absorber, which
#  boundary_test.py shows does not converge. It is used only as a fallback
#  and for the boundary comparison in the supplementary material.
#
#  Pass --results PATH to draw from a different run.
ENSEMBLE_CANDIDATES = (
    os.path.join("data", "ensemble_pml", "seed_results.json"),
    os.path.join("results", "seed_results.json"),
)

_ENSEMBLE_PATH = None


def ensemble_path(override=None):
    """Resolve which seed_results.json the figures are built from."""
    if override:
        if not os.path.isfile(override):
            raise SystemExit(f"--results: no such file: {override}")
        return os.path.abspath(override)
    for rel in ENSEMBLE_CANDIDATES:
        for base in (ROOT, os.path.dirname(ROOT)):
            p = os.path.join(base, rel)
            if os.path.isfile(p):
                return p
    raise SystemExit(
        "No seed_results.json found. Looked for:\n  "
        + "\n  ".join(ENSEMBLE_CANDIDATES)
        + "\nRun the ensemble first, or pass --results PATH.")


def load_ensemble():
    """The 34 usable realizations, and the regression the pipeline performs."""
    global _ENSEMBLE_PATH
    if _ENSEMBLE_PATH is None:
        _ENSEMBLE_PATH = ensemble_path()
    path = _ENSEMBLE_PATH
    recs = json.load(open(path))
    good = [r for r in recs
            if r.get("fit_ok") and r.get("decay_ok") and r.get("resonance_um")]
    lam = np.array([r["resonance_um"] * 1000 for r in good])      # nm
    nbar = np.array([r["mean_index_local"] for r in good])
    slope, inter = np.polyfit(nbar, lam, 1)
    resid = lam - (slope * nbar + inter)
    # ddof=2: two parameters were fitted, matching ensemble.analyse_ensemble
    sigma = float(np.std(resid, ddof=2))
    return lam, nbar, resid, slope, inter, sigma


def fig_ensemble():
    lam, nbar, resid, slope, inter, sigma = load_ensemble()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9),
                           gridspec_kw={"width_ratios": [1.35, 1]})
    a = ax[0]
    a.plot(nbar, lam, "o", ms=4, mfc="none", mec="k", mew=0.9)
    xs = np.linspace(nbar.min(), nbar.max(), 50)
    a.plot(xs, slope * xs + inter, "-", color="0.35", lw=1.2)
    a.set_xlabel("Local mean refractive index")
    a.set_ylabel("Resonance wavelength (nm)")
    a.set_title(f"(a) 34 tissue realizations\nslope {slope:.0f} nm/RIU", fontsize=9)
    a.ticklabel_format(useOffset=False, axis="x")

    b = ax[1]
    b.hist(resid * 1000, bins=9, color="0.8", edgecolor="k", lw=0.7)
    for s in (1, -1):
        b.axvline(s * sigma * 1000, color="k", ls="--", lw=0.9)
    b.axvline(0, color="k", lw=0.8)
    b.set_xlabel("Residual resonance shift (pm)")
    b.set_ylabel("Count")
    b.set_title(f"(b) After removing bulk drift\n$\\sigma$ = {sigma*1000:.0f} pm",
                fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_ensemble.pdf"))
    plt.close(fig)
    return sigma


def fig_signal_noise():
    """Signal against noise, with the contraction assumption made visible."""
    import sys
    sys.path.insert(0, ROOT)
    from redesign import MEASURED, signal_vs_contraction

    sigma_pm = MEASURED["speckle_pm"]
    lo, hi = MEASURED["speckle_ci_pm"]
    signal_pm = MEASURED["signal_pm"]
    rows = signal_vs_contraction()
    mass_only = min(abs(r["signal_pm"]) for r in rows)

    fig, a = plt.subplots(figsize=(3.6, 2.9))
    vals = [signal_pm, sigma_pm]
    a.bar([0, 1], vals, color=["0.45", "0.8"], edgecolor="k", lw=0.8,
          width=0.55)
    a.errorbar(1, sigma_pm, yerr=[[sigma_pm - lo], [hi - sigma_pm]],
               fmt="none", ecolor="k", capsize=4, lw=0.9)
    a.axhline(3 * sigma_pm, color="k", ls="--", lw=0.9)
    a.text(1.42, 3 * sigma_pm + 40, "3$\\sigma$", fontsize=8, ha="right")
    # Vertical bar: the range the signal spans across published layer
    # responses. Its lower end is where the sensor stops responding.
    hi_sig = max(abs(r["signal_pm"]) for r in rows)
    a.plot([0, 0], [mass_only, hi_sig], color="k", lw=1.2,
           solid_capstyle="butt")
    for y in (mass_only, hi_sig):
        a.plot([-0.13, 0.13], [y] * 2, color="k", lw=1.2)
    a.annotate("range across\npublished layer\nresponses",
               (0.14, (mass_only + hi_sig) / 2), fontsize=6.8,
               va="center", color="0.2")
    for i, v in enumerate(vals):
        a.text(i, v + 45, f"{v:.0f} pm", ha="center", fontsize=9)
    a.set_xticks([0, 1])
    a.set_xticklabels(["Signal at full\ncadmium coverage",
                       "Tissue noise\n$\\sigma$"], fontsize=8.5)
    a.set_ylabel("Resonance shift (pm)")
    a.set_ylim(0, 2000)
    a.set_xlim(-0.6, 1.6)
    a.set_title("A single site falls short of 3$\\sigma$,\n"
                "and the signal itself is an assumption", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_signal_noise.pdf"))
    plt.close(fig)


def fig_floor():
    """Independent sites required, against target occupancy and against signal.

    Panel (a) plots the number of independent sites needed to reach N_SIGMA
    against the target occupancy theta. Panel (b) plots the same requirement
    against the full-coverage shift, for two thresholds: detecting a fully
    occupied layer (theta = 1) and resolving the design target
    (theta = USABLE_THETA). The shift is used directly because the layer to
    water response ratio f is not bounded by 1.

    Every label is computed from the stored measurements.
    """
    import sys
    sys.path.insert(0, ROOT)
    from redesign import (MEASURED, MEASURED_LAYER_SERIES,
                          MEASURED_TISSUE_SIGNAL, N_SIGMA, USABLE_THETA,
                          effective_noise_pm)

    # The detection limit pairs the signal measured on tissue with the tissue
    # variability, and includes the signal's own scatter in the noise.
    sig = effective_noise_pm()
    dn_applied = MEASURED["dn_applied"]
    tissue_gain = 1.0 + MEASURED_TISSUE_SIGNAL["offset_vs_water"]

    def signal_pm(layer_nm, volume_functionalised=False):
        row = MEASURED_LAYER_SERIES[layer_nm]
        dn = (MEASURED_LAYER_SERIES[3.0]["dn_full"] if volume_functionalised
              else row["dn_full"])
        # The 7 nm runs were measured in water. The tissue offset measured at
        # 3 nm is applied as a single multiplicative factor, since it cancels
        # in the ratio between layer configurations.
        return row["layer_shift_pm"] * (dn / dn_applied) * tissue_gain

    def need(sig_pm, theta):
        return (N_SIGMA * sig / (theta * sig_pm)) ** 2

    cases = [
        ("3 nm layer, as built", signal_pm(3.0), "k", "-"),
        ("7 nm brush, volume functionalised", signal_pm(7.0, True), "#2E7D32", "--"),
        ("7 nm brush, outer face only", signal_pm(7.0), "#C62828", ":"),
    ]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # ---- (a) sites required against target occupancy ------------------
    a = ax[0]
    th = np.linspace(0.05, 1.0, 400)
    for lab, sp, col, ls in cases:
        a.plot(th, need(sp, th), ls, color=col, lw=1.3, label=lab)
    a.set_yscale("log")
    a.set_xlim(0, 1.02)
    a.set_ylim(0.5, 2e3)
    a.set_xlabel("Target occupancy $\\theta$")
    a.set_ylabel("Independent sites required")
    a.axhline(1, color="0.6", lw=0.8)

    base = cases[0][1]
    n_full = need(base, 1.0)
    n_design = need(base, USABLE_THETA)
    a.plot([1.0], [n_full], "o", ms=5, color="k", zorder=5)
    a.annotate("full coverage:\n%d sites" % int(np.ceil(n_full)),
               xy=(1.0, n_full), xytext=(-10, 34),
               textcoords="offset points", fontsize=7.4, ha="right")
    a.plot([USABLE_THETA], [n_design], "s", ms=5, color="crimson", zorder=5)
    a.annotate("design target $\\theta$ = %g:\n%d sites"
               % (USABLE_THETA, int(np.ceil(n_design))),
               xy=(USABLE_THETA, n_design), xytext=(20, 22),
               textcoords="offset points", fontsize=7.4, color="crimson",
               arrowprops=dict(arrowstyle="-", lw=0.6, color="crimson"))
    a.set_title("(a) Sites needed depend on the target occupancy",
                fontsize=8.5, loc="left")
    a.legend(fontsize=6.6, loc="upper right", frameon=True, framealpha=1.0,
             edgecolor="none", facecolor="w", borderpad=0.3, handlelength=2.2)

    # ---- (b) sites required against full-coverage shift ----------------
    b = ax[1]
    x = np.linspace(300, 3000, 400)
    for theta, ls, lab in ((1.0, "-", "detect full coverage"),
                           (USABLE_THETA, "--",
                            "resolve $\\theta$ = %g" % USABLE_THETA)):
        b.plot(x, need(x, theta), ls, color="k", lw=1.3, label=lab)
    for lab, sp, col, _ls in cases:
        b.axvline(sp, color=col, lw=0.9, alpha=0.55)
        b.text(sp - 45, 0.62, "%.0f pm" % sp, rotation=90, fontsize=6.6,
               color=col, ha="right", va="bottom")
    b.set_yscale("log")
    b.set_ylim(0.5, 2e3)
    b.set_xlim(300, 3000)
    b.set_xlabel("Shift at full coverage (pm)")
    b.set_ylabel("Independent sites required")
    b.axhline(1, color="0.6", lw=0.8)
    b.set_title("(b) The three measured configurations", fontsize=8.5,
                loc="left")
    b.legend(fontsize=6.8, loc="upper right", frameon=True, framealpha=1.0,
             edgecolor="none", facecolor="w", borderpad=0.3)

    for axis in ax:
        axis.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_floor.pdf"))
    plt.close(fig)


def fig_contraction():
    """The detection floor as a function of the assumed layer contraction.

    Under 2% of the full-coverage index change comes from the cadmium mass.
    The rest comes from the assumed folding of the layer, so the signal is
    plotted across the published range of thickness changes. The shaded band
    is the elongation side, where the one direct observation of a cadmium
    aptamer layer (Xue et al. 2020, dual polarization interferometry) places
    the chemistry at low Cd.
    """
    import sys
    sys.path.insert(0, ROOT)
    from redesign import (MEASURED, MEASURED_TISSUE_SIGNAL,
                          effective_noise_pm, signal_vs_contraction)
    from recognition_layer import CONFORMATIONAL_CONTRACTION_PERCENT as C0

    # Same pairing as fig_floor: the signal measured on tissue against the
    # effective noise, so site counts agree with those in the text.
    sigma = effective_noise_pm()
    tissue_gain = 1.0 + MEASURED_TISSUE_SIGNAL["offset_vs_water"]
    pcts = np.linspace(-32.0, 30.0, 260)
    rows = signal_vs_contraction(percents=tuple(pcts))
    sig = np.array([r["signal_pm"] for r in rows]) * tissue_gain
    mag = np.abs(sig)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))

    a = ax[0]
    a.plot(pcts, sig, "-", color="k", lw=1.3)
    a.axhline(0.0, color="0.6", lw=0.8)
    a.axhline(3 * sigma, color="k", ls="--", lw=0.9)
    a.axhline(-3 * sigma, color="k", ls="--", lw=0.9)
    a.text(-33, 3 * sigma + 70, "$+3\\sigma$", fontsize=7.5, ha="left")
    a.text(-33, -3 * sigma - 190, "$-3\\sigma$", fontsize=7.5, ha="left")
    # Mark the assumed value and two published measurements for scale. The
    # points carry the same tissue correction as the curve.
    exact = {r["contraction_percent"]: r["signal_pm"] * tissue_gain
             for r in signal_vs_contraction(percents=(C0, -15.0, 29.0))}
    # Labels are placed in data coordinates in the empty lower-left and
    # upper-right regions, clear of the curve and the 3-sigma lines.
    for pct, tx, ty, ha in [(C0,    -1.0,  1770.0, "left"),
                            (-15.0,  5.0,   830.0, "left"),
                            (29.0,  -20.0, -1000.0, "left")]:
        lab = {C0: "assumed here\n(fitted, not measured)",
               -15.0: "dopamine aptamer\n(measured)",
               29.0: "serotonin aptamer\n(measured)"}[pct]
        a.plot([pct], [exact[pct]], "o", ms=4.5, color="k")
        a.annotate(lab, xy=(pct, exact[pct]), xytext=(tx, ty),
                   textcoords="data", fontsize=7.6, ha=ha, va="center",
                   arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45",
                                   shrinkA=2, shrinkB=4))
    a.axvspan(0.0, 34.0, color="0.92", zorder=0)
    a.annotate("Cd aptamer stretches at low Cd (Xue 2020)",
               (17.0, -1810), fontsize=7.4, ha="center", va="bottom",
               color="0.25")
    a.set_xlim(-34, 34)
    a.set_ylim(-1900, 1900)
    a.set_xlabel("Assumed thickness change on binding (%)")
    a.set_ylabel("Signal at full coverage (pm)")
    a.set_title("(a) The signal changes sign inside\nthe published range",
                fontsize=9)

    b = ax[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        need = (3.0 * sigma / (0.25 * mag)) ** 2
    b.plot(pcts, need, "-", color="k", lw=1.3)
    b.set_yscale("log")
    j0 = int(np.argmin(mag))
    b.axvline(pcts[j0], color="0.6", lw=0.9, ls=":")
    b.axvspan(0.0, 34.0, color="0.92", zorder=0)
    b.text(pcts[j0] + 1.2, 3e5, "blind", fontsize=7.5, color="0.25",
           rotation=90, va="top")
    # Site count at the assumed contraction, from the same signal and noise
    # pairing as the curve.
    n_here = int(np.ceil((3.0 * sigma / (0.25 * abs(exact[C0]))) ** 2))
    b.plot([C0], [n_here], "o", ms=4.5, color="k")
    b.annotate(f"assumed here\n{n_here} sites", (C0, n_here),
               textcoords="offset points", xytext=(7, 10), fontsize=7.6)
    b.set_xlim(-34, 34)
    b.set_ylim(10, 1e6)
    b.set_xlabel("Assumed thickness change on binding (%)")
    b.set_ylabel("Independent sites for $\\theta \\leq 0.25$")
    b.set_title("(b) The array size the assumption implies", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_contraction.pdf"))
    plt.close(fig)


def fig_array(outer_um=7.0):
    fig, a = plt.subplots(figsize=(3.5, 2.8))
    pitch = np.logspace(-1, 1.7, 300)
    a.plot(pitch, np.clip((pitch / outer_um) ** 2, 0, 1), "-", color="k", lw=1.3)
    a.axvspan(4, 10, color="0.85", zorder=0)
    a.text(6.2, 0.10, "measured\nouter scale", fontsize=7.0, ha="center",
           bbox=dict(fc="w", ec="none", pad=1.0))
    a.axvline(15, color="0.4", ls="--", lw=1.0)
    a.text(16.5, 0.42, "design pitch\n15 $\\mu$m", fontsize=7.0)
    a.axvspan(0.2, 1.0, color="0.93", zorder=0)
    a.text(0.45, 0.80, "dense\nmetasurface", fontsize=7.0, ha="center")
    a.set_xscale("log")
    a.set_xlabel("Site pitch ($\\mu$m)")
    a.set_ylabel("Fraction of sites that are independent")
    a.set_ylim(-0.03, 1.05)
    a.set_title("Averaging needs sparse spacing", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_array.pdf"))
    plt.close(fig)


def fig_methods():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))

    # ---------------- (a) geometry, true cross-section ----------------
    #
    # This is a SIDE view, so the bowtie appears as two rectangles, not as two
    # triangles (the top view is in fig_device). Cutting along the bowtie axis
    # through both apices gives, for each triangle, a rectangle of width equal
    # to the triangle height (120 * sqrt(3)/2 = 104 nm) and height equal to
    # the metal thickness (40 nm). Drawn to scale in nanometres.
    a = ax[0]
    NM_H = 120.0 * math.sqrt(3) / 2.0        # 103.9 nm, triangle height
    T_AU = 40.0                              # metal thickness
    GAP = 30.0                               # physical tip-to-tip distance
    COAT = 7.0                               # 3 nm recognition + 4 nm brush

    a.set_xlim(-152, 152)
    a.set_ylim(-100, 182)
    a.set_aspect("equal")
    a.axis("off")

    # substrate below, tissue above, interface at y = 0
    a.add_patch(Rectangle((-152, -100), 304, 100, fc="0.88", ec="k", lw=0.7))
    a.add_patch(Rectangle((-152, 0), 304, 168, fc="0.97", ec="k", lw=0.7,
                          ls=(0, (4, 2))))

    # Conformal coating, drawn first so the metal sits on top of it. It
    # narrows the aqueous gap from 30 nm to 16 nm, which is the geometry the
    # steric accessibility mask acts on.
    for sgn in (-1, +1):
        x0 = sgn * (GAP / 2 + NM_H) if sgn < 0 else sgn * GAP / 2
        a.add_patch(Rectangle((x0 - COAT if sgn < 0 else x0 - COAT, 0),
                              NM_H + 2 * COAT, T_AU + COAT,
                              fc="#9DB8D2", ec="none"))
    # gold
    a.add_patch(Rectangle((-GAP / 2 - NM_H, 0), NM_H, T_AU,
                          fc="0.35", ec="k", lw=0.7))
    a.add_patch(Rectangle((GAP / 2, 0), NM_H, T_AU,
                          fc="0.35", ec="k", lw=0.7))

    a.text(-152 + 7, 161, "tissue\n(random medium)", fontsize=7.5, va="top")
    a.text(35, -88, "fused silica substrate", ha="center", fontsize=8)
    a.text(-GAP / 2 - NM_H / 2, T_AU / 2, "gold", ha="center", va="center",
           fontsize=7.5, color="w")

    # the source is launched INSIDE the substrate, so it arrives from below
    a.annotate("", xy=(-105, -22), xytext=(-105, -82),
               arrowprops=dict(arrowstyle="-|>", lw=1.1))
    a.text(-98, -52, "plane wave,\nlaunched in\nthe substrate", fontsize=7.0,
           va="center", ha="left")

    # metal thickness
    a.annotate("", xy=(-GAP / 2 - NM_H - 13, 0),
               xytext=(-GAP / 2 - NM_H - 13, T_AU),
               arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.2"))
    a.text(-GAP / 2 - NM_H - 17, T_AU / 2, "40 nm", rotation=90, fontsize=7.0,
           ha="right", va="center")

    # gap, measured metal to metal
    a.annotate("", xy=(-GAP / 2, T_AU / 2), xytext=(GAP / 2, T_AU / 2),
               arrowprops=dict(arrowstyle="<->", lw=0.8, color="crimson"))
    a.annotate("gap 30 nm metal to metal\n(20 nm nominal before the\n"
               "tip fillet is applied)",
               xy=(0, T_AU / 2), xytext=(-72, 92), fontsize=7.0,
               color="crimson", ha="center", va="bottom",
               arrowprops=dict(arrowstyle="-", lw=0.6, color="crimson",
                               shrinkA=2, shrinkB=2))

    # the coating
    a.annotate("3 nm recognition layer on a\n4 nm antifouling brush, which\n"
               "narrows the aqueous gap to 16 nm",
               xy=(GAP / 2 + 45, T_AU + COAT * 0.6), xytext=(148, 152),
               fontsize=7.0, ha="right", va="top", color="#2B4256",
               arrowprops=dict(arrowstyle="-", lw=0.6, color="#4F7095",
                               shrinkA=2, shrinkB=2))

    a.text(-152, 182, "(a) Simulation geometry, side view", fontsize=9,
           va="bottom")

    # ---------------- (b) paired protocol ----------------
    b = ax[1]
    b.set_xlim(0, 10); b.set_ylim(0, 10); b.axis("off")

    def box(x, y, w, h, txt, fc="w", fs=7.0):
        b.add_patch(Rectangle((x, y), w, h, fc=fc, ec="k", lw=0.8))
        b.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
               fontsize=fs, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        b.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                    mutation_scale=8, lw=0.8, color="k"))

    box(0.0, 8.4, 10.0, 0.95, "Three runs, one geometry, one grid",
        fc="0.92", fs=7.6)

    # The three simulations, named for what changes in each.
    box(0.0, 5.5, 3.1, 2.3, "1. Reference\nlayer unbound\nwater $n$ = 1.333")
    box(3.45, 5.5, 3.1, 2.3, "2. Water stepped\nlayer unbound\nwater $n + \\Delta n$")
    box(6.9, 5.5, 3.1, 2.3, "3. Layer bound\nlayer bound\nwater $n$ = 1.333")

    arrow(1.55, 8.4, 1.55, 7.8)
    arrow(5.0, 8.4, 5.0, 7.8)
    arrow(8.45, 8.4, 8.45, 7.8)

    # Each difference is taken against run 1, so run 1 feeds both results.
    # Drawn as vertical drops from a shared rail so the connectors never cross.
    b.plot([1.55, 1.55], [5.5, 4.95], lw=0.8, color="k")
    b.plot([5.0, 5.0], [5.5, 4.95], lw=0.8, color="k")
    b.plot([8.45, 8.45], [5.5, 4.95], lw=0.8, color="k")
    b.plot([1.55, 8.45], [4.95, 4.95], lw=0.8, color="k")
    arrow(3.0, 4.95, 3.0, 4.25)
    arrow(7.0, 4.95, 7.0, 4.25)

    box(0.9, 2.75, 4.2, 1.5, "run 2 minus run 1\ngives $S_{\\mathrm{bulk}}$",
        fc="0.92", fs=7.4)
    box(4.9, 2.75, 4.2, 1.5, "run 3 minus run 1\ngives $S_{\\mathrm{surface}}$",
        fc="0.92", fs=7.4)

    box(1.2, 0.5, 7.6, 1.3,
        "$f = S_{\\mathrm{surface}} \\, / \\, "
        "(S_{\\mathrm{bulk}} \\, \\Delta n_{\\mathrm{layer}})$",
        fc="0.85", fs=8.2)
    arrow(3.0, 2.75, 4.2, 1.8)
    arrow(7.0, 2.75, 5.8, 1.8)
    b.text(0.0, 10.35, "(b) Paired protocol", fontsize=9, va="bottom")

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_methods.pdf"))
    plt.close(fig)


def fig_workflow():
    fig, b = plt.subplots(figsize=(6.6, 1.9))
    b.set_xlim(0, 20); b.set_ylim(0, 5); b.axis("off")
    steps = [("Calibrate\ntissue model", "$\\mu_s'$ to $l_c$ = 1645 nm"),
             ("Generate\n34 realizations", "10 nm voxel grid"),
             ("FDTD\nsimulation", "5 x 5 x 2.5 $\\mu$m"),
             ("Extract\nresonance", "whole line fit"),
             ("Regress out\nbulk drift", "local mean index"),
             ("Variability\n$\\sigma$ = 482 pm", "bootstrap CI")]
    w, gap = 2.85, 0.45
    for i, (t, s) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        b.add_patch(Rectangle((x, 1.6), w, 2.0, fc="w" if i < 5 else "0.85",
                              ec="k", lw=0.8))
        b.text(x + w / 2, 2.85, t, ha="center", va="center", fontsize=7.6)
        b.text(x + w / 2, 1.95, s, ha="center", va="center", fontsize=6.6,
               color="0.3")
        if i < len(steps) - 1:
            b.add_patch(FancyArrowPatch((x + w, 2.6), (x + w + gap, 2.6),
                                        arrowstyle="-|>", mutation_scale=8, lw=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_workflow.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    import sys as _sys

    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--results", default=None,
                     help="seed_results.json to draw from. Defaults to the "
                          "PML ensemble, falling back to the absorber run.")
    _a = _ap.parse_args()
    _ENSEMBLE_PATH = ensemble_path(_a.results)

    # Always report which ensemble the figures were drawn from, and warn if it
    # disagrees with the stored value used by the other figures.
    print(f"  ensemble source       {_ENSEMBLE_PATH}")
    _sys.path.insert(0, ROOT)
    from redesign import MEASURED as _M

    sigma = fig_ensemble()
    print(f"  fig_ensemble.pdf      sigma = {sigma*1000:.1f} pm")
    if abs(sigma * 1000 - _M["speckle_pm"]) > 1.0:
        print(f"\n  WARNING: this ensemble gives {sigma*1000:.1f} pm but "
              f"redesign.MEASURED['speckle_pm'] is {_M['speckle_pm']:.1f} pm.")
        print("  Every figure that reads MEASURED -- fig_signal_noise and")
        print("  fig_floor -- will disagree with fig_ensemble. Update "
              "MEASURED,")
        print("  or pass --results for the run MEASURED describes.\n")
    fig_signal_noise();  print("  fig_signal_noise.pdf")
    fig_floor();         print("  fig_floor.pdf")
    fig_array();         print("  fig_array.pdf")
    fig_contraction();   print("  fig_contraction.pdf")
    fig_methods();       print("  fig_methods.pdf")
    fig_workflow();      print("  fig_workflow.pdf")

    # The explanatory figures take no ensemble and call no solver, but their
    # labels carry measured numbers, so they are regenerated together with the
    # figures above to keep every figure consistent.
    try:
        import make_explainer_figures as _ex
    except ImportError:
        _sys.path.insert(0, HERE)
        import make_explainer_figures as _ex
    _ex.fig_device();    print("  fig_device.pdf")
    _ex.fig_concept();   print("  fig_concept.pdf")
    _ex.fig_summary();   print("  fig_summary.pdf")

    print(f"\nAll figures regenerated from {_ENSEMBLE_PATH}.")
