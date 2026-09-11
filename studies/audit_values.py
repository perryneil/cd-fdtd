#!/usr/bin/env python3
"""
audit_values.py -- recompute the paper's derived numbers and look for them in
the manuscript source. Runs offline, calls no solver.

    python studies/audit_values.py --tex path/to/manuscript.tex --supp path/to/supplementary.tex

Every derived quantity (water shift, surface sensitivity, signal in tissue,
effective noise, S/N and required site counts) is computed from the measured
values stored in `redesign.MEASURED`, `redesign.MEASURED_LAYER_SERIES` and
`redesign.MEASURED_TISSUE_SIGNAL`. The script then lists each value and reports
whether a number within tolerance appears in the manuscript or the supplement.

Without the .tex files it still prints the derived values, which is useful on
its own as a summary of the results.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

import _bootstrap
_bootstrap.setup()


def derived():
    from redesign import (MEASURED, MEASURED_LAYER_SERIES,
                          MEASURED_TISSUE_SIGNAL, N_SIGMA, USABLE_THETA,
                          effective_noise_pm, tissue_signal_pm)

    sig = MEASURED["speckle_pm"]
    dna = MEASURED["dn_applied"]
    sb = MEASURED["s_bulk_nm_per_riu"]
    d = {}
    d["sigma_tissue_pm"] = sig
    d["s_bulk_nm_per_riu"] = sb
    d["bulk_step"] = 5.0e-3
    d["water_shift_pm"] = sb * d["bulk_step"] * 1000.0
    d["surface_sensitivity_nm_per_riu"] = (
        MEASURED_LAYER_SERIES[3.0]["layer_shift_pm"] / 1000.0) / dna

    def signal(layer, volume=False):
        row = MEASURED_LAYER_SERIES[layer]
        dn = MEASURED_LAYER_SERIES[3.0]["dn_full"] if volume else row["dn_full"]
        return row["layer_shift_pm"] * (dn / dna)

    # The detection limit pairs the signal measured on tissue with the
    # effective noise. The signal measured in water is also reported, because
    # the paper builds the tissue value from it term by term.
    gain = 1.0 + MEASURED_TISSUE_SIGNAL["offset_vs_water"]
    d["signal_3nm_water_pm"] = signal(3.0)
    d["signal_3nm_pm"] = tissue_signal_pm()
    d["effective_noise_pm"] = effective_noise_pm()
    d["signal_7nm_outer_pm"] = signal(7.0) * gain
    d["signal_7nm_volume_pm"] = signal(7.0, True) * gain
    eff = d["effective_noise_pm"]
    d["snr_3nm"] = d["signal_3nm_pm"] / eff
    d["snr_7nm_volume"] = d["signal_7nm_volume_pm"] / eff
    d["shortfall_3nm"] = N_SIGMA * eff / d["signal_3nm_pm"]

    def need(s, th):
        return math.ceil((N_SIGMA * eff / (th * s)) ** 2)

    d["N_full_3nm"] = need(d["signal_3nm_pm"], 1.0)
    d["N_design_3nm"] = need(d["signal_3nm_pm"], USABLE_THETA)
    d["N_design_7nm_volume"] = need(d["signal_7nm_volume_pm"], USABLE_THETA)
    d["N_design_7nm_outer"] = need(d["signal_7nm_outer_pm"], USABLE_THETA)
    d["f_3nm"] = MEASURED_LAYER_SERIES[3.0]["f_ratio"]
    d["f_7nm"] = MEASURED_LAYER_SERIES[7.0]["f_ratio"]
    return d


#: (value, label, tolerance in the same unit). Checked against both .tex files.
def checks(d):
    return [
        (d["sigma_tissue_pm"],            "sigma_tissue (pm)",            1.0),
        (d["s_bulk_nm_per_riu"],          "S_bulk (nm/RIU)",              0.2),
        (d["water_shift_pm"],             "water shift at 5e-3 (pm)",     1.0),
        (d["surface_sensitivity_nm_per_riu"], "surface sensitivity (nm/RIU)", 1.0),
        (d["signal_3nm_water_pm"],        "signal, 3 nm, water (pm)",     2.0),
        (d["signal_3nm_pm"],              "signal, 3 nm, tissue (pm)",    2.0),
        (d["effective_noise_pm"],         "effective noise (pm)",         1.0),
        (d["signal_7nm_outer_pm"],        "signal, 7 nm outer face (pm)", 2.0),
        (d["signal_7nm_volume_pm"],       "signal, 7 nm volume (pm)",     3.0),
        (d["snr_3nm"],                    "S/N, 3 nm",                    0.05),
        (d["shortfall_3nm"],              "shortfall vs 3 sigma",         0.02),
        (d["f_3nm"],                      "f at 3 nm",                    0.01),
        (d["f_7nm"],                      "f at 7 nm",                    0.01),
        (d["N_full_3nm"],                 "sites, full coverage",         0),
        (d["N_design_3nm"],               "sites, design target",         0),
        (d["N_design_7nm_volume"],        "sites, 7 nm volume",           0),
        (d["N_design_7nm_outer"],         "sites, 7 nm outer face",       0),
    ]


NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers_in(path):
    if not os.path.isfile(path):
        return set(), 0
    txt = open(path, encoding="utf-8", errors="ignore").read()
    txt = txt.split(r"\section*{References}")[0]
    return {float(m) for m in NUM.findall(txt)}, len(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default=os.path.join("..", "manuscript.tex"),
                    help="Path to manuscript.tex. Relative paths are taken "
                         "from the cdfdtd/ package directory.")
    ap.add_argument("--supp", default=os.path.join("..", "supplementary.tex"),
                    help="Path to supplementary.tex, resolved the same way.")
    args = ap.parse_args()

    d = derived()
    print()
    print("DERIVED FROM redesign.py")
    print("=" * 68)
    for k, v in d.items():
        print(f"  {k:34s} {v:12.4f}" if isinstance(v, float)
              else f"  {k:34s} {v:12d}")

    pools = {}
    for label, path in (("manuscript", args.tex), ("supplementary", args.supp)):
        nums, n = numbers_in(path)
        pools[label] = nums
        print(f"\n  {label:14s} {path}   ({n} chars scanned, "
              f"{len(nums)} distinct numbers)")

    print()
    print("PRESENCE CHECK  (is the correct value quoted anywhere?)")
    print("=" * 68)
    bad = 0
    for value, label, tol in checks(d):
        hits = []
        for name, pool in pools.items():
            if any(abs(p - value) <= max(tol, 0.5 * 10 ** -6) for p in pool):
                hits.append(name[:4])
        mark = "ok " if hits else "?? "
        if not hits:
            bad += 1
        print(f"  {mark} {label:32s} {value:10.3f}   "
              f"{'/'.join(hits) if hits else 'NOT FOUND in either file'}")

    print()
    if bad:
        print(f"  {bad} value(s) not found. Check the manuscript against the "
              "values above.")
    else:
        print("  Every derived value was found in the manuscript or supplement.")
