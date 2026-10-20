#!/usr/bin/env python3
"""The gap curve: every metric, train and held-out, against training-set size.

Reads the logs of job_gap.sbatch runs. For each run the best checkpoint is
the validation point with the lowest held-out v-loss (the one the job
evaluated); its v-loss and image error are read from that line, LPIPS / L1
from the metrics rows, the per-part colour score from the part_colour rows.

    python klein/plot_gap.py h200_pilot/gap5.log h200_pilot/gap10.log ... --out h200_pilot/gap_curve.png
"""
from __future__ import annotations

import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

INK, MUTED, GRID, BG = "#1a1a19", "#6b675f", "#e2e0da", "#fcfcfb"
TRAIN, HELD, SWAP, NONE, CEIL = "#2a78d6", "#eb6834", "#9a9590", "#c9c4bd", "#1f9e6e"


def parse(path):
    txt = open(path, errors="replace").read()
    d = {"path": path}
    m = re.search(r"\[data\] train: (\d+) frames over (\d+) objects", txt)
    if not m:
        return None
    d["frames"], d["N"] = int(m.group(1)), int(m.group(2))
    vals = [tuple(float(x) for x in v) for v in re.findall(
        r"^val step (\d+) fixed_train ([\d.]+) fixed_heldout ([\d.]+) x0_train ([\d.]+) x0_heldout ([\d.]+)", txt, re.M)]
    if not vals:
        return None
    best = min(vals, key=lambda v: v[2])            # lowest held-out v-loss, as the job selects
    d["step"], d["v_tr"], d["v_ho"], d["x0_tr"], d["x0_ho"] = best
    d["passes"] = d["step"] * 2 / d["frames"]
    d["last_step"] = vals[-1][0]
    for key, label in (("tr", "trained on"), ("own", r"held out \(own refs\)"), ("swap", r"held out \(SWAPPED refs\)"),
                       ("none", r"held out \(NO refs\)"), ("e_own", r"steps 1-3 \(own refs\)")):
        # the last block wins: a log may hold a superseded evaluation
        ms = re.findall(r"^\S+\s+%s\s+model\s+-\s+([\d.]+)\s+([\d.]+)" % label, txt, re.M)
        if ms:
            d["lpips_" + key], d["l1_" + key] = float(ms[-1][0]), float(ms[-1][1])
    for key, label in (("own", "early"), ("swap", "early_swap"), ("none", "early_noref"), ("ceil", "analytic")):
        ms = re.findall(r"^%s\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)" % label, txt, re.M)
        if ms:
            d["ident_" + key] = float(ms[-1][1])
    d["done"] = "GAP COMPLETE" in txt
    return d


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), "gap_curve.png")
    runs = [r for r in (parse(p) for p in args) if r]
    runs.sort(key=lambda r: r["N"])
    if not runs:
        raise SystemExit("no parsable gap logs")
    for r in runs:
        print("N=%3d  frames %5d  best step %6d (%.0f passes%s)  v %.4f/%.4f  x0 %.4f/%.4f  lpips %s/%s  ident %s"
              % (r["N"], r["frames"], r["step"], r["passes"], "" if r["done"] else ", RUNNING",
                 r["v_tr"], r["v_ho"], r["x0_tr"], r["x0_ho"],
                 r.get("lpips_tr", "-"), r.get("lpips_own", "-"), r.get("ident_own", "-")))
    Ns = np.array([r["N"] for r in runs])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4), dpi=200)
    fig.patch.set_facecolor(BG)
    halo = [matplotlib.patheffects.withStroke(linewidth=3, foreground=BG)]

    def line(ax, key, colour, label, style="-", marker="o"):
        xs = [r["N"] for r in runs if key in r]
        ys = [r[key] for r in runs if key in r]
        if not xs:
            return
        ax.plot(xs, ys, style, color=colour, marker=marker, linewidth=2, markersize=5, label=label)
        ax.annotate("%.3f" % ys[-1] if ys[-1] < 10 else "%.0f" % ys[-1], (xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", color=colour, fontsize=8, va="center",
                    fontweight="600", path_effects=halo)

    panels = [
        (axes[0, 0], "diffusion loss (v) at the best checkpoint", [("v_tr", TRAIN, "training frames", "-"), ("v_ho", HELD, "held-out objects", "-")]),
        (axes[0, 1], "image error (x0) at the best checkpoint", [("x0_tr", TRAIN, "training frames", "-"), ("x0_ho", HELD, "held-out objects", "-")]),
        (axes[0, 2], "LPIPS, lower is better", [("lpips_tr", TRAIN, "trained on, own photos", "-"), ("lpips_own", HELD, "held out, own photos", "-"),
                                              ("lpips_swap", SWAP, "held out, other object's photos", "--"), ("lpips_none", NONE, "held out, no photos", ":")]),
        (axes[1, 0], "L1, lower is better", [("l1_tr", TRAIN, "trained on", "-"), ("l1_own", HELD, "held out, own photos", "-"),
                                           ("l1_swap", SWAP, "other object's photos", "--"), ("l1_none", NONE, "no photos", ":")]),
        (axes[1, 1], "parts in own colour, steps 1-3, held out (%)", [("ident_ceil", CEIL, "ceiling (true colours)", "--"), ("ident_own", HELD, "own photos", "-"),
                                                                     ("ident_swap", SWAP, "other object's photos", "--"), ("ident_none", NONE, "no photos", ":")]),
        (axes[1, 2], "best checkpoint: passes over the data", [("passes", INK, "passes at best held-out", "-")]),
    ]
    for ax, title, series in panels:
        ax.set_facecolor(BG)
        for key, colour, label, style in series:
            line(ax, key, colour, label, style)
        ax.set_xscale("log")
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("training objects", color=MUTED, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, fontweight="600", loc="left")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.legend(frameon=False, fontsize=7.5, loc="best")
    fig.suptitle("Train / held-out gap against training-set size (same recipe, best checkpoint per run)",
                 color=INK, fontsize=13, fontweight="700", x=0.01, ha="left")
    note = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--note=")), None)
    note = (note or "Setup per point: rank-128 LoRA (alpha 32), regularisers off, warm-up then cosine 3e-4 to 3e-5; the checkpoint with the "
            "lowest held-out v-loss is kept and evaluated.\nHeld-out = the same 14 unseen objects for every point; ~210 frames per training object "
            "(4 colourings). Runs marked RUNNING in the console output are not finished.").replace("\\n", "\n")
    fig.text(0.01, 0.005, note, color=MUTED, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, facecolor=BG)
    print("->", out)


if __name__ == "__main__":
    main()
