#!/usr/bin/env python3
"""Training and held-out loss over steps for the gap runs, one panel each,
stacked so neither is squeezed. A run can be cut at a step (file:label:cut)
to drop a continuation that is not part of the reported result, and one
later segment of the log (a resumed run) can be drawn dashed on top with
file:label:cut:segN, where N counts the segments of the log from 0 (a new
segment starts wherever the step count goes backwards).

    python klein/plot_gap_curves.py h200_pilot/gap5.log:N=5 h200_pilot/gap40.log:N=40:60000:seg2 ... --out=h200_pilot/val_gap_runs.png
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_here, ".."), os.path.join(_here, "..", "h200_pilot")]   # plot_val.py
from plot_val import read_val_lines  # noqa: E402

INK, MUTED, GRID, BG = "#1a1a19", "#6b675f", "#e2e0da", "#fcfcfb"
COLOURS = ["#2a78d6", "#eb6834", "#1f9e6e", "#eda100", "#7b5cc9", "#e87ba4"]


def segments(path):
    """The log's validation lines split into segments (resumes / re-prints)."""
    segs, cur = [], []
    for line in open(path):
        m = re.search(r"val step (\d+) fixed_train ([\d.]+) fixed_heldout ([\d.]+)", line)
        if not m:
            continue
        row = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
        if cur and row[0] <= cur[-1][0]:
            segs.append(cur)
            cur = []
        cur.append(row)
    if cur:
        segs.append(cur)
    return segs


def main():
    specs = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), "val_gap_runs.png")
    note = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--note=")), "").replace("\\n", "\n")
    runs = []
    for i, spec in enumerate(specs):
        parts = spec.split(":")
        path, label = parts[0], parts[1] if len(parts) > 1 else os.path.basename(parts[0])
        cut = int(parts[2]) if len(parts) > 2 and parts[2] else None
        seg = int(parts[3][3:]) if len(parts) > 3 and parts[3].startswith("seg") else None
        rows = read_val_lines(path)
        if cut:
            rows = [r for r in rows if r[0] <= cut]
        extra = None
        if seg is not None:
            sg = segments(path)
            if seg < len(sg) and len(sg[seg]) > 1:
                extra = tuple(np.array(c) for c in zip(*sg[seg]))
        if rows:
            s, tr, ho = (np.array(c) for c in zip(*rows))
            runs.append((label, COLOURS[i % len(COLOURS)], s, tr, ho, extra))
    if not runs:
        raise SystemExit("nothing to plot")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=200, sharex=True)
    fig.patch.set_facecolor(BG)
    halo = [pe.withStroke(linewidth=3, foreground=BG)]
    xmax = max(max(r[2].max(), r[5][0].max() if r[5] is not None else 0) for r in runs)
    for ax, idx, title in ((axes[0], 4, "Held-out objects: the same 32 fixed frames (14 unseen objects) in every run"),
                           (axes[1], 3, "Training objects: 32 fixed frames of each run's own training set")):
        ax.set_facecolor(BG)
        ends = []
        for label, colour, s, tr, ho, extra in runs:
            y = ho if idx == 4 else tr
            ax.plot(s, y, color=colour, linewidth=1.8, label=label, zorder=3)
            x_end, y_end = s[-1], y[-1]
            if extra is not None:                 # a resumed segment, drawn as part of the run
                es, etr, eho = extra
                ey = eho if idx == 4 else etr
                keep = es > s[-1]                 # only the steps beyond the base run
                ax.plot(es[keep], ey[keep], color=colour, linewidth=1.8, zorder=3)
                x_end, y_end = es[-1], ey[-1]
            if idx == 4:
                b = int(np.argmin(y))
                ax.plot(s[b], y[b], "o", color=colour, markersize=8, fillstyle="none", zorder=4)
                ends.append((x_end, y[b], "%s  best %.4f at %dk" % (label, y[b], s[b] / 1000), colour))
            else:
                ends.append((x_end, y_end, "%s  %.4f" % (label, y_end), colour))
        lo, hi = ax.get_ylim()
        gap = 0.045 * (hi - lo)
        e = sorted(ends, key=lambda r: r[1])
        ys = [r[1] for r in e]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        for (x, y, label, colour), yl in zip(e, ys):
            ax.text(x + 0.01 * xmax, yl, label, color=colour, fontsize=9, va="center",
                    fontweight="600", path_effects=halo)
        ax.set_xlim(0, xmax * 1.28)
        ax.set_title(title, color=INK, fontsize=11, fontweight="600", loc="left")
        ax.set_ylabel("diffusion loss (fixed set)", color=MUTED, fontsize=9.5)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
    axes[1].set_xlabel("training step", color=MUTED, fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    fig.suptitle("Gap runs: training and held-out loss over steps (rank 128, regularisers off, one recipe)",
                 color=INK, fontsize=13, fontweight="700", x=0.01, ha="left")
    if note:
        fig.text(0.01, 0.005, note, color=MUTED, fontsize=8.5, va="bottom")
    fig.tight_layout(rect=(0, 0.06 if note else 0, 1, 0.96))
    fig.savefig(out, facecolor=BG)
    print("->", out)


if __name__ == "__main__":
    main()
