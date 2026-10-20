#!/usr/bin/env python3
"""Training and held-out loss curves from a run trained with --val-every.

Reads val_lines.txt ("val step N fixed_train X fixed_heldout Y", grepped from
the run's log) and plots both fixed-set losses against the training step.
Both are measured on a fixed set of frames with fixed noise and timesteps,
so unlike the per-step training loss they are comparable across steps and
with each other.

    python plot_val.py [val_lines.txt] [title]      -> val_curves.png
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

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID, BG = "#1a1a19", "#6b675f", "#e2e0da", "#fcfcfb"
TRAIN, HELD = "#2a78d6", "#eb6834"          # categorical slots 1-2, validated


def read_val_lines(src):
    """(step, train, held-out) tuples, one per step, restart-aware.

    Logs of resumed or restarted runs contain steps that go backwards: a
    crashed run's lines followed by the continuation's. Whenever a step is
    not greater than the last one seen, the run restarted there, and every
    stored step from that point on is dropped in favour of the new values.
    The job's final re-print of its last few lines is handled the same way.
    """
    segs, cur = [], []
    for line in open(src):
        m = re.search(r"val step (\d+) fixed_train ([\d.]+) fixed_heldout ([\d.]+)", line)
        if not m:
            continue
        row = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
        if cur and row[0] <= cur[-1][0]:      # steps went backwards: a new segment
            segs.append(cur)
            cur = []
        cur.append(row)
    if cur:
        segs.append(cur)
    rows, top = {}, -1
    for seg in segs:
        if seg[-1][0] > top:                  # extends the run: a restart or continuation
            # its own steps win over older values (a crashed segment re-run
            # from a checkpoint); steps it does not contain are kept, since a
            # log may open with a re-print of a few early lines and then jump
            # to where the continuation really starts
            rows.update({r[0]: r for r in seg})
            top = seg[-1][0]
        else:                                 # a re-print: only fill gaps
            for r in seg:
                rows.setdefault(r[0], r)
    return [rows[k] for k in sorted(rows)]


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    opts = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
    src = args[0] if args else os.path.join(HERE, "val_lines.txt")
    title = args[1] if len(args) > 1 else "N=40: training and held-out loss"
    out = opts.get("out", os.path.join(HERE, "val_curves.png"))
    note = opts.get("note", "Note: not the per-step training loss plotted earlier. That one uses a new batch, "
             "noise sample and timestep at every step,\nso it swings between about 0.04 and 0.18 "
             "and hides the trend. Here every point uses the same frames, noise and timesteps, so "
             "changes\nreflect the model. Levels differ from the earlier plot (different timestep "
             "mix); compare trends, not values. Same recipe as K40pr.").replace("\\n", "\n")
    rows = read_val_lines(src)
    if not rows:
        raise SystemExit("no 'val step' lines in %s" % src)
    s, tr, ho = (np.array(c) for c in zip(*rows))

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    halo = [pe.withStroke(linewidth=3, foreground=BG)]
    for y, colour, label in ((tr, TRAIN, "training objects"),
                             (ho, HELD, "held-out objects")):
        ax.plot(s, y, color=colour, linewidth=2, label=label, zorder=3)
        ax.plot(s[-1], y[-1], "o", color=colour, markersize=5, zorder=4,
                markeredgecolor=BG, markeredgewidth=2)
        ax.annotate("%s  %.4f" % (label, y[-1]), (s[-1], y[-1]), xytext=(8, 0),
                    textcoords="offset points", color=colour, fontsize=9,
                    va="center", fontweight="600", path_effects=halo)
    best = int(np.argmin(ho))
    ax.plot(s[best], ho[best], "o", color=HELD, markersize=8, fillstyle="none",
            zorder=4)
    ax.annotate("lowest held-out, step %d" % s[best], (s[best], ho[best]),
                xytext=(0, 12), textcoords="offset points", color=MUTED,
                fontsize=8, ha="center", va="bottom", path_effects=halo)

    ax.set_title(title, color=INK, fontsize=13, fontweight="600", loc="left", pad=16)
    ax.text(0, 1.015, opts.get("sub", "loss on a fixed set of 32 frames per side, with fixed noise "
            "and timesteps; gap = overfitting"), transform=ax.transAxes,
            color=MUTED, fontsize=9, va="bottom")
    ax.set_xlabel("training step", color=MUTED, fontsize=9)
    ax.set_ylabel("diffusion loss (fixed set)", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlim(0, s.max() * 1.22)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="lower left",
              ncol=2, bbox_to_anchor=(0, -0.24))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.012, 0.018, note, color=MUTED, fontsize=7.5, va="bottom", ha="left",
             linespacing=1.4)
    fig.savefig(out, facecolor=BG)
    print("-> %s  (%d points; last: train %.4f, held out %.4f)"
          % (out, len(s), tr[-1], ho[-1]))


if __name__ == "__main__":
    main()
