#!/usr/bin/env python3
"""Invented content: how much does a render draw where there is nothing?

LPIPS is scored on a crop around the object (true mask plus blockout
silhouette), so extra parts drawn elsewhere in the frame are not counted.
This scores the rest of the frame: pixels farther than `--margin` from both
the true object and the blockout. There the target is empty table, so any
strong difference from the target is content the model invented.

  invented %   share of the off-object area where the render differs from the
               target by more than --thresh (max over RGB, 0..1)
  frames %     share of frames with more than 0.5% of their off-object area
               invented (a visible extra part)
  off L1       mean absolute difference off the object

    python klein/offobject.py --data v5_colour --run runs/basecheck --tags d4_g40,d4_g100
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="folder holding gens_<tag>; or several, comma-separated")
    ap.add_argument("--tags", required=True)
    ap.add_argument("--margin", type=int, default=15, help="pixels kept clear around object and blockout")
    ap.add_argument("--thresh", type=float, default=0.15)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.data, "dataset", "index.json")))
    by = {(f["obj"], f["id"]): f for f in idx["frames"]}
    print("%-34s %6s %10s %9s %8s" % ("run/tag", "frames", "invented%", "frames%", "off L1"))
    for run in a.run.split(","):
        for tag in a.tags.split(","):
            gd = os.path.join(run, "gens_" + tag)
            if not os.path.isdir(gd):
                print("%-34s missing" % (os.path.basename(run) + "/" + tag))
                continue
            inv, bad, l1, n = [], 0, [], 0
            for fn in sorted(os.listdir(gd)):
                if not fn.endswith(".png"):
                    continue
                obj, _, fid = fn[:-4].rpartition("_")
                row = by.get((obj, fid))
                if row is None:
                    continue
                tgt_im = Image.open(os.path.join(a.data, row["tgt"])).convert("RGB")
                W, H = tgt_im.size
                tgt = np.asarray(tgt_im, np.float32) / 255
                try:
                    gen = np.asarray(Image.open(os.path.join(gd, fn)).convert("RGB").resize((W, H)), np.float32) / 255
                except (FileNotFoundError, OSError):          # removed while listing, or a partial write
                    continue
                mask = np.asarray(Image.open(os.path.join(a.data, row["mask"])).convert("L").resize((W, H))) > 64
                sil = np.asarray(Image.open(os.path.join(a.data, row["ppr"])).convert("RGB").resize((W, H), Image.NEAREST))[:, :, 0] > 1
                union = Image.fromarray(((mask | sil) * 255).astype(np.uint8))
                near = np.asarray(union.filter(ImageFilter.MaxFilter(2 * a.margin + 1))) > 0
                off = ~near
                if off.sum() < 1000:
                    continue
                d = np.abs(gen - tgt).max(-1)
                frac = float((d[off] > a.thresh).mean())
                inv.append(frac)
                bad += frac > 0.005
                l1.append(float(np.abs(gen - tgt)[off].mean()))
                n += 1
            if n:
                print("%-34s %6d %9.2f%% %8.1f%% %8.4f" % (os.path.basename(run) + "/" + tag, n,
                      100 * np.mean(inv), 100 * bad / n, np.mean(l1)))


if __name__ == "__main__":
    main()
