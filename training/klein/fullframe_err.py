#!/usr/bin/env python3
"""Generated frame vs real frame over the WHOLE image, pixels in [0, 1]:
L1 and RMSE, the way NeuralOS reports its frame errors, and with --lpips also
LPIPS (AlexNet) on the whole frame at full resolution. metrics.py scores only
a crop around the object, which makes our numbers much larger than a
whole-frame average over mostly empty table would be. The object's share of
the frame is printed alongside.

    python klein/fullframe_err.py --data v5_colour --run runs/gap40b --tags onewalk,ontrainwalk
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="run folders, comma-separated")
    ap.add_argument("--tags", default="onewalk,ontrainwalk")
    ap.add_argument("--lpips", action="store_true", help="also whole-frame LPIPS (needs torch + lpips)")
    a = ap.parse_args()
    lp = None
    if a.lpips:
        import torch
        import lpips
        net = lpips.LPIPS(net="alex", verbose=False)

        def lp(x, y):
            t = lambda z: torch.from_numpy(z).permute(2, 0, 1)[None] * 2 - 1
            with torch.no_grad():
                return float(net(t(x), t(y)).item())
    idx = json.load(open(os.path.join(a.data, "dataset", "index.json")))
    by = {(f["obj"], f["id"]): f for f in idx["frames"]}
    print("%-28s %6s %8s %8s %8s %9s" % ("run/tag", "frames", "L1", "RMSE", "LPIPS", "object %"))
    for run in a.run.split(","):
        for tag in a.tags.split(","):
            gd = os.path.join(run, "gens_" + tag)
            if not os.path.isdir(gd):
                print("%-28s missing" % (os.path.basename(run) + "/" + tag))
                continue
            l1, se, cover, lps = [], [], [], []
            for fn in sorted(os.listdir(gd)):
                if not fn.endswith(".png"):
                    continue
                obj, _, fid = fn[:-4].rpartition("_")
                row = by.get((obj, fid))
                if row is None:
                    continue
                tgt = Image.open(os.path.join(a.data, row["tgt"])).convert("RGB")
                t = np.asarray(tgt, np.float32) / 255
                try:
                    g = np.asarray(Image.open(os.path.join(gd, fn)).convert("RGB").resize(tgt.size), np.float32) / 255
                except OSError:
                    continue
                d = g - t
                if lp is not None:
                    lps.append(lp(g, t))
                l1.append(float(np.abs(d).mean()))
                se.append(float((d ** 2).mean()))
                m = np.asarray(Image.open(os.path.join(a.data, row["mask"])).convert("L").resize(tgt.size)) > 127
                cover.append(float(m.mean()))
            if l1:
                print("%-28s %6d %8.4f %8.4f %8s %8.1f%%" % (os.path.basename(run) + "/" + tag, len(l1),
                      np.mean(l1), np.sqrt(np.mean(se)), ("%.4f" % np.mean(lps)) if lps else "-",
                      100 * np.mean(cover)))


if __name__ == "__main__":
    main()
