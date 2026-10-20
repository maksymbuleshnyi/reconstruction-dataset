#!/usr/bin/env python3
"""Where is the remaining error? LPIPS split by region of the frame.

LPIPS with spatial=True gives a per-pixel distance map; its mean is the
LPIPS value. Summing the map over three regions splits that value:

  geometry  where the blockout and the true object disagree (the object
            exists in the truth but not in the blockout, or the other way
            round), widened by 3 px: shape the map does not get right
  interior  inside parts where blockout and truth agree: surface detail,
            texture, gloss, small features
  rest      everything else in the crop (table, shadows)

Each frame is cropped to the object (truth + blockout, plus a margin) and
scored at 256 px, like metrics.py, so the totals are close to its LPIPS.
If most of the error sits in "geometry", better data or a better model
cannot remove it; the map has to carry more shape (finer blockout, a real
progress photo). If it sits in "interior", the model is not using the
photos' detail well enough.

    python klein/error_regions.py --data v5_colour --run runs/gap40b,runs/gap100 --tag onewalk --limit 300
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


def dilate(m, r):
    if r <= 0:
        return m
    im = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * r + 1))
    return np.asarray(im) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="run folders, comma-separated")
    ap.add_argument("--tag", default="onewalk")
    ap.add_argument("--limit", type=int, default=300, help="frames per run, evenly spread")
    ap.add_argument("--margin", type=int, default=16)
    a = ap.parse_args()
    import lpips
    net = lpips.LPIPS(net="alex", spatial=True, verbose=False)
    idx = json.load(open(os.path.join(a.data, "dataset", "index.json")))
    by = {(f["obj"], f["id"]): f for f in idx["frames"]}

    def t(x):
        return torch.from_numpy(x).permute(2, 0, 1)[None].float() * 2 - 1

    print("%-22s %6s %7s | %-27s | %-27s | %-21s" % ("run", "frames", "LPIPS",
          "geometry: share / area", "interior: share / area", "rest: share / area"))
    for run in a.run.split(","):
        gd = os.path.join(run, "gens_" + a.tag)
        if not os.path.isdir(gd):
            print("%-22s missing %s" % (os.path.basename(run), gd))
            continue
        files = sorted(f for f in os.listdir(gd) if f.endswith(".png"))
        if len(files) > a.limit:
            files = [files[i] for i in np.linspace(0, len(files) - 1, a.limit).round().astype(int)]
        tot, contrib, area, n = [], {"geo": [], "int": [], "rest": []}, {"geo": [], "int": [], "rest": []}, 0
        per_obj = {}
        for fn in files:
            obj, _, fid = fn[:-4].rpartition("_")
            row = by.get((obj, fid))
            if row is None:
                continue
            tgt_im = Image.open(os.path.join(a.data, row["tgt"])).convert("RGB")
            W, H = tgt_im.size
            tgt = np.asarray(tgt_im, np.float32) / 255
            gen = np.asarray(Image.open(os.path.join(gd, fn)).convert("RGB").resize((W, H)), np.float32) / 255
            mask = np.asarray(Image.open(os.path.join(a.data, row["mask"])).convert("L").resize((W, H))) > 127
            sil = np.asarray(Image.open(os.path.join(a.data, row["ppr"])).convert("RGB").resize((W, H), Image.NEAREST))[:, :, 0] > 1
            union = mask | sil
            if union.sum() < 64:
                continue
            ys, xs = np.where(dilate(union, a.margin))
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = lambda z: z[y0:y1, x0:x1]
            g, tg = t(crop(gen)), t(crop(tgt))
            g = F.interpolate(g, size=(256, 256), mode="bilinear")
            tg = F.interpolate(tg, size=(256, 256), mode="bilinear")
            with torch.no_grad():
                m = net(g, tg)[0, 0].numpy()                          # (256, 256)
            rs = lambda z: np.asarray(Image.fromarray((crop(z) * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST)) > 0
            mk, sl = rs(mask), rs(sil)
            geo = dilate((mk & ~sl) | (sl & ~mk), 3)
            inner = mk & sl & ~geo
            rest = ~(geo | inner)
            total = float(m.mean())
            tot.append(total)
            for k, reg in (("geo", geo), ("int", inner), ("rest", rest)):
                contrib[k].append(float(m[reg].sum() / m.size))
                area[k].append(float(reg.mean()))
            d = per_obj.setdefault(obj, [])
            d.append(total)
            n += 1
        if not n:
            continue
        T = np.mean(tot)
        cell = lambda k: "%5.1f%% of LPIPS / %4.1f%%" % (100 * np.mean(contrib[k]) / T, 100 * np.mean(area[k]))
        print("%-22s %6d %7.4f | %-27s | %-27s | %-21s" % (os.path.basename(run), n, T, cell("geo"), cell("int"), cell("rest")))
        worst = sorted(per_obj, key=lambda o: -np.mean(per_obj[o]))
        print("%-22s per object: %s" % ("", "  ".join("%s %.3f" % (o[:8], np.mean(per_obj[o])) for o in worst)))


if __name__ == "__main__":
    main()
