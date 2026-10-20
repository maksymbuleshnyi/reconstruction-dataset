#!/usr/bin/env python3
"""Per-part colour fidelity: did each visible part get ITS colour?

For every rendered frame, the pack's part-id channel says which pixels
belong to which part. Inside each part's mask the mean colour of the render
is compared with the mean colour of the target (CIE Lab distance), and a
part counts as IDENTIFIED when its rendered CHROMA (Lab a, b: hue and
saturation, ignoring the lightness that shading changes) is closer to its own
true chroma than to that of any other part visible in the frame. The rate is
over parts that have a differently coloured neighbour, which is where "which
part is which" matters. The analytic render (albedo x shading) is scored the
same way as a reference row: it has the true colours by construction, so its
rate is the ceiling this metric can reach on these frames.

    python part_colour.py --data v5_colour --run runs/klein100b --tags early,early_swap,early_noref
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image


def srgb_to_lab(rgb):
    """rgb in 0..1, shape (..., 3) -> CIE Lab (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def part_masks(pack_path, n_parts, min_px=300):
    b = np.asarray(Image.open(pack_path).convert("RGB"))[:, :, 2].astype(np.float64)
    step = 255.0 / (n_parts + 1)
    k = np.clip(np.rint(b / step), 0, n_parts).astype(int)
    k[b < 0.5 * step] = 0
    out = {}
    for pid in np.unique(k):
        if pid == 0:
            continue
        m = k == pid
        if m.sum() >= min_px:
            out[int(pid)] = m
    return out


SIZE_BINS = [(0, 1000, "<1k px (<4 tokens)"), (1000, 4000, "1k-4k px"),
             (4000, 16000, "4k-16k px"), (16000, 10**9, ">16k px")]


def score(masks, true, got, des, ident, min_sep=15.0, by_size=None):
    ab = lambda v: v[1:]                          # chroma only
    for k in masks:
        des.append(float(np.linalg.norm(got[k] - true[k])))
        others = [j for j in masks if j != k
                  and np.linalg.norm(ab(true[j]) - ab(true[k])) >= min_sep]
        if others:
            own = np.linalg.norm(ab(got[k]) - ab(true[k]))
            best_other = min(np.linalg.norm(ab(got[k]) - ab(true[j])) for j in others)
            hit = float(own < best_other)
            ident.append(hit)
            if by_size is not None:
                area = int(masks[k].sum())
                for lo, hi, name in SIZE_BINS:
                    if lo <= area < hi:
                        by_size.setdefault(name, []).append(hit)


def analytic(row, data, gain=1.07):
    alb = np.asarray(Image.open(os.path.join(data, row["apr"])).convert("RGB"), np.float64) / 255
    sh = np.asarray(Image.open(os.path.join(data, row["ppr"])).convert("RGB"), np.float64)[:, :, 1] / 255
    return np.clip(alb * sh[:, :, None] * gain, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tags", default="early,early_swap,early_noref")
    ap.add_argument("--min-sep", type=float, default=15.0,
                    help="only count identification when the true colours of "
                         "the parts differ by at least this Lab distance")
    ap.add_argument("--by-size", action="store_true",
                    help="also print the identification rate per part size "
                         "(area of the part on screen at 512 px)")
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.data, "dataset", "index.json")))
    by = {(f["obj"], f["id"]): f for f in idx["frames"]}
    nparts = {o: v.get("n_parts") for o, v in idx["objects"].items()}

    print("%-14s %7s %8s %9s %9s" % ("condition", "frames", "parts", "dE mean", "ident %"))
    tags = a.tags.split(",")
    for tag in ["analytic:" + tags[0]] + tags:
        ref = tag.startswith("analytic:")
        gd = os.path.join(a.run, "gens_" + (tag.split(":")[1] if ref else tag))
        if not os.path.isdir(gd):
            print("%-14s missing" % tag)
            continue
        des, ident, frames, by_size = [], [], 0, ({} if a.by_size else None)
        for fn in sorted(os.listdir(gd)):
            if not fn.endswith(".png"):
                continue
            obj, _, fid = fn[:-4].rpartition("_")
            row = by.get((obj, fid))
            if row is None or not nparts.get(obj):
                continue
            masks = part_masks(os.path.join(a.data, row["ppr"]), nparts[obj])
            if not masks:
                continue
            tgt = np.asarray(Image.open(os.path.join(a.data, row["tgt"])).convert("RGB"),
                             np.float64) / 255
            ren = analytic(row, a.data) if ref else \
                np.asarray(Image.open(os.path.join(gd, fn)).convert("RGB")
                           .resize(tgt.shape[1::-1]), np.float64) / 255
            tl, rl = srgb_to_lab(tgt), srgb_to_lab(ren)
            true = {k: tl[m].mean(0) for k, m in masks.items()}
            got = {k: rl[m].mean(0) for k, m in masks.items()}
            frames += 1
            score(masks, true, got, des, ident, a.min_sep, by_size)
        print("%-14s %7d %8d %9.1f %9s" % (
            "analytic" if ref else tag, frames, len(des), float(np.mean(des)) if des else float("nan"),
            ("%.1f" % (100 * np.mean(ident))) if ident else "n/a"))
        if ident:
            print("%-14s %7s %8s %9s   (%d parts had a differently coloured neighbour)"
                  % ("", "", "", "", len(ident)))
        if by_size:
            print("%-14s %s" % ("  by part size", "   ".join(
                "%s: %.0f%% of %d" % (name, 100 * np.mean(by_size[name]), len(by_size[name]))
                for _, _, name in SIZE_BINS if name in by_size)))


if __name__ == "__main__":
    main()
