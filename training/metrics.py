#!/usr/bin/env python3
"""A metric set that compares renders fairly.

Everything reported so far measured the wrong thing, for three separate
reasons:

  * the objects are a single grey, so pixel L1 mostly scores average
    brightness, which a constant albedo times shading gets exactly right at
    zero cost. A flat grey blob lands within 3% of the full baseline.
  * the union region drags in the background, which is a colour photograph at
    48x the object's saturation. Neither predictor can know it, and a cruder
    blockout is charged more of it purely by having more spill.
  * gradient L1 saturates: once predicted edges stop overlapping the true
    ones, error stops growing however wrong the placement gets.

So this scores appearance INSIDE the object, where the background cannot
interfere, and scores geometry separately in the spill region, where no
segmentation of the render is needed. Every metric is reported for the model,
for the analytic baseline, and where meaningful against a no-information
floor, so no number stands without a bar under it.

APPEARANCE (object mask only)
  lpips     perceptual distance. The one that tracks what the pictures show.
  l1        brightness. Reported, not ranked on: it is what misled us.
  l1_tone   L1 after the render is rescaled to the target's mean brightness.
            The gap between l1 and l1_tone is how much of the error is a
            single scalar rather than wrong content.
  edge_f1   edges of the render against edges of the target, within a 2px
            tolerance, so it measures placement rather than magnitude and
            does not saturate the way gradient L1 does.
  grad_r    gradient error over the target's own gradient energy. 1.0 is what
            a blank image scores. Below 1 means real structure, above 1 means
            worse than predicting nothing.

GEOMETRY (spill region: blockout claims material, truth is background)
  spill_fix how much of the blockout's false material the render removed.
            0 = reproduced the blockout exactly, 1 = matched the truth.
            This is the number that says whether the render is bounded by the
            geometry it was handed.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

RUNS = (("g_ds", "cad", "exact CAD"), ("L_pr", "pr", "primitives"),
        ("L_bx", "bx", "boxes"))
TAGS = (("ontrain", "trained on"), ("one", "held out"))


def parse_runs(spec):
    """run:rung[:label[:objects.json]], comma separated.

    The optional objects file is the list the run was TRAINED on. Without it,
    "trained on" means the whole train split, which for an N=20 run is 100
    objects of which 80 were never seen - so the row would mostly be measuring
    generalisation, not fit."""
    out = []
    for part in spec.split(","):
        bits = [b.strip() for b in part.split(":")]
        if len(bits) == 2:
            bits.append(bits[0])
        if len(bits) == 3:
            bits.append("")
        out.append((bits[0], bits[1], bits[2], bits[3] or None))
    return tuple(out)


def pick_even(files, limit, allowed=None):
    """Up to `limit` renders spread evenly over objects.

    Taking the first N files in sorted order meant the first ~3 objects
    alphabetically, not the split: 80 files at 25 frames per object."""
    by = {}
    for fn in sorted(files):
        if not fn.endswith(".png"):
            continue
        obj = fn[:-4].rpartition("_")[0]
        if allowed is not None and obj not in allowed:
            continue
        by.setdefault(obj, []).append(fn)
    # Within an object, visit frames in a spread order (first, last, middle,
    # quarters, ...) rather than by id. Ids run along the capture session, so
    # taking the first few meant only the opening camera path of each object.
    def spread(v):
        n = len(v)
        seen, order, step = set(), [], n
        while step >= 1 and len(order) < n:
            for i in range(0, n, max(1, step)):
                if i not in seen:
                    seen.add(i)
                    order.append(v[i])
            step //= 2
        return order
    by = {o: spread(v) for o, v in by.items()}
    out, k = [], 0
    while len(out) < limit and any(len(v) > k for v in by.values()):
        for obj in sorted(by):
            if k < len(by[obj]) and len(out) < limit:
                out.append(by[obj][k])
        k += 1
    return out


def gmag(x):
    gx = np.abs(np.diff(x, axis=1))[:-1, :]
    gy = np.abs(np.diff(x, axis=0))[:, :-1]
    return (gx + gy).mean(2)


def dilate(b, r=2):
    """Binary dilation by r pixels, without scipy."""
    o = b.copy()
    for _ in range(r):
        o[1:, :] |= o[:-1, :]
        o[:-1, :] |= o[1:, :]
        o[:, 1:] |= o[:, :-1]
        o[:, :-1] |= o[:, 1:]
    return o


def erode(b, r=1):
    o = b.copy()
    for _ in range(r):
        o[1:, :] &= o[:-1, :]
        o[:-1, :] &= o[1:, :]
        o[:, 1:] &= o[:, :-1]
        o[:, :-1] &= o[:, 1:]
    return o


def edges(g, m, pct=88, floor=0.02):
    """Edge map inside the region.

    A percentile alone is not enough. Most of a smooth grey object has zero
    gradient, so the 88th percentile is frequently 0 and every pixel then
    counts as an edge - which scored a blank square a perfect 1.0 against a
    dotted one. The absolute floor is what makes the set mean "a real step in
    intensity" rather than "the top slice of whatever is here, including
    noise".
    """
    v = g[m]
    if v.size < 32:
        return np.zeros_like(g, bool)
    return (g >= max(float(np.percentile(v, pct)), floor)) & m


def edge_f1(pred, true, m, tol=2):
    # Erode first: the object's own silhouette is the strongest gradient in
    # the frame and both predictors get it roughly right, so leaving the rim
    # in measures outline agreement instead of interior detail.
    m = erode(m, 4)
    p, t = edges(pred, m), edges(true, m)
    if t.sum() < 4:
        return float("nan")          # the target has no detail here to find
    if p.sum() < 4:
        return 0.0                   # it does, and the prediction found none
    prec = float((p & dilate(t, tol)).sum()) / float(p.sum())
    rec = float((t & dilate(p, tol)).sum()) / float(t.sum())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/h200_pilot/v3_grey"))
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--gain", type=float, default=1.304)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--no-dreamsim", action="store_true")
    ap.add_argument("--tags", default=None,
                    help="eval tags to score as tag:label pairs, e.g. "
                         "'ontrain1k:trained on,one1k:held out'. Default is "
                         "the standard ontrain / one pair.")
    ap.add_argument("--run-list", default=None,
                    help="run:rung:label, comma separated. Defaults to the "
                         "v3 geometry ladder. (--runs is the directory those "
                         "run folders live in.)")
    ap.add_argument("--out", default="runs/metrics.json")
    a = ap.parse_args()

    lp = None
    if not a.no_lpips:
        import torch
        import torch.nn.functional as F
        import lpips
        net = lpips.LPIPS(net="alex")

        def lp(x, y):
            def t(z):
                z = torch.from_numpy(z).permute(2, 0, 1)[None]
                return F.interpolate(z * 2 - 1, size=(256, 256),
                                     mode="bilinear")
            with torch.no_grad():
                return float(net(t(x), t(y)).item())

    ds = None
    if not a.no_dreamsim:
        try:
            import torch
            from PIL import Image as PImage
            from dreamsim import dreamsim
            dsm, dspre = dreamsim(
                pretrained=True, device="cpu",
                cache_dir=os.path.expanduser("~/.dreamsim"))

            def ds(x, y):
                def t(z):
                    return dspre(PImage.fromarray(
                        (np.clip(z, 0, 1) * 255).astype(np.uint8)))
                with torch.no_grad():
                    return float(dsm(t(x), t(y)).item())
        except Exception as e:
            print("[dreamsim unavailable: %s]" % e, flush=True)
            ds = None

    ip = os.path.join(a.data, "index.json")
    if not os.path.exists(ip):
        ip = os.path.join(a.data, "dataset", "index.json")
    by = {(f["obj"], f["id"]): f for f in json.load(open(ip))["frames"]}

    def im(rel, mode="RGB"):
        return np.asarray(Image.open(os.path.join(a.data, rel)).convert(mode)
                          .resize((a.res, a.res)), np.float32) / 255

    hdr = ("%-12s %-11s %-9s %9s %8s %8s %8s %8s %9s"
           % ("rung", "split", "who", "dreamsim", "lpips", "l1", "l1_tone",
              "edge_f1", "spill_fix"))
    print(hdr)
    print("-" * len(hdr))
    out = []
    tags = TAGS
    if a.tags:
        tags = tuple(tuple(x.split(":", 1)) for x in a.tags.split(","))
    specs = (parse_runs(a.run_list) if a.run_list
             else tuple((r, g, n, None) for r, g, n in RUNS))
    for run, rung, name, objfile in specs:
        trained_objs = set(json.load(open(objfile))) if objfile else None
        for tag, split in tags:
            gd = os.path.join(a.runs, run, "gens_" + tag)
            if not os.path.isdir(gd):
                continue
            acc = {w: {k: [] for k in ("dreamsim", "lpips", "l1", "l1t",
                                       "f1", "gr", "sf")}
                   for w in ("model", "baseline")}
            allowed = trained_objs if tag.startswith("ontrain") else None
            chosen = pick_even(os.listdir(gd), a.limit, allowed)
            n_objs = len({c[:-4].rpartition("_")[0] for c in chosen})
            print("[metrics] %s %s: %d frames over %d objects"
                  % (run, split, len(chosen), n_objs), flush=True)
            for fn in chosen:
                if not fn.endswith(".png"):
                    continue
                obj, _, fid = fn[:-4].rpartition("_")
                r = by.get((obj, fid))
                if r is None:
                    continue
                gen = np.asarray(Image.open(os.path.join(gd, fn))
                                 .convert("RGB").resize((a.res, a.res)),
                                 np.float32) / 255
                tgt = im(r["tgt"])
                mkf = im(r["mask"], "L")[..., None]
                mk = mkf > 0.5
                if mk.sum() < 64:
                    continue
                pk = im(r["p" + rung])
                est = np.clip(im(r["a" + rung]) * pk[..., 1:2] * a.gain, 0, 1)
                sil = pk[..., 0:1] > 0.004
                spill = sil & ~mk
                # The union: the true part PLUS whatever the blockout claims
                # is material. Scoring only the true mask hands the baseline
                # the correct silhouette for free, which is its single largest
                # error (58% spill at boxes). The union charges it.
                unf = ((mk | sil).astype(np.float32))
                uys, uxs = np.where(unf[..., 0] > 0.5)
                upad = max(4, int(0.05 * max(uys.ptp() + 1, uxs.ptp() + 1)))
                uy0 = max(0, uys.min() - upad)
                uy1 = min(a.res, uys.max() + 1 + upad)
                ux0 = max(0, uxs.min() - upad)
                ux1 = min(a.res, uxs.max() + 1 + upad)

                def ucrop(z, y0=uy0, y1=uy1, x0=ux0, x1=ux1):
                    return np.ascontiguousarray(z[y0:y1, x0:x1])

                # EVERY metric scores the union, for the same reason the
                # perceptual ones do: inside the true mask a box fills the
                # silhouette as well as exact CAD, so the object mask hands
                # the baseline its largest error for free. The union counts
                # the region where the blockout claims material and the truth
                # is background.
                un = (mk | sil)
                m2 = un[:-1, :-1, 0]
                gt = gmag(tgt)
                tgt_e = float(gt[m2].mean()) if m2.sum() else float("nan")
                tmean = float((tgt * un).sum() / (un.sum() * 3))

                for who, pred in (("model", gen), ("baseline", est)):
                    d = acc[who]
                    d["l1"].append(float((np.abs(pred - tgt) * un).sum()
                                         / (un.sum() * 3)))
                    # the gain is fitted on the same region it is scored on
                    pm = float((pred * un).sum() / (un.sum() * 3))
                    sc = tmean / pm if pm > 1e-6 else 1.0
                    d["l1t"].append(float(
                        (np.abs(np.clip(pred * sc, 0, 1) - tgt) * un).sum()
                        / (un.sum() * 3)))
                    ge = gmag(pred)
                    # edge_f1 erodes by 4px, which removes the union's own
                    # outline but keeps the true object's boundary where it
                    # runs through the spill region - exactly the edge the
                    # baseline does not have and the target does.
                    d["f1"].append(edge_f1(ge, gt, m2))
                    d["gr"].append(float((np.abs(ge - gt) * m2).sum()
                                         / m2.sum()) / max(tgt_e, 1e-6))
                    if spill.sum() > 64:
                        # how much of the blockout's false material did this
                        # prediction remove? the baseline scores 0 by
                        # construction; matching the truth scores 1
                        den = float((np.abs(est - tgt) * spill).sum()
                                    / (spill.sum() * 3))
                        num = float((np.abs(pred - tgt) * spill).sum()
                                    / (spill.sum() * 3))
                        d["sf"].append(1.0 - num / den if den > 1e-6
                                       else float("nan"))
                    if lp is not None:
                        # Crop to the object before scoring. LPIPS averages
                        # over the frame, and the object is ~15% of it, so
                        # scoring the full image mostly averages identical
                        # black padding and compresses every score toward
                        # zero - and rewards objects that sit small in frame.
                        d["lpips"].append(lp(ucrop(pred * unf),
                                             ucrop(tgt * unf)))
                    if ds is not None:
                        # same crop and mask as LPIPS, so the two perceptual
                        # numbers are answering the same question on the same
                        # pixels and any disagreement is the metric, not the
                        # framing
                        d["dreamsim"].append(ds(ucrop(pred * unf),
                                                ucrop(tgt * unf)))
            if not acc["model"]["l1"]:
                continue
            for who in ("model", "baseline"):
                d = acc[who]
                v = {k: (float(np.nanmean(x)) if x else float("nan"))
                     for k, x in d.items()}
                def f(x):
                    return "%.4f" % x if x == x else "-"

                print("%-12s %-11s %-9s %9s %8s %8.4f %8.4f %8.3f %9s"
                      % (name, split, who,
                         f(v["dreamsim"]), f(v["lpips"]),
                         v["l1"], v["l1t"], v["f1"],
                         "%.2f" % v["sf"] if v["sf"] == v["sf"] else "-"))
                out.append({"run": run, "rung": rung, "split": split,
                            "who": who, **v})
            print()
    json.dump(out, open(a.out, "w"), indent=1)
    print("every metric scores the UNION of the true part and the blockout")
    print("silhouette, so the baseline is charged for claiming material that")
    print("is not there instead of being handed the correct outline.")
    print()
    print("lower is better: dreamsim, lpips, l1, l1_tone")
    print("higher is better: edge_f1, spill_fix")
    print("l1 is a diagnostic, not a ranking: on a single-grey corpus it")
    print("mostly scores average brightness. The l1 to l1_tone gap is the")
    print("size of the global gain error. spill_fix 0 = the render reproduced")
    print("the blockout's false material, 1 = it removed it.")
    print("(grad_r is still in the json; it rewards blankness, so it is not")
    print("shown and should not be used.)")
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
