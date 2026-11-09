#!/usr/bin/env python3
"""Image-error curves over training: every saved checkpoint of a run renders
the same fixed frames (distilled, 4 steps, like the final evaluation) and is
scored against the real frames over the WHOLE image, pixels in [0, 1]: L1,
RMSE and LPIPS. Three subsets, chosen once and evenly over the frames:
  heldout        held-out objects, own reference photos
  heldout_photo  the same frames plus the progress photo 3 frames away in the
                 walk (~35 degrees), only for runs trained with photos
  train          the run's own training objects
Rows go to <run>/curve.csv; checkpoints already there are skipped, so the
script can be rerun as new checkpoints appear.

    python klein/ckpt_curve.py --data v5_colour --run runs/hist40bo
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys

import numpy as np


def checkpoints(run):
    out = []
    for d in os.listdir(run):
        m = re.fullmatch(r"lora_step(\d+)", d)
        if m and os.path.isdir(os.path.join(run, d)):
            out.append((int(m.group(1)), os.path.join(run, d)))
    return sorted(out)


def spread(n_rows, n):
    return list(np.unique(np.linspace(0, n_rows - 1, min(n, n_rows)).round().astype(int)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--heldout", type=int, default=84, help="held-out frames (6 per object)")
    ap.add_argument("--train", type=int, default=60, help="training-object frames")
    ap.add_argument("--res", type=int, default=512)
    a = ap.parse_args()

    import torch
    from PIL import Image
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from render import DISTILLED
    from promo_eval import renderer_for
    from train_klein import Frames
    import lpips

    ta = json.load(open(os.path.join(a.run, "args.json")))
    n_refs, history = int(ta.get("n_refs", 3)), float(ta.get("history", 0.0)) > 0
    objs = None
    if ta.get("objects"):
        objs = json.load(open(ta["objects"]))
        objs = objs["objects"] if isinstance(objs, dict) else objs
    sets = {"heldout": Frames(a.data, "test_unseen", None, n_refs, a.res),
            "train": Frames(a.data, "train", objs, n_refs, a.res)}
    picks = {"heldout": spread(len(sets["heldout"].rows), a.heldout),
             "train": spread(len(sets["train"].rows), a.train)}
    subsets = ["heldout", "train"] + (["heldout_photo"] if history else [])
    sets["heldout"].hist_offset, sets["heldout"].hist_earlier = 3, 0.0

    out_f = os.path.join(a.run, "curve.csv")
    done = set()
    if os.path.exists(out_f):
        done = {(int(r["step"]), r["subset"]) for r in csv.DictReader(open(out_f))}
    new = not os.path.exists(out_f)
    w = csv.writer(open(out_f, "a", newline=""))
    if new:
        w.writerow(["step", "subset", "frames", "l1", "rmse", "lpips"])
    dev = "cuda"
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev)
    tt = lambda z: torch.from_numpy(z).permute(2, 0, 1)[None].to(dev) * 2 - 1

    for step, ck in checkpoints(a.run):
        todo = [s for s in subsets if (step, s) not in done]
        if not todo:
            continue
        R, _ = renderer_for(ck, DISTILLED)
        for s in todo:
            ds = sets["train" if s == "train" else "heldout"]
            l1, se, lv = [], [], []
            for n, i in enumerate(picks["train" if s == "train" else "heldout"]):
                r = ds.rows[i]
                item = ds.item(i, rng=None, history=(s == "heldout_photo"))
                g = R.render(item["ctx"][0], item["ctx"][1:], steps=4, seed=n, res=a.res,
                             ref_id_maps=item.get("ref_ids"), n_parts=item.get("n_parts"))
                gt = np.asarray(item["target"], np.float32) / 255
                ge = np.asarray(g.convert("RGB").resize((a.res, a.res)), np.float32) / 255
                l1.append(float(np.abs(ge - gt).mean()))
                se.append(float(((ge - gt) ** 2).mean()))
                with torch.no_grad():
                    lv.append(float(lp(tt(ge), tt(gt)).item()))
            w.writerow([step, s, len(l1), "%.5f" % np.mean(l1), "%.5f" % np.sqrt(np.mean(se)),
                        "%.5f" % np.mean(lv)])
            print("[curve] step %d %-14s %3d frames  L1 %.4f  RMSE %.4f  LPIPS %.4f"
                  % (step, s, len(l1), np.mean(l1), np.sqrt(np.mean(se)), np.mean(lv)), flush=True)
        del R
        gc.collect()
        torch.cuda.empty_cache()
    print("CURVE COMPLETE %s" % a.run, flush=True)


if __name__ == "__main__":
    main()
