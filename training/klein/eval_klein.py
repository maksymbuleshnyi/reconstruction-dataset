#!/usr/bin/env python3
"""Render evaluation frames with the klein LoRA, in the layout metrics.py scores.

Writes runs/<run>/gens_<tag>/<object>_<frame>.png for whole camera paths of
each object (--walks frames per object), with the distilled 4-step model and
the trained LoRA. The reference photos can be the object's own (--refs own),
another object's (--refs swap: sorted objects, shifted by one) or none
(--refs none), which is the reference-usage test.

    python eval_klein.py --data v5_colour --lora runs/klein_v5/lora \\
        --out runs/klein_v5 --split test_unseen --walks 70 --tag onewalk
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import DISTILLED, Renderer, pixel_part_ids, substrate_image   # noqa: E402
import numpy as np                                      # noqa: E402
from PIL import Image                                   # noqa: E402


def permute_labels(idmap, n):
    """Part k -> k+1 (cyclic) in an id-map image, keeping the level encoding."""
    if not n:
        return idmap
    k = pixel_part_ids(idmap, 0, n)
    k2 = np.where(k > 0, (k % n) + 1, 0)
    lv = np.rint(255.0 * k2 / (n + 1)).astype(np.uint8)
    return Image.fromarray(np.stack([lv, lv, lv], -1))
from train_klein import Frames           # noqa: E402


def pick_walks(ds, walks):
    """Whole camera paths per object until at least `walks` frames of it."""
    by = {}
    for i, r in enumerate(ds.rows):
        by.setdefault(r["obj"], {}).setdefault(str(r.get("traj", 0)), []).append(i)
    picks = []
    for obj in sorted(by):
        got = 0
        for key in sorted(by[obj], key=lambda k: (len(k), k)):
            if got >= walks:
                break
            idxs = sorted(by[obj][key], key=lambda i: ds.rows[i].get("k", 0))
            picks += idxs
            got += len(idxs)
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--lora", default=None, help="LoRA folder; omit for the plain model")
    ap.add_argument("--model", default=DISTILLED)
    ap.add_argument("--out", required=True, help="run folder (gets gens_<tag>)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test_unseen")
    ap.add_argument("--objects", default=None)
    ap.add_argument("--walks", type=int, default=70)
    ap.add_argument("--refs", default="own", choices=["own", "swap", "swapvar", "none"],
                    help="swap: another object's photos; swapvar: the same "
                         "object in another appearance")
    ap.add_argument("--n-refs", type=int, default=3)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--states", default=None,
                    help="only frames whose assembly state is in this range, "
                         "e.g. 1-3 for the first three steps")
    ap.add_argument("--variants", default=None,
                    help="only frames of these appearance variants, e.g. 16-19")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="same subset rule as the trainer's --max-frames")
    ap.add_argument("--save-substrate", action="store_true",
                    help="also write the reference-coloured analytic render")
    ap.add_argument("--substrate-stat", default=None, choices=["mean", "robust"],
                    help="override the substrate colour estimator recorded with "
                         "the LoRA (to test 'robust' on a model trained with 'mean')")
    ap.add_argument("--bias-bg", type=float, default=None,
                    help="override the background bias recorded with the LoRA (e.g. to "
                         "try it at inference on a model trained without it)")
    ap.add_argument("--history", action="store_true",
                    help="add the history image (real image of a nearby frame of the same "
                         "walk) as one more reference")
    ap.add_argument("--hist-offset", type=int, default=3)
    ap.add_argument("--hist-earlier", action="store_true",
                    help="the history image comes from an earlier assembly step")
    ap.add_argument("--hist-require", action="store_true",
                    help="skip frames that have no history image (e.g. no earlier step)")
    ap.add_argument("--hist-no-labels", action="store_true",
                    help="give the history image without part labels (as a real photo of "
                         "unknown camera would come): no bias, no substrate colour from it")
    ap.add_argument("--label-perm", action="store_true",
                    help="permute the part labels of the reference id maps (part k "
                         "-> k+1): a model that follows the labels must now paint "
                         "the wrong colours, a model that ignores them is unchanged")
    a = ap.parse_args()

    # the token layout the LoRA was trained with is recorded next to it
    pack_ids, part_ids, substrate = "aligned", "none", False
    part_bias, bias_beta, bias_gamma, id_images = "none", 3.0, 3.0, False
    substrate_stat, bias_bg = "mean", 0.0
    if a.lora:
        arg_file = os.path.join(os.path.dirname(a.lora.rstrip("/")), "args.json")
        if os.path.exists(arg_file):
            ta = json.load(open(arg_file))
            pack_ids, part_ids = ta.get("pack_ids", "offset"), ta.get("part_ids", "none")
            substrate = bool(ta.get("substrate", False))
            part_bias = ta.get("part_bias", "none")
            bias_beta, bias_gamma = ta.get("bias_beta", 3.0), ta.get("bias_gamma", 3.0)
            id_images = bool(ta.get("id_images", False))
            substrate_stat = ta.get("substrate_stat", "mean")
            bias_bg = float(ta.get("bias_bg", 0.0))
    if a.substrate_stat:
        substrate_stat = a.substrate_stat
    if a.bias_bg is not None:
        bias_bg = a.bias_bg
    R = Renderer(a.model, a.lora, pack_ids=pack_ids, part_ids=part_ids, substrate=substrate,
                 part_bias=part_bias, bias_beta=bias_beta, bias_gamma=bias_gamma,
                 id_images=id_images, substrate_stat=substrate_stat, bias_bg=bias_bg)
    print("[eval] LoRA %s, pack ids %s, part ids %s, substrate %s (%s), part bias %s (bg %.1f), id images %s"
          % (a.lora, pack_ids, part_ids, substrate, substrate_stat, part_bias, bias_bg, id_images), flush=True)

    objs = json.load(open(a.objects)) if a.objects else None
    if isinstance(objs, dict):
        objs = objs["objects"]
    ds = Frames(a.data, a.split, objs, a.n_refs, a.res, a.variants, a.max_frames)
    ds.hist_offset, ds.hist_earlier = a.hist_offset, (1.0 if a.hist_earlier else 0.0)
    if a.states:
        lo, hi = (int(x) for x in a.states.split("-"))
        ds.rows = [r for r in ds.rows if lo <= int(r.get("st", 0)) <= hi]
        print("[eval] states %d-%d: %d frames" % (lo, hi, len(ds.rows)), flush=True)
    picks = pick_walks(ds, a.walks)
    if a.history and a.hist_require:
        before = len(picks)
        picks = [i for i in picks if ds.history_row(ds.rows[i]) is not None]
        print("[eval] %d of %d frames have a history image" % (len(picks), before), flush=True)
    objects = sorted({ds.rows[i]["obj"] for i in picks})
    swap = {o: objects[(k + 1) % len(objects)] for k, o in enumerate(objects)} \
        if a.refs == "swap" else {}
    if a.refs == "swap" and len(objects) < 2:
        raise SystemExit("--refs swap needs at least two objects")
    gd = os.path.join(a.out, "gens_" + a.tag)
    os.makedirs(gd, exist_ok=True)
    print("[eval] %d frames over %d objects, refs=%s -> %s"
          % (len(picks), len(objects), a.refs, gd), flush=True)

    t0 = time.time()
    for n, i in enumerate(picks):
        r = ds.rows[i]
        fn = os.path.join(gd, "%s_%s.png" % (r["obj"], r["id"]))
        if os.path.exists(fn):
            continue
        vto = None
        if a.refs == "swapvar":       # same object, the next appearance in the set
            vs = sorted({int(x.get("variant", 0)) for x in ds.rows if x["obj"] == r["obj"]})
            vto = vs[(vs.index(int(r.get("variant", 0))) + 1) % len(vs)]
        item = ds.item(i, rng=None, swap_to=swap.get(r["obj"]),
                       no_refs=(a.refs == "none"), variant_to=vto, history=a.history)
        if a.history and a.hist_no_labels and item.get("ref_ids") and ds.history_row(r) is not None:
            item["ref_ids"][-1] = None
        if a.label_perm and item.get("ref_ids"):
            item["ref_ids"] = [permute_labels(m, item.get("n_parts")) if m is not None else None
                               for m in item["ref_ids"]]
        im = R.render(item["ctx"][0], item["ctx"][1:], steps=a.steps, cfg=a.cfg,
                      seed=a.seed + n, res=a.res, ref_id_maps=item.get("ref_ids"),
                      n_parts=item.get("n_parts"))
        im.save(fn)
        if substrate or a.save_substrate:
            # the reference-coloured analytic render, scored as its own row
            sd = os.path.join(a.out, "gens_" + a.tag + "_sub")
            os.makedirs(sd, exist_ok=True)
            substrate_image(item["ctx"][0], item["ctx"][1:], item.get("ref_ids"),
                            item.get("n_parts"), stat=substrate_stat) \
                .save(os.path.join(sd, os.path.basename(fn)))
        if n % 20 == 0:
            print("  %d/%d (%.0fs)" % (n, len(picks), time.time() - t0), flush=True)
    json.dump({"frames": len(picks), "objects": objects, "refs": a.refs,
               "steps": a.steps, "lora": a.lora, "seconds": time.time() - t0},
              open(os.path.join(a.out, "eval_%s.json" % a.tag), "w"), indent=1)
    print("EVAL COMPLETE %s" % a.tag, flush=True)


if __name__ == "__main__":
    main()
