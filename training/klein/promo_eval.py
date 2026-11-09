#!/usr/bin/env python3
"""The promo sequence: a fixed session over a few held-out objects, rendered
the same way by every model, so models can be compared frame by frame and by
eye.

Per object the session plays its camera walks in assembly order (4 steps, 15
frames each, ~11 degrees apart). The object's 3 reference photos are always
given. Now and then the "user" takes a progress photo: after rendering walk
frame k in --photo-at, the real image of that frame goes into the memory.
With --memory nearest every later frame of that object also gets the stored
photo closest to it as a 4th reference: a photo of the same assembly step if
one exists (the smallest camera angle), else one from the latest earlier
step. The memory only holds photos taken BEFORE the frame being rendered, so
no frame ever sees its own target. Seeds are the frame index, so every model
gets the same noise.

Writes <out>/promo_<tag>/NNNN.png and meta.json (per frame: object, step,
camera, which photo was used and at what angle, whole-frame L1, RMSE, LPIPS).

    python klein/promo_eval.py --data v5_colour --lora runs/gap40b/lora_best --out runs/gap40b --tag mem --memory nearest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

OBJECTS = "20505_017d4cc4,22491_a33562ad,22852_dc1dbed6"   # 15, 29 and 42 parts


def sequence(step_walks, objects, variant):
    """Rows in playing order: per object its walks by assembly step, each walk
    in camera order."""
    seq = []
    for o in objects:
        sw = step_walks.get((o, variant), {})
        for st in sorted(sw):
            seq += [sw[st][k] for k in sorted(sw[st])]
    return seq


def view_dir(r):
    ce = math.cos(r["el"])
    return np.array([math.cos(r["az"]) * ce, math.sin(r["az"]) * ce, math.sin(r["el"])])


def angle(a, b):
    return math.degrees(math.acos(float(np.clip(view_dir(a) @ view_dir(b), -1.0, 1.0))))


def retrieve(memory, row):
    """The stored photo closest to this frame: same assembly step first (smallest
    camera angle), else the latest earlier step. None if nothing fits."""
    st = int(row["st"])
    pool = [m for m in memory if int(m["st"]) == st]
    if not pool:
        earlier = [m for m in memory if int(m["st"]) < st]
        if not earlier:
            return None
        last = max(int(m["st"]) for m in earlier)
        pool = [m for m in earlier if int(m["st"]) == last]
    return min(pool, key=lambda m: angle(m, row))


def renderer_for(lora, model):
    """A Renderer with the token layout the LoRA was trained with (args.json),
    and the number of reference photos it was trained with."""
    from render import Renderer
    kw, n_refs = {}, 3
    arg_file = os.path.join(os.path.dirname(lora.rstrip("/")), "args.json")
    if os.path.exists(arg_file):
        ta = json.load(open(arg_file))
        kw = dict(pack_ids=ta.get("pack_ids", "offset"), part_ids=ta.get("part_ids", "none"),
                  substrate=bool(ta.get("substrate", False)), part_bias=ta.get("part_bias", "none"),
                  bias_beta=ta.get("bias_beta", 3.0), bias_gamma=ta.get("bias_gamma", 3.0),
                  id_images=bool(ta.get("id_images", False)), id_grey=bool(ta.get("id_grey", False)),
                  substrate_stat=ta.get("substrate_stat", "mean"), bias_bg=float(ta.get("bias_bg", 0.0)))
        n_refs = int(ta.get("n_refs", 3))
    print("[promo] %s: %d reference photos, %s" % (lora, n_refs, kw), flush=True)
    return Renderer(model, lora, **kw), n_refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--model", default=None, help="default: the distilled klein")
    ap.add_argument("--out", required=True, help="run folder (gets promo_<tag>)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--objects", default=OBJECTS)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--memory", default="nearest", choices=["none", "nearest"])
    ap.add_argument("--photo-at", default="4,11", help="walk frames after which a photo is stored")
    ap.add_argument("--no-photo-labels", action="store_true",
                    help="stored photos without part labels (a real photo of unknown camera)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=512)
    a = ap.parse_args()

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from render import DISTILLED
    from train_klein import Frames
    import lpips

    objects = a.objects.split(",")
    R, n_refs = renderer_for(a.lora, a.model or DISTILLED)
    ds = Frames(a.data, "test_unseen", objects, n_refs, a.res)
    seq = sequence(ds.step_walks, objects, a.variant)
    photo_at = {int(x) for x in a.photo_at.split(",") if x}
    lp = lpips.LPIPS(net="alex", verbose=False).to(R.dev)
    od = os.path.join(a.out, "promo_" + a.tag)
    os.makedirs(od, exist_ok=True)
    print("[promo] %d frames over %d objects, memory %s, photos after walk frames %s -> %s"
          % (len(seq), len(objects), a.memory, sorted(photo_at), od), flush=True)

    def im(rel):
        return Image.open(os.path.join(a.data, rel)).convert("RGB").resize((a.res, a.res), Image.BILINEAR)

    meta, memory, cur, t0 = [], [], None, time.time()
    for t, r in enumerate(seq):
        if r["obj"] != cur:                       # a new object: an empty memory
            cur, memory = r["obj"], []
        files, pids = ds.ref_files(r, None, with_ids=True)
        refs = [im(f) for f in files]
        ids = [Image.open(os.path.join(a.data, p)).convert("RGB") if p else None for p in pids]
        used = retrieve(memory, r) if a.memory == "nearest" else None
        if used is not None:
            refs.append(im(used["tgt"]))
            b = Image.open(os.path.join(a.data, used["ppr"])).convert("RGB").split()[2]
            ids.append(None if a.no_photo_labels else Image.merge("RGB", (b, b, b)))
        fn = os.path.join(od, "%04d.png" % t)
        if os.path.exists(fn):
            g = Image.open(fn).convert("RGB")
        else:
            g = R.render(im(r["ppr"]), refs, steps=a.steps, seed=a.seed + t, res=a.res,
                         ref_id_maps=ids, n_parts=ds.objects[r["obj"]].get("n_parts"))
            g.save(fn)
        gt = np.asarray(im(r["tgt"]), np.float32) / 255
        ge = np.asarray(g.convert("RGB").resize((a.res, a.res)), np.float32) / 255
        tt = lambda z: torch.from_numpy(z).permute(2, 0, 1)[None].to(R.dev) * 2 - 1
        with torch.no_grad():
            lv = float(lp(tt(ge), tt(gt)).item())
        meta.append({"t": t, "obj": r["obj"], "id": r["id"], "st": int(r["st"]), "k": int(r.get("k", 0)),
                     "az": r["az"], "el": r["el"], "tgt": r["tgt"],
                     "photo": used["tgt"] if used is not None else None,
                     "photo_st": int(used["st"]) if used is not None else None,
                     "photo_angle": round(angle(used, r), 1) if used is not None else None,
                     "l1": float(np.abs(ge - gt).mean()), "rmse": float(np.sqrt(((ge - gt) ** 2).mean())),
                     "lpips": lv})
        if int(r.get("k", 0)) in photo_at:        # the user photographs this view, after it was shown
            memory.append(r)
        if t % 20 == 0:
            print("  %d/%d (%.0fs)" % (t, len(seq), time.time() - t0), flush=True)

    def mean(rows, k):
        return float(np.mean([m[k] for m in rows])) if rows else float("nan")
    withp = [m for m in meta if m["photo"]]
    summary = {"frames": len(meta), "lora": a.lora, "memory": a.memory, "photo_at": sorted(photo_at),
               "all": {k: mean(meta, k) for k in ("l1", "rmse", "lpips")},
               "with_photo": {k: mean(withp, k) for k in ("l1", "rmse", "lpips")} | {"n": len(withp)}}
    json.dump({"summary": summary, "frames": meta}, open(os.path.join(od, "meta.json"), "w"), indent=1)
    print("[promo] %s: all %d frames L1 %.4f LPIPS %.4f | %d frames with a photo L1 %.4f LPIPS %.4f"
          % (a.tag, len(meta), summary["all"]["l1"], summary["all"]["lpips"], len(withp),
             summary["with_photo"]["l1"], summary["with_photo"]["lpips"]), flush=True)
    print("PROMO COMPLETE %s" % a.tag, flush=True)


if __name__ == "__main__":
    main()
