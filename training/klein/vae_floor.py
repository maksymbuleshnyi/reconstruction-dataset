#!/usr/bin/env python3
"""The VAE floor: the best any klein-based renderer can score.

Every generated image is decoded from the VAE latent, so a model whose latent
prediction were PERFECT would still lose whatever the VAE loses. This
round-trips the true target of every frame a run was evaluated on (encode,
decode, nothing else) and writes the result as a run of its own,
runs/vaefloor/gens_<tag>, so metrics.py scores it with the same crop, mask
and LPIPS as the models. The distance between a model's row and this row is
the part of the error the model is responsible for.

    python klein/vae_floor.py --data v5_colour --like runs/klein100bs_cd --tags onewalk,ontrainwalk
    python metrics.py --data v5_colour --limit 100000 --gain 1.07 --no-dreamsim \\
        --tags "ontrainwalk:trained on,onewalk:held out" --run-list "vaefloor:pr:vaefloor:"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import DISTILLED, Codec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--like", required=True, help="run folder whose gens_<tag> frames to mirror")
    ap.add_argument("--tags", default="onewalk,ontrainwalk")
    ap.add_argument("--out", default="runs/vaefloor")
    ap.add_argument("--model", default=DISTILLED)
    ap.add_argument("--res", type=int, default=512)
    a = ap.parse_args()

    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    pipe.vae.to("cuda")
    codec = Codec(pipe, "cuda", torch.bfloat16)

    idx = json.load(open(os.path.join(a.data, "dataset", "index.json")))
    by = {(f["obj"], f["id"]): f for f in idx["frames"]}
    for tag in a.tags.split(","):
        src = os.path.join(a.like, "gens_" + tag)
        if not os.path.isdir(src):
            print("skip", src, "(missing)")
            continue
        dst = os.path.join(a.out, "gens_" + tag)
        os.makedirs(dst, exist_ok=True)
        files = sorted(f for f in os.listdir(src) if f.endswith(".png"))
        t0 = time.time()
        for n, fn in enumerate(files):
            obj, _, fid = fn[:-4].rpartition("_")
            row = by.get((obj, fid))
            if row is None or os.path.exists(os.path.join(dst, fn)):
                continue
            tgt = Image.open(os.path.join(a.data, row["tgt"])).convert("RGB") \
                .resize((a.res, a.res), Image.BILINEAR)
            lat = codec.encode(tgt)                                # (1, C, h, w)
            h, w = lat.shape[-2:]
            packed = codec.P._pack_latents(lat)                     # (1, hw, C)
            codec.decode(packed, h, w).save(os.path.join(dst, fn))
            if n % 100 == 0:
                print("  %s %d/%d (%.0fs)" % (tag, n, len(files), time.time() - t0), flush=True)
        print("->", dst, len(files), "frames", flush=True)
    json.dump({"like": a.like, "tags": a.tags}, open(os.path.join(a.out, "eval_vaefloor.json"), "w"))
    print("VAE FLOOR COMPLETE")


if __name__ == "__main__":
    main()
