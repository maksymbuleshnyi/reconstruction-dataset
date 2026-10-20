#!/usr/bin/env python3
"""One JPEG comparing renders of held-out frames across reference conditions.

Columns: blockout pack | reference photo | target | own refs | swapped refs |
no refs. Rows: frames spread over the held-out objects.

    python sheet.py --data v5_colour --run runs/klein100 --n 8 --out sheet_klein100.jpg
"""
from __future__ import annotations

import argparse
import json
import os

from PIL import Image, ImageDraw

TAGS = [("onewalk", "own refs"), ("onewalk_swap", "swapped refs"),
        ("onewalk_noref", "no refs")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="onewalk",
                    help="gens folders <prefix>, <prefix>_swap, <prefix>_noref")
    ap.add_argument("--also", default="",
                    help="more gens folders as columns, 'tag:label,tag:label'")
    a = ap.parse_args()
    tags = [(a.prefix, "own refs" if not a.also else a.prefix)]
    for suf, lab in (("_sub", "ref-coloured render"), ("_swap", "swapped refs"),
                     ("_swapvar", "other appearance"), ("_noref", "no refs")):
        if os.path.isdir(os.path.join(a.run, "gens_" + a.prefix + suf)):
            tags.append((a.prefix + suf, lab))
    for spec in filter(None, a.also.split(",")):
        t, _, lab = spec.partition(":")
        tags.append((t, lab or t))
    idx = json.load(open(os.path.join(a.data, "dataset", "index.json")))
    by = {(f["obj"], f["id"]): f for f in idx["frames"]}
    gd = os.path.join(a.run, "gens_" + a.prefix)
    files = sorted(os.listdir(gd))
    # spread over objects: one frame per object, then wrap
    per = {}
    for fn in files:
        obj = fn[:-4].rpartition("_")[0]
        per.setdefault(obj, []).append(fn)
    objs = sorted(per)
    picks = files if len(files) <= a.n else []      # a tiny run: show every frame
    k = 0
    while len(picks) < a.n and k < 50:
        for o in objs:
            if len(picks) >= a.n:
                break
            fl = per[o]
            if k * 7 + 3 < len(fl):
                picks.append(fl[k * 7 + 3])
        k += 1
    S = a.size
    cols = ["pack", "reference", "target"] + [t[1] for t in tags]
    sheet = Image.new("RGB", (S * len(cols), S * len(picks) + 20), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        d.text((c * S + 6, 4), name, fill=(230, 230, 230))
    for r, fn in enumerate(picks):
        obj, _, fid = fn[:-4].rpartition("_")
        row = by[(obj, fid)]
        refs = [x for x in idx["objects"][obj]["refs"]
                if x.get("variant", 0) == row.get("variant", 0)] or idx["objects"][obj]["refs"]
        ims = [Image.open(os.path.join(a.data, row["ppr"])),
               Image.open(os.path.join(a.data, refs[0]["file"])),
               Image.open(os.path.join(a.data, row["tgt"]))]
        for tag, _ in tags:
            p = os.path.join(a.run, "gens_" + tag, fn)
            ims.append(Image.open(p) if os.path.exists(p) else Image.new("RGB", (S, S)))
        for c, im in enumerate(ims):
            sheet.paste(im.convert("RGB").resize((S, S)), (c * S, 20 + r * S))
        d.text((6, 24 + r * S), obj, fill=(230, 230, 230))
    sheet.save(a.out, quality=88)
    print("->", a.out, len(picks), "frames")


if __name__ == "__main__":
    main()
