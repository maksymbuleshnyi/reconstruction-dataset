#!/usr/bin/env python3
"""Side-by-side video of promo sessions (klein/promo_eval.py): the real frame,
then one column per model, and the progress photo the memory handed over.
Each column carries its frame's LPIPS; the end card holds the means. Needs
ffmpeg; runs anywhere the renders and the dataset are.

    python klein/promo_video.py --data v5_colour \
        --arms "gap40b, 3 photos=h200_pilot/runs/gap40b/promo_none;gap40b + memory=h200_pilot/runs/gap40b/promo_nearest" \
        --out h200_pilot/promo.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK, MUTED, BG, ACCENT = (26, 26, 25), (107, 103, 95), (252, 252, 251), (42, 120, 214)
TILE, HEAD, FOOT = 384, 46, 40


def font(sz, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", required=True, help="'label=promo folder;label=promo folder;...'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=3.0)
    a = ap.parse_args()
    arms = [(s.split("=", 1)[0].strip(), s.split("=", 1)[1].strip()) for s in a.arms.split(";") if s.strip()]
    metas = [json.load(open(os.path.join(p, "meta.json"))) for _, p in arms]
    n = min(len(m["frames"]) for m in metas)
    # the memory column shows the photo of the first arm that used one
    cols = 2 + len(arms)
    W, H = cols * TILE, HEAD + TILE + FOOT
    f_head, f_small, f_big = font(17, True), font(15), font(30, True)
    tmp = os.path.splitext(a.out)[0] + "_frames"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    def tile(img):
        return img.convert("RGB").resize((TILE, TILE), Image.BILINEAR)

    for t in range(n):
        fr = [m["frames"][t] for m in metas]
        f0 = fr[0]
        c = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(c)
        photo = next((f["photo"] for f in fr if f["photo"]), None)
        pf = next((f for f in fr if f["photo"]), None)
        heads = ["real frame", "memory: stored photo"] + [lab for lab, _ in arms]
        for i, h in enumerate(heads):
            d.text((i * TILE + 10, 13), h, fill=INK, font=f_head)
        c.paste(tile(Image.open(os.path.join(a.data, f0["tgt"]))), (0, HEAD))
        if photo:
            c.paste(tile(Image.open(os.path.join(a.data, photo))), (TILE, HEAD))
            same = pf["photo_st"] == f0["st"]
            d.text((TILE + 10, HEAD + TILE + 10), "%s, %.0f deg away" % ("same step" if same else "earlier step",
                   pf["photo_angle"]), fill=MUTED, font=f_small)
        else:
            d.rectangle([TILE, HEAD, 2 * TILE - 1, HEAD + TILE - 1], fill=(236, 234, 229))
            d.text((TILE + 110, HEAD + TILE // 2 - 10), "no photo yet", fill=MUTED, font=f_head)
        for i, ((lab, p), f) in enumerate(zip(arms, fr)):
            x = (2 + i) * TILE
            c.paste(tile(Image.open(os.path.join(p, "%04d.png" % t))), (x, HEAD))
            d.text((x + 10, HEAD + TILE + 10), "LPIPS %.3f   L1 %.3f" % (f["lpips"], f["l1"]),
                   fill=INK, font=f_small)
        d.text((10, HEAD + TILE + 10), "%s  step %d  frame %d/15" % (f0["obj"], f0["st"], f0["k"] + 1),
               fill=MUTED, font=f_small)
        c.save(os.path.join(tmp, "%04d.png" % t))

    # end card: the means, overall and on the frames that had a photo
    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    d.text((40, 40), "Mean over %d frames (same frames, photos and seeds for every column)" % n,
           fill=INK, font=f_head)
    y = 100
    for (lab, _), m in zip(arms, metas):
        fr = m["frames"][:n]
        wp = [f for f in fr if f["photo"]]
        s = "%-34s all: LPIPS %.3f  L1 %.4f" % (lab, np.mean([f["lpips"] for f in fr]), np.mean([f["l1"] for f in fr]))
        if m["summary"]["memory"] != "none":
            s += "    with a photo (%d): LPIPS %.3f" % (len(wp), np.mean([f["lpips"] for f in wp]))
        d.text((40, y), s, fill=INK, font=font(19))
        y += 40
    for k in range(int(a.fps * 4)):
        c.save(os.path.join(tmp, "%04d.png" % (n + k)))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i",
                    os.path.join(tmp, "%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out], check=True)
    print("->", a.out)


if __name__ == "__main__":
    main()
