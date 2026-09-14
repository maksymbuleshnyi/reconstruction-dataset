#!/usr/bin/env python3
"""Composite captured targets over backgrounds and assemble the training
index. This is where the synthetic data is pushed toward photographs.

Realism levers, in order of effect:
  - backgrounds: real photos if you drop any into backgrounds/ (jpg/png),
    otherwise procedurally generated desk/wall scenes -- wood grain, wall
    gradients, vignette. One background per SESSION, like one photo session
    on one desk.
  - tone: per-session gamma / white-balance / brightness jitter applied to
    the composite, so object and background share one grade and the model
    cannot separate them by statistics.
  - sensor: per-frame gaussian noise, occasional slight defocus, JPEG at
    quality 84-92. References get the same treatment (their own sessions):
    style must flow from the reference, so the reference must live in the
    same visual world as the target.

v2: lighting is flat and fixed, and the conditioning carries everything the
agent can know -- depth, normals, per-part colour (albedo) and per-part
material class. The renderer's only remaining job is object detail.

Writes dataset/comp/<obj>/<id>.jpg plus dataset/index.json:
  objects: refs (with camera pose, for nearest-view selection), progress
  refs (with state), split membership
  frames: paths to the composite target, all nine conditioning maps, the
  mask, camera and state metadata

  .venv/bin/python compose_dataset.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
# CAPTURE_DIR / DATASET_DIR let a subset be composed somewhere separate,
# so an experiment never reads or overwrites the live corpus.
CAP = os.environ.get("CAPTURE_DIR") or os.path.join(HERE, "capture")
OUT = os.environ.get("DATASET_DIR") or os.path.join(HERE, "dataset")
BG_USER = os.path.join(HERE, "backgrounds")
# Paths in the index are relative to the DATA ROOT (the parent of the
# dataset dir), so a subset composed elsewhere is self-contained and the
# trainer can resolve it with --data pointing at that root. For the live
# corpus ROOT == HERE, so nothing changes.
ROOT = os.path.dirname(OUT)
RES = 512


# ------------------------------------------------------------ backgrounds

def _smooth(n, k):
    """1-D noise smoothed by a running mean: cheap band-limited variation."""
    x = np.random.normal(0, 1, n + k)
    c = np.cumsum(x)
    return (c[k:] - c[:-k]) / k


def _wood(rng, w=768):
    base = np.array([rng.uniform(110, 165), rng.uniform(75, 115),
                     rng.uniform(45, 80)])
    im = np.zeros((w, w, 3), np.float32) + base
    plank = rng.randint(80, 150)
    for y0 in range(0, w, plank):
        tone = rng.uniform(-14, 14)                 # each plank its own tone
        grain = _smooth(w, 40) * 9                  # grain along the plank
        wob = _smooth(w, 90) * 4
        for y in range(y0, min(y0 + plank, w)):
            im[y] += (tone + grain + wob * math.sin(y * 0.05))[:, None]
        im[y0:y0 + 2] -= 30                         # seam
    im += np.random.normal(0, 2.5, im.shape)
    from PIL import Image as _I, ImageFilter as _F
    sm = _I.fromarray(np.clip(im, 0, 255).astype(np.uint8))         .filter(_F.GaussianBlur(1.2))
    return np.asarray(sm, np.float32)


def _wall_desk(rng, w=768):
    wall = np.array([rng.uniform(170, 225)] * 3) \
        + np.array([rng.uniform(-8, 8) for _ in range(3)])
    desk = wall * rng.uniform(0.55, 0.8)
    im = np.zeros((w, w, 3), np.float32)
    horizon = int(w * rng.uniform(0.35, 0.55))
    im[:horizon] = wall
    im[horizon:] = desk
    for y in range(w):                              # soft vertical falloff
        im[y] *= 1 - 0.15 * y / w
    im += np.random.normal(0, 4, im.shape)
    return im


def _paper(rng, w=768):
    im = np.zeros((w, w, 3), np.float32) + rng.uniform(225, 248)
    im += np.random.normal(0, 3, im.shape)
    return im


def make_backgrounds(n=14, seed=7):
    rng = random.Random(seed)
    outs = []
    user = sorted(glob.glob(os.path.join(BG_USER, "*")))
    for k in range(n):
        if user:
            im = Image.open(user[k % len(user)]).convert("RGB")
            s = min(im.size)
            im = im.crop((0, 0, s, s)).resize((768, 768))
            outs.append(np.asarray(im, np.float32))
            continue
        f = [_wood, _wall_desk, _wood, _wall_desk, _paper][k % 5]
        im = f(rng)
        yy, xx = np.mgrid[0:768, 0:768]
        v = 1 - 0.35 * (((xx - 384) ** 2 + (yy - 384) ** 2) / 384 ** 2)
        outs.append(np.clip(im * v[..., None], 0, 255))
    return outs


# ------------------------------------------------------------ compositing

def grade(rng):
    """Lighting is fixed, so the colour grade is fixed too: only the
    background and mild sensor noise vary. Anything that changed the object's
    appearance unpredictably was asking the model to guess something absent
    from its inputs."""
    return {"gamma": 1.0,
            "wb": np.array([1.0, 1.0, 1.0]),
            "gain": 1.0,
            "bg_shift": (rng.randint(0, 255), rng.randint(0, 255))}


def composite(png_path, bg, g, rng, out_path, clean=False):
    """Place the render on a background.

    clean=True is used for TARGETS. Grain, defocus and jpeg are unpredictable
    from the conditioning, so they are pure noise in the supervision, and blur
    attacks the fine detail the renderer exists to produce. They also set an
    error floor nothing can beat. Targets are therefore lossless and
    undegraded.

    clean=False is used for REFERENCE images, which stand in for photographs
    the user took on a phone. Those really do have grain, compression and
    imperfect focus, and the model must be able to read them at deployment.
    """
    fg = Image.open(png_path).convert("RGBA")
    x0, y0 = g["bg_shift"]
    base = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)) \
        .crop((x0, y0, x0 + RES, y0 + RES))
    base = base.convert("RGBA")
    base.alpha_composite(fg)
    a = np.asarray(base.convert("RGB"), np.float32) / 255.0
    a = np.clip((a ** g["gamma"]) * g["wb"] * g["gain"], 0, 1) * 255
    if clean:
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(out_path)
        return
    a = a + np.random.normal(0, rng.uniform(1.0, 3.0), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.25:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1)))
    im.save(out_path, quality=rng.randint(84, 92))


def main():
    sw = os.path.join(HERE, "sweep_splits.json")
    if os.path.exists(sw):
        spl = json.load(open(sw))
        corpus = json.load(open(os.path.join(HERE, "meta",
                                             "sweep_corpus.json")))
        membership = {}
        for i in corpus:
            membership[i] = ("test_unseen" if i in set(spl["test"]) else
                             "val_unseen" if i in set(spl["val"]) else
                             "train")
        # only compose what has actually been captured
        membership = {i: m for i, m in membership.items()
                      if os.path.exists(os.path.join(CAP, i, "index.json"))}
    else:
        split = json.load(open(os.path.join(HERE, "pilot_split.json")))
        membership = {i: "train" for i in split["train"]}
        membership.update({i: "test_unseen" for i in split["test"]})
    bgs = make_backgrounds()
    index = {"objects": {}, "frames": []}
    rngg = random.Random(11)

    for obj, memb in membership.items():
        odir = os.path.join(CAP, obj)
        rows = json.load(open(os.path.join(odir, "index.json")))
        cdir = os.path.join(OUT, "comp", obj)
        os.makedirs(cdir, exist_ok=True)
        # crc32, not hash(): salted string hashing reshuffled every
        # background and grade on each recompose
        rng = random.Random(zlib.crc32(obj.encode()) & 0xffff)
        grades = {}      # session -> (bg index, grade)
        refs, prefs, n_pairs = [], [], 0
        for r in rows:
            sess = r["sess"]
            if sess not in grades:
                grades[sess] = (rng.randrange(len(bgs)), grade(rng))
            bgi, g = grades[sess]
            src = os.path.join(odir, r["id"] + "_tgt.png")
            is_ref = r["kind"] == "ref"
            dst = os.path.join(cdir, r["id"]
                               + (".jpg" if is_ref else ".png"))
            # Resume, but never keep a target older than the render it came
            # from. The filename does not change when the capture is rebuilt,
            # so an existence check silently pairs new conditioning with old
            # targets -- which is exactly what invalidated the first v2 run.
            if (not os.path.exists(dst)
                    or os.path.getmtime(dst) < os.path.getmtime(src)):
                composite(src, bgs[bgi], g, rng, dst, clean=not is_ref)
            rel = os.path.relpath(dst, ROOT)
            cam = {k: r[k] for k in ("az", "el", "dist", "st")}
            if r["kind"] == "ref":
                refs.append(dict(cam, file=rel,
                                 shifted=bool(r.get("shift")),
                                 disassembled=bool(r.get("layout"))))
            elif r["kind"] == "pref":
                prefs.append(dict(cam, file=rel))
            else:
                n_pairs += 1
                cond = {}
                # p<rung> packs depth into R and shading into G; older
                # captures wrote a separate d<rung>. Index whichever exists.
                for tag in ("dcad", "pcad", "ncad", "acad", "mcad",
                            "dpr", "ppr", "npr", "apr", "mpr",
                            "dbx", "pbx", "nbx", "abx", "mbx", "mask"):
                    fp = os.path.join(odir, f'{r["id"]}_{tag}.png')
                    if tag[0] in "dp" and not os.path.exists(fp):
                        continue
                    cond[tag] = os.path.relpath(fp, ROOT)
                index["frames"].append(dict(
                    cam, obj=obj, id=r["id"], split=memb,
                    traj=r.get("traj", 0), k=r.get("k", 0),
                    tx=r["tx"], ty=r["ty"], tz=r["tz"],
                    near=r["near"], far=r["far"],
                    tgt=rel, **cond))
        index["objects"][obj] = {"split": memb, "refs": refs,
                                 "prefs": prefs}
        print(f"[compose] {obj}: {n_pairs} pairs, {len(refs)} refs, "
              f"{len(prefs)} progress refs ({memb})", flush=True)

    json.dump(index, open(os.path.join(OUT, "index.json"), "w"))
    tr = sum(1 for f in index["frames"] if f["split"] == "train")
    te = len(index["frames"]) - tr
    print(f"\n[compose] index: {tr} train / {te} test_unseen frames "
          f"-> dataset/index.json")


if __name__ == "__main__":
    main()
