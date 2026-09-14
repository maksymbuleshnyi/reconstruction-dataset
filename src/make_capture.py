#!/usr/bin/env python3
"""Bake capture pages and drive headless Chrome through them.

Per object this produces, under capture/<id>/:
    fNNNN_tgt.png            realistic target, alpha (composited later)
    fNNNN_{d,n,a}{cad,pr,bx} depth / normal / albedo conditioning at three
                             coarseness rungs: exact CAD, decomposed
                             primitives, single boxes
    fNNNN_mask.png           exact object silhouette from the CAD
    rK_tgt / pK_tgt          reference views (full object) and progress
                             views (mid-assembly states, other angles)
    index.json               one row per frame: kind, state, camera, session

Design decisions that matter:
  - lighting is randomized per SESSION (one session per assembly state, one
    per reference view), the way real photos of one working moment share one
    light; exposure and env intensity ride along
  - close-up cameras aim at the newest part, mirroring the manuals' action
    camera, so the model sees the detail it will be asked to render
  - references deliberately DIFFER in state from most pairs: the bank teaches
    view transfer, the progress views teach state transfer

  .venv/bin/python capture_server.py 8877        # terminal 1
  .venv/bin/python make_capture.py --one 23466_1cf60d09
  .venv/bin/python make_capture.py --pilot       # all 16, sequential
"""
from __future__ import annotations

import json
import math
import os
import random
import zlib
import subprocess
import sys
import time

import numpy as np

from fusion_demo import Prepped, _fit_one

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "site")
CAP = os.environ.get("CAPTURE_DIR") or os.path.join(HERE, "capture")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 8877
RES = 512


def rnd(a, nd=4):
    return [round(float(x), nd) for x in np.asarray(a).reshape(-1)]


def mesh_json(v, f):
    return {"v": rnd(v), "f": [int(x) for x in np.asarray(f).reshape(-1)]}


def build_script(pp: Prepped, rng: random.Random):
    K = len(pp.order)
    R = pp.fit[1]
    ctr = pp.fit[0]
    centers = [c[0].mean(0) for c in pp.cad]
    rad = [float(np.linalg.norm(np.ptp(c[0], axis=0)) / 2) for c in pp.cad]

    if K <= 14:
        states = list(range(1, K + 1))
    else:
        states = sorted({int(round(x)) for x in np.linspace(1, K, 14)})

    sessions, rows = [], []

    def new_session():
        t = rng.uniform(-1, 1)
        sessions.append({
            "ki": round(rng.uniform(0.9, 1.4), 3),
            "kc": [1.0, round(0.93 + 0.05 * t, 3),
                   round(0.85 + 0.12 * t, 3)],
            "kd": [round(math.cos(a := rng.uniform(0, 2 * math.pi)), 3),
                   round(math.sin(a), 3), round(rng.uniform(0.8, 1.8), 3)],
            "env": round(rng.uniform(0.75, 1.25), 3),
            "exp": round(rng.uniform(0.85, 1.1), 3),
            # A colourway for this session, applied to the TARGET and to the
            # albedo conditioning together. With one colour scheme per object
            # the model identifies the object from its shape and recalls the
            # colours, so the albedo channel is redundant and stays unlearned:
            # blanking it changed L1 by 0.0017. Varying colour across sessions
            # breaks that shortcut and leaves reading albedo as the only route.
            "cw": rng.randint(1, 10 ** 6)})
        return len(sessions) - 1

    # v2: frames come in CONTINUOUS TRAJECTORIES, not independent samples.
    # Memory conditioning is only learnable if consecutive training frames are
    # genuinely consecutive -- the memory image at frame k must be what would
    # actually have accumulated by then. Each state gets one smooth walk:
    # an orbit with a drifting elevation and a slow dolly, plus a close-up
    # approach to the newest part at the end.
    fid = 0
    for si, st in enumerate(states):
        sess = new_session()
        newest = pp.order[st - 1]
        az0 = rng.uniform(0, 2 * math.pi)
        el0 = rng.uniform(0.25, 0.75)
        d0 = rng.uniform(1.9, 2.4) * R
        sweep = rng.choice([1, -1]) * rng.uniform(1.5, 2.6)   # radians walked
        for c in range(12):                      # the orbit walk
            u = c / 11
            rows.append({"id": f"f{fid:04d}", "kind": "pair", "st": st,
                         "sess": sess, "traj": si, "k": c,
                         "az": round(az0 + sweep * u, 4),
                         "el": round(el0 + 0.22 * math.sin(math.pi * u), 4),
                         "dist": round(d0 * (1 - 0.18 * math.sin(
                             math.pi * u)), 4),
                         "tx": round(float(ctr[0]), 4),
                         "ty": round(float(ctr[1]), 4),
                         "tz": round(float(ctr[2]), 4)})
            fid += 1
        tgt = centers[newest] * 0.85 + ctr * 0.15
        dclose = min(1.9 * R, max(0.5 * R, rad[newest] * 7))
        for c in range(3):                       # approach the new part
            u = (c + 1) / 3
            rows.append({"id": f"f{fid:04d}", "kind": "pair", "st": st,
                         "sess": sess, "traj": si, "k": 12 + c,
                         "az": round(az0 + sweep + 0.35 * u, 4),
                         "el": round(el0 + 0.10 + 0.25 * u, 4),
                         "dist": round(d0 + (dclose - d0) * u, 4),
                         "tx": round(float(ctr[0] + (tgt[0]-ctr[0]) * u), 4),
                         "ty": round(float(ctr[1] + (tgt[1]-ctr[1]) * u), 4),
                         "tz": round(float(ctr[2] + (tgt[2]-ctr[2]) * u), 4)})
            fid += 1

    # Reference bank: what a user can actually photograph.
    #
    # A larger POOL than any single session uses. The model is shown a random
    # few each step, so it never comes to depend on one particular photo and
    # gets robustness to framing and angle for free -- the cost is capture
    # time, not training time. The test set draws a frozen selection from the
    # same pool, so evaluation stays comparable across models.
    #
    # Half the pool is the parts on a bench (the real input at session start,
    # since the object does not exist yet); half is the finished product in a
    # SHIFTED colourway, standing in for box art, which shows the product
    # rather than the user's unit.
    N_PARTS_REF, N_PROD_REF = 6, 4
    for k in range(N_PARTS_REF):
        sess = new_session()
        rows.append({"id": f"r{k}", "kind": "ref", "st": K, "sess": sess,
                     "layout": 17 + k * 91,
                     "subset": 1.0,
                     "wide": round(0.82 - 0.05 * k, 3),
                     "az": round(math.pi / 2 + (k - 2.5) * 0.55
                                 + rng.uniform(-0.15, 0.15), 4),
                     "el": round(1.34 - 0.06 * k
                                 + rng.uniform(-0.04, 0.04), 4),
                     "dist": round(2.2 * R, 4),
                     "tx": round(float(ctr[0]), 4),
                     "ty": round(float(ctr[1]), 4),
                     "tz": round(float(ctr[2]), 4)})

    allv = np.vstack([c[0] for c in pp.cad])
    ext = np.ptp(allv, axis=0)
    elong = float(max(ext) / max(np.median(ext), 1e-6))
    ref_dist = R * (1.75 + 0.42 * min(max(elong - 1.0, 0.0), 2.2))
    shift = rng.randint(2, 10 ** 6)          # one colourway for all of them
    base_az = rng.uniform(0, 2 * math.pi)
    for k in range(N_PROD_REF):
        sess = new_session()
        rows.append({"id": f"r{N_PARTS_REF + k}", "kind": "ref", "st": K,
                     "sess": sess, "shift": shift,
                     "az": round(base_az + k * 1.6, 4),
                     "el": round([0.60, 0.32, 0.82, 0.45][k]
                                 + rng.uniform(-0.05, 0.05), 4),
                     "dist": round(ref_dist * [1.0, 0.88, 0.95, 0.8][k], 4),
                     "tx": round(float(ctr[0]), 4),
                     "ty": round(float(ctr[1]), 4),
                     "tz": round(float(ctr[2]), 4)})

    # progress references: every other captured state, two other angles
    pk = 0
    for st in states[::2]:
        for _ in range(2):
            sess = new_session()
            rows.append({"id": f"p{pk}", "kind": "pref", "st": st,
                         "sess": sess,
                         "az": round(rng.uniform(0, 2 * math.pi), 4),
                         "el": round(rng.uniform(0.25, 0.8), 4),
                         "dist": round(2.3 * R, 4),
                         "tx": round(float(ctr[0]), 4),
                         "ty": round(float(ctr[1]), 4),
                         "tz": round(float(ctr[2]), 4)})
            pk += 1
    return sessions, rows


def bake(aid: str) -> str:
    # zlib.crc32, not hash(): Python salts string hashing per process,
    # so hash() gave a different camera walk on every run and any
    # partial re-render silently desynced that object from its targets.
    rng = random.Random(zlib.crc32(aid.encode()) & 0xffff)
    pp = Prepped(os.path.join(HERE, "data", aid))
    parts = []
    for i, (buid, name, m, col, met) in enumerate(pp.parts):
        d = mesh_json(m.vertices, m.faces)
        d["c"], d["m"] = list(col), met
        # parsed from the Fusion material name: metalness, roughness,
        # clearcoat. Drives BOTH the target render and the material
        # conditioning map, so what the model is told and what it must
        # produce agree by construction.
        d["pbr"] = list(pp.info["pbr"][i]) if i < len(pp.info.get("pbr", [])) else None
        parts.append(d)
    prims = [mesh_json(b[0], b[1]) for b in pp.blocks]
    boxes = []
    for _, _, m, _, _ in pp.parts:
        pr, _, _ = _fit_one(m)
        boxes.append(mesh_json(pr.vertices, pr.faces))
    sessions, rows = build_script(pp, rng)
    scene = {"object": aid, "res": RES, "parts": parts, "prims": prims,
             "boxes": boxes, "order": pp.order, "inner": pp.inner,
             "center": rnd(pp.fit[0]), "radius": round(pp.fit[1], 4),
             "sessions": sessions, "script": rows}
    tpl = open(os.path.join(SITE, "capture_template.html")).read()
    three = open(os.path.join(SITE, "vendor", "three.min.js")).read()
    out = tpl.replace("/*__THREE__*/", three).replace(
        "/*__SCENE__*/", json.dumps(scene, separators=(",", ":")))
    path = os.path.join(SITE, f"capture_{aid}.html")
    open(path, "w").write(out)
    n_pair = sum(1 for r in rows if r["kind"] == "pair")
    print(f"[bake] {aid}: {n_pair} pairs + "
          f"{len(rows) - n_pair} refs -> {os.path.basename(path)}",
          flush=True)
    return path


def run(aid: str, timeout=1200) -> bool:
    done = os.path.join(CAP, aid, "DONE")
    if os.path.exists(done):
        os.remove(done)
    bake(aid)
    proc = subprocess.Popen(
        [CHROME, "--headless=new", "--mute-audio", "--disable-gpu-sandbox",
         f"--window-size={RES},{RES + 40}",
         f"http://localhost:{PORT}/site/capture_{aid}.html"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    ok = False
    while time.time() - t0 < timeout:
        if os.path.exists(done):
            ok = True
            break
        if proc.poll() is not None:
            break
        time.sleep(2)
    proc.terminate()
    n = len([f for f in os.listdir(os.path.join(CAP, aid))
             if f.endswith(".png")]) if os.path.isdir(
        os.path.join(CAP, aid)) else 0
    print(f"[run] {aid}: {'ok' if ok else 'TIMEOUT/CRASH'}, {n} png, "
          f"{time.time() - t0:.0f}s", flush=True)
    return ok


def main():
    os.makedirs(CAP, exist_ok=True)
    if "--ids" in sys.argv:
        ids = json.load(open(sys.argv[sys.argv.index("--ids") + 1]))
        if isinstance(ids, dict):
            ids = ids.get("new") or ids.get("pilot") or []
        okc = 0
        for aid in ids:
            if os.path.exists(os.path.join(CAP, aid, "DONE")):
                print(f"[run] {aid}: already captured", flush=True)
                okc += 1
                continue
            okc += run(aid)
        print(f"[capture] {okc}/{len(ids)} objects complete", flush=True)
    elif "--one" in sys.argv:
        run(sys.argv[sys.argv.index("--one") + 1])
    elif "--pilot" in sys.argv:
        ids = json.load(open(os.path.join(HERE,
                                          "pilot_selection.json")))["pilot"]
        okc = 0
        for aid in ids:
            if os.path.exists(os.path.join(CAP, aid, "DONE")):
                print(f"[run] {aid}: already captured", flush=True)
                okc += 1
                continue
            okc += run(aid)
        print(f"[capture] {okc}/{len(ids)} objects complete", flush=True)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
