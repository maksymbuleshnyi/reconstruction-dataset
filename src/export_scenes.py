"""Export the source scenes, not the rendered frames.

Every image in the dataset is generated from one of these: the part meshes,
the blockout approximations of them, the assembly order, and each part's
colour and material. Shipping the scenes instead of 334k PNGs makes the
release about 70 MB rather than 12 GB, and anyone can regenerate the frames
at whatever resolution they want.

    python export_scenes.py ../reconstruction-dataset/scenes
"""
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fusion_demo import Prepped, _fit_one, pbr_for      # noqa: E402


def mesh(v, f, nd=3):
    return {"v": [round(float(x), nd) for x in np.asarray(v).reshape(-1)],
            "f": [int(x) for x in np.asarray(f).reshape(-1)]}


def export(obj):
    pp = Prepped(os.path.join(HERE, "data", obj))
    parts, boxes = [], []
    for i, (buid, name, m, col, met) in enumerate(pp.parts):
        d = mesh(m.vertices, m.faces)
        d["name"] = str(name)
        d["colour"] = [int(c) for c in col]
        # metalness, roughness, clearcoat, parsed from the Fusion material name
        d["pbr"] = [float(x) for x in (pp.info["pbr"][i]
                                       if i < len(pp.info.get("pbr", []))
                                       else pbr_for(""))]
        parts.append(d)
        pr, _, _ = _fit_one(m)
        boxes.append(mesh(pr.vertices, pr.faces))
    return {
        "id": obj,
        "parts": parts,                                   # exact CAD geometry
        "primitives": [mesh(b[0], b[1]) for b in pp.blocks],   # mid blockout
        "boxes": boxes,                                   # coarsest blockout
        "order": [int(i) for i in pp.order],              # assembly sequence
        "centre": [round(float(x), 4) for x in pp.fit[0]],
        "radius": round(float(pp.fit[1]), 4),
    }


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "scenes"
    os.makedirs(out, exist_ok=True)
    ids = sorted(d for d in os.listdir(os.path.join(HERE, "data"))
                 if os.path.isdir(os.path.join(HERE, "data", d)))
    index, total = [], 0
    for n, obj in enumerate(ids, 1):
        dst = os.path.join(out, obj + ".json.gz")
        try:
            if not os.path.exists(dst):
                sc = export(obj)
                with gzip.open(dst, "wt", compresslevel=6) as fh:
                    json.dump(sc, fh)
            else:
                with gzip.open(dst, "rt") as fh:
                    sc = json.load(fh)
        except Exception as exc:
            print("  [%3d/%d] %s skipped (%s)" % (n, len(ids), obj, exc))
            continue
        sz = os.path.getsize(dst)
        total += sz
        index.append({"id": obj, "parts": len(sc["parts"]),
                      "steps": len(sc["order"]),
                      "tris": sum(len(p["f"]) // 3 for p in sc["parts"]),
                      "bytes": sz})
        print("  [%3d/%d] %-18s %2d parts  %.1f MB"
              % (n, len(ids), obj, len(sc["parts"]), sz / 1e6), flush=True)
    index.sort(key=lambda r: -r["parts"])
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
    print("\n%d scenes, %.0f MB total" % (len(index), total / 1e6))


if __name__ == "__main__":
    main()
