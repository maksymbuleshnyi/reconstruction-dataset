#!/usr/bin/env python3
"""Load a scene and print what is in it. The smallest possible example of
reading this dataset.

    python3 src/read_scene.py scenes/23310_7daded69.json.gz
"""
import gzip
import json
import sys


def load(path):
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def main(path):
    s = load(path)
    print("object      %s" % s["id"])
    print("parts       %d" % len(s["parts"]))
    print("steps       %d" % len(s["order"]))
    print("triangles   %d" % sum(len(p["f"]) // 3 for p in s["parts"]))
    print("radius      %.3f" % s["radius"])
    print()
    print("assembly order (first ten):")
    for n, i in enumerate(s["order"][:10], 1):
        p = s["parts"][i]
        met, rough, coat = p["pbr"]
        kind = "metal" if met > 0.5 else "non-metal"
        print("  %2d. %-30s %-10s rough %.2f  rgb%s"
              % (n, p["name"][:30], kind, rough, tuple(p["colour"])))
    print()
    print("blockout levels, triangles per level:")
    for k in ("boxes", "primitives", "parts"):
        print("  %-11s %7d" % (k, sum(len(m["f"]) // 3 for m in s[k])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "scenes/23310_7daded69.json.gz")
