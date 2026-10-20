#!/usr/bin/env python3
"""Bake alpha/rank into LoRA checkpoints saved before the trainer did it itself.

A checkpoint trained with --lora-alpha != rank was applied at scale alpha/rank
during training, but the saved file has no alpha and loads at scale 1. This
multiplies the B matrices by the scale once and records it in lora_meta.json,
so it is never applied twice.

    python klein/bake_lora_scale.py 0.25 runs/gap5/lora_best runs/gap5/lora ...
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file


def main():
    scale = float(sys.argv[1])
    for d in sys.argv[2:]:
        meta_f = os.path.join(d, "lora_meta.json")
        if os.path.exists(meta_f) and json.load(open(meta_f)).get("baked_scale", 1.0) != 1.0:
            print("skip %s (already baked)" % d)
            continue
        f = os.path.join(d, "pytorch_lora_weights.safetensors")
        sd = load_file(f)
        n = 0
        for k in list(sd):
            if "lora_B" in k:
                sd[k] = (sd[k].float() * scale).to(sd[k].dtype)
                n += 1
        save_file(sd, f)
        json.dump({"baked_scale": scale}, open(meta_f, "w"))
        print("baked %s: %d B matrices x %.3f" % (d, n, scale))


if __name__ == "__main__":
    main()
