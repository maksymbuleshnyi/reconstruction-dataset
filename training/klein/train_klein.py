#!/usr/bin/env python3
"""LoRA-train FLUX.2 [klein] 4B as a blockout-to-photo renderer.

The whole conditioning goes in as CONTEXT IMAGES, the way klein already does
multi-reference editing: image 0 is the blockout pack (R depth, G shading,
B part id), images 1..n are reference photos of the object, and optionally
the last one is a history picture. Each context image is VAE-encoded and
appended to the token sequence with its own position offset (the pipeline's
own helpers do this); the loss is computed on the target tokens only. A LoRA
on the attention projections is the only thing trained, against the BASE
checkpoint, and loads on the distilled 4-step model afterwards.

Every N steps the loss on a FIXED set of frames (fixed noise, fixed
timesteps) is printed for training objects and held-out objects:

    val step 250 fixed_train 0.0912 fixed_heldout 0.0987

which is the train / validation curve.

    python train_klein.py --data v5_colour --out runs/klein_v5 --steps 6000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import (BASE, PROMPT, Codec, PartEmbed, attention_bias,   # noqa: E402
                    id_colour_images, part_ids_from_map, substrate_image, to_tensor)


# ----------------------------------------------------------------- data
def degrade_photo(photo, lab, rng):
    """A real photo is never as clean as a render: random lighting and colour
    cast, blur, sensor noise, and up to two patches blocked out (a hand, a
    tool). Blocked pixels lose their part label too. Geometry is untouched, so
    the labels still line up with the photo."""
    from PIL import ImageDraw, ImageEnhance, ImageFilter
    im = ImageEnhance.Brightness(photo).enhance(rng.uniform(0.7, 1.3))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.75, 1.25))
    a = np.asarray(im, np.float32) * np.array([rng.uniform(0.85, 1.15) for _ in range(3)], np.float32)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))
    if rng.random() < 0.5:
        a = np.asarray(im, np.float32)
        a = a + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 8), a.shape)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    n_occ = rng.choice([0, 0, 1, 2])
    if n_occ:
        W, H = im.size
        im = im.copy()
        lab = lab.copy() if lab is not None else None
        d, dl = ImageDraw.Draw(im), (ImageDraw.Draw(lab) if lab is not None else None)
        for _ in range(n_occ):
            w, h = int(W * rng.uniform(0.1, 0.3)), int(H * rng.uniform(0.1, 0.3))
            x, y = rng.randrange(0, W - w), rng.randrange(0, H - h)
            g = rng.randrange(20, 120)
            d.rectangle((x, y, x + w, y + h), fill=(g, g, g))
            if dl is not None:
                dl.rectangle((x, y, x + w, y + h), fill=(0, 0, 0))
    return im, lab


class Frames:
    """Pairs of (control pack, target) with the reference photos of the same
    object, and of the same appearance variant when the corpus has them."""

    def __init__(self, root, split, objects=None, n_refs=3, res=512, variants=None,
                 max_frames=None):
        self.root, self.res, self.n_refs = root, res, n_refs
        idx = json.load(open(os.path.join(root, "dataset", "index.json")))
        self.objects = idx["objects"]
        rows = [f for f in idx["frames"] if f["split"] == split and "ppr" in f]
        if objects is not None:
            allowed = set(objects)
            rows = [f for f in rows if f["obj"] in allowed]
        if variants:                                  # "a-b": appearance variants
            lo, hi = (int(x) for x in variants.split("-"))
            rows = [f for f in rows if lo <= int(f.get("variant", 0)) <= hi]
        if max_frames:                                # a fixed, evenly spread subset
            pick = np.unique(np.linspace(0, len(rows) - 1, min(max_frames, len(rows))).round().astype(int))
            rows = [rows[i] for i in pick]
        self.rows = rows
        # (object, variant) -> reference photos of that appearance
        self.refs = {}
        for obj, meta in self.objects.items():
            for r in meta["refs"]:
                self.refs.setdefault((obj, r.get("variant", 0)), []).append(r)
        self.objs = sorted({f["obj"] for f in rows})
        # camera walks, for the history image: (obj, variant, traj) -> {k: row};
        # each walk shows one assembly step: (obj, variant) -> {step: walk}
        self.walks, self.hist_offset, self.hist_earlier = {}, 3, 0.0
        self.step_walks = {}
        for f in idx["frames"]:
            if "ppr" in f:
                key = (f["obj"], f.get("variant", 0), f.get("traj", 0))
                self.walks.setdefault(key, {})[int(f.get("k", 0))] = f
                self.step_walks.setdefault((f["obj"], f.get("variant", 0)), {}).setdefault(
                    int(f.get("st", 0)), self.walks[key])
        print("[data] %s: %d frames over %d objects, %d reference sets"
              % (split, len(rows), len(self.objs), len(self.refs)), flush=True)

    def _im(self, rel):
        return Image.open(os.path.join(self.root, rel)).convert("RGB") \
            .resize((self.res, self.res), Image.BILINEAR)

    def ref_files(self, row, rng=None, swap_to=None, with_ids=False, variant_to=None):
        """Frozen (rng None) or random selection of n_refs references.
        The parts-spread photos and the product photos are interleaved, as
        the SDXL evaluation did. swap_to: use another object's references.
        with_ids: also return each photo's part-id map path (or None)."""
        obj = swap_to or row["obj"]
        var = row.get("variant", 0) if variant_to is None else variant_to
        pool = self.refs.get((obj, var)) or \
            self.refs.get((obj, 0)) or \
            [r for k, v in self.refs.items() if k[0] == obj for r in v]
        parts = [r for r in pool if r.get("disassembled")]
        prod = [r for r in pool if not r.get("disassembled")]
        if rng is not None:
            parts, prod = list(parts), list(prod)
            rng.shuffle(parts)
            rng.shuffle(prod)
        out, pids = [], []
        for i in range(self.n_refs):
            src = parts if (i % 2 == 0 or not prod) else prod
            if not src:
                src = pool
            x = src[(i // 2) % len(src)]
            out.append(x["file"])
            pids.append(x.get("pid"))
        return (out, pids) if with_ids else out

    def history_row(self, r):
        """The frame hist_offset steps earlier in the same camera walk (or
        later, near the start): same object, appearance and assembly step,
        seen from a camera ~11 degrees per step away. Its real image is the
        'progress photo' an interactive session would have."""
        rng = getattr(self, "_hist_rng", None) or random
        k = int(r.get("k", 0))
        if self.hist_earlier and rng.random() < self.hist_earlier:
            # a photo from an EARLIER assembly step (the latest one with a walk):
            # fewer parts, another camera path; the frame at the same walk index
            sw = self.step_walks.get((r["obj"], r.get("variant", 0)), {})
            earlier = [st for st in sw if st < int(r.get("st", 0))]
            if not earlier:
                return None
            w = sw[max(earlier)]
            return w.get(k) or w[min(w, key=lambda c: abs(c - k))]
        w = self.walks.get((r["obj"], r.get("variant", 0), r.get("traj", 0)), {})
        off = self.hist_offset
        if isinstance(off, tuple):                    # a range: random distance per sample
            off = rng.randint(off[0], off[1])
        for c in (k - off, k + off):
            if c in w:
                return w[c]
        return None

    def item(self, i, rng=None, swap_to=None, no_refs=False, variant_to=None, history=False):
        r = self.rows[i]
        ctx = [self._im(r["ppr"])]
        ref_ids = []
        if not no_refs:
            files, pids = self.ref_files(r, rng, swap_to, with_ids=True, variant_to=variant_to)
            ctx += [self._im(f) for f in files]
            ref_ids = [Image.open(os.path.join(self.root, p)).convert("RGB")
                       if p else None for p in pids]
        if history:
            # the neighbouring frame's real image as one more reference; its
            # part labels come from that frame's own blockout (pack B channel),
            # moved to channel 0 where reference id maps keep them. With
            # hist_label_drop the labels are withheld (a real photo of unknown
            # camera has none): no bias and no substrate colour from it
            self._hist_rng = rng
            h = self.history_row(r)
            if h is not None:
                photo = self._im(h["tgt"])
                lab = None
                if not (rng is not None and rng.random() < getattr(self, "hist_label_drop", 0.0)):
                    b = Image.open(os.path.join(self.root, h["ppr"])).convert("RGB").split()[2]
                    lab = Image.merge("RGB", (b, b, b))
                if rng is not None and rng.random() < getattr(self, "hist_augment", 0.0):
                    photo, lab = degrade_photo(photo, lab, rng)
                ctx.append(photo)
                ref_ids.append(lab)
        mask = Image.open(os.path.join(self.root, r["mask"])).convert("L") \
            .resize((self.res, self.res), Image.BILINEAR)
        return {"target": self._im(r["tgt"]), "ctx": ctx, "mask": mask,
                "ref_ids": ref_ids, "obj": r["obj"], "id": r["id"],
                "n_parts": self.objects[r["obj"]].get("n_parts")}


# ----------------------------------------------------------------- model
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--objects", default=None, help="json list of training objects")
    ap.add_argument("--variants", default=None,
                    help="train on these appearance variants only, e.g. 0-15")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="train on only this many frames (evenly spread over the "
                         "selection); the training fixed set is then exactly these "
                         "frames. For memorisation tests")
    ap.add_argument("--val-split", default="test_unseen",
                    help="split of the held-out fixed set")
    ap.add_argument("--val-variants", default=None,
                    help="variants of the held-out fixed set, e.g. 16-19")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation (batch is 1)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sched-start", type=int, default=0,
                    help="step at which the warm-up and cosine start (default 0). For a "
                         "continuation set it to the resume step: the rate then ramps "
                         "from --lr-warm-from to --lr and decays to --lr-final by --steps")
    ap.add_argument("--lr-warm-from", type=float, default=0.0,
                    help="learning rate at the start of the warm-up (0 for a fresh run; "
                         "the previous final rate for a continuation)")
    ap.add_argument("--lr-warmup", type=int, default=0,
                    help="linear learning-rate warm-up over this many steps from "
                         "0 (starting cold at 3e-4 gave a loss spike at 5k)")
    ap.add_argument("--lr-final", type=float, default=None,
                    help="cosine-decay the learning rate from --lr to this value "
                         "over the whole run (0..--steps; a resume continues the "
                         "schedule). Default: constant. A constant 3e-4 left the "
                         "N=100 fixed-set loss noisy and 0.01 above its 1e-4 level")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=None,
                    help="LoRA scaling numerator (scale = alpha / rank). Default: "
                         "= rank, i.e. scale 1 at any rank, which makes a rank-128 "
                         "update ~4x larger than a rank-32 one at the same lr and "
                         "made the N=40 run oscillate by 0.05; use 32 for rank 128")
    ap.add_argument("--select", default="v", choices=["v", "x0"],
                    help="held-out quantity that picks lora_best and drives --patience: "
                         "the v-loss (all noise levels equally; tracks LPIPS) or the "
                         "image error (dominated by the low-noise points; plateaus early)")
    ap.add_argument("--n-refs", type=int, default=3)
    ap.add_argument("--ref-drop", type=float, default=0.10,
                    help="fraction of steps trained with no reference photos")
    ap.add_argument("--bg-weight", type=float, default=1.0,
                    help="loss weight off the object (1 on it). Below 1 the "
                         "model paints reference objects into the empty "
                         "background almost for free")
    ap.add_argument("--topbg", type=float, default=0.5,
                    help="extra weight on the worst 10%% of background tokens: "
                         "invented structure off the object is punished where "
                         "it appears, not averaged away (0 = off)")
    ap.add_argument("--ref-warmup", type=float, default=0.2,
                    help="fraction of training at the start with NO reference "
                         "photos, so following the pack is learned before the "
                         "shortcut of copying the photos is available")
    ap.add_argument("--history", type=float, default=0.0,
                    help="chance per training sample of adding the history image (the "
                         "real image of a nearby frame of the same walk) as one more "
                         "reference; fixed validation sets then include it. 0 = off")
    ap.add_argument("--hist-offset", default="3",
                    help="how many frames away the history image is (~11 degrees each); "
                         "a range 'a-b' draws a random distance per training sample")
    ap.add_argument("--hist-earlier", type=float, default=0.0,
                    help="chance that the history image comes from an EARLIER assembly "
                         "step (the latest one with a walk) instead of the same step")
    ap.add_argument("--hist-augment", type=float, default=0.0,
                    help="chance that the history photo is degraded (lighting, colour "
                         "cast, blur, noise, blocked patches), as real photos would be")
    ap.add_argument("--hist-label-drop", type=float, default=0.0,
                    help="chance per sample that the history image comes without part "
                         "labels, as a real photo of unknown camera would")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-frames", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping: end training when the held-out image "
                         "error has not improved for this many validations "
                         "(0 = run to --steps). Needs --save-best")
    ap.add_argument("--save-best", action="store_true",
                    help="at every validation, keep the checkpoint with the lowest "
                         "held-out image error as <out>/lora_best (+ best.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="LoRA folder to continue from")
    ap.add_argument("--eval-only", action="store_true",
                    help="load --resume, compute the held-out fixed-set loss split by "
                         "noise level (4 bands of sigma) and exit; use with a larger "
                         "--val-frames than training (e.g. 128)")
    ap.add_argument("--start-step", type=int, default=0,
                    help="with --resume: the step that checkpoint was saved at, "
                         "so the schedule (warm-up, total) continues rather than restarts")
    ap.add_argument("--part-bias", default="none", choices=["none", "soft", "hard"],
                    help="attention bias from the part maps: a target token of "
                         "part k is pushed towards reference tokens of part k "
                         "(+beta) and away from other parts and clutter (-gamma)")
    ap.add_argument("--bias-beta", type=float, default=3.0)
    ap.add_argument("--bias-gamma", type=float, default=3.0)
    ap.add_argument("--bias-bg", type=float, default=0.0,
                    help="also push background screen tokens (where the map shows "
                         "nothing) away from the photos' part tokens by this much, so "
                         "empty regions cannot copy parts from the photos")
    ap.add_argument("--bias-drop", type=float, default=0.15,
                    help="per reference: chance its bias is withheld, so photos "
                         "without labels still work")
    ap.add_argument("--substrate", action="store_true",
                    help="also feed the reference-coloured analytic render "
                         "(part colours read from the photos through their "
                         "id maps) as an input aligned with the target")
    ap.add_argument("--substrate-stat", default="mean", choices=["mean", "robust"],
                    help="how a part's colour is read from the photos for the "
                         "substrate: pixel mean, or median chroma with "
                         "bright-quartile lightness (robust to gloss, shadow, "
                         "thin parts)")
    ap.add_argument("--id-images", action="store_true",
                    help="the same random colour for the same part, as one image "
                         "aligned with the blockout and one aligned with each "
                         "reference photo (from its id map); attention has to "
                         "learn to match the colours")
    ap.add_argument("--part-ids", default="rope", choices=["none", "rope", "embed"],
                    help="rope: the part id under each token goes on the 4th "
                         "position axis; embed: it is added to every image "
                         "token as a learned vector; for the target, the pack "
                         "and the reference photos (from their id maps)")
    ap.add_argument("--ref-id-drop", type=float, default=0.3,
                    help="fraction of samples whose reference id maps are "
                         "withheld, so real photos without maps still work")
    ap.add_argument("--pack-ids", default="aligned", choices=["aligned", "offset"],
                    help="aligned: the pack shares the target's position ids "
                         "(OminiControl); offset: the pack is offset like a "
                         "reference, as in the first runs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev, wdt = "cuda", torch.bfloat16

    from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    pipe = Flux2KleinPipeline.from_pretrained(a.model, torch_dtype=wdt)
    vae, transformer = pipe.vae, pipe.transformer
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(a.model, subfolder="scheduler")

    # the prompt is fixed, so its embedding is computed once and the text
    # encoder is dropped from the GPU
    pipe.text_encoder.to(dev)
    with torch.no_grad():
        enc = pipe.encode_prompt(prompt=PROMPT, device=dev, max_sequence_length=512)
    prompt_embeds, text_ids = enc[0], enc[1]
    pipe.text_encoder.to("cpu")
    del pipe.text_encoder
    torch.cuda.empty_cache()
    prompt_embeds, text_ids = prompt_embeds.to(dev, wdt), text_ids.to(dev)

    vae.to(dev).requires_grad_(False)
    transformer.to(dev).requires_grad_(False)
    n_single = len(transformer.single_transformer_blocks)
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.lora_alpha or a.rank, init_lora_weights="gaussian",
                      target_modules=["to_k", "to_q", "to_v", "to_out.0",
                                      "to_qkv_mlp_proj"]
                      + [f"single_transformer_blocks.{i}.attn.to_out"
                         for i in range(n_single)])
    transformer.add_adapter(lcfg)
    if a.resume:
        from peft import set_peft_model_state_dict
        from diffusers.utils import convert_unet_state_dict_to_peft
        sd = Flux2KleinPipeline.lora_state_dict(a.resume)
        sd = {k.replace("transformer.", ""): v for k, v in sd.items()
              if k.startswith("transformer.")}
        meta_f = os.path.join(a.resume, "lora_meta.json")
        baked = json.load(open(meta_f)).get("baked_scale", 1.0) if os.path.exists(meta_f) else 1.0
        if baked != 1.0:                    # undo the baked alpha/rank (see save)
            sd = {k: (v / baked if "lora_B" in k else v) for k, v in sd.items()}
        set_peft_model_state_dict(transformer, convert_unet_state_dict_to_peft(sd),
                                  adapter_name="default")
        print("[model] resumed LoRA from", a.resume, flush=True)
    params = [p for p in transformer.parameters() if p.requires_grad]
    for p in params:                       # LoRA weights train in fp32
        p.data = p.data.float()
    embed = None
    if a.part_ids == "embed":
        embed = PartEmbed(transformer.config.num_attention_heads
                          * transformer.config.attention_head_dim).to(dev).install(transformer)
        if a.resume and os.path.exists(os.path.join(a.resume, "part_embed.pt")):
            embed.load_state_dict(torch.load(os.path.join(a.resume, "part_embed.pt"), map_location=dev))
        params += list(embed.parameters())
    print("[model] LoRA rank %d: %.1fM trainable of %.2fB"
          % (a.rank, sum(p.numel() for p in params) / 1e6,
             sum(p.numel() for p in transformer.parameters()) / 1e9), flush=True)
    transformer.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    guidance_embeds = bool(getattr(transformer.config, "guidance_embeds", False))

    codec = Codec(pipe, dev, wdt)
    P, encode = codec.P, codec.encode

    pal_rng = np.random.default_rng(a.seed)

    def prepare(item, drop_ref_ids=False, pal=None):
        """Everything the transformer needs for one sample."""
        tgt = encode(item["target"])                           # (1, C, h, w)
        pack, refs = encode(item["ctx"][0]), [encode(im) for im in item["ctx"][1:]]
        use = a.part_ids in ("rope", "embed")
        npart = item.get("n_parts")
        tgt_ids = codec.target_ids(tgt, item["ctx"][0] if use else None, npart)   # (1, hw, 4)
        h, w = tgt.shape[-2:]
        rpid = None
        if (use or a.part_bias != "none") and item.get("ref_ids"):
            rpid = [part_ids_from_map(m, h, w, 0, npart) if m is not None else None
                    for m in item["ref_ids"]]
        extra, tail = [], []
        if a.substrate:
            sub = substrate_image(item["ctx"][0], item["ctx"][1:], item.get("ref_ids"), npart,
                                  stat=a.substrate_stat)
            extra = [encode(sub)]
        if a.id_images:
            # the photos' id images are withheld with the same chance as the
            # other id-map uses, so photos without maps still work
            maps = [] if (drop_ref_ids or not refs) else item.get("ref_ids", [])
            pim, rims = id_colour_images(item["ctx"][0], maps, npart, pal or pal_rng)
            extra.append(encode(pim))
            tail = [encode(x) if x is not None else None for x in rims]
        ctx_packed, ctx_ids = codec.context(pack, refs, tgt_ids, a.pack_ids,
                                            rpid if (use and not drop_ref_ids) else None,
                                            extra, tail)
        n_tail = sum(h * w for x in tail if x is not None)
        bias = None
        if a.part_bias != "none" and rpid is not None and refs:
            labs = [None if (r is None or (rng.random() < a.bias_drop)) else r[0] for r in rpid]
            if any(l is not None for l in labs):
                tgt_lab = part_ids_from_map(item["ctx"][0], h, w, 2, npart)[0]
                bias = attention_bias(tgt_lab, labs, prompt_embeds.shape[1], 1 + len(extra),
                                      a.bias_beta, a.bias_gamma, a.part_bias == "hard", dev, wdt,
                                      n_tail, a.bias_bg)
        # loss weight per target token: object vs background
        h, w = tgt.shape[-2:]
        m = torch.from_numpy(np.asarray(item["mask"], np.float32) / 255.0)[None, None]
        m = F.max_pool2d(m.to(dev), 9, 1, 4)
        m = F.interpolate(m, size=(h, w), mode="bilinear").flatten(1)   # (1, hw)
        obj = (m > 0.2).float()
        # row 0: loss weight per token; row 1: 1 where the token is background
        wgt = torch.cat([a.bg_weight + (1 - a.bg_weight) * obj, 1 - obj], 0)
        return tgt, tgt_ids, ctx_packed, ctx_ids, wgt, bias

    def predict(noisy_packed, tgt_ids, ctx_packed, ctx_ids, t, bias=None):
        hs = torch.cat([noisy_packed, ctx_packed], 1)
        ids = torch.cat([tgt_ids, ctx_ids], 1)
        if embed is not None:                 # identity via the embedding, not RoPE
            embed.ids = ids[..., 3]
            ids = ids.clone()
            ids[..., 3] = 0
        kw = {}
        if bias is not None:
            kw["joint_attention_kwargs"] = {"attention_mask": bias}
        if guidance_embeds:
            kw["guidance"] = torch.full((hs.shape[0],), 1.0, device=dev, dtype=torch.float32)
        with torch.autocast("cuda", dtype=wdt):
            pred = transformer(hidden_states=hs, timestep=t / 1000,
                               encoder_hidden_states=prompt_embeds,
                               txt_ids=text_ids, img_ids=ids,
                               return_dict=False, **kw)[0]
        return pred[:, :noisy_packed.shape[1]]

    sigmas_all = sched.sigmas.to(dev).float()          # shifted, 1000 entries
    tsteps_all = sched.timesteps.to(dev).float()

    def loss_at(tgt, tgt_ids, ctx_packed, ctx_ids, wgt, noise, idx, bias=None, with_x0=False):
        sigma = sigmas_all[idx].view(1, 1, 1, 1)
        t = tsteps_all[idx].view(1)
        noisy = (1 - sigma) * tgt.float() + sigma * noise
        pred = predict(P._pack_latents(noisy.to(wdt)), tgt_ids, ctx_packed, ctx_ids, t, bias)
        target = P._pack_latents(noise - tgt.float())
        err = ((pred.float() - target) ** 2).mean(-1)          # (1, hw)
        w, bgm = wgt[0:1], wgt[1:2]
        loss = (err * w).sum() / w.sum()
        if a.topbg > 0:
            bg = err * bgm                                       # background tokens
            k = max(1, int(0.1 * bg.shape[1]))
            loss = loss + a.topbg * torch.topk(bg, k, dim=1).values.mean()
        if with_x0:
            # the image the prediction implies, against the true latent: a
            # fit measure that is well conditioned at every noise level
            # (the v target itself blows up as sigma -> 0, so the v loss has
            # a floor near 1 at the least-noisy fixed points however good
            # the fit; that floor is what keeps fixed_train from reaching 0)
            x0_hat = P._pack_latents(noisy) - float(sigma) * pred.float()
            x0_err = ((x0_hat - P._pack_latents(tgt.float())) ** 2).mean()
            return loss, x0_err
        return loss

    # ------------------------------------------------------------ data
    objs = json.load(open(a.objects)) if a.objects else None
    if isinstance(objs, dict):
        objs = objs["objects"]
    ds = Frames(a.data, "train", objs, a.n_refs, a.res, a.variants, a.max_frames)
    ho = str(a.hist_offset)
    hist_off = tuple(int(x) for x in ho.split("-")) if "-" in ho else int(ho)
    ds.hist_offset, ds.hist_label_drop, ds.hist_earlier = hist_off, a.hist_label_drop, a.hist_earlier
    ds.hist_augment = a.hist_augment
    rng = random.Random(a.seed)

    # ------------------------------------------------------------ fixed sets
    # Same frames, same noise, same timesteps at every report, so the two
    # curves are comparable with each other and across steps.
    def build_fixed(split, objects, n, tag, variants=None, max_frames=None, history=None):
        d = Frames(a.data, split, objects, a.n_refs, a.res, variants, max_frames)
        # fixed sets: always the middle of the range, always with labels
        d.hist_offset = (hist_off[0] + hist_off[1]) // 2 if isinstance(hist_off, tuple) else hist_off
        if not d.rows:
            return []
        picks = np.unique(np.linspace(0, len(d.rows) - 1, min(n, len(d.rows))).round().astype(int))
        ts = np.linspace(20, 980, len(picks)).round().astype(int)
        g = torch.Generator(); g.manual_seed(1234)
        out = []
        fixed_pal = np.random.default_rng(1234)     # fixed id colours per report
        for i, ti in zip(picks, ts):
            tgt, tgt_ids, ctx_packed, ctx_ids, wgt, bias = prepare(
                d.item(int(i), history=(a.history > 0) if history is None else history), pal=fixed_pal)
            out.append((tgt, tgt_ids, ctx_packed, ctx_ids, wgt,
                        torch.randn(tgt.shape, generator=g).to(dev), int(999 - ti), bias))
        print("[val] %s: %d frames over %d objects" % (tag, len(picks),
              len({d.rows[int(i)]["obj"] for i in picks})), flush=True)
        return out

    @torch.no_grad()
    def fixed_loss(batches):
        """(v-loss, image error) averaged over the fixed set."""
        transformer.eval()
        tot, tot_x0 = 0.0, 0.0
        for tgt, tgt_ids, ctx_packed, ctx_ids, wgt, noise, idx, bias in batches:
            l, x0 = loss_at(tgt, tgt_ids, ctx_packed, ctx_ids, wgt, noise, idx, bias, with_x0=True)
            tot += float(l)
            tot_x0 += float(x0)
        transformer.train()
        n = max(len(batches), 1)
        return tot / n, tot_x0 / n

    fixed_tr = build_fixed("train", objs, a.val_frames, "training frames", a.variants,
                           a.max_frames) if (a.val_every and not a.eval_only) else []
    fixed_ho = build_fixed(a.val_split, None, a.val_frames, "held-out frames",
                           a.val_variants) if a.val_every else []
    # with photos in training, also track the held-out loss WITHOUT a photo and
    # pick the checkpoint on the mean: the model must not get worse without one
    fixed_ho_np = build_fixed(a.val_split, None, a.val_frames, "held-out frames, no photo",
                              a.val_variants, history=False) \
        if (a.val_every and a.history > 0 and not a.eval_only) else []

    if a.eval_only:
        # held-out loss by noise level: each fixed frame has its own timestep,
        # spread evenly from almost clean to almost pure noise
        transformer.eval()
        bands = {}
        with torch.no_grad():
            for tgt, tgt_ids, ctx_packed, ctx_ids, wgt, noise, idx, bias in fixed_ho:
                l, x0 = loss_at(tgt, tgt_ids, ctx_packed, ctx_ids, wgt, noise, idx, bias, with_x0=True)
                sig = float(sigmas_all[idx])
                b = min(int(sig * 4), 3)
                bands.setdefault(b, []).append((float(l), float(x0)))
        names = ["sigma 0.00-0.25 (fine detail)", "sigma 0.25-0.50", "sigma 0.50-0.75",
                 "sigma 0.75-1.00 (layout, colour)"]
        allv = [v for b in bands.values() for v in b]
        print("EVALONLY %s  frames %d  v-loss %.4f  x0 %.4f" % (a.resume, len(allv),
              np.mean([v[0] for v in allv]), np.mean([v[1] for v in allv])), flush=True)
        for b in sorted(bands):
            print("EVALONLY   %-34s n=%3d  v-loss %.4f  x0 %.4f" % (names[b], len(bands[b]),
                  np.mean([v[0] for v in bands[b]]), np.mean([v[1] for v in bands[b]])), flush=True)
        return

    # ------------------------------------------------------------ train
    lora_scale = (a.lora_alpha or a.rank) / a.rank

    def save(tag):
        d = os.path.join(a.out, tag)
        os.makedirs(d, exist_ok=True)
        sd = get_peft_model_state_dict(transformer)
        if lora_scale != 1.0:
            # the saved file carries no alpha, so every loader applies the
            # adapter at scale 1: bake alpha/rank into the B matrices
            sd = {k: (v * lora_scale if "lora_B" in k else v) for k, v in sd.items()}
        Flux2KleinPipeline.save_lora_weights(d, transformer_lora_layers=sd)
        json.dump({"rank": a.rank, "alpha": a.lora_alpha or a.rank, "baked_scale": lora_scale},
                  open(os.path.join(d, "lora_meta.json"), "w"))
        if embed is not None:
            torch.save(embed.state_dict(), os.path.join(d, "part_embed.pt"))
        print("[save] %s" % d, flush=True)

    transformer.train()
    order = list(range(len(ds.rows)))
    rng.shuffle(order)
    pos, step, t0 = 0, a.start_step, time.time()
    for _ in range(step):                  # replay the shuffles so data order continues
        pos += a.accum
        if pos >= len(order):
            rng.shuffle(order); pos = 0
    run_loss = []
    done_before = step
    best_ho, since_best = float("inf"), 0
    best_f = os.path.join(a.out, "best.json")
    if a.resume and a.save_best and os.path.exists(best_f):
        bj = json.load(open(best_f))                  # a continued run keeps its best
        best_ho = bj.get("fixed_heldout" if a.select == "v" else "x0_heldout", float("inf"))
        print("[resume] previous best held-out %.4f at step %d" % (best_ho, bj.get("step", -1)), flush=True)
    while step < a.steps:
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(a.accum):
            if pos >= len(order):
                rng.shuffle(order); pos = 0
            # curriculum: pack only at first, then references with dropout
            drop = 1.0 if step < a.ref_warmup * a.steps else a.ref_drop
            item = ds.item(order[pos], rng, no_refs=rng.random() < drop,
                           history=a.history > 0 and rng.random() < a.history)
            pos += 1
            tgt, tgt_ids, ctx_packed, ctx_ids, wgt, bias = prepare(
                item, drop_ref_ids=rng.random() < a.ref_id_drop)
            # logit-normal timestep density, as the klein trainer uses
            u = float(torch.sigmoid(torch.randn(1)))
            idx = min(int(u * 1000), 999)
            noise = torch.randn(tgt.shape, device=dev)
            loss = loss_at(tgt, tgt_ids, ctx_packed, ctx_ids, wgt, noise, idx, bias) / a.accum
            loss.backward()
            acc += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if a.lr_final is not None or a.lr_warmup:
            # the schedule runs from --sched-start (0, or the resume step for a
            # continuation with its own warm-up and decay) to --steps
            s0 = a.sched_start
            frac = min(max((step - s0) / max(a.steps - s0, 1), 0.0), 1.0)
            lr_now = a.lr if a.lr_final is None else \
                a.lr_final + 0.5 * (a.lr - a.lr_final) * (1 + math.cos(math.pi * frac))
            if a.lr_warmup and step - s0 < a.lr_warmup:
                w = (step - s0 + 1) / a.lr_warmup
                lr_now = a.lr_warm_from + (lr_now - a.lr_warm_from) * w
            for g in opt.param_groups:
                g["lr"] = lr_now
        opt.step()
        step += 1
        run_loss.append(acc)
        if step % 25 == 0:
            print("step %d/%d loss %.4f (avg25 %.4f) %.1f s/step mem %.0fG lr %.1e"
                  % (step, a.steps, acc, float(np.mean(run_loss[-25:])),
                     (time.time() - t0) / max(step - done_before, 1),
                     torch.cuda.max_memory_allocated() / 1e9, opt.param_groups[0]["lr"]), flush=True)
        if a.val_every and (step % a.val_every == 0 or step == a.steps):
            (ltr, xtr), (lho, xho) = fixed_loss(fixed_tr), fixed_loss(fixed_ho)
            extra = ""
            if fixed_ho_np:
                lnp, _ = fixed_loss(fixed_ho_np)
                extra = " nophoto_heldout %.4f" % lnp
            print("val step %d fixed_train %.4f fixed_heldout %.4f x0_train %.4f x0_heldout %.4f%s"
                  % (step, ltr, lho, xtr, xho, extra), flush=True)
            if fixed_ho_np:
                lho = 0.5 * (lho + lnp)           # selection on with- and without-photo
            crit = lho if a.select == "v" else xho
            if a.save_best and fixed_ho and crit < best_ho:
                best_ho, since_best = crit, 0
                save("lora_best")
                json.dump({"step": step, "x0_heldout": xho, "fixed_heldout": lho,
                           "x0_train": xtr, "fixed_train": ltr},
                          open(os.path.join(a.out, "best.json"), "w"), indent=1)
            else:
                since_best += 1
            if a.patience and a.save_best and since_best >= a.patience:
                print("early stop at step %d: no held-out improvement for %d validations"
                      % (step, a.patience), flush=True)
                save("lora")
                break
        if step % a.save_every == 0 or step == a.steps:
            save("lora" if step == a.steps else "lora_step%05d" % step)
    print("TRAIN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
