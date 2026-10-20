#!/usr/bin/env python3
"""Shared klein pieces: image encoding, context position ids, sampling.

Training, evaluation and the demo all go through this file, so the layout of
the token sequence is the same everywhere:

    [ target tokens | pack tokens | ref 1 tokens | ref 2 | ref 3 | ... ]

Position ids decide what the model can do with each block. The pack is
SPATIALLY ALIGNED with the target, so it shares the target's (h, w) ids
(OminiControl's finding: aligned conditions must share ids, or the model has
no cheap way to learn that pack pixel (x, y) speaks about target pixel
(x, y)). The references are photos from other cameras, so they get the
pipeline's usual offset ids (t = 10, 20, 30, ...), which lets attention look
for appearance anywhere in them. `pack_ids="offset"` reproduces the first
runs, where the pack was offset like a reference and the model tended to
draw the object in the photos instead of the geometry in the pack.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

PROMPT = "a photograph of a mechanical assembly on a studio table"
DISTILLED = "black-forest-labs/FLUX.2-klein-4B"
BASE = "black-forest-labs/FLUX.2-klein-base-4B"


def to_tensor(im):
    return torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1.0).permute(2, 0, 1)


def part_ids_from_map(im: Image.Image, h: int, w: int, channel: int = 2,
                      n_parts: int | None = None):
    """Per-token part index from a part-id image, 0 for background.

    The capture writes part k as level (k + 1) / (n_parts + 1) in 0..255.
    With n_parts known every pixel is snapped to the nearest level and the
    id is the small integer k + 1, which is what goes on the position axis.
    Without it the raw level is used, after dropping rare (edge-blend) values.
    Each token's id is the most common non-zero id in its 16x16 patch.
    """
    a = np.asarray(im.convert("RGB").resize((w * 16, h * 16), Image.NEAREST))[:, :, channel]
    if n_parts:
        step = 255.0 / (n_parts + 1)
        k = np.clip(np.rint(a / step), 0, n_parts).astype(np.int64)
        # values well below the first level are edge blends with black
        k[a < 0.5 * step] = 0
        a = k
    else:
        hist = np.bincount(a.reshape(-1), minlength=256)
        valid = np.zeros(256, bool)
        valid[hist >= 24] = True
        valid[0] = False
        a = np.where(valid[a], a, 0).astype(np.int64)
    a = a.reshape(h, 16, w, 16).transpose(0, 2, 1, 3).reshape(h, w, 256)
    out = np.zeros((h, w), np.int64)
    for i in range(h):
        for j in range(w):
            v = a[i, j]
            v = v[v > 0]
            if v.size:
                vals, cnt = np.unique(v, return_counts=True)
                out[i, j] = int(vals[np.argmax(cnt)])
    return torch.from_numpy(out.reshape(1, h * w))


def pixel_part_ids(im: Image.Image, channel: int, n_parts: int, size=None):
    """Per-pixel part index (0 = background) from a part-id image."""
    a = np.asarray(im.convert("RGB") if size is None else
                   im.convert("RGB").resize(size, Image.NEAREST))[:, :, channel].astype(np.float64)
    step = 255.0 / (n_parts + 1)
    k = np.clip(np.rint(a / step), 0, n_parts).astype(np.int64)
    k[a < 0.5 * step] = 0
    return k


def srgb_to_lab(rgb):
    """rgb in 0..1, (..., 3) -> CIE Lab (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab):
    """CIE Lab (D65) -> rgb in 0..1, the inverse of srgb_to_lab."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    f = np.stack([fy + a / 500, fy, fy - b / 200], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) \
        * np.array([0.95047, 1.0, 1.08883])
    Mi = np.array([[3.2404542, -1.5371385, -0.4985314],
                   [-0.9692660, 1.8760108, 0.0415560],
                   [0.0556434, -0.2040259, 1.0572252]])
    c = np.clip(xyz @ Mi.T, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def part_colour_robust(pixels):
    """Albedo estimate of one part from its photo pixels (N, 3) in 0..255.

    The mean darkens glossy and shadowed parts (a lit surface averages to
    albedo x ~0.6, highlights and reflections pull the hue) and the model
    then multiplies by the map's shading a second time. Instead: chroma is
    the median of Lab a, b (robust to highlights, reflections and edge
    blends), lightness is the 80th percentile of L (the well-lit pixels,
    below the specular top), which is close to the unshaded albedo.
    """
    lab = srgb_to_lab(pixels / 255.0)
    est = np.array([np.percentile(lab[:, 0], 80), np.median(lab[:, 1]), np.median(lab[:, 2])])
    return lab_to_srgb(est) * 255.0


def substrate_image(pack: Image.Image, refs, ref_id_maps, n_parts, gain=1.07,
                    grey=0.5, stat="mean"):
    """The reference-coloured analytic render.

    Every part's colour is read from the reference photos DIRECTLY: the mean
    of the photo pixels whose id map says "this part", over all photos that
    show it. That flat colour is painted onto the pack and multiplied by the
    pack's shading, which is the analytic baseline with the true albedo
    replaced by what the photos say. Parts no photo shows stay grey. Fed to
    the renderer as an input aligned with the target, this makes "which
    colour goes where" a lookup we compute instead of one attention has to
    discover; it is also the fair baseline for the paper.
    stat: "mean" (the pixel mean, as trained so far) or "robust" (median
    chroma, bright-quartile lightness: see part_colour_robust).
    """
    W, H = pack.size
    pk = np.asarray(pack.convert("RGB")).astype(np.float64)
    ids = pixel_part_ids(pack, 2, n_parts)
    shade = pk[:, :, 1:2] / 255.0
    colour = {}
    if refs and ref_id_maps and n_parts:
        acc, cnt, pix = {}, {}, {}
        for im, idm in zip(refs, ref_id_maps):
            if idm is None:
                continue
            rgb = np.asarray(im.convert("RGB").resize((W, H), Image.BILINEAR)).astype(np.float64)
            rid = pixel_part_ids(idm, 0, n_parts, (W, H))
            for k in np.unique(rid):
                if k == 0:
                    continue
                m = rid == k
                acc[k] = acc.get(k, 0) + rgb[m].sum(0)
                cnt[k] = cnt.get(k, 0) + int(m.sum())
                if stat == "robust":
                    pix.setdefault(k, []).append(rgb[m])
        if stat == "robust":
            colour = {k: part_colour_robust(np.concatenate(pix[k], 0))
                      for k in pix if cnt[k] >= 30}
        else:
            colour = {k: acc[k] / cnt[k] for k in acc if cnt[k] >= 30}
    alb = np.full((H, W, 3), grey * 255.0)
    for k in np.unique(ids):
        if k and k in colour:
            alb[ids == k] = colour[k]
    out = np.clip(alb / 255.0 * shade * gain, 0, 1) * 255.0
    out[ids == 0] = 0
    return Image.fromarray(out.astype(np.uint8))


def id_colour_images(pack: Image.Image, ref_id_maps, n_parts, rng=None):
    """The same random colour for the same part, on the blockout and on every
    reference photo that has an id map: one image aligned with the blockout
    (from the pack's B channel) and one per reference (from its id map, or
    None). Colours are drawn afresh per call so a colour never means a part;
    only "equal on both sides" carries information. This is the id-image
    variant: the link is handed to the model as pictures and attention has to
    learn to match them, unlike the bias which applies the match itself.
    """
    rng = rng or np.random.default_rng()
    n = int(n_parts or 0)
    pal = np.zeros((n + 1, 3), np.uint8)
    if n:
        hue = rng.permutation(n) / n + rng.random() / n            # spread hues
        for k, hh in enumerate(hue):
            i = int(hh * 6) % 6
            f = hh * 6 - int(hh * 6)
            v, s = 0.55 + 0.45 * rng.random(), 0.75 + 0.25 * rng.random()
            p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
            r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
            pal[k + 1] = np.rint(np.array([r, g, b]) * 255)
    W, H = pack.size
    out = [Image.fromarray(pal[pixel_part_ids(pack, 2, n)])]
    refs = []
    for idm in ref_id_maps or []:
        refs.append(None if idm is None else
                    Image.fromarray(pal[pixel_part_ids(idm, 0, n, (W, H))]))
    return out[0], refs


def attention_bias(tgt_lab, ref_labs, n_txt, n_aligned, beta=3.0, gamma=3.0,
                   hard=False, device="cuda", dtype=torch.bfloat16, n_tail=0, bg=0.0):
    """Part-correspondence bias for the joint attention, (1, 1, L, L).

    The joint sequence is [text | target | aligned images (map, substrate) |
    reference photos]. Only the rows of the target and the aligned images
    against the columns of the reference photos are touched: a query token of
    part k gets +beta on reference tokens of part k and -gamma on reference
    tokens of any other part or of the photo's clutter (label 0). Background
    queries (label 0), references without labels, text, self and reference
    rows all get 0, so the model's prior is left alone there and no row is
    ever fully masked. `hard` uses -1e4 instead of -gamma (a Parts2Whole /
    EasyControl style mask). Labels are per token, (hw,) tensors. n_aligned
    counts the aligned context images (map, substrate, id image), not the
    target; n_tail counts tokens appended after the references (the
    reference id images), which get 0 everywhere.
    bg > 0 also pushes BACKGROUND screen tokens (label 0: where the map shows
    nothing) away from the photos' part tokens by -bg, so empty regions cannot
    copy parts out of the photos; they keep 0 towards the photos' table.
    """
    hw = int(tgt_lab.numel())
    n_ref = len(ref_labs)
    n_rows = 1 + n_aligned                    # the target block, then the aligned images
    L = n_txt + n_rows * hw + n_ref * hw + n_tail
    bias = torch.zeros(L, L, device=device, dtype=torch.float32)
    q = tgt_lab.view(hw, 1).to(device)
    k0 = n_txt + n_rows * hw
    for i, rl in enumerate(ref_labs):
        if rl is None:
            continue
        r = rl.view(1, hw).to(device)
        same = (q == r) & (q > 0)
        diff = (q != r) & (q > 0)
        blk = torch.zeros(hw, hw, device=device)
        blk[diff] = -1e4 if hard else -float(gamma)
        blk[same] = 0.0 if hard else float(beta)
        if bg:
            blk[(q == 0) & (r > 0)] = -float(bg)
        for a in range(n_rows):
            r0 = n_txt + a * hw
            bias[r0:r0 + hw, k0 + i * hw:k0 + (i + 1) * hw] = blk
    return bias[None, None].to(dtype)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """The pipeline's timestep-shift rule (copied so the sampler matches)."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


class PartEmbed(torch.nn.Module):
    """Part identity added to every image token as a learned vector.

    A trainable table indexed by part number (0 = background), added to the
    output of the transformer's image input projection, for the target, the
    pack and the reference tokens alike. Unlike the rotary axis, this puts
    the identity straight into the residual stream, where a LoRA can use it
    at once. Zero-initialised, so training starts from the plain model.
    """

    def __init__(self, dim, n_max=128):
        super().__init__()
        self.table = torch.nn.Embedding(n_max, dim)
        torch.nn.init.zeros_(self.table.weight)
        self.ids = None                       # (1, n_image_tokens), set per forward

    def install(self, transformer):
        name = "x_embedder" if hasattr(transformer, "x_embedder") else "img_in"
        mod = getattr(transformer, name)

        def hook(_m, _inp, out):
            if self.ids is None:
                return out
            return out + self.table(self.ids.to(out.device).clamp(0, self.table.num_embeddings - 1)).to(out.dtype)
        mod.register_forward_hook(hook)
        return self


class Codec:
    """VAE encode/decode with klein's patchify and batch-norm statistics."""

    def __init__(self, pipe, device="cuda", dtype=torch.bfloat16):
        from diffusers import Flux2KleinPipeline
        self.P = Flux2KleinPipeline
        self.pipe, self.dev, self.wdt = pipe, device, dtype
        self.vae = pipe.vae
        eps = getattr(self.vae.config, "batch_norm_eps", 1e-4)
        self.bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device, torch.float32)
        self.bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + eps) \
            .to(device, torch.float32)

    @torch.no_grad()
    def encode(self, im: Image.Image):
        """PIL -> normalised, patchified latent (1, C, H/16, W/16)."""
        x = to_tensor(im)[None].to(self.dev, self.wdt)
        lat = self.vae.encode(x).latent_dist.mode()
        lat = self.P._patchify_latents(lat).float()
        return ((lat - self.bn_mean) / self.bn_std).to(self.wdt)

    @torch.no_grad()
    def decode(self, packed, h, w):
        """(1, h*w, C) target tokens in raster order -> PIL image."""
        lat = packed.float().permute(0, 2, 1).reshape(1, -1, h, w)
        lat = lat * self.bn_std + self.bn_mean
        lat = self.P._unpatchify_latents(lat).to(self.wdt)
        img = self.vae.decode(lat, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(img, output_type="pil")[0]

    def context(self, pack_lat, ref_lats, tgt_ids, pack_ids="aligned", ref_part_ids=None,
                aligned_lats=(), ref_aligned_lats=()):
        """Packed context tokens and their ids for one sample.

        ref_part_ids: per reference, a (1, hw) tensor of part ids for the
        fourth position axis, or None for a reference without an id map.
        aligned_lats: further images aligned with the target (they share its
        ids), e.g. the reference-coloured substrate render.
        ref_aligned_lats: per reference, an image aligned with THAT reference
        (shares its ids), e.g. its id-colour image, or None. They go after
        the references in the sequence.
        """
        P = self.P
        if pack_ids == "aligned":
            packed = [P._pack_latents(pack_lat)] + [P._pack_latents(x) for x in aligned_lats]
            ids = [tgt_ids] * (1 + len(aligned_lats))
            if ref_lats:
                packed.append(torch.cat([P._pack_latents(r) for r in ref_lats], 1))
                rid = P._prepare_image_ids([r[0] for r in ref_lats]).to(self.dev)
                n = ref_lats[0].shape[-2] * ref_lats[0].shape[-1]
                if ref_part_ids is not None:
                    rid = rid.clone()
                    for k, pid in enumerate(ref_part_ids):
                        if pid is not None:
                            rid[0, k * n:(k + 1) * n, 3] = pid[0].to(self.dev)
                ids.append(rid)
                for k, x in enumerate(ref_aligned_lats):
                    if x is not None:
                        packed.append(P._pack_latents(x))
                        ids.append(rid[:, k * n:(k + 1) * n])
        else:                       # the first runs: pack offset like a reference
            allc = [pack_lat] + list(ref_lats)
            packed = [torch.cat([P._pack_latents(c) for c in allc], 1)]
            ids = [P._prepare_image_ids([c[0] for c in allc]).to(self.dev)]
        return torch.cat(packed, 1), torch.cat(ids, 1)

    def target_ids(self, lat, pack_im=None, n_parts=None):
        """Position ids of the target (and the aligned pack). With a pack
        image, the fourth axis carries the part id under each token, so a
        target token and the reference tokens of the same part sit at zero
        distance on that axis and attention can match by part."""
        ids = self.P._prepare_latent_ids(lat).to(self.dev)
        if pack_im is not None:
            h, w = lat.shape[-2:]
            ids = ids.clone()
            ids[0, :, 3] = part_ids_from_map(pack_im, h, w, 2, n_parts)[0].to(self.dev)
        return ids


class Renderer:
    """Distilled klein + LoRA, sampling with our own token layout."""

    def __init__(self, model=DISTILLED, lora=None, pack_ids="aligned", part_ids="none",
                 substrate=False, part_bias="none", bias_beta=3.0, bias_gamma=3.0,
                 device="cuda", dtype=torch.bfloat16, compile=False, id_images=False,
                 substrate_stat="mean", bias_bg=0.0):
        """part_ids: none | rope (4th position axis) | embed (learned vector).
        part_bias: none | soft | hard, the attention bias from the part maps.
        id_images: the same-colour id image on the blockout and each photo.
        substrate_stat: mean | robust, how a part's colour is read from the photos."""
        from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
        pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=dtype)
        if lora:
            pipe.load_lora_weights(lora)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe, self.dev, self.wdt, self.pack_ids = pipe, device, dtype, pack_ids
        self.part_ids, self.substrate, self.id_images = part_ids, substrate, id_images
        self.substrate_stat = substrate_stat
        self.bias_bg = bias_bg
        self.part_bias, self.bias_beta, self.bias_gamma = part_bias, bias_beta, bias_gamma
        self.embed = None
        if part_ids == "embed":
            self.embed = PartEmbed(pipe.transformer.config.num_attention_heads
                                   * pipe.transformer.config.attention_head_dim).to(device)
            pe_path = os.path.join(lora, "part_embed.pt") if lora else None
            if pe_path and os.path.exists(pe_path):
                self.embed.load_state_dict(torch.load(pe_path, map_location=device))
            self.embed.install(pipe.transformer)
        self.codec = Codec(pipe, device, dtype)
        self.transformer = pipe.transformer
        if compile:
            self.transformer = torch.compile(self.transformer)
        self.sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        with torch.no_grad():
            self.pe, self.tids = pipe.encode_prompt(prompt=PROMPT, device=device,
                                                    max_sequence_length=512)
            self.npe, self.ntids = pipe.encode_prompt(prompt="", device=device,
                                                      max_sequence_length=512)
        self.pe, self.npe = self.pe.to(device, dtype), self.npe.to(device, dtype)
        self.guidance_embeds = bool(getattr(self.transformer.config, "guidance_embeds", False))

    def _pred(self, hs, ids, t, pe, tids, bias=None):
        if self.embed is not None:            # identity via the embedding, not RoPE
            self.embed.ids = ids[..., 3]
            ids = ids.clone()
            ids[..., 3] = 0
        kw = {}
        if bias is not None:
            kw["joint_attention_kwargs"] = {"attention_mask": bias}
        if self.guidance_embeds:
            kw["guidance"] = torch.full((hs.shape[0],), 1.0, device=self.dev, dtype=torch.float32)
        return self.transformer(hidden_states=hs, timestep=t / 1000,
                                encoder_hidden_states=pe, txt_ids=tids, img_ids=ids,
                                return_dict=False, **kw)[0]

    @torch.no_grad()
    def render(self, pack: Image.Image, refs, steps=4, cfg=1.0, seed=0, res=512,
               ref_id_maps=None, n_parts=None):
        """ref_id_maps: per reference, its part-id image or None.
        n_parts: the object's part count, to snap id levels exactly."""
        P = self.codec.P
        pack_lat = self.codec.encode(pack)
        ref_lats = [self.codec.encode(r) for r in refs]
        C, h, w = pack_lat.shape[1:]
        g = torch.Generator(self.dev).manual_seed(int(seed))
        lat = torch.randn((1, C, h, w), generator=g, device=self.dev, dtype=torch.float32)
        use = self.part_ids in ("rope", "embed")
        tgt_ids = self.codec.target_ids(lat, pack if use else None, n_parts)
        rpid = None
        if (use or self.part_bias != "none") and ref_id_maps:
            rpid = [part_ids_from_map(m, h, w, 0, n_parts) if m is not None else None
                    for m in ref_id_maps]
        extra, tail = [], []
        if self.substrate:
            extra = [self.codec.encode(substrate_image(pack, refs, ref_id_maps, n_parts,
                                                       stat=self.substrate_stat))]
        if self.id_images:
            pim, rims = id_colour_images(pack, ref_id_maps if ref_lats else [], n_parts,
                                         np.random.default_rng(int(seed)))
            extra.append(self.codec.encode(pim))
            tail = [self.codec.encode(x) if x is not None else None for x in rims]
        ctx, ctx_ids = self.codec.context(pack_lat, ref_lats, tgt_ids, self.pack_ids,
                                          rpid if use else None, extra, tail)
        n_tail = sum(h * w for x in tail if x is not None)
        bias = None
        if self.part_bias != "none" and rpid is not None and ref_lats:
            tgt_lab = part_ids_from_map(pack, h, w, 2, n_parts)[0]
            bias = attention_bias(tgt_lab, rpid and [r[0] if r is not None else None for r in rpid],
                                  self.pe.shape[1], 1 + len(extra), self.bias_beta,
                                  self.bias_gamma, self.part_bias == "hard", self.dev, self.wdt,
                                  n_tail, self.bias_bg)
        x = P._pack_latents(lat)                                   # (1, hw, C)
        ids = torch.cat([tgt_ids, ctx_ids], 1)
        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = compute_empirical_mu(x.shape[1], steps)
        self.sched.set_timesteps(steps, device=self.dev, sigmas=sigmas, mu=mu)
        for t in self.sched.timesteps:
            hs = torch.cat([x.to(self.wdt), ctx], 1)
            tt = t.expand(1).to(self.wdt)
            pred = self._pred(hs, ids, tt, self.pe, self.tids, bias)[:, :x.shape[1]]
            if cfg > 1.0:
                nb = None
                if bias is not None and self.npe.shape[1] != self.pe.shape[1]:
                    nb = attention_bias(tgt_lab, [r[0] if r is not None else None for r in rpid],
                                        self.npe.shape[1], 1 + len(extra), self.bias_beta,
                                        self.bias_gamma, self.part_bias == "hard", self.dev, self.wdt,
                                        n_tail, self.bias_bg)
                else:
                    nb = bias
                npred = self._pred(hs, ids, tt, self.npe, self.ntids, nb)[:, :x.shape[1]]
                pred = npred + cfg * (pred - npred)
            x = self.sched.step(pred.float(), t, x, return_dict=False)[0]
        return self.codec.decode(x, h, w)
