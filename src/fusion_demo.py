#!/usr/bin/env python3
"""F1-manual style demo from the Fusion 360 Gallery assembly dataset:

    left   the agent's high-level blockout: one fitted primitive per part
           (box or cylinder, whichever wraps tighter), schematic style
    right  the real CAD, shaded -- the render target

`--manual` produces the animated version: the camera orbits while parts fly
in one step at a time, with a step counter, like the F1 deck GIFs. The
dataset has no assembly instructions, so the order is DERIVED: largest part
first, then whichever remaining part shares the most annotated contacts with
what is already placed (nearest part as the tie-break). Honest label: a
plausible order, not ground truth.

Pure numpy + PIL painter renderer: no GL, no GPU, runs anywhere.

  .venv/bin/python fusion_demo.py --inspect             # schema of one assembly
  .venv/bin/python fusion_demo.py --n 6                 # stills of six best
  .venv/bin/python fusion_demo.py --manual auto         # animate the best one
  .venv/bin/python fusion_demo.py --manual <dir name>   # animate a chosen one
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "demo")

INK = (30, 30, 32)
PAPER = (252, 252, 251)
RULE = (210, 210, 206)
MUTE = (120, 120, 118)

# appearance names carry no RGB in the dataset, so colour comes from keywords
# in the user-assigned material name, with a stable pastel fallback per part
NAME_COLORS = [
    ("red", (196, 74, 60)), ("blue", (70, 110, 190)), ("green", (86, 150, 96)),
    ("yellow", (222, 188, 70)), ("orange", (226, 140, 60)),
    ("black", (60, 60, 64)), ("white", (235, 235, 232)),
    ("steel", (150, 155, 162)), ("iron", (120, 122, 128)),
    ("aluminum", (176, 180, 186)), ("brass", (190, 160, 90)),
    ("copper", (188, 120, 90)), ("chrome", (200, 205, 212)),
    ("gold", (212, 175, 96)), ("wood", (168, 128, 88)),
    ("plastic", (140, 150, 165)),
]
PALETTE = [(168, 140, 120), (120, 140, 168), (140, 160, 130), (180, 160, 110),
           (150, 130, 160), (130, 160, 160), (180, 130, 130), (140, 140, 150)]


def color_for(name: str, i: int):
    n = (name or "").lower()
    for key, c in NAME_COLORS:
        if key in n:
            return c
    return PALETTE[i % len(PALETTE)]


# ------------------------------------------------------------ assembly loading

def transform_of(occ: dict) -> np.ndarray:
    """Occurrence transform: Fusion gives a coordinate system (origin + axes);
    some exports give a flat 4x4. Accept both."""
    t = occ.get("transform")
    if t is None:
        return np.eye(4)
    if isinstance(t, dict):
        def vec(*keys, d=(0, 0, 0)):
            for k in keys:
                if k in t:
                    v = t[k]
                    if isinstance(v, dict):
                        return np.array([v.get("x", 0), v.get("y", 0),
                                         v.get("z", 0)], float)
                    return np.asarray(v, float)
            return np.asarray(d, float)
        m = np.eye(4)
        m[:3, 0] = vec("x_axis", "xAxis", d=(1, 0, 0))
        m[:3, 1] = vec("y_axis", "yAxis", d=(0, 1, 0))
        m[:3, 2] = vec("z_axis", "zAxis", d=(0, 0, 1))
        m[:3, 3] = vec("origin", d=(0, 0, 0))
        return m
    return np.asarray(t, float).reshape(4, 4)


def _uids_in(obj, known: set, found: list):
    """Recursively collect strings that are body uids (contact entries vary in
    shape across exports; this reads any of them)."""
    if isinstance(obj, str):
        if obj in known:
            found.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _uids_in(v, known, found)
    elif isinstance(obj, list):
        for v in obj:
            _uids_in(v, known, found)


# Fastener vocabulary. Whole words only: "KINGPIN" is a structural pin and must
# not be caught by "pin", nor "PIVOT BUSHING" by anything here.
FASTENER_WORDS = {
    "screw", "bolt", "nut", "washer", "rivet", "stud", "dowel", "grub",
    "setscrew", "capscrew", "fastener", "circlip", "clip", "cotter",
    "vis", "boulon", "rondelle", "tornillo", "schraube", "mutter",
}
# Hardware named by its standard rather than by what it is: "DIN 128 - A5",
# "ANSI B18.2.4.5M - M5 x 0", "IFI 513 - M5x0", "GB 273.3-87".
FASTENER_STD = re.compile(
    r"^\s*(iso|din|ansi|asme|jis|gb|bs|nf|uni|sae|ifi|gost)\b", re.I)
# A bare metric designation: "M5x12", "M8 x 20".
FASTENER_METRIC = re.compile(r"^\s*m\d+(\s*[x×]\s*\d+)?\s*$", re.I)


def is_fastener(name: str) -> bool:
    """Whether a part is hardware joining other parts rather than a part."""
    n = (name or "").lower()
    if FASTENER_STD.match(n) or FASTENER_METRIC.match(n):
        return True
    return any(w in FASTENER_WORDS for w in re.findall(r"[a-z]+", n))


def load_assembly(adir: str, max_tris=200_000):
    """-> (parts, contact_pairs). parts = [(buid, name, mesh, colour)] with
    meshes posed in assembly coordinates."""
    j = json.load(open(os.path.join(adir, "assembly.json")))
    comps, bodies, occs = (j.get("components", {}), j.get("bodies", {}),
                           j.get("occurrences", {}))

    def body_mesh(buid: str):
        p = os.path.join(adir, buid + ".obj")
        if not os.path.exists(p):
            return None
        m = trimesh.load(p, force="mesh", process=False)
        return m if isinstance(m, trimesh.Trimesh) and len(m.faces) else None

    raw = []

    def comp_bodies(cuid):
        return [b for b in comps.get(cuid, {}).get("bodies", [])
                if b in bodies]

    paths: dict = {}

    def walk(node: dict, parent_tf: np.ndarray, path: tuple = ()):
        for ouid, child in (node or {}).items():
            occ = occs.get(ouid, {})
            tf = parent_tf @ transform_of(occ)
            if not occ.get("is_visible", True):
                continue
            here = path + (ouid,)
            paths[ouid] = here
            for buid in comp_bodies(occ.get("component", "")):
                m = body_mesh(buid)
                if m is not None:
                    m = m.copy()
                    m.apply_transform(tf)
                    raw.append([buid, m, ouid])
            walk(child, tf, here)

    tree = (j.get("tree") or {}).get("root", {})
    if tree:
        root_comp = j.get("root", {}).get("component")
        for buid in comp_bodies(root_comp) if root_comp else []:
            m = body_mesh(buid)
            if m is not None:
                raw.append([buid, m, ""])
        walk(tree, np.eye(4), ())
    if not raw:                        # no tree: every body where it stands
        for buid in bodies:
            m = body_mesh(buid)
            if m is not None:
                raw.append([buid, m, ""])

    tris = sum(len(m.faces) for _, m, _o in raw)
    if tris > max_tris:                # keep the painter fast: drop tiny parts
        raw.sort(key=lambda p: -len(p[1].faces))
        keep, acc = [], 0
        for p in raw:
            if acc + len(p[1].faces) > max_tris and keep:
                break
            keep.append(p)
            acc += len(p[1].faces)
        raw = keep

    METAL = ("steel", "iron", "alumin", "chrome", "brass", "copper",
             "gold", "metal", "titan", "nickel", "silver")
    parts = []
    # Facts the file already records about each occurrence, which the body
    # record does not carry: bodies are called "Body1", occurrences are called
    # "M8 Jam nut". Kept parallel to `parts` rather than widened into it, so
    # every existing consumer of the 5-tuple is untouched.
    info: dict = {"occ_name": [], "grounded": [], "volume": [],
                  "path": [], "occ": [], "pbr": []}
    for i, (buid, m, ouid) in enumerate(raw):
        b = bodies.get(buid, {})
        app = (b.get("appearance") or {}).get("name", "") or \
            (b.get("material") or {}).get("name", "")
        metal = int(any(k in (app or "").lower() for k in METAL))
        info["pbr"].append(pbr_for(app))
        parts.append((buid, b.get("name", buid), m, color_for(app, i), metal))

        occ = occs.get(ouid, {})
        # ":1" is Fusion's instance suffix, not part of the name.
        oname = re.sub(r":\d+$", "", occ.get("name") or "").strip()
        info["occ_name"].append(oname or b.get("name", buid))
        info["grounded"].append(bool(occ.get("is_grounded")))
        # Real solid volume. The heuristics used bounding-box volume, which
        # calls a long thin bolt one of the biggest parts in the assembly.
        vol = ((occ.get("physical_properties") or {}).get("volume")
               or (b.get("physical_properties") or {}).get("volume"))
        if not vol:
            try:
                vol = abs(float(m.volume))
            except Exception:                                  # noqa: BLE001
                vol = 0.0
        info["volume"].append(float(vol or 0.0))
        # The branch of the assembly tree this part hangs off. Everything under
        # one node is a subassembly - "Truck ASM" and its baseplate, kingpin,
        # bushings and hanger - and a manual that finishes one before starting
        # the next reads the way the thing is actually built.
        info["path"].append(paths.get(ouid, ()))
        info["occ"].append(ouid)

    # annotated contacts, resolved to exact part INSTANCES via the
    # occurrence uid, so an instanced screw pairs with its own hole
    by_key: dict = {}
    by_body: dict = {}
    for idx, (buid, _m, ouid) in enumerate(raw):
        by_key.setdefault((buid, ouid), []).append(idx)
        by_body.setdefault(buid, []).append(idx)

    def resolve(ent):
        if not isinstance(ent, dict):
            return []
        b, o = ent.get("body"), ent.get("occurrence") or ""
        return by_key.get((b, o)) or by_body.get(b) or []

    by_occ: dict = {}
    for idx, (buid, _m, ouid) in enumerate(raw):
        if ouid:
            by_occ.setdefault(ouid, []).append(idx)

    pairs = []

    # JOINTS first: a joint is the file stating outright that two occurrences
    # are fixed together, which is stronger evidence than any contact and far
    # stronger than surfaces measured to lie close. It also fills a gap nothing
    # else can - the skateboard records 23 joints and ZERO contacts, so without
    # these its entire connectivity was inferred from proximity.
    for field in ("joints", "as_built_joints"):
        for jt in (j.get(field) or {}).values():
            if not isinstance(jt, dict):
                continue
            ends = []
            for side in ("one", "two"):
                occ = jt.get(f"occurrence_{side}")
                idxs = by_occ.get(occ) or []
                if not idxs:
                    # Fall back to the body named in the joint's geometry.
                    g = jt.get(f"geometry_or_origin_{side}") or {}
                    ent = g.get("entity_one") or g.get("entity_two") or {}
                    idxs = by_body.get(ent.get("body")) or []
                ends.append(idxs)
            for a in ends[0][:2]:
                for b in ends[1][:2]:
                    if a != b:
                        pairs.append((a, b))
                        pairs.append((a, b))   # counted twice: joints outrank contacts

    for c in j.get("contacts", []) or []:
        ia = resolve(c.get("entity_one"))
        ib = resolve(c.get("entity_two"))
        for a in ia[:2]:
            for b in ib[:2]:
                if a != b:
                    pairs.append((a, b))

    # Holes say which parts a fastener passes through: every hole belongs to a
    # body, so a bolt sharing a hole's axis is going into that body and cannot
    # arrive first. Kept raw here; `Prepped` turns them into precedence.
    info["holes"] = [h for h in (j.get("holes") or []) if h.get("body")]
    info["hole_body_index"] = {b: idxs for b, idxs in by_body.items()}
    return parts, pairs, info


# ------------------------------------------------------------ blockout fitting

# Fusion records a material name per body, and the names carry two independent
# facts: what it is made of and how it is finished. "Brass - Polished" and
# "Steel - Satin" are both metal but look nothing alike. Collapsing them to a
# single metal/not bit threw that away and rendered every metal part
# identically; these tables recover it as the three PBR scalars the renderer
# and the material conditioning map both want.
SUBSTANCE = (
    # keyword         metalness  base roughness  clearcoat
    (("steel", "iron", "alumin", "chrome", "brass", "copper", "gold",
      "titan", "nickel", "silver", "platinum", "metal"), 0.92, 0.38, 0.00),
    (("paint", "enamel", "lacquer"),                     0.05, 0.35, 0.75),
    (("plastic", "nylon", "abs", "acrylic"),             0.04, 0.55, 0.40),
    (("pine", "oak", "walnut", "birch", "maple", "wood"), 0.02, 0.72, 0.15),
    (("rubber", "foam"),                                 0.02, 0.92, 0.00),
    (("glass", "gemstone", "ruby", "diamond"),           0.08, 0.08, 0.85),
)
FINISH = (
    (("polished", "glossy", "mirror"), 0.12),
    (("semi-polished", "semigloss", "satin", "anodized", "brushed"), 0.42),
    (("matte", "flat", "rough", "cast"), 0.82),
)


def pbr_for(app: str):
    """material name -> (metalness, roughness, clearcoat)"""
    a = (app or "").lower()
    met, rough, coat = 0.06, 0.55, 0.30              # unnamed: generic plastic
    for keys, m, r, c in SUBSTANCE:
        if any(k in a for k in keys):
            met, rough, coat = m, r, c
            break
    for keys, r in FINISH:                           # finish overrides roughness
        if any(k in a for k in keys):
            rough = r
            break
    return round(met, 3), round(rough, 3), round(coat, 3)


def _fit_one(m: trimesh.Trimesh):
    """One primitive for one blob: oriented box, or bounding cylinder when
    that wraps tighter. Returns (mesh, edges, primitive volume)."""
    try:
        obb = m.bounding_box_oriented
    except Exception:
        obb = m.bounding_box
    vol_b = float(np.prod(obb.primitive.extents)) if hasattr(obb, "primitive") \
        else obb.volume
    try:
        c = trimesh.bounds.minimum_cylinder(m, sample_count=6)
        vol_c = math.pi * c["radius"] ** 2 * c["height"]
    except Exception:
        c, vol_c = None, np.inf
    if c is not None and vol_c < vol_b * 0.9:
        prim = trimesh.creation.cylinder(radius=c["radius"],
                                         height=c["height"], sections=24)
        prim.apply_transform(c["transform"])
        return prim, cylinder_edges(c), vol_c
    prim = trimesh.Trimesh(vertices=np.asarray(obb.vertices),
                           faces=np.asarray(obb.faces), process=False)
    try:
        tf, ext = obb.primitive.transform, obb.primitive.extents
    except AttributeError:
        tf, ext = np.eye(4), np.ptp(np.asarray(obb.vertices), axis=0)
    return prim, box_edges(tf, ext), vol_b


def fit_primitives(m: trimesh.Trimesh, allow_decomp=True, max_pieces=5):
    """The agent's view of a part. One primitive when it wraps tight; when it
    does not (an L-bracket, a claw, a curved arm), a handful of convex pieces
    each get their own box or cylinder. Still crude on purpose -- the point is
    a blockout an agent could author, not a mesh. Returns (prims, edges)."""
    prim, edges, pvol = _fit_one(m)
    if not allow_decomp:
        return [prim], edges
    try:
        tight = float(m.convex_hull.volume) / max(pvol, 1e-12)
    except Exception:
        return [prim], edges
    if tight >= 0.8:                    # the hull fills the primitive: keep it
        return [prim], edges
    try:
        hulls = trimesh.decomposition.convex_decomposition(
            m, maxConvexHulls=max_pieces, resolution=60000,
            maxNumVerticesPerCH=48)
    except Exception:
        return [prim], edges
    prims, segs, vols = [], [], []
    for h in hulls:
        try:
            hm = trimesh.Trimesh(vertices=h["vertices"], faces=h["faces"],
                                 process=False)
            p, e, v = _fit_one(hm)
            prims.append(p)
            segs.append(e)
            vols.append(v)
        except Exception:
            continue
    if len(prims) < 2:
        return [prim], edges
    keep = [i for i, v in enumerate(vols) if v > 0.02 * sum(vols)]
    if not keep:
        return [prim], edges
    out_e = []
    for i in keep:
        out_e += segs[i]
    return [prims[i] for i in keep], out_e


def box_edges(tf: np.ndarray, ext: np.ndarray):
    """The 12 frame edges of an oriented box, from its local frame: corners
    are the sign combinations of +-extents/2, an edge joins corners whose
    signs differ on exactly one axis. No diagonals, ever."""
    signs = [np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1)
             for sz in (-1, 1)]
    corner = {tuple(s): trimesh.transform_points(
        [s * ext / 2.0], tf)[0] for s in signs}
    out = []
    for s in signs:
        for ax in range(3):
            if s[ax] < 0:
                t = s.copy()
                t[ax] = 1
                out.append(np.stack([corner[tuple(s)], corner[tuple(t)]]))
    return out


def cylinder_edges(c: dict):
    tf, r, h = c["transform"], c["radius"], c["height"]
    th = np.linspace(0, 2 * math.pi, 33)
    ring = np.stack([np.cos(th) * r, np.sin(th) * r, np.zeros_like(th)], 1)
    out = []
    for z in (-h / 2, h / 2):
        w = trimesh.transform_points(ring + [0, 0, z], tf)
        out += [np.stack([w[i], w[i + 1]]) for i in range(len(w) - 1)]
    for a in (0, 8, 16, 24):
        p = np.stack([ring[a] + [0, 0, -h / 2], ring[a] + [0, 0, h / 2]])
        out.append(trimesh.transform_points(p, tf))
    return out


# ------------------------------------------------------------ painter renderer

def look_at(center, az, el, dist):
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    eye = center + dist * np.array([ca * ce, sa * ce, se])
    f = center - eye
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    return eye, r, u, f


def render(items, res, az, el, fit, flat=False):
    """items: [(verts, faces, normals, colour, edge_segs_or_None, offset)].
    Painter's algorithm; Lambert for the CAD panel, flat fills + frame edges
    for the schematic panel."""
    center, radius = fit
    eye, r, u, f = look_at(center, az, el, dist=radius * 2.7)
    focal = res * 1.15

    def project(v):
        p = v - eye
        cam = np.stack([p @ r, p @ u, p @ f], 1)
        z = np.maximum(cam[:, 2], 1e-6)
        return (cam[:, 0] / z * focal + res / 2,
                -cam[:, 1] / z * focal + res / 2, z)

    tris, lines = [], []
    for verts, faces, normals, col, edges, off in items:
        v = verts + off
        x, y, z = project(v)
        zm = z[faces].mean(1)
        light = normals @ (-f)
        fill2 = normals @ np.array([0.4, 0.5, 0.76])
        s = np.clip(np.abs(light) * 0.62 + np.clip(fill2, 0, 1) * 0.30 + 0.22,
                    0, 1)
        for k in range(len(faces)):
            if flat:
                shade = 0.72 + 0.28 * float(np.abs(light[k]))
                c = tuple(int((cc * 0.55 + 255 * 0.45) * shade)
                          for cc in col)
            else:
                c = tuple(int(cc * s[k]) for cc in col)
            tris.append((zm[k], [(x[i], y[i]) for i in faces[k]], c))
        if edges:
            for seg in edges:
                ex, ey, ez = project(np.asarray(seg) + off)
                if (ez <= 1e-5).any():
                    continue
                # edges join the depth-sorted queue with the fills, slightly
                # nearer than their true depth so a face and its own frame
                # resolve frame-on-top -- but a box IN FRONT still hides them
                lines.append((ez.mean() * 0.985, list(zip(ex, ey))))
    queue = [(-z, 0, ("p", poly, c)) for z, poly, c in tris] +             [(-z, 1, ("l", pts, None)) for z, pts in lines]
    queue.sort(key=lambda t: (t[0], t[1]))
    im = Image.new("RGB", (res, res), PAPER)
    d = ImageDraw.Draw(im)
    for _, _, (kind, geom, c) in queue:
        if kind == "p":
            d.polygon(geom, fill=c)
        else:
            d.line(geom, fill=INK, width=2)
    return im


# ------------------------------------------------------------ demo assembly

def font(sz, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Helvetica.ttc",
              "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz, index=1 if bold else 0)
            except Exception:
                pass
    return ImageFont.load_default()


class Prepped:
    """Everything derivable once per assembly, before any frame is drawn."""

    def __init__(self, adir: str):
        self.parts, pairs, self.info = load_assembly(adir)
        self._contact_pairs = [(a, b) for a, b in pairs
                               if a < len(self.parts) and b < len(self.parts)]
        if len(self.parts) < 2:
            raise ValueError("fewer than 2 renderable parts")
        self.metal = [p[4] for p in self.parts]
        self.cad = []
        for buid, name, m, col, met in self.parts:
            self.cad.append((np.asarray(m.vertices), np.asarray(m.faces),
                             np.asarray(m.face_normals), col, None))
        allv = np.vstack([c[0] for c in self.cad])
        ctr = (allv.min(0) + allv.max(0)) / 2
        self.fit = (ctr, float(np.linalg.norm(allv - ctr, axis=1).max()))
        self.blocks = []
        for buid, name, m, col, met in self.parts:
            diag = float(np.linalg.norm(np.ptp(np.asarray(m.vertices),
                                               axis=0)))
            prims, edges = fit_primitives(
                m, allow_decomp=diag > 0.10 * self.fit[1])
            comb = trimesh.util.concatenate(prims) if len(prims) > 1 \
                else prims[0]
            self.blocks.append((np.asarray(comb.vertices),
                                np.asarray(comb.faces),
                                np.asarray(comb.face_normals), col, edges))
        self.adj = self._adjacency()  # needs _contact_pairs set first
        self.fastener = [is_fastener(nm) for nm in self.info["occ_name"]]
        self.contained = self._containment()   # needs `fastener`
        self.hole_of = self._hole_precedence()
        self.inner = sorted({i for i, _ in self.contained})
        self.order = self._order()

    def _adjacency(self):
        """Which parts actually touch: the dataset's annotated contact pairs
        (resolved per instance) plus true surface proximity via KD-trees.
        Bounding boxes only pre-filter; they never decide."""
        from scipy.spatial import cKDTree
        n = len(self.parts)
        bbs = [(c[0].min(0), c[0].max(0)) for c in self.cad]
        eps = 0.012 * self.fit[1]
        pts, trees = [], []
        for c in self.cad:
            v = c[0]
            step = max(1, len(v) // 400)
            pv = v[::step]
            pts.append(pv)
            trees.append(cKDTree(pv))
        w = np.zeros((n, n))
        for i, j in self._contact_pairs:
            w[i, j] += 3.0
            w[j, i] += 3.0
        for i in range(n):
            for j in range(i + 1, n):
                lo = np.maximum(bbs[i][0], bbs[j][0]) - eps
                hi = np.minimum(bbs[i][1], bbs[j][1]) + eps
                if not ((hi - lo) > 0).all():
                    continue
                a, b = (i, j) if len(pts[i]) < len(pts[j]) else (j, i)
                d, _ = trees[b].query(pts[a], k=1,
                                      distance_upper_bound=eps)
                frac = float(np.isfinite(d).mean())
                if frac > 0.005:
                    w[i, j] += frac
                    w[j, i] += frac
        return [(i, j, float(w[i, j])) for i in range(n)
                for j in range(i + 1, n) if w[i, j] > 0]

    def _containment(self):
        """(inner, outer) pairs: inner really sits inside outer's envelope. You
        cannot drop a part into a housing that is already closed, so inner must
        be placed before outer.

        Boxes only pre-filter. Deciding on boxes alone counts anything that
        merely passes through a larger part's box - a bolt through a bracket, a
        shaft across an opening - and each false pair is a hard ordering
        constraint, so the effect is not cosmetic: one 68-part assembly came out
        with 114 containment pairs over 52 parts, enough to strip every
        supported candidate out of the pool and strand parts in mid-air.

        The envelope is the outer part's convex hull, which is what makes a
        housing a housing: it spans the cavity, so a part seated in that cavity
        falls inside it, while a fastener passing through a plate mostly does
        not."""
        n = len(self.parts)
        bbs = [(c[0].min(0), c[0].max(0)) for c in self.cad]
        vols = [float(np.prod(mx - mn)) for mn, mx in bbs]
        from scipy.spatial import ConvexHull

        hulls: dict = {}

        def envelope(j):
            """Half-space form (A, b) of the outer part's convex hull, built
            once. A point p is inside when A @ p + b <= tol for every facet.

            Deliberately not trimesh's own `contains`: that needs an optional
            spatial-index backend, and without it every query raises. Caught and
            skipped, it silently reported zero containment everywhere - a
            constraint that had quietly stopped existing. scipy is already a
            dependency here and the half-space test needs nothing further."""
            if j not in hulls:
                try:
                    hull = ConvexHull(np.asarray(self.cad[j][0]))
                    hulls[j] = (hull.equations[:, :-1], hull.equations[:, -1],
                                float(np.ptp(self.cad[j][0], axis=0).max()))
                except Exception as exc:                       # noqa: BLE001
                    # Degenerate (coplanar) part: it encloses nothing.
                    hulls[j] = None
                    del exc
            return hulls[j]

        out = []
        for i in range(n):
            # A bolt lying inside a plate's hull is passing THROUGH it, not
            # sealed inside it. Counting it as contained inverts the one rule
            # that matters for hardware - "a fastener comes after what it joins"
            # - and the two then deadlock: the plate waits for its own screws
            # while the screws wait for the plate. On the skateboard that left
            # the deck, the largest part in the assembly by 24x, stranded at
            # step 34 with the trucks already built on top of nothing.
            if self.fastener[i]:
                continue
            verts = np.asarray(self.cad[i][0])
            step = max(1, len(verts) // 200)
            probe = verts[::step]
            for j in range(n):
                if i == j or vols[j] < vols[i] * 1.5:
                    continue
                lo = np.maximum(bbs[i][0], bbs[j][0])
                hi = np.minimum(bbs[i][1], bbs[j][1])
                d = hi - lo
                if not ((d > 0).all() and float(np.prod(d)) > 0.85 * vols[i]):
                    continue
                hull = envelope(j)
                if hull is None:
                    continue          # cannot verify: do not assert a constraint
                normals, offsets, extent = hull
                # A small slack, or a part seated flush against a cavity wall
                # reads as outside on the facet it touches.
                tol = 1e-3 * extent
                inside = float(np.mean(
                    (probe @ normals.T + offsets <= tol).all(axis=1)))
                # Nearly all of it enclosed, not just most: a part half in and
                # half out is passing through, and that is the case the box test
                # could not tell apart.
                if inside > 0.9:
                    out.append((i, j))
        return out

    def _hole_precedence(self):
        """fastener -> the parts whose holes it passes through.

        A hole record carries an origin, an axis and a diameter, and belongs to
        a body. Matching on the origin alone - which is all the first version
        did - pairs a bolt with any hole that happens to be nearby, including
        holes on the far side of the part and holes far too small for it. All
        three fields are checked here:

          axis       the fastener's own long axis must run along the hole's,
                     because a bolt lying across a hole is not in it
          line       its centre must sit near the hole's axis LINE, not merely
                     near the mouth - a long bolt is seated deep
          diameter   its shaft must actually fit, within a tolerance either way
                     for threads and clearance
        """
        holes = self.info.get("holes") or []
        by_body = self.info.get("hole_body_index") or {}
        if not holes:
            return {}

        need: dict = {}
        for i, fast in enumerate(self.fastener):
            if not fast:
                continue
            verts = np.asarray(self.cad[i][0])
            centre = verts.mean(0)
            extent = np.ptp(verts, axis=0)
            length = float(extent.max())
            # Long axis of the fastener, and its cross-section across the other
            # two directions.
            long_axis = np.zeros(3)
            long_axis[int(np.argmax(extent))] = 1.0
            cross = float(np.sort(extent)[:2].mean())

            for h in holes:
                targets = [t for t in (by_body.get(h.get("body")) or [])
                           if t != i]
                if not targets:
                    continue
                o = h.get("origin") or {}
                d = h.get("direction") or {}
                origin = np.array([o.get("x", 0.0), o.get("y", 0.0),
                                   o.get("z", 0.0)], dtype=float)
                axis = np.array([d.get("x", 0.0), d.get("y", 0.0),
                                 d.get("z", 0.0)], dtype=float)
                norm = float(np.linalg.norm(axis))
                if norm < 1e-9:
                    continue
                axis = axis / norm

                # Aligned, either way up.
                if abs(float(np.dot(long_axis, axis))) < 0.8:
                    continue
                # Near the hole's axis line, and not further along it than the
                # fastener is long.
                delta = centre - origin
                along = float(np.dot(delta, axis))
                radial = float(np.linalg.norm(delta - along * axis))
                if radial > max(cross, 1e-9) or abs(along) > max(length, 1e-9):
                    continue
                # Fits the hole: a shaft much fatter than the bore is not in it,
                # and one much thinner is passing through something else.
                bore = float(h.get("diameter") or 0.0)
                if bore > 0 and not (0.5 * bore <= cross <= 1.8 * bore):
                    continue
                need.setdefault(i, set()).update(targets)
        return need

    def _order(self):
        """Assembly order from what the file actually records.

        Earlier versions inferred all of this from geometry - biggest bounding
        box first, grow along contacts, guess a fastener by its size - while the
        assembly.json carries the answers:

          occurrence names   "M8 Jam nut", "Baseplate", "WASHER", not "Body1"
          is_grounded        the part that is fixed in space: the base
          physical volume    real material, so a long bolt stops outranking a deck
          holes              which body a fastener goes into
          tree               "Truck ASM" and its children are one subassembly

        Four rules, hard first:

          1. a fastener comes after every part it touches, and after every part
             whose hole it passes through - the nuts-before-the-deck case
          2. a part enclosed by another comes before it closes over
          3. a part rests on something already placed
          4. finish the subassembly you are in before starting another

        1 and 3 cannot always both hold, and 2 and 3 likewise: a one-piece
        housing has to enclose its internals and support them. Where they
        conflict, support wins - a part with nothing under it is visible in the
        manual, while the others read as insertion.

        Still a derived order: the dataset records no assembly sequence. The
        timeline_index is modelling order, and on the skateboard it puts the
        deck last, after all eight of its own bolts.
        """
        n = len(self.parts)
        vol = self.info["volume"]
        cen = [c[0].mean(0) for c in self.cad]
        w = np.zeros((n, n))
        for i, j, ov in self.adj:
            w[i, j] = w[j, i] = ov
        deg = [int(len(np.nonzero(w[i])[0])) for i in range(n)]

        must_precede: dict = {}
        for i, j in self.contained:
            must_precede.setdefault(j, set()).add(i)
        # A fastener joins what it touches; it cannot lead.
        for i in range(n):
            if self.fastener[i]:
                nbrs = set(int(x) for x in np.nonzero(w[i])[0])
                nbrs |= set(self.hole_of.get(i, set()))
                if nbrs:
                    must_precede.setdefault(i, set()).update(nbrs)

        placed: list = []
        rest = list(range(n))
        while rest:
            done = set(placed)
            allowed = [i for i in rest
                       if not (must_precede.get(i, set()) - done)]
            supported = [i for i in rest
                         if float(w[i, placed].sum()) > 0
                         or not deg[i]]
            supported_set = set(supported)
            pool = ([i for i in allowed if i in supported_set]
                    or supported or allowed or rest)

            if not placed:
                # The grounded occurrence IS the base - the file says so. Only
                # fall back to size when nothing is grounded, and then to real
                # volume rather than to a bounding box.
                grounded = [i for i in pool
                            if self.info["grounded"][i] and not self.fastener[i]]
                body = [i for i in pool if not self.fastener[i]] or pool
                nxt = max(grounded or body,
                          key=lambda i: (vol[i], deg[i], -cen[i][2]))
            else:
                zlo, zhi = (min(c[2] for c in cen), max(c[2] for c in cen))
                zr = max(zhi - zlo, 1e-9)
                here = self.info["path"][placed[-1]]

                def score(i):
                    touch = float(w[i, placed].sum())
                    # Depth of shared tree branch: staying inside the current
                    # subassembly beats jumping to another and coming back.
                    path = self.info["path"][i]
                    shared = 0
                    for a, b in zip(path, here):
                        if a != b:
                            break
                        shared += 1
                    dist = min(np.linalg.norm(cen[i] - cen[j]) for j in placed)
                    zpos = 1 - (cen[i][2] - zlo) / zr      # low parts first
                    return (touch > 0, shared, not self.fastener[i],
                            touch * 0.001 + zpos * 0.2
                            - dist / (self.fit[1] * 4), vol[i])

                nxt = max(pool, key=score)

            placed.append(nxt)
            rest.remove(nxt)

            # Identical bodies are one operation. A twin is taken only if it
            # passes the same tests the pick above passed, so grouping can never
            # introduce a part that floats or that jumps its own constraints.
            buid = self.parts[nxt][0]
            while True:
                done = set(placed)
                twin = next(
                    (i for i in rest
                     if self.parts[i][0] == buid
                     and not (must_precede.get(i, set()) - done)
                     and (float(w[i, placed].sum()) > 0 or not deg[i])),
                    None)
                if twin is None:
                    break
                placed.append(twin)
                rest.remove(twin)
        return placed


def draw_chrome(im, res, bar, step, total, title):
    d = ImageDraw.Draw(im)
    d.text((14, 9), title, font=font(17, True), fill=INK)
    lab = f"step {step:02d} / {total:02d}"
    d.text((im.width - 14 - d.textlength(lab, font=font(16)), 10), lab,
           font=font(16), fill=MUTE)
    d.line([(0, bar - 1), (im.width, bar - 1)], fill=RULE)
    d.rectangle([res, bar, res + 2, im.height], fill=RULE)
    d.text((12, bar + 8), "agent blockout", font=font(14), fill=MUTE)
    d.text((res + 15, bar + 8), "render target (real CAD)", font=font(14),
           fill=MUTE)


def frame_pair(pp: Prepped, subset, res, az, el, fly=None):
    """fly = (part_index, 0..1 progress): translate that part in from
    outside along its outward direction while it eases in."""
    ctr, radius = pp.fit

    def items(source):
        out = []
        for i in subset:
            v, f, nrm, col, edges = source[i]
            off = np.zeros(3)
            if fly and fly[0] == i:
                t = fly[1]
                ease = 1 - (1 - t) ** 3
                outward = v.mean(0) - ctr
                nn = np.linalg.norm(outward)
                outward = outward / nn if nn > 1e-9 else np.array([0, 0, 1.0])
                off = outward * (1 - ease) * radius * 0.9
            out.append((v, f, nrm, col, edges, off))
        return out

    left = render(items(pp.blocks), res, az, el, pp.fit, flat=True)
    right = render(items(pp.cad), res, az, el, pp.fit)
    bar = 38
    im = Image.new("RGB", (res * 2 + 3, res + bar), PAPER)
    im.paste(left, (0, bar))
    im.paste(right, (res + 3, bar))
    return im, bar


def make_manual(adir, res=480, spf=10, fps=12, hold_end=18):
    name = os.path.basename(adir.rstrip("/"))
    pp = Prepped(adir)
    order, K = pp.order, len(pp.order)
    total = K * spf + hold_end
    print(f"[manual] {name}: {K} parts, {total} frames", flush=True)
    frames = []
    for fidx in range(total):
        az = 0.15 + 2.2 * math.pi * fidx / total
        el = 0.40 + 0.12 * math.sin(2 * math.pi * fidx / total)
        step = min(fidx // spf, K - 1)
        within = (fidx - step * spf) / max(1, spf) if fidx < K * spf else 1.0
        subset = order[:step + 1]
        fly = (order[step], min(1.0, within * 2)) \
            if fidx < K * spf and within < 0.5 else None
        im, bar = frame_pair(pp, subset, res, az, el, fly)
        draw_chrome(im, res, bar, step + 1, K, "Assembly manual")
        frames.append(im)
        if (fidx + 1) % 20 == 0:
            print(f"  {fidx+1}/{total}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    gif = os.path.join(OUT, f"{name}_manual.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"[manual] {gif}  ({len(frames)} frames)")
    try:
        fdir = os.path.join(OUT, f"_{name}_frames")
        os.makedirs(fdir, exist_ok=True)
        for i, f in enumerate(frames):
            f.save(os.path.join(fdir, f"f{i:04d}.png"))
        mp4 = os.path.join(OUT, f"{name}_manual.mp4")
        subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i",
                        os.path.join(fdir, "f%04d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", "-vf",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4],
                       check=True, capture_output=True)
        for f in os.listdir(fdir):
            os.remove(os.path.join(fdir, f))
        os.rmdir(fdir)
        print(f"[manual] {mp4}")
    except Exception as e:
        print(f"[manual] mp4 skipped ({e})")
    return gif


def rank_assemblies(min_parts=8, max_parts=48):
    rows = []
    for aj in glob.glob(os.path.join(DATA, "**", "assembly.json"),
                        recursive=True):
        adir = os.path.dirname(aj)
        try:
            j = json.load(open(aj))
        except Exception:
            continue
        nb = len(j.get("bodies", {}))
        objs = glob.glob(os.path.join(adir, "*.obj"))
        size = sum(os.path.getsize(p) for p in objs)
        if not (min_parts <= nb <= max_parts) or len(objs) < min_parts \
                or size > 60e6:
            continue
        rows.append((nb, size, adir))
    rows.sort(key=lambda r: (-r[0], r[1]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--res", type=int, default=520)
    ap.add_argument("--manual", default=None,
                    help="'auto' or an assembly dir name: animated manual")
    ap.add_argument("--spf", type=int, default=10, help="frames per step")
    a = ap.parse_args()

    if a.inspect:
        ajs = sorted(glob.glob(os.path.join(DATA, "**", "assembly.json"),
                               recursive=True))
        print(f"{len(ajs)} assemblies under data/")
        if ajs:
            j = json.load(open(ajs[0]))
            print("keys:", sorted(j.keys()))
            occ = next(iter(j.get("occurrences", {}).values()), {})
            print("occurrence keys:", sorted(occ.keys()))
            print("transform sample:",
                  json.dumps(occ.get("transform"), default=str)[:300])
            b = next(iter(j.get("bodies", {}).values()), {})
            print("body keys:", sorted(b.keys()))
            cs = j.get("contacts", [])
            print(f"contacts: {len(cs)}; sample:",
                  json.dumps(cs[0], default=str)[:300] if cs else "-")
        return

    os.makedirs(OUT, exist_ok=True)
    if a.manual:
        if a.manual == "auto":
            rows = rank_assemblies()
            if not rows:
                raise SystemExit("no suitable assemblies extracted yet")
            adir = rows[0][2]
        else:
            hits = glob.glob(os.path.join(DATA, "**", a.manual),
                             recursive=True)
            adir = hits[0] if hits else os.path.join(DATA, a.manual)
        make_manual(adir, res=a.res, spf=a.spf)
        return

    rows = rank_assemblies()
    print(f"[demo] {len(rows)} candidate assemblies (8-48 parts)")
    made = 0
    for nb, size, adir in rows:
        if made >= a.n:
            break
        try:
            pp = Prepped(adir)
            im, bar = frame_pair(pp, pp.order, a.res, az=0.8, el=0.42)
            draw_chrome(im, a.res, bar, len(pp.order), len(pp.order),
                        "Assembly manual")
        except Exception as e:
            print(f"  skip {os.path.basename(adir)}: {e}")
            continue
        name = os.path.basename(adir.rstrip("/"))
        p = os.path.join(OUT, f"{name}_{nb}parts.png")
        im.save(p)
        made += 1
        print(f"  [{made}/{a.n}] {name}: {nb} parts -> {p}", flush=True)


if __name__ == "__main__":
    main()
