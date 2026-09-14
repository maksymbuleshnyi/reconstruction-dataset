# Conditioning format

What the renderer is given for one frame, and where each piece comes from.

## Channels

Each frame is rendered at a chosen geometry level (`boxes`, `primitives` or
`parts`) and produces:

| map | channels | contents |
| --- | --- | --- |
| packed | 2 | red = depth, green = the lighting term |
| normals | 3 | view-space surface direction |
| colour | 3 | per-part albedo, flat within a part |
| material | 3 | metalness, roughness, clearcoat |
| mask | 1 | object silhouette, used for scoring rather than as input |

Depth is one number, so it occupies one channel rather than a greyscale image
replicated three times. The lighting term is rendered by drawing the geometry
with a white base colour under the scene's fixed light, which makes
`target ≈ colour × lighting` true by construction.

## Lighting

One fixed key light, identical in every frame of every object, plus ambient
and two fills. Nothing about the illumination varies per session, so nothing
in the target depends on information absent from the inputs.

## Targets

Targets are rendered with the real CAD geometry and the parsed materials, then
composited onto a background. They are lossless and carry no synthetic grain
or defocus: unpredictable degradation is noise in the supervision, and it sets
an error floor that no model can beat.

Reference photographs are treated differently and do carry grain, compression
and occasional defocus, because at deployment those stand in for pictures
taken on a phone.

## Assembly order

`order` is computed from the CAD: contact relationships between components,
containment, and the distinction between fasteners and structural parts.
Placing parts in that order produces a build that is physically sensible
rather than an arbitrary permutation.
