# Training the blockout-to-photo renderer

The code and settings behind the data-size curves: a renderer that turns a
blockout of an assembly step into a photograph of the specific object, given
three reference photos of it, trained on 5, 20, 40 and 100 objects and
evaluated on 14 objects it never saw.

## Model

- **Backbone:** FLUX.2 [klein] 4B. A LoRA is trained on the base checkpoint
  (`FLUX.2-klein-base-4B`) and applied to the 4-step distilled checkpoint
  (`FLUX.2-klein-4B`) for rendering. Sampling with the base model at 28 steps
  gives the same LPIPS.
- **Inputs, as image tokens in one sequence:** the blockout map (R depth,
  G shading, B part id), the *substrate* (the map painted with each part's
  colour read from the photos), and three reference photos (two of the parts
  laid out, one assembled). The text prompt is fixed:
  "a photograph of a mechanical assembly on a studio table".
- **Part correspondence:** every screen pixel and every photo pixel carries
  the id of the CAD part it belongs to. In every attention layer, a screen
  token gets +3 towards photo tokens of its own part and -3 towards other
  parts (`attention_bias` in `klein/render.py`).
- **Loss:** flow matching. The model predicts `noise - x0` from
  `(1 - s) x0 + s noise` and the conditioning tokens; the loss is the
  per-token squared error, plus 0.5 x the mean of the worst 10% of background
  tokens. Only the target tokens are scored, and only the LoRA is trained.

## Setup of the curves

The same recipe for every point (`klein/job_gap.sbatch`):

| setting | value |
|---|---|
| LoRA | rank 128, alpha 32, on the attention projections (and the fused attention/MLP input projection of the single-stream blocks) |
| learning rate | linear warm-up 1,000 steps, cosine 3e-4 to 3e-5 over the budget |
| batch | 1 frame, gradients accumulated over 2 |
| regularisers | off (no photo dropout, no label dropout) |
| inputs | attention bias (soft, +3/-3) and substrate |
| validation | every 250 steps, fixed 32 frames with fixed noise and noise levels, training and held-out objects |
| checkpoint | the one with the lowest held-out validation loss |
| rendering | distilled checkpoint, 4 steps, 512 px |

| run | training objects | frames | steps | passes at best checkpoint |
|---|---|---|---|---|
| gap5 | 5 (`meta/n5.json`) | 1,050 | 15,000 | 27 |
| gap20 | 20 (`meta/n20.json`) | 4,200 | 60,000 | 27 |
| gap40b | 40 (`meta/n40.json`) | 8,400 | 120,000 | 27 |
| gap100 | all 100 | 20,910 | 120,000 | 11 |

`meta/n5.json` is the first five objects of `meta/n40.json`.

## Results

LPIPS, lower is better. Held out = the 14 unseen objects, 1,035 frames.

| training objects | held out, own photos | held out, another object's photos | held out, no photos | training objects, own photos |
|---|---|---|---|---|
| 5 | 0.294 | 0.401 | 0.425 | 0.228 |
| 20 | 0.207 | 0.379 | 0.411 | 0.108 |
| 40 | **0.198** | 0.387 | 0.415 | 0.109 |
| 100 | 0.219 | 0.411 | 0.428 | 0.158 |

- `results/loss_curves.png`: training and held-out validation loss over
  steps for the four runs.
- `results/gap_curve.png`: every metric against the number of training
  objects (loss, image error, LPIPS, L1, share of parts rendered in their own
  colour, passes).
- `results/logs/`: the training logs. The `val step` lines are the curves;
  the `[metrics]` blocks at the end are the evaluations. Home-directory paths
  are replaced by `~/`. `gap100.log` also contains the start of a later
  continuation beyond step 120,000, which is not part of these results.

## Layout and use

```
klein/render.py          model wrapper, token layout, attention bias, substrate, sampler
klein/train_klein.py     LoRA training with fixed-set validation
klein/eval_klein.py      renders held-out or training frames with a trained LoRA
klein/job_gap.sbatch     Slurm job: train one point of the curve, then evaluate it
klein/part_colour.py     share of parts rendered in their own colour
klein/offobject.py       content invented away from the object
klein/error_regions.py   LPIPS split into blockout-mismatch, part interior and rest
klein/plot_gap*.py       the two figures
klein/bake_lora_scale.py fixes LoRAs saved before the trainer stored alpha/rank in them
metrics.py, plot_val.py  LPIPS / L1 scoring; validation-curve reader
meta/                    the object lists
```

`klein/job_gap.sbatch` is the job exactly as it ran: it changes into
`~/h200_pilot` (this folder's contents), reads the corpus from
`~/h200_pilot/v5_colour` and uses the Python environment set at its top;
adjust those lines for another machine. For example:

```
sbatch -J gap20 -o gap20.log klein/job_gap.sbatch gap20 n20.json 60000
```

Runs longer than the Slurm time limit are submitted as a chain that resumes
from the latest checkpoint; see the header of `klein/job_gap.sbatch`.

The training corpus (`v5_colour`, about 12 GB) is not in this repository:
114 objects (100 for training, 14 held out), each in 4 random colourings,
about 210 frames per object, and per colouring 5 reference photos with
per-pixel part-id maps.

Software: PyTorch, a diffusers build that includes `Flux2KleinPipeline`,
peft, and `lpips` for the metrics. The Slurm settings (partition, excluded
nodes, paths) are those of the cluster the runs were made on.
