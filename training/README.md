# Training the blockout-to-photo renderer

The code and settings behind the data-size curves: a renderer that turns a
blockout of an assembly step into a photograph of the specific object, given
three reference photos of it, trained on 5, 20, 40 and 100 objects and
evaluated on 14 objects it never saw. The model that followed the curves,
which also gets a progress photo of the object, is described in
[The progress-photo model](#the-progress-photo-model).

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
  parts (`attention_bias` in `klein/render.py`). The curve runs also put
  each token's part id on the 4th RoPE position axis (`--part-ids rope`, the
  trainer default).
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
| inputs | attention bias (soft, +3/-3), substrate, part id on the 4th RoPE axis |
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

Error over the whole frame, generated vs real, pixels in [0, 1], the way
NeuralOS reports frame errors (`klein/job_fullframe.sbatch`,
`klein/fullframe_err.py`). LPIPS above is scored on a crop around the object,
so these numbers are much smaller: most of the frame is empty table.

| training objects | held out, L1 | held out, RMSE | training objects, L1 | training objects, RMSE |
|---|---|---|---|---|
| 5 | 0.062 | 0.147 | 0.036 | 0.092 |
| 40 | **0.043** | **0.118** | **0.018** | **0.070** |
| 100 | 0.047 | 0.128 | 0.031 | 0.098 |

- `results/loss_curves.png`: training and held-out validation loss over
  steps for the four runs.
- `results/gap_curve.png`: every metric against the number of training
  objects (loss, image error, LPIPS, L1, share of parts rendered in their own
  colour, passes).
- `results/overfit5/`: the five-frame overfitting test: log, settings,
  renders and scores (see [below](#the-five-frame-overfitting-test)).
- `results/logs/`: the training logs. The `val step` lines are the curves;
  the `[metrics]` blocks at the end are the evaluations. Home-directory paths
  are replaced by `~/`. `gap100.log` also contains the start of a later
  continuation beyond step 120,000, which is not part of these results.

## Reproducing the curves

About 7 GPU-days on an H200 in total (1.7 to 1.9 s per training step):
N=5 about 8 h, N=20 about 32 h, N=40 about 64 h, N=100 about 57 h, plus about
3 h of evaluation per run.

### 1. Software and models

Python 3.11 with the packages in `requirements.txt`. The klein classes
(`Flux2KleinPipeline`, `Flux2Transformer2DModel`, `AutoencoderKLFlux2`) need a
recent diffusers; the runs used diffusers installed from its GitHub main
branch. `klein/job_klein_setup.sbatch` is the setup job that was used: it
installs those packages into `~/kleinlib`, downloads both checkpoints
(`black-forest-labs/FLUX.2-klein-4B` and `black-forest-labs/FLUX.2-klein-base-4B`,
about 30 GB) and checks that the classes import.

### 2. Data

Put the training corpus `v5_colour/` next to `klein/`. It is about 12 GB and
not in this repository: 114 objects (100 for training, 14 held out), each in 4
random colourings, about 210 frames per object, and per colouring 5 reference
photos with per-pixel part-id maps. The object lists in `meta/` refer to it.
The copy used for these runs is on the training cluster; ask for access.

### 3. Train and evaluate the four points

The jobs are written for Slurm and change into `~/h200_pilot` (where this
folder's contents and `v5_colour/` live); edit `cd`, `PYTHONPATH`, `PY` and
`DATA` at the top of `klein/job_gap.sbatch` for another machine. Each job
trains, keeps the checkpoint with the lowest held-out validation loss, then
renders and scores it (held out with own / another object's / no photos,
training objects, first three assembly steps, part colours).

```
sbatch -J gap5  -o gap5.log  klein/job_gap.sbatch gap5  n5.json  15000
sbatch -J gap20 -o gap20.log klein/job_gap.sbatch gap20 n20.json 60000

# 120k steps exceed a 36 h time limit: a chain of pieces, each resuming from
# the latest checkpoint (pieces started after the run is done exit at once)
for R in "gap40b n40.json" "gap100 all"; do
  set -- $R
  J=$(sbatch --parsable -J $1 -o $1.log klein/job_gap.sbatch $1 $2 120000)
  for i in 1 2; do
    J=$(sbatch --parsable --dependency=afterany:$J -J $1 -o $1.log --open-mode=append \
        klein/job_gap.sbatch $1 $2 120000 auto)
  done
done
```

Without Slurm, the training command inside `klein/job_gap.sbatch` can be run
directly; the evaluation commands follow it in the same file.

The job stops early if the held-out loss has not improved for 25k steps (the
runs used 15k). This is a safety net; it never fired in these runs.

### 4. Plots

```
python klein/plot_gap_curves.py gap5.log:N=5 gap20.log:N=20:60000 gap40b.log:N=40 gap100.log:N=100:120000 --out=loss_curves.png
python klein/plot_gap.py gap5.log gap20.log gap40b.log gap100.log --out=gap_curve.png
```

A `:60000` or `:120000` after a log cuts it at that step: the logs in
`results/logs/` also contain later continuations that are not part of the
curves. The table above is the `[metrics]` block at the end of each log (the
`lpips` column of the held-out and training rows).

## What has already been checked

The open question is why held-out LPIPS stops improving after about 20
objects, and gets worse at 100, while the held-out loss keeps falling. What
has been ruled in or out so far, with the job that tests each point:

| check | result |
|---|---|
| Can the model fit its data? 5 frames, regularisers off (`job_overfit5.sbatch`, see [below](#the-five-frame-overfitting-test)) | yes: LPIPS 0.051 on the memorised frames |
| VAE ceiling (`vae_floor.py`) | 0.004 to 0.008 LPIPS: the latent is not the limit |
| Sampler: base model, 28 steps, vs distilled, 4 steps (`job_basecheck.sbatch`) | same LPIPS (N=40, 60k run: 0.178 vs 0.177; N=100: 0.211 vs 0.211, on 210 held-out frames) |
| Checkpoint-to-checkpoint noise (`job_ckptsweep.sbatch`) | about +/-0.006 LPIPS between neighbouring checkpoints while the loss falls smoothly; N=100 at 80k to 120k: 0.227, 0.212, 0.214, 0.205, 0.211 |
| Lower learning rate, N=20 (peak 1.5e-4) | lower loss (0.2387 vs 0.2413), same LPIPS (0.207) |
| Twice the budget, N=40 (60k to 120k steps) | 0.200 to 0.198: within the noise |
| What N=100 does differently (sheets) | adds confident detail the object does not have (grills, extra plates, screws), with both samplers; with the 4-step sampler it also draws extra parts away from the object in 37% of frames (`offobject.py`), which LPIPS, scored on a crop around the object, does not see |
| Is information missing from the inputs? (`job_histinfer.sbatch`) | a real image of a nearby frame (same object and step, about 35 degrees away) given as a fourth reference at inference only, no retraining: held-out LPIPS 0.282 to 0.175 (N=5), 0.186 to 0.117 (N=20), 0.176 to 0.111 (N=40), 0.211 to 0.121 (N=100), on 210 held-out frames |

The last row is the strongest lead: most of the remaining error is detail the
inputs (blockout plus three photos) do not determine, and adding training
objects cannot supply it. Not run yet: the held-out loss split by noise level
(`job_evalonly.sbatch`) and the LPIPS split by image region
(`error_regions.py`).

### The five-frame overfitting test

Everything from the run is in `results/overfit5/`, except the checkpoint and
the data.

- **Command:** `sbatch -J overfit5 -o overfit5.log klein/job_overfit5.sbatch overfit5 128`
  (run on 7 October 2026). `args.json` is the trainer's record of every
  setting.
- **Data:** the small corpus `v5_unit` (not in this repository), object
  `21590_7d092186` (`meta/one.json`), colouring 0, its first five frames
  (`--variants 0-0 --max-frames 5`).
- **Training:** 3,000 steps with 2 accumulated frames each (1,200 passes over
  the five frames), rank 128 with alpha = rank, warm-up 100 steps then cosine
  3e-4 to 3e-5, every regulariser off, attention bias, substrate and RoPE part
  ids as in the curves.
- **Log:** `overfit5.log`. Validation every 100 steps on 32 fixed points
  drawn from the five frames (`fixed_train`), and on unseen colourings of the
  same object (16 to 19) for reference. The summary near the end gives both
  every 500 steps: the five frames fall from 0.486 to 0.220 while the unseen
  colourings rise from 0.508 to 1.003, so the model memorises. (The job file
  now prints a longer summary than the one in this log; the training and
  evaluation commands are unchanged.)
- **Outputs:** `renders/` holds the five frames rendered by the final
  checkpoint (distilled model, 4 steps). `metrics.json` scores them against
  the real frames: LPIPS 0.051 and L1 0.027, against 0.330 and 0.166 for the
  baseline, which paints the blockout with the reference colours. The VAE
  round trip alone costs 0.0075 LPIPS on training frames. `sheet.jpg` shows,
  per frame, the blockout map, a reference photo, the real frame, the render
  and the baseline.
- **Checkpoint:** on the training cluster at `~/h200_pilot/runs/overfit5/lora`.
  It is too large for a GitHub file (a rank-128 LoRA is a few hundred MB);
  ask for it.

The history option (`--history`, `--hist-offset`, `--hist-earlier`,
`--hist-label-drop`, `--hist-augment` in `train_klein.py`, `--history` in
`eval_klein.py`) is off by default and was not used for the curves. The
progress-photo model below is trained with it.

## The progress-photo model

The last check above says the inputs, not the model or the number of
objects, limit held-out quality. The interface can supply what is missing:
photos of the object taken during assembly. So the next model is trained with
one (`klein/job_hist.sbatch`). Changes from the curve recipe:

- **Inputs:** two reference photos (one of the parts laid out, one assembled;
  a second laid-out photo added almost nothing when measured) plus a
  *progress photo*. In training it is absent 20% of the time, from the same
  assembly step 1 to 7 frames away in the same camera walk (11 to 80 degrees)
  50%, and from an earlier assembly step 30% (`--history 0.8 --hist-offset 1-7
  --hist-earlier 0.375`). It carries part labels like the other photos and is
  not degraded (`--hist-label-drop 0 --hist-augment 0`).
- **Part correspondence:** the attention bias only: no substrate
  (`--no-substrate`) and no part id on the RoPE axis (`--part-ids none`; in
  FLUX.2 the 4th position axis is the text-index axis).
- **Schedule:** 40 objects (`meta/n40.json`), 180,000 steps, warm-up then
  cosine from 1.5e-4 to 1.5e-5 (half the curves' rate; at N=20 it gave lower
  loss and the same LPIPS). Rank 128, alpha 32. Gradient checkpointing off
  (`--grad-ckpt off`): 1.0 s per step and 39 GB on an H200 NVL.
- **Checkpoint:** the lowest mean of the held-out loss with and without the
  progress photo, so the no-photo case cannot be traded away.
- **Variant `hist40bo_r256`:** the same with rank 256 and alpha 64 (the same
  alpha/rank), to test whether the adapter's size is a limit. 42 GB, same
  speed.

```
X="--n-refs 2 --no-substrate --part-ids none --lr 1.5e-4 --lr-final 1.5e-5 --hist-label-drop 0 --hist-augment 0 --grad-ckpt off"
H="-p ALL --constraint=H200 --open-mode=append"
J=$(sbatch --parsable $H -J hist40bo -o hist40bo.log klein/job_hist.sbatch hist40bo n40.json 180000 auto "$X")
for i in 2 3 4; do
  J=$(sbatch --parsable $H --dependency=afterany:$J -J hist40bo -o hist40bo.log \
      klein/job_hist.sbatch hist40bo n40.json 180000 auto "$X")
done
sbatch -p ALL --open-mode=append --dependency=afterany:$J -o curve_hist40bo.log klein/job_curve.sbatch hist40bo
# rank 256: the same with hist40bo_r256 and "$X --rank 256 --lora-alpha 64"
```

On a preemptible partition (`-p ALL` above), a preempted piece is requeued
and resumes from the latest `lora_step` checkpoint (saved every 10,000
steps); the optimizer state is not saved and restarts. After training, the
job evaluates the best checkpoint with no photo, with a photo 11, 35 and 80
degrees away, without labels, from an earlier step, and on the first three
assembly steps. `klein/job_curve.sbatch` then scores every saved checkpoint by
whole-frame L1, RMSE and LPIPS (`klein/ckpt_curve.py`, into
`runs/RUN/curve.csv`).

**Status, 9 November 2026:** both runs are still training (rank 128 at about
70,000 steps, rank 256 at about 117,000). Fixed-set validation loss, mean of
the last 2,500 steps up to step 69,750:

| run | held out, no photo | held out, with photo | training objects, with photo |
|---|---|---|---|
| progress-photo model, rank 128 | 0.2368 | **0.2306** | 0.1831 |
| progress-photo model, rank 256 | 0.2410 | 0.2363 | 0.1826 |
| gap40b, same step | 0.2407 | | |
| gap100, same step | 0.2376 | | |

These are diffusion losses, not image errors, and the photo's effect on them
is small because most of this loss comes from low noise levels, where no
photo can help. The image metrics come when training ends. So far rank 256 is
no better than rank 128.

**Open: noise levels.** Training draws the noise level from a logit-normal
around 0.5; the validation sets use evenly spaced levels from 0.02 to 0.98;
the 4-step sampler renders at about 1.0, 0.96, 0.88 and 0.72 (computed from
the scheduler settings, not yet checked on the cluster). Only about 18% of
training samples fall at 0.72 or above, where layout and colour are decided
and where the progress photo matters most.

### Fixes since the curves

- Chained pieces resumed from the alphabetically last checkpoint
  (`lora_step90000` after `lora_step150000`); `auto` now sorts by number.
- Resuming before the end of the warm-up gave a negative learning rate; the
  warm-up factor is now clamped to [0, 1].
- New trainer options, all off by default so the curve recipe is unchanged:
  `--no-substrate`, `--grad-ckpt`, `--lora-targets`, `--train-blocks` (with
  `--block-lr`), `--eval-levels`, `--id-grey`. `job_gap.sbatch` takes extra
  trainer flags as a 5th argument.

## Files

```
klein/render.py          model wrapper, token layout, attention bias, substrate, sampler
klein/train_klein.py     LoRA training with fixed-set validation
klein/eval_klein.py      renders held-out or training frames with a trained LoRA
klein/job_gap.sbatch     Slurm job: train one point of the curve, then evaluate it
klein/job_hist.sbatch    Slurm job: train the progress-photo model, then evaluate it
klein/job_curve.sbatch, ckpt_curve.py   whole-frame L1 / RMSE / LPIPS of every saved checkpoint
klein/job_fullframe.sbatch, job_fullframe_hist.sbatch, fullframe_err.py
                         whole-frame error of the best checkpoint, without / with the photo
klein/promo_eval.py, promo_video.py, job_promo.sbatch
                         a session as the interface would run it, with a memory of photos
klein/job_klein_setup.sbatch  installs the packages and downloads the checkpoints
klein/job_basecheck.sbatch, job_ckptsweep.sbatch, job_evalonly.sbatch,
klein/job_histinfer.sbatch, job_histdist.sbatch, job_overfit5.sbatch
                         the diagnostic jobs (job_overfit5 uses the small v5_unit corpus)
klein/part_colour.py     share of parts rendered in their own colour
klein/offobject.py       content invented away from the object
klein/error_regions.py   LPIPS split into blockout-mismatch, part interior and rest
klein/vae_floor.py       the VAE round-trip ceiling
klein/plot_gap*.py       the two figures
klein/bake_lora_scale.py fixes LoRAs saved before the trainer stored alpha/rank in them
metrics.py, plot_val.py  LPIPS / L1 scoring; validation-curve reader
meta/                    the object lists
requirements.txt         Python packages
```

The Slurm settings (partition, excluded nodes, paths) are those of the cluster
the runs were made on.
