#!/bin/bash
#SBATCH -J hist
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 36:00:00
# Is the remaining error missing INFORMATION? Same setup as the gap runs,
# plus a progress photo as a fourth reference, in the mix the interface will
# see: no photo 20%, a photo of the same step 1-7 frames away (11-80 degrees)
# 50%, a photo from an earlier assembly step 30%; 30% of photos come without
# part labels and half are degraded (lighting, colour cast, blur, noise,
# blocked patches). The checkpoint is picked on the mean held-out loss with
# and without a photo, so it cannot trade the no-photo case away. Evaluated as a matrix: no photo / 11, 35, 80 degrees / no
# labels / earlier step, plus the first three assembly steps. Compare with gap40 (same N and
# budget, no history): held-out LPIPS 0.200. A large drop means the inputs,
# not the model or the data size, were the limit.
#   J=$(sbatch --parsable -J hist40 -o hist40.log klein/job_hist.sbatch hist40 n40.json 120000 0 "--n-refs 2")
#   then chained pieces with "auto" as for the gap runs. A 5th argument passes
#   further trainer flags; "--n-refs 2" gives the three-image setup: one
#   laid-out photo, one assembled photo, and the progress photo from memory.
#   Evaluation reads the reference count from the run's args.json.
#
# (setup of the gap runs, for reference:)
#
# Trains on N objects with the setup that can overfit (rank 128 with alpha 32,
# regularisers off, warm-up + cosine), keeps the checkpoint with the lowest held-out image
# v-loss (lora_best); every run trains its full budget (the 25k-step patience
# is a safety net that should not fire), then evaluates THAT checkpoint on held-out objects (own /
# swapped / no photos), on training objects, and on the first three assembly
# steps (per-part colour score). Held-out is always the same 14 unseen objects.
#
#   N:      5      20      40      100
#   steps:  60000  120000  120000  120000   (planned cosine over the whole budget)
#   sbatch -J gap5b -o gap5b.log klein/job_gap.sbatch gap5b n5.json 60000
# usage: sbatch -J <name> -o <name>.log klein/job_gap.sbatch RUN OBJECTS STEPS [RESUME_STEP|auto]
#   Runs longer than the wall-time: submit a chain, each piece resuming from
#   the latest checkpoint ("auto"); pieces after completion are no-ops:
#     J=$(sbatch --parsable -J gap100 -o gap100.log klein/job_gap.sbatch gap100 all 120000)
#     for i in 1 2; do J=$(sbatch --parsable --dependency=afterany:$J -J gap100 -o gap100.log \
#         --open-mode=append klein/job_gap.sbatch gap100 all 120000 auto); done
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$HOME/kleinlib:$HOME/.local/lib/python3.11/site-packages:$HOME/dsenv
PY=$HOME/py311/bin/python
N=${1:?run name}; OBJ=${2:?objects json or all}; STEPS=${3:-60000}; RS=${4:-0}; EXTRA=${5:-}
DATA=$HOME/h200_pilot/v5_colour
set +e
OBJARG=""; OBJTAG=""
if [ "$OBJ" != "all" ]; then
  for c in $OBJ meta/$OBJ $DATA/meta/$OBJ; do [ -f "$c" ] && OBJ="$c" && break; done
  [ -f "$OBJ" ] || { echo "cannot find $OBJ"; exit 1; }
  OBJARG="--objects $OBJ"; OBJTAG="$OBJ"
fi
RESUME=""
if [ "$RS" = "auto" ]; then          # continue from the latest saved step (chained pieces)
  CK=$(ls -d runs/$N/lora_step* 2>/dev/null | sort -V | tail -1)
  if [ -n "$CK" ]; then RS=$(echo $CK | sed "s/.*lora_step0*//"); else RS=0; fi
fi
if [ "$RS" -gt 0 ]; then
  CK=$(printf "runs/%s/lora_step%05d" $N $RS)
  [ -d "$CK" ] || { echo "no checkpoint $CK"; exit 1; }
  RESUME="--resume $CK --start-step $RS"; echo "resuming from $CK"
fi
echo "run=$N objects=$OBJ steps=$STEPS extra=$EXTRA"

if [ ! -f runs/$N/DONE_TRAIN ]; then
  $PY klein/train_klein.py --data $DATA --out runs/$N $OBJARG $RESUME \
      --steps $STEPS --accum 2 --val-every 250 --val-frames 32 --save-every 10000 --save-best --select v --patience 100 \
      --rank 128 --lora-alpha 32 --ref-warmup 0 --lr 3e-4 --lr-warmup 1000 --lr-final 3e-5 \
      --ref-drop 0 --bias-drop 0 --ref-id-drop 0 --part-bias soft --substrate --history 0.8 --hist-offset 1-7 --hist-earlier 0.375 --hist-label-drop 0.3 --hist-augment 0.5 $EXTRA \
    && touch runs/$N/DONE_TRAIN
fi
[ -f runs/$N/DONE_TRAIN ] || { echo "training did not finish"; exit 1; }
[ -d runs/$N/lora_best ] || { echo "no lora_best"; exit 1; }
echo "== best checkpoint"; cat runs/$N/best.json

ev() {   # tag split walks refs [extra]
  local T=$1 S=$2 W=$3 R=$4; shift 4
  [ -f runs/$N/eval_$T.json ] && return
  rm -rf runs/$N/gens_$T
  $PY klein/eval_klein.py --data $DATA --lora runs/$N/lora_best --out runs/$N \
      --split $S --walks $W --refs $R --tag $T "$@"
}
ev onewalk test_unseen 70 own
ev onewalk_d1 test_unseen 70 own --history --hist-offset 1
ev onewalk_d3 test_unseen 70 own --history --hist-offset 3
ev onewalk_d7 test_unseen 70 own --history --hist-offset 7
ev onewalk_d3nolab test_unseen 70 own --history --hist-offset 3 --hist-no-labels
ev onewalk_early test_unseen 70 own --history --hist-earlier --hist-require
rm -rf runs/$N/gens_onewalk_e; mkdir -p runs/$N/gens_onewalk_e
for f in runs/$N/gens_onewalk_early/*.png; do ln -sf ../gens_onewalk/$(basename $f) runs/$N/gens_onewalk_e/; done
ev ontrainwalk train 30 own $OBJARG
ev early test_unseen 30 own --states 1-3
ev early_hist test_unseen 30 own --states 1-3 --history --hist-offset 3

echo; echo "================ metrics (best checkpoint) ================"
$PY metrics.py --data $DATA --limit 100000 --gain 1.07 --no-dreamsim \
  --tags "ontrainwalk:trained on,onewalk:held out no photo,onewalk_d1:held out photo 11 deg,onewalk_d3:held out photo 35 deg,onewalk_d7:held out photo 80 deg,onewalk_d3nolab:held out photo 35 deg no labels,onewalk_e:held out no photo (frames with earlier step),onewalk_early:held out photo from earlier step,early:steps 1-3 no photo,early_hist:steps 1-3 photo 35 deg" \
  --out runs/metrics_$N.json --run-list "$N:pr:$N:$OBJTAG" 2>/dev/null | grep -A2 "^\[metrics\]"
echo; echo "================ per-part colour, steps 1-3 ================"
$PY klein/part_colour.py --data $DATA --run runs/$N --tags early,early_hist
$PY klein/offobject.py --data $DATA --run runs/$N --tags onewalk,onewalk_d3,onewalk_early
echo "HIST COMPLETE $N"
