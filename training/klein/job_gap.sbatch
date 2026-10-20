#!/bin/bash
#SBATCH -J gap
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 36:00:00
# One point of the train / held-out gap curve against data size.
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
N=${1:?run name}; OBJ=${2:?objects json or all}; STEPS=${3:-60000}; RS=${4:-0}
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
  CK=$(ls -d runs/$N/lora_step* 2>/dev/null | sort | tail -1)
  if [ -n "$CK" ]; then RS=$(echo $CK | sed "s/.*lora_step0*//"); else RS=0; fi
fi
if [ "$RS" -gt 0 ]; then
  CK=$(printf "runs/%s/lora_step%05d" $N $RS)
  [ -d "$CK" ] || { echo "no checkpoint $CK"; exit 1; }
  RESUME="--resume $CK --start-step $RS"; echo "resuming from $CK"
fi
echo "run=$N objects=$OBJ steps=$STEPS"

if [ ! -f runs/$N/DONE_TRAIN ]; then
  $PY klein/train_klein.py --data $DATA --out runs/$N $OBJARG $RESUME \
      --steps $STEPS --accum 2 --val-every 250 --val-frames 32 --save-every 10000 --save-best --select v --patience 100 \
      --rank 128 --lora-alpha 32 --ref-warmup 0 --lr 3e-4 --lr-warmup 1000 --lr-final 3e-5 \
      --ref-drop 0 --bias-drop 0 --ref-id-drop 0 --part-bias soft --substrate \
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
ev onewalk_swap test_unseen 70 swap
ev onewalk_noref test_unseen 70 none
ev ontrainwalk train 30 own $OBJARG
ev early test_unseen 30 own --states 1-3
ev early_swap test_unseen 30 swap --states 1-3
ev early_noref test_unseen 30 none --states 1-3

echo; echo "================ metrics (best checkpoint) ================"
$PY metrics.py --data $DATA --limit 100000 --gain 1.07 --no-dreamsim \
  --tags "ontrainwalk:trained on,onewalk:held out (own refs),onewalk_swap:held out (SWAPPED refs),onewalk_noref:held out (NO refs),early:steps 1-3 (own refs),early_swap:steps 1-3 (SWAPPED refs),early_noref:steps 1-3 (NO refs)" \
  --out runs/metrics_$N.json --run-list "$N:pr:$N:$OBJTAG" 2>/dev/null | grep -A2 "^\[metrics\]"
echo; echo "================ per-part colour, steps 1-3 ================"
$PY klein/part_colour.py --data $DATA --run runs/$N --tags early,early_swap,early_noref
echo "GAP COMPLETE $N"
