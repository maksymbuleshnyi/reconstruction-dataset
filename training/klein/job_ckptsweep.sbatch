#!/bin/bash
#SBATCH -J ckptsweep
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o ckptsweep.log
# How much does held-out LPIPS move between neighbouring checkpoints of one
# run? Renders the saved checkpoints of RUN on the same held-out subset as
# basecheck (distilled, 4 steps, own photos) and scores LPIPS and invented
# content for each. If neighbours differ by a few hundredths, one checkpoint
# per run is not enough to compare runs, and picking by loss is not picking by
# image quality.
# usage: sbatch klein/job_ckptsweep.sbatch RUN "STEP STEP ..."     (the final one is lora_best)
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$HOME/kleinlib:$HOME/.local/lib/python3.11/site-packages:$HOME/dsenv
PY=$HOME/py311/bin/python
DATA=$HOME/h200_pilot/v5_colour
RUN=${1:-gap100}; STEPS=${2:-"80000 90000 100000 110000"}
OUT=runs/sweep_$RUN
mkdir -p $OUT
set +e
TAGS=""; LABELS=""
for S in $STEPS; do
  CK=$(printf "runs/%s/lora_step%05d" $RUN $S)
  [ -d $CK ] || { echo "no $CK"; continue; }
  T=s$((S / 1000))k
  if [ ! -f $OUT/eval_$T.json ]; then
    rm -rf $OUT/gens_$T
    $PY klein/eval_klein.py --data $DATA --lora $CK --out $OUT --split test_unseen \
        --walks 15 --refs own --tag $T
  fi
  TAGS="$TAGS,$T"; LABELS="$LABELS,$T:$RUN at $((S / 1000))k"
done
# the evaluated (best-loss) checkpoint, rendered by basecheck on the same subset
[ -d runs/basecheck/gens_d4_g100 ] && [ "$RUN" = gap100 ] && ln -sfn ../basecheck/gens_d4_g100 $OUT/gens_best \
  && TAGS="$TAGS,best" && LABELS="$LABELS,best:$RUN best (by loss)"
echo; echo "================ LPIPS by checkpoint, held-out subset ================"
$PY metrics.py --data $DATA --limit 100000 --gain 1.07 --no-dreamsim \
  --tags "${LABELS#,}" --out runs/metrics_sweep_$RUN.json --run-list "sweep_$RUN:pr:sweep_$RUN:" 2>/dev/null \
  | grep -A1 "^\[metrics\]" | grep model
echo; echo "================ invented content by checkpoint ================"
$PY klein/offobject.py --data $DATA --run $OUT --tags "${TAGS#,}"
echo; echo "================ held-out loss at those steps ================"
for S in $STEPS 120000; do grep "^val step $S " $RUN.log | tail -1; done
echo "CKPTSWEEP COMPLETE"
