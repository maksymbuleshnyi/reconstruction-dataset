#!/bin/bash
#SBATCH -J overfit5
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
# The sanity test: can the model memorise FIVE frames to a loss near zero?
# Five frames of one object (one appearance), every regulariser off, 3000
# steps = 1200 passes. fixed_train is the loss on exactly those five frames;
# fixed_heldout (unseen appearances) is printed for reference only. At the
# end the five frames are rendered and scored against the truth (LPIPS; the
# VAE floor on training frames is 0.0075) and put on a sheet next to it.
#   sbatch -J overfit5 -o overfit5.log klein/job_overfit5.sbatch overfit5 128
# usage: sbatch -J <name> -o <name>.log klein/job_overfit5.sbatch RUN [RANK] [STEPS] [EXTRA]
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$HOME/kleinlib:$HOME/.local/lib/python3.11/site-packages:$HOME/dsenv
PY=$HOME/py311/bin/python
N=${1:?run name}; RANK=${2:-128}; STEPS=${3:-3000}; EXTRA=${4:-}
DATA=$HOME/h200_pilot/v5_unit
SEL="--objects meta/one.json --variants 0-0 --max-frames 5"
set +e
rm -rf runs/$N
$PY klein/train_klein.py --data $DATA --out runs/$N $SEL \
    --val-split train --val-variants 16-19 --val-every 100 --val-frames 32 \
    --steps $STEPS --accum 2 --save-every $STEPS --rank $RANK \
    --ref-warmup 0 --lr 3e-4 --lr-warmup 100 --lr-final 3e-5 \
    --ref-drop 0 --bias-drop 0 --ref-id-drop 0 --part-bias soft --substrate $EXTRA \
  || { echo "training failed"; exit 1; }
echo "== step, v-loss five frames, v-loss unseen, IMAGE error five frames, image error unseen"
grep "^val step" $N.log | awk 'NR % 5 == 0 {print $3, $5, $7, $9, $11}'
$PY klein/eval_klein.py --data $DATA --lora runs/$N/lora --out runs/$N --split train $SEL \
    --walks 100 --refs own --tag fit
$PY metrics.py --data $DATA --limit 100000 --gain 1.07 --no-dreamsim \
  --tags "fit:the five memorised frames" --out runs/metrics_$N.json --run-list "$N:pr:$N:" 2>/dev/null \
  | grep -A1 "^\[metrics\]" | grep model
$PY klein/sheet.py --data $DATA --run runs/$N --n 5 --prefix fit --out sheet_$N.jpg
echo "OVERFIT5 COMPLETE $N"
