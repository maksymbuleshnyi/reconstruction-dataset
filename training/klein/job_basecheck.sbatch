#!/bin/bash
#SBATCH -J basecheck
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o basecheck.log
# Why does gap100 have the lowest held-out LOSS but a worse held-out LPIPS
# than gap40? The loss is measured on the BASE model the LoRA was trained
# against; the images come from the 4-step DISTILLED model with the same LoRA.
# If the LoRA transfers worse to the distilled model the more it is trained,
# sampling with the base model should reverse the order. Same held-out subset
# (about 15 frames per unseen object, own photos) for every row:
#   d4_*   distilled, 4 steps (what all evaluations so far used)
#   b28_*  base, 28 steps, CFG 4 (the model the LoRA was trained on)
#   d4_g100_60k  gap100 at 60k steps, distilled: does LPIPS get worse with training?
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$HOME/kleinlib:$HOME/.local/lib/python3.11/site-packages:$HOME/dsenv
PY=$HOME/py311/bin/python
DATA=$HOME/h200_pilot/v5_colour
BASE=black-forest-labs/FLUX.2-klein-base-4B
OUT=runs/basecheck
mkdir -p $OUT
set +e
ev() {   # tag lora [extra eval flags]
  local T=$1 L=$2; shift 2
  [ -f $OUT/eval_$T.json ] && return
  rm -rf $OUT/gens_$T
  $PY klein/eval_klein.py --data $DATA --lora $L --out $OUT --split test_unseen \
      --walks 15 --refs own --tag $T "$@"
}
ev d4_g40 runs/gap40/lora_best
ev d4_g100 runs/gap100/lora_best
ev d4_g100_60k runs/gap100/lora_step60000
ev b28_g40 runs/gap40/lora_best --model $BASE --steps 28 --cfg 4.0
ev b28_g100 runs/gap100/lora_best --model $BASE --steps 28 --cfg 4.0
echo; echo "================ held-out subset, own photos ================"
$PY metrics.py --data $DATA --limit 100000 --gain 1.07 --no-dreamsim \
  --tags "d4_g40:gap40 distilled 4 steps,d4_g100:gap100 distilled 4 steps,d4_g100_60k:gap100 at 60k distilled 4 steps,b28_g40:gap40 BASE 28 steps cfg4,b28_g100:gap100 BASE 28 steps cfg4" \
  --out runs/metrics_basecheck.json --run-list "basecheck:pr:basecheck:" 2>/dev/null | grep -A1 "^\[metrics\]" | grep model
$PY klein/sheet.py --data $DATA --run $OUT --n 8 --prefix d4_g40 \
    --also "d4_g100:gap100 distilled,b28_g40:gap40 BASE,b28_g100:gap100 BASE" --out sheet_basecheck.jpg
echo "BASECHECK COMPLETE"
