#!/bin/bash
#SBATCH -J promo
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
# The promo session (klein/promo_eval.py) for each run's best checkpoint, twice:
# the 3 reference photos only, and with the memory of progress photos (the
# stored photo closest to each frame as a 4th reference). Same 180 frames,
# same photos, same seeds for every run. Renders that exist are reused.
# usage: sbatch -o promo.log klein/job_promo.sbatch gap40b [hist40 ...]
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$HOME/kleinlib:$HOME/.local/lib/python3.11/site-packages:$HOME/dsenv
PY=$HOME/py311/bin/python
DATA=$HOME/h200_pilot/v5_colour
set +e
for R in "$@"; do
  [ -d runs/$R/lora_best ] || { echo "no runs/$R/lora_best, skipped"; continue; }
  for M in none nearest; do
    $PY klein/promo_eval.py --data $DATA --lora runs/$R/lora_best --out runs/$R --tag $M --memory $M 2>&1 \
      | grep "\[promo\]\|PROMO COMPLETE\|rror"
  done
done
echo "PROMOS COMPLETE"
