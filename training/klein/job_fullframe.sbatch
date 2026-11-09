#!/bin/bash
#SBATCH -J fullframe
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o fullframe.log
# Generated vs real frame over the WHOLE image, pixels in [0, 1] (L1 and
# RMSE, as NeuralOS reports), for the best checkpoint of each curve run, on
# held-out frames (onewalk) and on training frames (ontrainwalk). Renders are
# reused where they exist; missing ones are rendered with the run's lora_best.
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$HOME/kleinlib:$HOME/.local/lib/python3.11/site-packages:$HOME/dsenv
PY=$HOME/py311/bin/python
DATA=$HOME/h200_pilot/v5_colour
set +e
for spec in "gap5|meta/n5.json" "gap20|n20.json" "gap40b|n40.json" "gap100|"; do
  R=${spec%%|*}; O=${spec#*|}; OBJ=""; [ -n "$O" ] && OBJ="--objects $O"
  echo "== $R: best checkpoint $(cat runs/$R/best.json 2>/dev/null | tr -d '\n ' | cut -c1-60)"
  for T in onewalk ontrainwalk; do
    n=$(ls runs/$R/gens_$T 2>/dev/null | wc -l)
    if [ "$n" -lt 50 ]; then
      echo "   rendering $T (found $n frames)"
      rm -rf runs/$R/gens_$T
      if [ $T = onewalk ]; then
        $PY klein/eval_klein.py --data $DATA --lora runs/$R/lora_best --out runs/$R --split test_unseen \
            --walks 70 --refs own --tag $T 2>&1 | grep "EVAL COMPLETE\|rror"
      else
        $PY klein/eval_klein.py --data $DATA --lora runs/$R/lora_best --out runs/$R --split train \
            --walks 30 --refs own --tag $T $OBJ 2>&1 | grep "EVAL COMPLETE\|rror"
      fi
    else
      echo "   $T: reusing $n renders"
    fi
  done
done
echo; echo "================ whole-frame error, generated vs real, pixels in [0, 1] ================"
$PY klein/fullframe_err.py --data $DATA --run runs/gap5,runs/gap20,runs/gap40b,runs/gap100 --tags onewalk,ontrainwalk --lpips
echo "FULLFRAME COMPLETE"
