#!/bin/bash
#SBATCH -J curve
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN,ALL
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
# Whole-frame L1 / RMSE / LPIPS of every saved checkpoint of RUN on fixed
# frames (klein/ckpt_curve.py), into runs/RUN/curve.csv. Rerunnable: only
# checkpoints not scored yet are rendered. Usually queued after the run:
#   sbatch --dependency=afterany:<last training piece> -o curve_hist40bo.log klein/job_curve.sbatch hist40bo
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$HOME/kleinlib:$HOME/.local/lib/python3.11/site-packages:$HOME/dsenv
$HOME/py311/bin/python klein/ckpt_curve.py --data $HOME/h200_pilot/v5_colour --run runs/${1:?run} 2>&1 \
  | grep "\[curve\]\|\[promo\]\|COMPLETE\|rror"
