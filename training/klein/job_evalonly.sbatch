#!/bin/bash
#SBATCH -J evalonly
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:30:00
#SBATCH -o evalonly.log
# Fixed-set loss of finished models on 128 fixed frames (4x the training-time
# set), on each run's own training objects and on the held-out objects, split
# into four noise bands plus the lowest-noise points alone. Does N=100's lower loss hold on more
# frames, and is it lower at every noise level or only at high noise (layout)
# while worse at low noise (fine detail, what LPIPS sees)?
# usage: sbatch klein/job_evalonly.sbatch "gap5 gap20 gap40b gap100"
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$HOME/kleinlib:$HOME/.local/lib/python3.11/site-packages:$HOME/dsenv
PY=$HOME/py311/bin/python
for R in ${1:-gap5 gap20 gap40b gap100}; do
  case $R in                         # each run's own training objects
    gap5*)  OBJ="--objects meta/n5.json" ;;
    gap20*) OBJ="--objects n20.json" ;;
    gap40*) OBJ="--objects n40.json" ;;
    *)      OBJ="" ;;                # all 100
  esac
  L=runs/$R/lora_best                # gap100: the 120k checkpoint of the curves
  [ "$R" = gap100 ] && [ -d runs/gap100/lora_best_120k ] && L=runs/gap100/lora_best_120k
  $PY klein/train_klein.py --data $HOME/h200_pilot/v5_colour --out runs/evalonly_$R --eval-only \
      --resume $L $OBJ --val-frames 128 --rank 128 --lora-alpha 32 \
      --part-bias soft --substrate --bias-drop 0 2>&1 | grep "EVALONLY\|Error\|error:"
done
echo "EVALONLY COMPLETE"
