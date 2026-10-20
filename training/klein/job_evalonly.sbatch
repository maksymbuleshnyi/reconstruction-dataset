#!/bin/bash
#SBATCH -J evalonly
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -o evalonly.log
# Held-out loss of finished models on 128 fixed frames (4x the training-time
# set), split into four noise bands. Does N=100's lower loss hold on more
# frames, and is it lower at every noise level or only at high noise (layout)
# while worse at low noise (fine detail, what LPIPS sees)?
# usage: sbatch klein/job_evalonly.sbatch "gap20 gap40b gap100"
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$HOME/kleinlib:$HOME/.local/lib/python3.11/site-packages:$HOME/dsenv
PY=$HOME/py311/bin/python
for R in ${1:-gap20 gap40b gap100}; do
  $PY klein/train_klein.py --data $HOME/h200_pilot/v5_colour --out runs/evalonly_$R --eval-only \
      --resume runs/$R/lora_best --val-frames 128 --rank 128 --lora-alpha 32 \
      --part-bias soft --substrate --bias-drop 0 2>&1 | grep "EVALONLY\|Error\|error:"
done
echo "EVALONLY COMPLETE"
