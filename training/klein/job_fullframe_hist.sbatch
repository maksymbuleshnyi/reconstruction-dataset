#!/bin/bash
#SBATCH -J ffhist
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
# Whole-frame error with and without the nearby photo, on the SAME frames:
# all held-out test frames (onewalk) and the run's training frames
# (ontrainwalk), each rendered twice by the run's best checkpoint: own photos
# only, and own photos plus the real image of the frame 3 steps away in the
# same camera walk (~35 degrees, same assembly step), given at inference only.
# Renders that already exist are reused.
# usage: sbatch -J ffh_<run> -o ffhist_<run>.log klein/job_fullframe_hist.sbatch RUN [OBJECTS_JSON]
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$HOME/kleinlib:$HOME/.local/lib/python3.11/site-packages:$HOME/dsenv
PY=$HOME/py311/bin/python
DATA=$HOME/h200_pilot/v5_colour
R=${1:?run}; O=${2:-}; OBJ=""; [ -n "$O" ] && OBJ="--objects $O"
set +e
echo "== $R: best checkpoint $(cat runs/$R/best.json 2>/dev/null | tr -d '\n ' | cut -c1-80)"
ev() {   # tag split walks [extra]   renders only the frames not there yet
  local T=$1 S=$2 W=$3; shift 3
  echo "   $T: $(ls runs/$R/gens_$T 2>/dev/null | wc -l) renders before"
  $PY klein/eval_klein.py --data $DATA --lora runs/$R/lora_best --out runs/$R --split $S \
      --walks $W --refs own --tag $T "$@" 2>&1 | grep "EVAL COMPLETE\|rror"
}
ev onewalk test_unseen 70
ev onewalk_hist test_unseen 70 --history --hist-offset 3
ev ontrainwalk train 30 $OBJ
ev ontrainwalk_hist train 30 $OBJ --history --hist-offset 3
echo; echo "================ $R: whole frame, generated vs real, pixels in [0, 1] ================"
$PY klein/fullframe_err.py --data $DATA --run runs/$R --tags onewalk,onewalk_hist,ontrainwalk,ontrainwalk_hist --lpips
echo "FFHIST COMPLETE $R"
