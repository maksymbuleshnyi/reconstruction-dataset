#!/bin/bash
#SBATCH -J histinfer
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
# Does the missing information help, without any training? The finished
# model gets the history image (the real image of the frame 3 steps away in
# the same walk: same object and step, ~35 degrees away) as a fourth
# reference at inference only. Same held-out subset as the earlier checks.
# A clear drop in LPIPS means the inputs were the limit. No drop is
# inconclusive: the model never learned to use such a view (then train).
# usage: sbatch -J hi_<run> -o histinfer_<run>.log klein/job_histinfer.sbatch RUN [LORA_DIR]
#   (default gap40b, runs/RUN/lora_best)
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$HOME/kleinlib:$HOME/.local/lib/python3.11/site-packages:$HOME/dsenv
PY=$HOME/py311/bin/python
DATA=$HOME/h200_pilot/v5_colour
RUN=${1:-gap40b}; LORA=${2:-runs/$RUN/lora_best}
OUT=runs/histinfer_$RUN
mkdir -p $OUT
set +e
ev() {   # tag [extra eval flags]
  local T=$1; shift
  [ -f $OUT/eval_$T.json ] && return
  rm -rf $OUT/gens_$T
  $PY klein/eval_klein.py --data $DATA --lora $LORA --out $OUT --split test_unseen \
      --walks 15 --tag $T "$@"
}
ev refs --refs own
ev refs_hist --refs own --history
ev hist_only --refs none --history
echo; echo "================ $RUN ($LORA), held-out subset ================"
$PY metrics.py --data $DATA --limit 100000 --gain 1.07 --no-dreamsim \
  --tags "refs:own photos,refs_hist:own photos + HISTORY (no training),hist_only:history only" \
  --out runs/metrics_histinfer_$RUN.json --run-list "histinfer_$RUN:pr:histinfer_$RUN:" 2>/dev/null \
  | grep -A1 "^\[metrics\]" | grep model
$PY klein/sheet.py --data $DATA --run $OUT --n 8 --prefix refs \
    --also "refs_hist:+ history,hist_only:history only" --out sheet_histinfer_$RUN.jpg
echo "HISTINFER COMPLETE"
