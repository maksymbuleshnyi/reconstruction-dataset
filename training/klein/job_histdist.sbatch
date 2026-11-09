#!/bin/bash
#SBATCH -J histdist
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:30:00
#SBATCH -o histdist.log
# How far away may the progress photo be? gap40b with the history image taken
# 1, 3, 5, 7 frames away in the same walk (about 11, 35, 57, 80 degrees), and
# at 3 frames WITHOUT part labels (a real photo of unknown camera), and from an
# earlier assembly step (scored on the frames that have one). Same
# held-out subset as histinfer, whose own-photos and 3-frames renders are
# reused. No training.
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$HOME/kleinlib:$HOME/.local/lib/python3.11/site-packages:$HOME/dsenv
PY=$HOME/py311/bin/python
DATA=$HOME/h200_pilot/v5_colour
RUN=gap40b; OUT=runs/histdist_$RUN; SRC=runs/histinfer_$RUN
mkdir -p $OUT
ln -sfn ../histinfer_$RUN/gens_refs $OUT/gens_refs
ln -sfn ../histinfer_$RUN/gens_refs_hist $OUT/gens_d3
set +e
ev() {   # tag [extra eval flags]
  local T=$1; shift
  [ -f $OUT/eval_$T.json ] && return
  rm -rf $OUT/gens_$T
  $PY klein/eval_klein.py --data $DATA --lora runs/$RUN/lora_best --out $OUT --split test_unseen \
      --walks 15 --refs own --tag $T "$@"
}
ev d1 --history --hist-offset 1
ev d5 --history --hist-offset 5
ev d7 --history --hist-offset 7
ev d3nolab --history --hist-offset 3 --hist-no-labels
ev early --history --hist-earlier --hist-require
# the no-history renders of exactly the frames that have an earlier step
rm -rf $OUT/gens_refs_e; mkdir -p $OUT/gens_refs_e
for f in $OUT/gens_early/*.png; do b=$(basename $f); ln -sf ../../$SRC/gens_refs/$b $OUT/gens_refs_e/$b; done
echo; echo "================ $RUN, held-out subset: history distance ================"
$PY metrics.py --data $DATA --limit 100000 --gain 1.07 --no-dreamsim \
  --tags "refs:no history,d1:history 1 frame away (~11 deg),d3:history 3 frames (~35 deg),d5:history 5 frames (~57 deg),d7:history 7 frames (~80 deg),d3nolab:history 3 frames WITHOUT labels,refs_e:no history (frames with an earlier step),early:history from an EARLIER step (same frames)" \
  --out runs/metrics_histdist_$RUN.json --run-list "histdist_$RUN:pr:histdist_$RUN:" 2>/dev/null \
  | grep -A1 "^\[metrics\]" | grep model
echo "HISTDIST COMPLETE"
