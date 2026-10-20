#!/bin/bash
#SBATCH -J kleinsetup
#SBATCH -x watgpu208,watgpu708
#SBATCH -p YUNTIAN
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH -o kleinsetup.log
# One-off: newer diffusers/transformers/peft into their own folder (the SDXL
# jobs keep the versions they were run with), download FLUX.2 [klein] 4B
# (distilled, for serving and evaluation) and its base checkpoint (for LoRA
# training), then time the distilled model with 0-5 reference images.
#
# Downloads about 30 GB into $HF_HOME. Nothing outside ~/h200_pilot,
# ~/kleinlib and the HF cache is touched.
cd ~/h200_pilot
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
PY=$HOME/py311/bin/python
LIB=$HOME/kleinlib
# kleinlib first so its diffusers/transformers shadow the older ones in ~/.local
export PYTHONPATH=$LIB:$HOME/.local/lib/python3.11/site-packages
set +e

if [ ! -f $LIB/DONE ]; then
  mkdir -p $LIB
  # --no-deps: torch and the rest already exist in the cluster environment;
  # only the packages that must be newer are installed here
  $PY -m pip install --target $LIB --no-deps --upgrade \
      "git+https://github.com/huggingface/diffusers" \
      "transformers>=4.57" peft accelerate "huggingface_hub>=0.34" \
      safetensors tokenizers regex sentencepiece \
    && touch $LIB/DONE
fi
$PY - <<'EOF'
import torch, diffusers, transformers, peft, huggingface_hub
print("[env] torch", torch.__version__, "cuda", torch.cuda.is_available())
print("[env] diffusers", diffusers.__version__, "transformers", transformers.__version__,
      "peft", peft.__version__, "hub", huggingface_hub.__version__)
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2
print("[env] klein classes import ok")
EOF
[ $? -eq 0 ] || { echo "ENV CHECK FAILED"; exit 1; }

$PY - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ("black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-base-4B"):
    p = snapshot_download(repo)
    print("[download]", repo, "->", p, flush=True)
EOF
[ $? -eq 0 ] || { echo "DOWNLOAD FAILED"; exit 1; }
du -sh $HF_HOME/hub/models--black-forest-labs--FLUX.2-klein-* 2>/dev/null

$PY klein/profile_klein.py --res 512 --steps 4 --max-ctx 5
echo "KLEIN SETUP COMPLETE"
