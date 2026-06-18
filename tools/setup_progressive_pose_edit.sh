#!/usr/bin/env bash
set -euo pipefail

# Environment setup for inference/progressive_pose_edit.py.
#
# Usage:
#   bash tools/setup_progressive_pose_edit.sh
#
# Optional:
#   ENV_NAME=motionedit bash tools/setup_progressive_pose_edit.sh
#   SKIP_FLASH_ATTN=1 bash tools/setup_progressive_pose_edit.sh
#   DOWNLOAD_MODELS=1 HF_TOKEN=... bash tools/setup_progressive_pose_edit.sh
#
# Models layout (shared storage):
#   /data/shared-vilab/pretrained_models/Qwen-Image-Edit-2511   (already present)
#   /data/shared-vilab/pretrained_models/Qwen3-VL-8B-Instruct   (already present)
#   /data/shared-vilab/pretrained_models/motionedit_vlm/        (downloaded by script)

ENV_NAME="${ENV_NAME:-motionedit}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  else
    echo "conda was not found. Install Miniconda/Anaconda first." >&2
    exit 1
  fi
fi

cd "${REPO_ROOT}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] Updating conda env: ${ENV_NAME}"
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  echo "[setup] Creating conda env from environment.yml: ${ENV_NAME}"
  if [ "${ENV_NAME}" = "motionedit" ]; then
    conda env create -f environment.yml
  else
    conda env create -n "${ENV_NAME}" -f environment.yml
  fi
fi

echo "[setup] Installing/updating runtime helpers for Qwen3-VL + MotionEdit"
conda run -n "${ENV_NAME}" python -m pip install \
  "transformers>=4.57.0,<5.0" \
  "peft>=0.18.0" \
  "diffusers>=0.37.0,<0.39.0" \
  "torchao>=0.16.0,<0.17.0" \
  "huggingface-hub>=0.34.0" \
  "qwen-vl-utils" \
  "accelerate" \
  "safetensors" \
  "packaging" \
  "cffi" \
  "soundfile"

if [ "${SKIP_FLASH_ATTN}" != "1" ]; then
  echo "[setup] Installing flash-attn. Set SKIP_FLASH_ATTN=1 to skip this."
  conda run -n "${ENV_NAME}" python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
else
  echo "[setup] Skipping flash-attn install"
fi

echo "[setup] Verifying key imports"
conda run -n "${ENV_NAME}" python - <<'PY'
import transformers
import peft
import torchao
from packaging.version import Version
from transformers import AutoModelForImageTextToText, AutoProcessor, HybridCache
from diffusers import QwenImageEditPlusPipeline
from huggingface_hub import snapshot_download, hf_hub_download

tv = Version(transformers.__version__)
assert Version("4.57.0") <= tv < Version("5.0.0")
assert Version(peft.__version__) >= Version("0.18.0")
assert Version("0.16.0") <= Version(torchao.__version__) < Version("0.17.0")
_ = HybridCache
print(f"transformers {transformers.__version__}, peft {peft.__version__}, torchao {torchao.__version__}")
print("Key imports OK")
PY

if [ "${DOWNLOAD_MODELS}" = "1" ]; then
  echo "[setup] Downloading required model files"
  conda run -n "${ENV_NAME}" python tools/download_progressive_pose_models.py
else
  echo "[setup] Model download skipped. Run this when ready:"
  echo "        conda run -n ${ENV_NAME} python tools/download_progressive_pose_models.py"
fi

echo "[setup] Done"
