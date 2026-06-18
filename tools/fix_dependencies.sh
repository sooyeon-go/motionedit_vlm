#!/usr/bin/env bash
set -euo pipefail

# Fix dependencies for progressive_pose_edit.py in the motionedit env.
#
# Compatible stack:
#   - transformers>=4.57,<5   (Qwen3-VL needs 4.57+; 5.x removes HybridCache)
#   - peft>=0.18.0            (HybridCache import fix for newer transformers)
#   - diffusers==0.36.0       (Qwen Image Edit pipeline)
#
# Usage:
#   bash tools/fix_dependencies.sh
#   ENV_NAME=motionedit bash tools/fix_dependencies.sh

ENV_NAME="${ENV_NAME:-motionedit}"

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  else
    echo "conda was not found." >&2
    exit 1
  fi
fi

echo "[fix] Installing compatible transformers / peft / diffusers / torchao stack"
conda run -n "${ENV_NAME}" python -m pip install \
  "transformers>=4.57.0,<5.0" \
  "peft>=0.18.0" \
  "diffusers==0.36.0" \
  "torchao>=0.16.0" \
  "huggingface-hub>=0.34.0" \
  "qwen-vl-utils" \
  "accelerate" \
  "safetensors" \
  "packaging"

echo "[fix] Verifying imports"
conda run -n "${ENV_NAME}" python - <<'PY'
import huggingface_hub
import transformers
import peft
import torchao
from packaging.version import Version
from transformers import AutoModelForImageTextToText, AutoProcessor, HybridCache
from diffusers import QwenImageEditPlusPipeline

print(f"huggingface-hub {huggingface_hub.__version__}")
print(f"transformers {transformers.__version__}")
print(f"peft {peft.__version__}")
print(f"torchao {torchao.__version__}")

tv = Version(transformers.__version__)
assert Version("4.57.0") <= tv < Version("5.0.0"), (
    f"Need transformers in [4.57, 5.0), got {transformers.__version__}"
)
assert Version(peft.__version__) >= Version("0.18.0")
assert Version(torchao.__version__) >= Version("0.16.0"), (
    f"Need torchao>=0.16.0 for peft LoRA loading, got {torchao.__version__}"
)
_ = HybridCache
print("Qwen3-VL + MotionEdit dependency stack OK")
PY

echo "[fix] Done. Re-run your script."
