"""Default local paths for progressive pose editing models."""

from pathlib import Path

SHARED_PRETRAINED_ROOT = Path("/data/shared-vilab/pretrained_models")
MOTIONEDIT_VLM_DIR = SHARED_PRETRAINED_ROOT / "motionedit_vlm"

# Already present on shared storage; not downloaded by tools/download_progressive_pose_models.py
EDITOR_BASE_MODEL = SHARED_PRETRAINED_ROOT / "Qwen-Image-Edit-2511"
PLANNER_VLM_MODEL = SHARED_PRETRAINED_ROOT / "Qwen3-VL-8B-Instruct"

# Downloaded into motionedit_vlm/
MOTIONEDIT_LORA_DIR = MOTIONEDIT_VLM_DIR / "motionedit-lora"
DINOV2_MODEL = MOTIONEDIT_VLM_DIR / "dinov2-base"
UNIMATCH_DIR = MOTIONEDIT_VLM_DIR / "unimatch"
UNIMATCH_CKPT = (
    UNIMATCH_DIR
    / "pretrained"
    / "gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
)

MANIFEST_PATH = MOTIONEDIT_VLM_DIR / "progressive_pose_models_manifest.json"
