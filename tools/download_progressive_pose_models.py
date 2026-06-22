"""Download models needed by inference/progressive_pose_edit.py from Hugging Face.

Models already on shared storage (not downloaded here):
  - /data/shared-vilab/pretrained_models/Qwen-Image-Edit-2511
  - /data/shared-vilab/pretrained_models/Qwen3-VL-8B-Instruct

Downloaded into /data/shared-vilab/pretrained_models/motionedit_vlm/:
  - motionedit-lora/          (elaine1wan/motionedit adapter)
  - qwen-angles-lora/         (fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA)
  - dinov2-base/              (facebook/dinov2-base)
  - unimatch/pretrained/...   (UniMatch optical flow checkpoint)

Example:
  python tools/download_progressive_pose_models.py

If a gated model requires auth:
  HF_TOKEN=hf_xxx python tools/download_progressive_pose_models.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_paths import (  # noqa: E402
    DINOV2_MODEL,
    EDITOR_BASE_MODEL,
    MOTIONEDIT_LORA_DIR,
    MOTIONEDIT_VLM_DIR,
    PLANNER_VLM_MODEL,
    QWEN_ANGLES_LORA_DIR,
    QWEN_ANGLES_LORA_REPO,
    QWEN_ANGLES_LORA_WEIGHT,
    UNIMATCH_CKPT,
    UNIMATCH_DIR,
)

DEFAULT_UNIMATCH_FILENAME = (
    "pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
)


def _token(explicit_token: Optional[str]) -> Optional[str]:
    return explicit_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def download_snapshot(
    repo_id: str,
    destination: Path,
    token: Optional[str],
    repo_type: Optional[str] = None,
    allow_patterns: Optional[list[str]] = None,
) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[download] snapshot {repo_id} -> {destination}")
    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(destination),
        token=token,
        allow_patterns=allow_patterns,
    )


def download_file(
    repo_id: str,
    filename: str,
    destination_root: Path,
    token: Optional[str],
    repo_type: Optional[str] = None,
) -> str:
    destination_root.mkdir(parents=True, exist_ok=True)
    print(f"[download] file {repo_id}/{filename} -> {destination_root}")
    return hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        local_dir=str(destination_root),
        token=token,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face assets for progressive pose editing.",
    )
    parser.add_argument(
        "--models_dir",
        default=str(MOTIONEDIT_VLM_DIR),
        help="Directory for motionedit_vlm assets (LoRA, DINOv2, UniMatch).",
    )
    parser.add_argument("--hf_token", default=None, help="Optional Hugging Face token.")

    parser.add_argument("--motionedit_lora_repo", default="elaine1wan/motionedit")
    parser.add_argument(
        "--angles_lora_repo",
        default=QWEN_ANGLES_LORA_REPO,
        help="HF repo for Qwen-Image-Edit multi-angle camera LoRA.",
    )
    parser.add_argument(
        "--angles_lora_filename",
        default=QWEN_ANGLES_LORA_WEIGHT,
        help="LoRA weight filename inside the angles repo.",
    )
    parser.add_argument("--dinov2_repo", default="facebook/dinov2-base")

    parser.add_argument(
        "--unimatch_repo_id",
        default="haofeixu/unimatch",
        help="HF repo or Space containing the UniMatch checkpoint.",
    )
    parser.add_argument(
        "--unimatch_repo_type",
        default="space",
        choices=["model", "dataset", "space"],
        help="Repo type for --unimatch_repo_id.",
    )
    parser.add_argument(
        "--unimatch_filename",
        default=DEFAULT_UNIMATCH_FILENAME,
        help="Checkpoint path inside the UniMatch HF repo.",
    )
    parser.add_argument(
        "--unimatch_output_root",
        default=str(UNIMATCH_DIR),
        help="Download root for UniMatch (file lands under pretrained/).",
    )

    parser.add_argument("--skip_motionedit_lora", action="store_true")
    parser.add_argument("--skip_angles_lora", action="store_true")
    parser.add_argument(
        "--angles_lora_only",
        action="store_true",
        help="Download only the Qwen multi-angle LoRA.",
    )
    parser.add_argument("--skip_dinov2", action="store_true")
    parser.add_argument("--skip_unimatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = _token(args.hf_token)
    models_dir = Path(args.models_dir)
    manifest: dict[str, str] = {
        "editor_base": str(EDITOR_BASE_MODEL),
        "planner_vlm": str(PLANNER_VLM_MODEL),
        "note": (
            "editor_base and planner_vlm are expected to already exist on shared "
            "storage and are not downloaded by this script."
        ),
    }

    angles_only = args.angles_lora_only

    if not args.skip_motionedit_lora and not angles_only:
        lora_dir = models_dir / "motionedit-lora"
        manifest["motionedit_lora_adapter"] = download_file(
            repo_id=args.motionedit_lora_repo,
            filename="adapter_model_converted.safetensors",
            destination_root=lora_dir,
            token=token,
        )
        manifest["motionedit_lora_config"] = download_file(
            repo_id=args.motionedit_lora_repo,
            filename="adapter_config.json",
            destination_root=lora_dir,
            token=token,
        )

    if not args.skip_angles_lora:
        angles_dir = models_dir / "qwen-angles-lora"
        manifest["qwen_angles_lora"] = download_file(
            repo_id=args.angles_lora_repo,
            filename=args.angles_lora_filename,
            destination_root=angles_dir,
            token=token,
        )

    if not args.skip_dinov2 and not angles_only:
        manifest["dinov2"] = download_snapshot(
            repo_id=args.dinov2_repo,
            destination=models_dir / "dinov2-base",
            token=token,
        )

    if not args.skip_unimatch and not angles_only:
        manifest["unimatch"] = download_file(
            repo_id=args.unimatch_repo_id,
            repo_type=args.unimatch_repo_type,
            filename=args.unimatch_filename,
            destination_root=Path(args.unimatch_output_root),
            token=token,
        )

    manifest_path = models_dir / "progressive_pose_models_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n[download] Done. Manifest:")
    print(json.dumps(manifest, indent=2))
    print(f"\n[download] Wrote {manifest_path}")
    print("\nRun progressive pose editing with defaults (no extra model args needed):")
    print("  python inference/progressive_pose_edit.py \\")
    print("    --source_image path/to/source.png \\")
    print("    --target_image path/to/target.png")


if __name__ == "__main__":
    main()
