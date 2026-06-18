"""Download models needed by inference/progressive_pose_edit.py from Hugging Face.

The progressive pipeline uses:
  - Qwen/Qwen-Image-Edit-2509 as the editor base model
  - elaine1wan/motionedit as the MotionEdit LoRA adapter
  - Qwen/Qwen3-VL-8B-Instruct as the planner/verifier VLM
  - facebook/dinov2-base as the source identity scorer
  - haofeixu/unimatch Space checkpoint for optical flow

Example:
  python tools/download_progressive_pose_models.py

If a gated model requires auth:
  HF_TOKEN=hf_xxx python tools/download_progressive_pose_models.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
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
        default=str(REPO_ROOT / "models" / "hf"),
        help="Local directory for large HF snapshots.",
    )
    parser.add_argument("--hf_token", default=None, help="Optional Hugging Face token.")

    parser.add_argument("--editor_base_repo", default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--motionedit_lora_repo", default="elaine1wan/motionedit")
    parser.add_argument("--planner_vlm_repo", default="Qwen/Qwen3-VL-8B-Instruct")
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
        default=str(REPO_ROOT / "train" / "scripts" / "unimatch"),
        help=(
            "Download root for UniMatch. With the default filename, the final file "
            "lands in train/scripts/unimatch/pretrained/."
        ),
    )

    parser.add_argument("--skip_editor_base", action="store_true")
    parser.add_argument("--skip_motionedit_lora", action="store_true")
    parser.add_argument("--skip_planner_vlm", action="store_true")
    parser.add_argument("--skip_dinov2", action="store_true")
    parser.add_argument("--skip_unimatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = _token(args.hf_token)
    models_dir = Path(args.models_dir)
    manifest: dict[str, str] = {}

    if not args.skip_editor_base:
        manifest["editor_base"] = download_snapshot(
            repo_id=args.editor_base_repo,
            destination=models_dir / "Qwen-Image-Edit-2509",
            token=token,
        )

    if not args.skip_motionedit_lora:
        lora_dir = models_dir / "motionedit-lora"
        # Only the adapter is needed for inference. The HF repo also contains
        # optimizer/scaler training artifacts that are not required here.
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

    if not args.skip_planner_vlm:
        manifest["planner_vlm"] = download_snapshot(
            repo_id=args.planner_vlm_repo,
            destination=models_dir / "Qwen3-VL-8B-Instruct",
            token=token,
        )

    if not args.skip_dinov2:
        manifest["dinov2"] = download_snapshot(
            repo_id=args.dinov2_repo,
            destination=models_dir / "dinov2-base",
            token=token,
        )

    if not args.skip_unimatch:
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
    print("\nExample local run arguments:")
    print(f"  --editor_base_model {models_dir / 'Qwen-Image-Edit-2509'}")
    print(f"  --motionedit_lora_path {models_dir / 'motionedit-lora'}")
    print(f"  --planner_vlm {models_dir / 'Qwen3-VL-8B-Instruct'}")
    print(f"  --dinov2_model {models_dir / 'dinov2-base'}")
    print(
        "  --unimatch_ckpt "
        f"{Path(args.unimatch_output_root) / args.unimatch_filename}"
    )


if __name__ == "__main__":
    main()
