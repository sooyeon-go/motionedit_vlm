#!/usr/bin/env python3
"""Download the strongest official local Grounding DINO checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_paths import GROUNDING_DINO_MODEL, GROUNDING_DINO_REPO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Grounding DINO Base for local text-conditioned detection."
    )
    parser.add_argument("--repo_id", default=GROUNDING_DINO_REPO)
    parser.add_argument("--output_dir", type=Path, default=GROUNDING_DINO_MODEL)
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[grounding-dino] {args.repo_id} -> {args.output_dir}", flush=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.output_dir),
        token=args.hf_token,
    )
    print(f"[grounding-dino] ready: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
