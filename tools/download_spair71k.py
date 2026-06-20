#!/usr/bin/env python3
"""Download and extract the SPair-71k semantic correspondence dataset.

Official page: https://cvlab.postech.ac.kr/research/SPair-71k/
Archive:       https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
Size:          ~227 MB

After extraction, the dataset root contains:
  JPEGImages/      - 1800 images (18 categories)
  ImageAnnotation/ - image-level JSON annotations
  Segmentation/    - segmentation PNG masks
  PairAnnotation/  - trn/, val/, test/ pair JSON annotations
  Layout/          - train/val/test splits
  devkit/          - evaluation / visualization scripts
  Visualization/   - output folder for devkit scripts

Example:
  python tools/download_spair71k.py
  python tools/download_spair71k.py --output_dir /data/datasets/spair-71k
  python tools/download_spair71k.py --split val --list_pairs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

SPAIR71K_URL = "https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"
SPAIR71K_PAGE = "https://cvlab.postech.ac.kr/research/SPair-71k/"

EXPECTED_TOP_LEVEL = (
    "JPEGImages",
    "ImageAnnotation",
    "Segmentation",
    "PairAnnotation",
    "Layout",
    "devkit",
    "Visualization",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        log(f"[download] Archive already exists: {destination}")
        return

    log(f"[download] Fetching {url}")
    log(f"[download] Saving to {destination}")

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r[download] {mb_done:.1f}/{mb_total:.1f} MB ({pct:.1f}%)", end="", flush=True)

    tmp_path = destination.with_suffix(destination.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_report)
        print(flush=True)
        tmp_path.replace(destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    log(f"[download] Done: {destination}")


def extract_archive(archive_path: Path, output_dir: Path, force: bool = False) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force and (output_dir / "JPEGImages").is_dir():
            log(f"[extract] Dataset already present at {output_dir}")
            return output_dir
        if force:
            log(f"[extract] Removing existing directory: {output_dir}")
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[extract] Extracting {archive_path} -> {output_dir}")

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=output_dir)

    # Some archives contain a single top-level folder; normalize to output_dir root.
    root = resolve_dataset_root(output_dir)
    return root


def resolve_dataset_root(output_dir: Path) -> Path:
    if (output_dir / "JPEGImages").is_dir():
        return output_dir

    children = [p for p in output_dir.iterdir() if p.is_dir()]
    for child in children:
        if (child / "JPEGImages").is_dir():
            log(f"[extract] Found nested dataset root: {child}")
            return child

    return output_dir


def validate_dataset_root(root: Path) -> list[str]:
    missing = [name for name in EXPECTED_TOP_LEVEL if not (root / name).exists()]
    return missing


def list_pair_examples(root: Path, split: str, limit: int = 5) -> None:
    pair_dir = root / "PairAnnotation" / split
    if not pair_dir.is_dir():
        raise FileNotFoundError(f"PairAnnotation split not found: {pair_dir}")

    json_files = sorted(pair_dir.glob("*.json"))
    log(f"[pairs] {split}: {len(json_files)} annotation files")
    for path in json_files[:limit]:
        data = json.loads(path.read_text(encoding="utf-8"))
        src = data.get("src_imname") or data.get("src_image") or "?"
        tgt = data.get("trg_imname") or data.get("trg_image") or "?"
        category = data.get("category") or data.get("class") or "?"
        log(f"  - {path.name}: category={category}, src={src}, trg={tgt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract SPair-71k from the official POSTECH page.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/spair-71k"),
        help="Directory where the extracted dataset will live.",
    )
    parser.add_argument(
        "--archive_path",
        type=Path,
        default=None,
        help="Where to store the downloaded .tar.gz (default: <output_dir>/SPair-71k.tar.gz).",
    )
    parser.add_argument(
        "--url",
        default=SPAIR71K_URL,
        help="Download URL for SPair-71k.tar.gz.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download; only extract/validate an existing archive.",
    )
    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="Download only; do not extract.",
    )
    parser.add_argument(
        "--force_extract",
        action="store_true",
        help="Remove existing output_dir contents before extracting.",
    )
    parser.add_argument(
        "--list_pairs",
        action="store_true",
        help="Print a few pair annotation examples after setup.",
    )
    parser.add_argument(
        "--split",
        choices=("trn", "val", "test"),
        default="val",
        help="PairAnnotation split to inspect with --list_pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    archive_path = (args.archive_path or output_dir / "SPair-71k.tar.gz").expanduser().resolve()

    log("========== SPair-71k Download ==========")
    log(f"page   : {SPAIR71K_PAGE}")
    log(f"url    : {args.url}")
    log(f"output : {output_dir}")

    if not args.skip_download:
        download_file(args.url, archive_path)
    elif not archive_path.exists() and not (output_dir / "JPEGImages").is_dir():
        raise FileNotFoundError(
            f"Archive not found at {archive_path}. Run without --skip_download."
        )

    root = output_dir
    if not args.skip_extract:
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive missing: {archive_path}")
        root = extract_archive(archive_path, output_dir, force=args.force_extract)

    missing = validate_dataset_root(root)
    if missing:
        raise RuntimeError(
            "Dataset extraction looks incomplete. Missing top-level folders:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    log("[validate] Dataset layout OK")
    log(f"[validate] root = {root}")
    log(f"[validate] images = {root / 'JPEGImages'}")
    log(f"[validate] pairs  = {root / 'PairAnnotation'}")

    if args.list_pairs:
        list_pair_examples(root, split=args.split)

    manifest = {
        "dataset": "SPair-71k",
        "page": SPAIR71K_PAGE,
        "url": args.url,
        "archive_path": str(archive_path),
        "root": str(root),
        "splits": {
            split: len(list((root / "PairAnnotation" / split).glob("*.json")))
            for split in ("trn", "val", "test")
        },
    }
    manifest_path = root / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[done] Wrote {manifest_path}")
    log("[done] SPair-71k is ready.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[download] Interrupted.", file=sys.stderr)
        sys.exit(130)
