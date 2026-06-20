"""SPair-71k pair enumeration and image path resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


SPLITS = ("test", "val", "trn")
CANONICAL_SPLIT_ORDER = SPLITS


@dataclass(frozen=True)
class SpairPair:
    split: str
    pair_id: int
    filename: str
    category: str
    src_imname: str
    trg_imname: str
    annotation_path: Path
    src_image_path: Path
    trg_image_path: Path

    @property
    def output_name(self) -> str:
        return self.filename


def parse_category_from_filename(path: Path) -> Optional[str]:
    stem = path.stem
    if ":" not in stem:
        return None
    return stem.rsplit(":", 1)[-1]


def resolve_image_path(dataset_root: Path, category: str, imname: str) -> Path:
    return dataset_root / "JPEGImages" / category / imname


def load_spair_pair(annotation_path: Path, dataset_root: Path) -> SpairPair:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    split = annotation_path.parent.name
    category = (
        data.get("category")
        or data.get("class")
        or parse_category_from_filename(annotation_path)
    )
    if not category:
        raise ValueError(f"Could not resolve category for {annotation_path}")

    src_imname = data.get("src_imname") or data.get("src_image")
    trg_imname = data.get("trg_imname") or data.get("trg_image")
    if not src_imname or not trg_imname:
        raise ValueError(f"Missing src/trg image names in {annotation_path}")

    filename = data.get("filename") or annotation_path.stem
    pair_id = int(data.get("pair_id", -1))

    src_image_path = resolve_image_path(dataset_root, category, src_imname)
    trg_image_path = resolve_image_path(dataset_root, category, trg_imname)

    return SpairPair(
        split=split,
        pair_id=pair_id,
        filename=filename,
        category=category,
        src_imname=src_imname,
        trg_imname=trg_imname,
        annotation_path=annotation_path,
        src_image_path=src_image_path,
        trg_image_path=trg_image_path,
    )


def normalize_splits(splits: Iterable[str]) -> tuple[str, ...]:
    """Return splits in canonical order: test -> val -> trn."""
    selected = {split.strip() for split in splits if split.strip()}
    unknown = selected.difference(CANONICAL_SPLIT_ORDER)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}. Expected subset of {CANONICAL_SPLIT_ORDER}")
    return tuple(split for split in CANONICAL_SPLIT_ORDER if split in selected)


def iter_pair_annotation_files(
    pair_annotation_dir: Path,
    splits: Iterable[str] = SPLITS,
) -> Iterator[Path]:
    for split in normalize_splits(splits):
        split_dir = pair_annotation_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"PairAnnotation split not found: {split_dir}")
        yield from sorted(split_dir.glob("*.json"))


def shard_items(items: list[SpairPair], worker_id: int, num_workers: int) -> list[SpairPair]:
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError(f"worker_id must be in [0, {num_workers - 1}], got {worker_id}")
    return [item for index, item in enumerate(items) if index % num_workers == worker_id]
