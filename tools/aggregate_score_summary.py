#!/usr/bin/env python3
"""Aggregate VLM score summaries from progressive pose edit outputs.

This scans result.json files under an output root and computes dataset-level
means for the 0-5 VLM score keys. It supports both normal outputs and
COMPARE_ANGLE_LORA=1 outputs with angle_lora/ and base_qwen_angle/ variants.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCORE_KEYS = (
    "interpolation_quality",
    "target_geometry_match",
    "source_identity_preservation",
    "overall_quality",
)

COMPARISON_VARIANTS = {"angle_lora", "base_qwen_angle"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate score_summary fields from result.json files.",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Output root to scan, e.g. outputs/angle_lora_ablation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to write the aggregate JSON. Default: "
            "<output_root>/logs/score_summary_from_results.json"
        ),
    )
    parser.add_argument(
        "--variant",
        default="all",
        choices=["all", "main", "angle_lora", "base_qwen_angle"],
        help="Only aggregate one variant. Default aggregates all and also reports by_variant.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include result.json files even if sibling final.png is missing.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def score_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key in SCORE_KEYS:
        if key not in value:
            continue
        try:
            out[key] = float(value[key])
        except (TypeError, ValueError):
            pass
    return out


def average_score_dicts(score_dicts: list[dict[str, float]]) -> dict[str, float]:
    valid = [scores for scores in score_dicts if scores]
    if not valid:
        return {}

    averaged: dict[str, float] = {}
    for key in SCORE_KEYS:
        values = [float(scores[key]) for scores in valid if key in scores]
        if values:
            averaged[key] = round(sum(values) / len(values), 4)
    return averaged


def infer_variant(result_path: Path) -> str:
    parent_name = result_path.parent.name
    if parent_name in COMPARISON_VARIANTS:
        return parent_name
    return "main"


def is_complete_result(result_path: Path, include_incomplete: bool) -> bool:
    if include_incomplete:
        return True
    return (result_path.parent / "final.png").is_file()


def record_from_summary(
    result_path: Path,
    result_data: dict[str, Any],
    variant: str,
) -> dict[str, Any] | None:
    summary = result_data.get("score_summary", {})
    if not isinstance(summary, dict) or not summary:
        return None

    overall = score_dict(summary.get("overall_mean", {}))
    per_step = score_dict(summary.get("per_step_mean", {}))
    trajectory = score_dict(summary.get("trajectory_mean", summary.get("trajectory", {})))
    if not overall and not per_step and not trajectory:
        return None

    return {
        "result_path": str(result_path),
        "sample_dir": str(result_path.parent.parent if variant in COMPARISON_VARIANTS else result_path.parent),
        "variant": variant,
        "overall_mean": overall,
        "per_step_mean": per_step,
        "trajectory": trajectory,
        "num_scored_steps": int(summary.get("num_scored_steps", 0) or 0),
    }


def collect_records(output_root: Path, include_incomplete: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for result_path in sorted(output_root.rglob("result.json")):
        resolved = result_path.resolve()
        if resolved in seen:
            continue

        data = read_json(result_path)
        if data is None:
            continue

        variants = data.get("variants", {})
        if isinstance(variants, dict) and variants:
            # Comparison manifest: count variant result files, not the manifest itself.
            seen.add(resolved)
            for variant_name, variant_info in sorted(variants.items()):
                if not isinstance(variant_info, dict):
                    continue
                variant_result = result_path.parent / str(
                    variant_info.get("result", f"{variant_name}/result.json")
                )
                variant_result = variant_result.resolve()
                if variant_result in seen:
                    continue
                variant_data = read_json(variant_result)
                if variant_data is None:
                    continue
                if not is_complete_result(variant_result, include_incomplete):
                    continue
                record = record_from_summary(variant_result, variant_data, str(variant_name))
                if record is not None:
                    records.append(record)
                    seen.add(variant_result)
            continue

        if not is_complete_result(result_path, include_incomplete):
            continue
        variant = infer_variant(result_path)
        record = record_from_summary(result_path, data, variant)
        if record is not None:
            records.append(record)
            seen.add(resolved)

    return records


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scale": "0.0-5.0",
        "num_results": len(records),
        "num_samples_with_scores": sum(1 for record in records if record["overall_mean"]),
        "num_scored_steps_total": sum(int(record["num_scored_steps"]) for record in records),
        "overall_mean": average_score_dicts([record["overall_mean"] for record in records]),
        "per_step_mean": average_score_dicts([record["per_step_mean"] for record in records]),
        "trajectory_mean": average_score_dicts([record["trajectory"] for record in records]),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    records = collect_records(output_root, include_incomplete=args.include_incomplete)
    if args.variant != "all":
        records = [record for record in records if record["variant"] == args.variant]

    by_variant_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_variant_records[record["variant"]].append(record)

    payload = {
        "output_root": str(output_root),
        "variant_filter": args.variant,
        "summary": aggregate_records(records),
        "by_variant": {
            variant: aggregate_records(variant_records)
            for variant, variant_records in sorted(by_variant_records.items())
        },
        "records": records,
    }

    output_path = args.output
    if output_path is None:
        output_path = output_root / "logs" / "score_summary_from_results.json"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[aggregate] scanned: {output_root}")
    print(f"[aggregate] wrote: {output_path}")
    print(f"[aggregate] num_results: {payload['summary']['num_results']}")
    print(f"[aggregate] overall_mean: {payload['summary']['overall_mean']}")
    if payload["by_variant"]:
        print("[aggregate] by_variant:")
        for variant, summary in payload["by_variant"].items():
            print(f"  - {variant}: n={summary['num_results']} overall={summary['overall_mean']}")


if __name__ == "__main__":
    main()
