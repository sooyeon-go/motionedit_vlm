#!/usr/bin/env python3
"""Measure whole-trajectory VLM verify scores on DAVIS sequences at multiple strides.

This script treats real video sequences as ground-truth interpolation trajectories.
For each sequence and stride S, it builds trajectory windows of keyframes spaced
exactly S frames apart (frame k, k+S, k+2S, ...) and asks the same trajectory VLM
verifier used by inference/progressive_pose_edit.py to judge the whole sequence at
once. Per-pair (single-transition) evaluation is intentionally not done here, because
judging a whole progressive trajectory is far more meaningful than scoring isolated
image-to-image transitions.

Larger strides = larger temporal gaps between keyframes = harder interpolation, so the
by-stride summary shows how trajectory quality degrades as the gap grows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = REPO_ROOT / "inference"
TOOLS_DIR = REPO_ROOT / "tools"
for path in (INFERENCE_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import progressive_pose_edit as ppe  # noqa: E402
from model_paths import PLANNER_VLM_MODEL  # noqa: E402


DEFAULT_DAVIS_ROOT = Path("/data/shared-vilab/datasets/DAVIS/JPEGImages/Full-Resolution")
DEFAULT_OUTPUT = Path("outputs/davis_stride_verify_scores/score_summary.json")


def parse_strides(raw: str) -> list[int]:
    strides = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not strides or any(stride <= 0 for stride in strides):
        raise ValueError("--strides must contain positive integers")
    return strides


def parse_sequences(raw: str | None) -> set[str] | None:
    if raw is None or not raw.strip():
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def list_sequence_dirs(frames_root: Path, selected: set[str] | None) -> list[Path]:
    if not frames_root.is_dir():
        raise FileNotFoundError(f"DAVIS frame root not found: {frames_root}")
    sequence_dirs = [
        path
        for path in sorted(frames_root.iterdir())
        if path.is_dir() and (selected is None or path.name in selected)
    ]
    if selected is not None:
        found = {path.name for path in sequence_dirs}
        missing = sorted(selected - found)
        if missing:
            raise FileNotFoundError(f"Requested sequence(s) not found: {', '.join(missing)}")
    return sequence_dirs


def list_frames(sequence_dir: Path) -> list[Path]:
    return sorted(sequence_dir.glob("*.jpg"))


def _uniform_sample(values: list[int], max_count: int) -> list[int]:
    if max_count <= 0 or len(values) <= max_count:
        return values
    if max_count == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    picked: list[int] = []
    for k in range(max_count):
        pos = round(last * k / (max_count - 1))
        if values[pos] not in picked:
            picked.append(values[pos])
    return picked


def build_trajectory_windows(
    frames: list[Path],
    stride: int,
    window: int,
    max_windows: int,
) -> list[list[Path]]:
    """Build trajectory windows of `window` keyframes spaced `stride` frames apart.

    Each returned window is a list of frame paths [f_k, f_{k+S}, f_{k+2S}, ...] with
    exactly `window` keyframes. If the sequence is too short to fit a full window at
    this stride, a single fallback trajectory of all stride-spaced keyframes (plus the
    last frame) is returned so short sequences are not silently dropped.
    """
    num_frames = len(frames)
    if num_frames < 2 or window < 2:
        return []

    span = (window - 1) * stride
    if num_frames <= span:
        indices = list(range(0, num_frames, stride))
        if indices[-1] != num_frames - 1:
            indices.append(num_frames - 1)
        if len(indices) < 2:
            return []
        return [[frames[idx] for idx in indices]]

    max_start = num_frames - 1 - span
    starts = _uniform_sample(list(range(0, max_start + 1)), max_windows)
    windows: list[list[Path]] = []
    for start in starts:
        indices = [start + step * stride for step in range(window)]
        windows.append([frames[idx] for idx in indices])
    return windows


def bool_rate(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if key in record]
    if not values:
        return None
    return round(sum(1 for value in values if bool(value)) / len(values), 4)


def trajectory_record_to_dict(result: ppe.TrajectoryVLMVerifyResult) -> dict[str, Any]:
    return {
        "progressive_toward_target": result.progressive_toward_target,
        "abrupt_jumps": result.abrupt_jumps,
        "scores": result.scores,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    score_dicts = [
        record.get("scores", {})
        for record in records
        if isinstance(record.get("scores", {}), dict) and record.get("scores", {})
    ]
    return {
        "num_trajectories": len(records),
        "num_scored": len(score_dicts),
        "mean_scores": ppe.average_score_dicts(score_dicts),
        "yes_rates": {
            "progressive_toward_target": bool_rate(records, "progressive_toward_target"),
            "abrupt_jumps_present": bool_rate(records, "abrupt_jumps"),
        },
    }


def evaluate_trajectory(
    frame_paths: list[Path],
    stride: int,
    planner_vlm: ppe.QwenVLMClient,
) -> dict[str, Any]:
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    result = ppe.vlm_verify_trajectory(
        trajectory=images,
        target_img=images[-1],
        planner_vlm=planner_vlm,
        shared_parts=[],
    )
    payload = trajectory_record_to_dict(result)
    payload.update(
        {
            "stride": stride,
            "num_keyframes": len(frame_paths),
            "start_frame": frame_paths[0].name,
            "end_frame": frame_paths[-1].name,
            "frame_names": [path.name for path in frame_paths],
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whole-trajectory VLM verify scores on DAVIS sequences by stride.",
    )
    parser.add_argument("--frames_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--planner_vlm", default=str(PLANNER_VLM_MODEL))
    parser.add_argument("--vlm_device_map", default="auto")
    parser.add_argument("--strides", default="1,2,4,8,16")
    parser.add_argument(
        "--sequences",
        default=None,
        help="Optional comma-separated sequence names. Default: all sequences.",
    )
    parser.add_argument(
        "--trajectory_window",
        type=int,
        default=6,
        help="Number of keyframes per trajectory window (spaced `stride` frames apart).",
    )
    parser.add_argument(
        "--max_trajectories_per_sequence_stride",
        type=int,
        default=3,
        help="Uniformly sample at most this many trajectory windows per sequence per stride; 0 means all.",
    )
    parser.add_argument(
        "--max_trajectories_per_stride",
        type=int,
        default=0,
        help="Stop after this many total trajectories per stride; 0 means all.",
    )
    parser.add_argument("--jsonl_path", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.trajectory_window < 2:
        raise ValueError("--trajectory_window must be >= 2")

    ppe.VERBOSE = not args.quiet
    frames_root = args.frames_root.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    jsonl_path = (
        args.jsonl_path.expanduser().resolve()
        if args.jsonl_path is not None
        else output_path.with_suffix(".jsonl")
    )
    strides = parse_strides(args.strides)
    sequence_dirs = list_sequence_dirs(frames_root, parse_sequences(args.sequences))

    planned: dict[int, list[tuple[list[Path], str]]] = {stride: [] for stride in strides}
    for sequence_dir in sequence_dirs:
        frames = list_frames(sequence_dir)
        for stride in strides:
            windows = build_trajectory_windows(
                frames,
                stride=stride,
                window=args.trajectory_window,
                max_windows=args.max_trajectories_per_sequence_stride,
            )
            for frame_paths in windows:
                if (
                    args.max_trajectories_per_stride > 0
                    and len(planned[stride]) >= args.max_trajectories_per_stride
                ):
                    break
                planned[stride].append((frame_paths, sequence_dir.name))

    plan_counts = {str(stride): len(items) for stride, items in planned.items()}
    if args.dry_run:
        print(
            json.dumps(
                {
                    "frames_root": str(frames_root),
                    "trajectory_window": args.trajectory_window,
                    "planned_trajectories": plan_counts,
                },
                indent=2,
            )
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"[davis-eval] frames_root={frames_root}", flush=True)
    print(
        f"[davis-eval] sequences={len(sequence_dirs)}, "
        f"trajectory_window={args.trajectory_window}, planned_trajectories={plan_counts}",
        flush=True,
    )
    print(f"[davis-eval] loading planner VLM: {args.planner_vlm}", flush=True)
    planner_vlm = ppe.QwenVLMClient(
        model_id=args.planner_vlm,
        device_map=args.vlm_device_map,
        torch_dtype="auto",
    )

    all_records: list[dict[str, Any]] = []
    by_stride_records: dict[int, list[dict[str, Any]]] = {stride: [] for stride in strides}
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for stride in strides:
            items = planned[stride]
            iterator = tqdm(items, desc=f"stride={stride}", unit="traj", disable=args.quiet)
            for frame_paths, sequence_name in iterator:
                try:
                    record = evaluate_trajectory(frame_paths, stride, planner_vlm)
                    record["sequence"] = sequence_name
                    record["status"] = "ok"
                except Exception as exc:
                    record = {
                        "status": "error",
                        "sequence": sequence_name,
                        "stride": stride,
                        "frame_names": [path.name for path in frame_paths],
                        "error": str(exc),
                    }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                all_records.append(record)
                if record.get("status") == "ok":
                    by_stride_records[stride].append(record)

    ok_records = [record for record in all_records if record.get("status") == "ok"]
    payload = {
        "frames_root": str(frames_root),
        "strides": strides,
        "trajectory_window": args.trajectory_window,
        "planner_vlm": args.planner_vlm,
        "jsonl_path": str(jsonl_path),
        "elapsed_sec": round(time.time() - started, 2),
        "num_sequences": len(sequence_dirs),
        "num_records": len(all_records),
        "num_ok": len(ok_records),
        "num_errors": len(all_records) - len(ok_records),
        "overall": aggregate_records(ok_records),
        "by_stride": {
            str(stride): aggregate_records(records)
            for stride, records in by_stride_records.items()
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[davis-eval] wrote summary: {output_path}", flush=True)
    print(f"[davis-eval] wrote trajectory records: {jsonl_path}", flush=True)
    print(f"[davis-eval] overall mean: {payload['overall']['mean_scores']}", flush=True)


if __name__ == "__main__":
    main()
