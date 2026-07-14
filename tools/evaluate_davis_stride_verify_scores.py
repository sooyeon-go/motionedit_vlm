#!/usr/bin/env python3
"""Measure whole-trajectory VLM verify scores on DAVIS sequences at multiple strides.

Reads every sequence folder under:
  /data/shared-vilab/datasets/DAVIS/JPEGImages/Full-Resolution/<sequence>/*.jpg

For each sequence and stride S, builds trajectory windows of keyframes spaced
exactly S frames apart and asks the same trajectory VLM verifier used by
inference/progressive_pose_edit.py to judge the whole sequence at once.

Multi-GPU:
  One worker process is launched per GPU. Trajectory jobs are sharded round-robin
  across workers; each worker loads its own VLM and writes a shard JSONL. The
  parent process merges shards into the final summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
DEFAULT_GPUS = "0,1,2,3,4,5,6,7"


def parse_strides(raw: str) -> list[int]:
    strides = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not strides or any(stride <= 0 for stride in strides):
        raise ValueError("--strides must contain positive integers")
    return strides


def parse_sequences(raw: str | None) -> set[str] | None:
    if raw is None or not raw.strip():
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def parse_gpu_ids(raw: str | None) -> list[int]:
    """Parse GPU ids. Empty/None falls back to DEFAULT_GPUS (0-7)."""
    text = DEFAULT_GPUS if raw is None or not str(raw).strip() else str(raw)
    ids = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not ids or any(gpu_id < 0 for gpu_id in ids):
        raise ValueError("--gpus must contain non-negative integers")
    return ids


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
    if not sequence_dirs:
        raise FileNotFoundError(f"No sequence folders found under: {frames_root}")
    return sequence_dirs


def list_frames(sequence_dir: Path) -> list[Path]:
    frames = sorted(sequence_dir.glob("*.jpg")) + sorted(sequence_dir.glob("*.jpeg"))
    # Prefer *.jpg; avoid double-counting if both patterns somehow match.
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in frames:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


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
    """Build trajectory windows of `window` keyframes spaced `stride` frames apart."""
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


def plan_jobs(
    sequence_dirs: list[Path],
    strides: list[int],
    trajectory_window: int,
    max_trajectories_per_sequence_stride: int,
    max_trajectories_per_stride: int,
) -> list[dict[str, Any]]:
    """Flatten all trajectory windows into a job list for sharding across GPUs."""
    by_stride_count: dict[int, int] = {stride: 0 for stride in strides}
    jobs: list[dict[str, Any]] = []
    for sequence_dir in sequence_dirs:
        frames = list_frames(sequence_dir)
        if len(frames) < 2:
            print(
                f"[davis-eval] skip {sequence_dir.name}: fewer than 2 jpg frames",
                flush=True,
            )
            continue
        for stride in strides:
            if (
                max_trajectories_per_stride > 0
                and by_stride_count[stride] >= max_trajectories_per_stride
            ):
                continue
            windows = build_trajectory_windows(
                frames,
                stride=stride,
                window=trajectory_window,
                max_windows=max_trajectories_per_sequence_stride,
            )
            for frame_paths in windows:
                if (
                    max_trajectories_per_stride > 0
                    and by_stride_count[stride] >= max_trajectories_per_stride
                ):
                    break
                jobs.append(
                    {
                        "sequence": sequence_dir.name,
                        "stride": stride,
                        "frame_paths": [str(path) for path in frame_paths],
                    }
                )
                by_stride_count[stride] += 1
    return jobs


def shard_jobs(jobs: list[dict[str, Any]], worker_id: int, num_workers: int) -> list[dict[str, Any]]:
    if num_workers <= 1:
        return jobs
    return [job for idx, job in enumerate(jobs) if idx % num_workers == worker_id]


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


def run_worker(
    jobs: list[dict[str, Any]],
    *,
    planner_vlm_path: str,
    vlm_device_map: str,
    jsonl_path: Path,
    quiet: bool,
    worker_id: int,
    gpu_id: int | None,
) -> list[dict[str, Any]]:
    print(
        f"[davis-eval][worker {worker_id}] gpu={gpu_id} jobs={len(jobs)} "
        f"device_map={vlm_device_map}",
        flush=True,
    )
    if not jobs:
        jsonl_path.write_text("", encoding="utf-8")
        return []

    print(f"[davis-eval][worker {worker_id}] loading planner VLM: {planner_vlm_path}", flush=True)
    planner_vlm = ppe.QwenVLMClient(
        model_id=planner_vlm_path,
        device_map=vlm_device_map,
        torch_dtype="auto",
    )

    records: list[dict[str, Any]] = []
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        iterator = tqdm(
            jobs,
            desc=f"gpu={gpu_id if gpu_id is not None else 'local'}",
            unit="traj",
            disable=quiet,
        )
        for job in iterator:
            frame_paths = [Path(path) for path in job["frame_paths"]]
            sequence_name = job["sequence"]
            stride = int(job["stride"])
            try:
                record = evaluate_trajectory(frame_paths, stride, planner_vlm)
                record["sequence"] = sequence_name
                record["status"] = "ok"
                record["worker_id"] = worker_id
                if gpu_id is not None:
                    record["gpu_id"] = gpu_id
            except Exception as exc:
                record = {
                    "status": "error",
                    "sequence": sequence_name,
                    "stride": stride,
                    "frame_names": [path.name for path in frame_paths],
                    "error": str(exc),
                    "worker_id": worker_id,
                }
                if gpu_id is not None:
                    record["gpu_id"] = gpu_id
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
    return records


def write_summary(
    *,
    frames_root: Path,
    strides: list[int],
    trajectory_window: int,
    planner_vlm: str,
    jsonl_path: Path,
    output_path: Path,
    started: float,
    num_sequences: int,
    all_records: list[dict[str, Any]],
    gpu_ids: list[int],
) -> dict[str, Any]:
    ok_records = [record for record in all_records if record.get("status") == "ok"]
    by_stride_records: dict[int, list[dict[str, Any]]] = {stride: [] for stride in strides}
    by_sequence_records: dict[str, list[dict[str, Any]]] = {}
    for record in ok_records:
        stride = int(record["stride"])
        if stride in by_stride_records:
            by_stride_records[stride].append(record)
        sequence_name = str(record.get("sequence", "unknown"))
        by_sequence_records.setdefault(sequence_name, []).append(record)

    by_sequence: dict[str, Any] = {}
    for sequence_name in sorted(by_sequence_records):
        seq_records = by_sequence_records[sequence_name]
        seq_by_stride: dict[int, list[dict[str, Any]]] = {stride: [] for stride in strides}
        for record in seq_records:
            stride = int(record["stride"])
            if stride in seq_by_stride:
                seq_by_stride[stride].append(record)
        by_sequence[sequence_name] = {
            **aggregate_records(seq_records),
            "by_stride": {
                str(stride): aggregate_records(records)
                for stride, records in seq_by_stride.items()
                if records
            },
        }

    payload = {
        "frames_root": str(frames_root),
        "strides": strides,
        "trajectory_window": trajectory_window,
        "planner_vlm": planner_vlm,
        "gpu_ids": gpu_ids,
        "jsonl_path": str(jsonl_path),
        "elapsed_sec": round(time.time() - started, 2),
        "num_sequences": num_sequences,
        "num_records": len(all_records),
        "num_ok": len(ok_records),
        "num_errors": len(all_records) - len(ok_records),
        "overall": aggregate_records(ok_records),
        "by_stride": {
            str(stride): aggregate_records(records)
            for stride, records in by_stride_records.items()
        },
        "by_sequence": by_sequence,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def merge_jsonl_files(shard_paths: list[Path], out_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_handle:
        for shard_path in shard_paths:
            if not shard_path.is_file():
                continue
            with shard_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    records.append(record)
                    out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def launch_multi_gpu(args: argparse.Namespace, jobs: list[dict[str, Any]], gpu_ids: list[int]) -> int:
    """Spawn one subprocess per GPU, then merge shard outputs."""
    output_path = args.output_path.expanduser().resolve()
    jsonl_path = (
        args.jsonl_path.expanduser().resolve()
        if args.jsonl_path is not None
        else output_path.with_suffix(".jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    jobs_path = output_path.with_suffix(".jobs.json")
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")

    started = time.time()
    num_workers = len(gpu_ids)
    shard_paths = [
        jsonl_path.with_name(f"{jsonl_path.stem}.gpu{gpu_id}.jsonl") for gpu_id in gpu_ids
    ]
    script_path = Path(__file__).resolve()
    procs: list[subprocess.Popen[str]] = []

    print(
        f"[davis-eval] launching {num_workers} workers on GPUs {gpu_ids} "
        f"({len(jobs)} trajectory jobs)",
        flush=True,
    )
    for worker_id, (gpu_id, shard_path) in enumerate(zip(gpu_ids, shard_paths)):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable,
            str(script_path),
            "--frames_root",
            str(args.frames_root),
            "--output_path",
            str(output_path),
            "--planner_vlm",
            str(args.planner_vlm),
            "--vlm_device_map",
            "cuda:0",
            "--strides",
            args.strides,
            "--trajectory_window",
            str(args.trajectory_window),
            "--max_trajectories_per_sequence_stride",
            str(args.max_trajectories_per_sequence_stride),
            "--max_trajectories_per_stride",
            str(args.max_trajectories_per_stride),
            "--jsonl_path",
            str(shard_path),
            "--worker_id",
            str(worker_id),
            "--num_workers",
            str(num_workers),
            "--jobs_path",
            str(jobs_path),
            "--gpu_id",
            str(gpu_id),
        ]
        if args.sequences:
            cmd.extend(["--sequences", args.sequences])
        if args.quiet:
            cmd.append("--quiet")
        print(f"[davis-eval] start worker {worker_id} on GPU {gpu_id}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))

    exit_codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more workers failed: exit_codes={exit_codes}")

    all_records = merge_jsonl_files(shard_paths, jsonl_path)
    for shard_path in shard_paths:
        try:
            shard_path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        jobs_path.unlink(missing_ok=True)
    except OSError:
        pass

    frames_root = args.frames_root.expanduser().resolve()
    sequence_dirs = list_sequence_dirs(frames_root, parse_sequences(args.sequences))
    payload = write_summary(
        frames_root=frames_root,
        strides=parse_strides(args.strides),
        trajectory_window=args.trajectory_window,
        planner_vlm=str(args.planner_vlm),
        jsonl_path=jsonl_path,
        output_path=output_path,
        started=started,
        num_sequences=len(sequence_dirs),
        all_records=all_records,
        gpu_ids=gpu_ids,
    )
    print(f"[davis-eval] wrote summary: {output_path}", flush=True)
    print(f"[davis-eval] wrote trajectory records: {jsonl_path}", flush=True)
    print(f"[davis-eval] overall mean: {payload['overall']['mean_scores']}", flush=True)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whole-trajectory VLM verify scores on all DAVIS sequences "
            "under JPEGImages/Full-Resolution, with optional multi-GPU sharding."
        ),
    )
    parser.add_argument(
        "--frames_root",
        type=Path,
        default=DEFAULT_DAVIS_ROOT,
        help="DAVIS JPEGImages/Full-Resolution root (one folder per sequence, jpg frames inside).",
    )
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--planner_vlm", default=str(PLANNER_VLM_MODEL))
    parser.add_argument(
        "--vlm_device_map",
        default="cuda:0",
        help="device_map for QwenVLMClient. Multi-GPU launcher overrides this to cuda:0 per worker.",
    )
    parser.add_argument("--strides", default="1,2,4,8,16")
    parser.add_argument(
        "--sequences",
        default=None,
        help="Optional comma-separated sequence names. Default: ALL sequence folders.",
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
    parser.add_argument(
        "--gpus",
        default=DEFAULT_GPUS,
        help="Comma-separated GPU ids. Default: 0,1,2,3,4,5,6,7 (8 GPUs).",
    )
    parser.add_argument("--jsonl_path", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # Internal worker flags (set by multi-GPU launcher).
    parser.add_argument("--worker_id", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--jobs_path", type=Path, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.trajectory_window < 2:
        raise ValueError("--trajectory_window must be >= 2")
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")

    ppe.VERBOSE = not args.quiet
    frames_root = args.frames_root.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    jsonl_path = (
        args.jsonl_path.expanduser().resolve()
        if args.jsonl_path is not None
        else output_path.with_suffix(".jsonl")
    )
    strides = parse_strides(args.strides)

    # Worker mode: already-launched child process with CUDA_VISIBLE_DEVICES set.
    if args.worker_id is not None:
        if args.jobs_path is None:
            raise ValueError("--jobs_path is required in worker mode")
        jobs = json.loads(args.jobs_path.read_text(encoding="utf-8"))
        jobs = shard_jobs(jobs, args.worker_id, args.num_workers)
        run_worker(
            jobs,
            planner_vlm_path=str(args.planner_vlm),
            vlm_device_map=args.vlm_device_map,
            jsonl_path=jsonl_path,
            quiet=args.quiet,
            worker_id=args.worker_id,
            gpu_id=args.gpu_id,
        )
        return

    sequence_dirs = list_sequence_dirs(frames_root, parse_sequences(args.sequences))
    jobs = plan_jobs(
        sequence_dirs,
        strides=strides,
        trajectory_window=args.trajectory_window,
        max_trajectories_per_sequence_stride=args.max_trajectories_per_sequence_stride,
        max_trajectories_per_stride=args.max_trajectories_per_stride,
    )
    plan_counts: dict[str, int] = {str(stride): 0 for stride in strides}
    for job in jobs:
        plan_counts[str(job["stride"])] += 1

    gpu_ids = parse_gpu_ids(args.gpus)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "frames_root": str(frames_root),
                    "num_sequences": len(sequence_dirs),
                    "sequence_names": [path.name for path in sequence_dirs],
                    "trajectory_window": args.trajectory_window,
                    "planned_trajectories": plan_counts,
                    "num_jobs": len(jobs),
                    "gpus": gpu_ids,
                },
                indent=2,
            )
        )
        return

    # Multi-GPU parent: spawn one process per GPU.
    if len(gpu_ids) > 1:
        launch_multi_gpu(args, jobs, gpu_ids)
        return

    # Single-GPU / CPU path (torch is already imported; select device via device_map).
    gpu_id = gpu_ids[0] if gpu_ids else None
    if gpu_id is not None:
        vlm_device_map = f"cuda:{gpu_id}"
    else:
        vlm_device_map = args.vlm_device_map

    started = time.time()
    print(f"[davis-eval] frames_root={frames_root}", flush=True)
    print(
        f"[davis-eval] sequences={len(sequence_dirs)}, "
        f"trajectory_window={args.trajectory_window}, planned_trajectories={plan_counts}",
        flush=True,
    )
    print(f"[davis-eval] gpu_ids={gpu_ids}", flush=True)

    all_records = run_worker(
        jobs,
        planner_vlm_path=str(args.planner_vlm),
        vlm_device_map=vlm_device_map,
        jsonl_path=jsonl_path,
        quiet=args.quiet,
        worker_id=0,
        gpu_id=gpu_id,
    )
    payload = write_summary(
        frames_root=frames_root,
        strides=strides,
        trajectory_window=args.trajectory_window,
        planner_vlm=str(args.planner_vlm),
        jsonl_path=jsonl_path,
        output_path=output_path,
        started=started,
        num_sequences=len(sequence_dirs),
        all_records=all_records,
        gpu_ids=gpu_ids,
    )
    print(f"[davis-eval] wrote summary: {output_path}", flush=True)
    print(f"[davis-eval] wrote trajectory records: {jsonl_path}", flush=True)
    print(f"[davis-eval] overall mean: {payload['overall']['mean_scores']}", flush=True)


if __name__ == "__main__":
    main()
