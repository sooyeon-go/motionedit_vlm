#!/usr/bin/env python3
"""Test the pre-align module's ability to recover a known random flip/rotation on DAVIS.

Experiment (per DAVIS sequence, per stride S):
  1. source = frame_0, target = frame_S  (video first frame vs the frame S steps later).
     Real video frames of the same object are assumed to be roughly orientation-aligned,
     so the "correct" pre-align of a perturbed source is to undo the perturbation.
  2. Apply a RANDOM coarse D4 transform (horizontal/vertical flip x {0,90,180,270}
     rotation, excluding identity) to the source. The applied perturbation is recorded.
  3. Run the existing pre-align pipeline (pre_align_source_until_verified, i.e. landmark
     -> VLM verify -> bruteforce fallback, optionally Orient Anything) to align the
     perturbed source toward the target.
  4. Check whether the pipeline recovered the original orientation. Recovery is judged
     objectively in the D4 group via a 2x2 color fingerprint: composing the perturbation
     with the transform the pipeline applied must return to identity.

Multi-GPU: one worker per GPU, jobs sharded round-robin, shards merged into a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
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
DEFAULT_OUTPUT = Path("outputs/davis_prealign_recovery/summary.json")
DEFAULT_GPUS = "0,1,2,3,4,5,6,7"


# ------------------------------------------------------------------
# DAVIS discovery
# ------------------------------------------------------------------
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
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in frames:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def target_index(num_frames: int, stride: int, target_mode: str) -> int:
    last = num_frames - 1
    if target_mode == "stride":
        return min(stride, last)
    if target_mode == "last":
        return last
    if target_mode == "last_strided":
        return (last // stride) * stride
    raise ValueError(f"Unknown target_mode: {target_mode}")


# ------------------------------------------------------------------
# D4 orientation group: perturbation + objective recovery check
# ------------------------------------------------------------------
def _canonical_fingerprint_image() -> Image.Image:
    """2x2 image with 4 distinct colors; each D4 element yields a distinct arrangement."""
    image = Image.new("RGB", (2, 2))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 255, 0))
    image.putpixel((0, 1), (0, 0, 255))
    image.putpixel((1, 1), (255, 255, 0))
    return image


CANONICAL_FP = _canonical_fingerprint_image()


def apply_d4(
    image: Image.Image,
    flip_horizontal: bool,
    flip_vertical: bool,
    rotation_degrees: float,
) -> Image.Image:
    """Apply a D4 transform using the same op order as apply_coarse_orientation_transform.

    Rotation is snapped to the nearest multiple of 90 (D4). PIL rotate is CCW; NEAREST
    keeps the 2x2 fingerprint exact.
    """
    out = image
    if flip_horizontal:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_vertical:
        out = out.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    snapped = (int(round(rotation_degrees / 90.0)) * 90) % 360
    if snapped:
        out = out.rotate(snapped, resample=Image.Resampling.NEAREST, expand=True)
    return out


def d4_fingerprint(flip_horizontal: bool, flip_vertical: bool, rotation_degrees: float) -> bytes:
    return apply_d4(CANONICAL_FP, flip_horizontal, flip_vertical, rotation_degrees).tobytes()


IDENTITY_FP = d4_fingerprint(False, False, 0)


def enumerate_d4_perturbations() -> list[tuple[bool, bool, int]]:
    """Return the 7 non-identity D4 elements as (flip_h, flip_v, rotation) params."""
    seen: dict[bytes, tuple[bool, bool, int]] = {}
    for flip_horizontal in (False, True):
        for flip_vertical in (False, True):
            for rotation in (0, 90, 180, 270):
                fingerprint = d4_fingerprint(flip_horizontal, flip_vertical, rotation)
                if fingerprint not in seen:
                    seen[fingerprint] = (flip_horizontal, flip_vertical, rotation)
    return [params for fingerprint, params in seen.items() if fingerprint != IDENTITY_FP]


D4_PERTURBATIONS = enumerate_d4_perturbations()


def rotation_residual_deg(rotation_degrees: float) -> float:
    """Distance from the nearest multiple of 90 degrees (in-plane tilt not in D4)."""
    return round(abs(rotation_degrees - round(rotation_degrees / 90.0) * 90.0), 3)


def is_orientation_recovered(
    perturbation: tuple[bool, bool, int],
    applied_flip_horizontal: bool,
    applied_flip_vertical: bool,
    applied_rotation_degrees: float,
) -> bool:
    """True if applying the pipeline's transform on top of the perturbation is identity."""
    perturbed = apply_d4(CANONICAL_FP, perturbation[0], perturbation[1], perturbation[2])
    restored = apply_d4(
        perturbed,
        applied_flip_horizontal,
        applied_flip_vertical,
        applied_rotation_degrees,
    )
    return restored.tobytes() == CANONICAL_FP.tobytes()


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------
def rate(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if key in record and record[key] is not None]
    if not values:
        return None
    return round(sum(1 for value in values if bool(value)) / len(values), 4)


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    mode_counts: dict[str, int] = {}
    for record in records:
        mode = str(record.get("prealign_mode", "unknown"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    return {
        "num_samples": len(records),
        "recovery_rate": rate(records, "recovered"),
        "verify_apply_rate": rate(records, "verify_apply"),
        "mode_counts": mode_counts,
    }


# ------------------------------------------------------------------
# Per-sample experiment
# ------------------------------------------------------------------
def sample_perturbation(sequence: str, stride: int, perturb_seed: int) -> tuple[bool, bool, int]:
    rng = random.Random(f"{sequence}|{stride}|{perturb_seed}")
    return rng.choice(D4_PERTURBATIONS)


def derive_prealign_mode(parsed: dict[str, Any]) -> str:
    if parsed.get("bruteforce_fallback"):
        return "bruteforce"
    if parsed.get("rolled_back"):
        return "rolled_back"
    if parsed.get("phase") == "orient_anything_pre_align":
        return "orient_anything"
    return "landmark"


def evaluate_sample(
    job: dict[str, Any],
    planner_vlm: ppe.QwenVLMClient,
    *,
    samples_root: Path,
    min_confidence: float,
    max_rotation: float,
    max_prealign_verify_attempts: int,
    prealign_bruteforce_after_attempts: int,
    perturb_seed: int,
    orient_anything: Any | None,
    orient_confidence_threshold: float,
    save_images: bool,
) -> dict[str, Any]:
    sequence = job["sequence"]
    stride = int(job["stride"])
    source_path = Path(job["source_path"])
    target_path = Path(job["target_path"])

    source_img = Image.open(source_path).convert("RGB")
    target_img = Image.open(target_path).convert("RGB")

    perturbation = sample_perturbation(sequence, stride, perturb_seed)
    flip_h, flip_v, rot = perturbation
    transform = ppe.CoarseOrientationTransform(
        candidate_id=-1,
        flip_horizontal=flip_h,
        flip_vertical=flip_v,
        rotation_degrees=rot,
    )
    perturbed_source = ppe.apply_coarse_orientation_transform(source_img, transform)

    sample_dir = samples_root / sequence / f"stride_{stride:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "perturbation.json").write_text(
        json.dumps(
            {
                "sequence": sequence,
                "stride": stride,
                "source_frame": source_path.name,
                "target_frame": target_path.name,
                "perturbation": {
                    "flip_horizontal": flip_h,
                    "flip_vertical": flip_v,
                    "rotation_degrees": rot,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    aligned, decision, parsed = ppe.pre_align_source_until_verified(
        source_img=perturbed_source,
        target_img=target_img,
        planner_vlm=planner_vlm,
        output_dir=sample_dir,
        min_confidence=min_confidence,
        max_rotation=max_rotation,
        max_attempts=max_prealign_verify_attempts,
        bruteforce_after_attempts=prealign_bruteforce_after_attempts,
        orient_anything=orient_anything,
        orient_confidence_threshold=orient_confidence_threshold,
    )

    recovered = is_orientation_recovered(
        perturbation,
        decision.applied_horizontal_flip,
        decision.applied_vertical_flip,
        decision.applied_rotation_degrees,
    )
    verify = parsed.get("verify") or {}
    verify_apply = bool(verify.get("overall_ok")) and str(verify.get("recommendation", "")).lower() == "apply"

    if save_images:
        source_img.save(sample_dir / "source_frame0.png")
        perturbed_source.save(sample_dir / "perturbed_source.png")
        aligned.save(sample_dir / "aligned.png")
        target_img.save(sample_dir / "target.png")

    return {
        "sequence": sequence,
        "stride": stride,
        "source_frame": source_path.name,
        "target_frame": target_path.name,
        "perturbation": {
            "flip_horizontal": flip_h,
            "flip_vertical": flip_v,
            "rotation_degrees": rot,
        },
        "applied": {
            "flip_horizontal": decision.applied_horizontal_flip,
            "flip_vertical": decision.applied_vertical_flip,
            "rotation_degrees": round(float(decision.applied_rotation_degrees), 3),
        },
        "applied_rotation_residual_deg": rotation_residual_deg(
            float(decision.applied_rotation_degrees)
        ),
        "recovered": recovered,
        "prealign_mode": derive_prealign_mode(parsed),
        "confidence": round(float(decision.confidence), 4),
        "verify": {
            "overall_ok": verify.get("overall_ok"),
            "recommendation": verify.get("recommendation"),
            "failure_reason": verify.get("failure_reason"),
        },
        "verify_apply": verify_apply,
        "sample_dir": str(sample_dir),
    }


# ------------------------------------------------------------------
# Planning + sharding
# ------------------------------------------------------------------
def plan_jobs(
    sequence_dirs: list[Path],
    strides: list[int],
    target_mode: str,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for sequence_dir in sequence_dirs:
        frames = list_frames(sequence_dir)
        if len(frames) < 2:
            print(f"[prealign-eval] skip {sequence_dir.name}: fewer than 2 jpg frames", flush=True)
            continue
        source_path = frames[0]
        for stride in strides:
            tgt_idx = target_index(len(frames), stride, target_mode)
            if tgt_idx <= 0:
                continue
            jobs.append(
                {
                    "sequence": sequence_dir.name,
                    "stride": stride,
                    "source_path": str(source_path),
                    "target_path": str(frames[tgt_idx]),
                    "target_index": tgt_idx,
                }
            )
    return jobs


def shard_jobs(jobs: list[dict[str, Any]], worker_id: int, num_workers: int) -> list[dict[str, Any]]:
    if num_workers <= 1:
        return jobs
    return [job for idx, job in enumerate(jobs) if idx % num_workers == worker_id]


def maybe_build_orient_anything(args: argparse.Namespace) -> Any | None:
    if not args.use_orient_anything:
        return None
    return ppe.OrientAnythingClient(
        repo_dir=Path(args.orient_anything_repo),
        checkpoint_path=(Path(args.orient_anything_ckpt) if args.orient_anything_ckpt else None),
        model_size=args.orient_anything_model_size,
        device="cuda",
        cache_dir=(Path(args.orient_anything_cache_dir) if args.orient_anything_cache_dir else None),
    )


def run_worker(
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    jsonl_path: Path,
    samples_root: Path,
    worker_id: int,
    gpu_id: int | None,
) -> list[dict[str, Any]]:
    print(f"[prealign-eval][worker {worker_id}] gpu={gpu_id} jobs={len(jobs)}", flush=True)
    if not jobs:
        jsonl_path.write_text("", encoding="utf-8")
        return []

    print(f"[prealign-eval][worker {worker_id}] loading planner VLM: {args.planner_vlm}", flush=True)
    planner_vlm = ppe.QwenVLMClient(
        model_id=args.planner_vlm,
        device_map=args.vlm_device_map,
        torch_dtype="auto",
    )
    orient_anything = maybe_build_orient_anything(args)

    records: list[dict[str, Any]] = []
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        iterator = tqdm(
            jobs,
            desc=f"gpu={gpu_id if gpu_id is not None else 'local'}",
            unit="sample",
            disable=args.quiet,
        )
        for job in iterator:
            try:
                record = evaluate_sample(
                    job,
                    planner_vlm,
                    samples_root=samples_root,
                    min_confidence=args.pre_align_min_confidence,
                    max_rotation=args.max_pre_align_rotation,
                    max_prealign_verify_attempts=args.max_prealign_verify_attempts,
                    prealign_bruteforce_after_attempts=args.prealign_bruteforce_after_attempts,
                    perturb_seed=args.perturb_seed,
                    orient_anything=orient_anything,
                    orient_confidence_threshold=args.orient_anything_confidence_threshold,
                    save_images=not args.no_save_images,
                )
                record["status"] = "ok"
                record["worker_id"] = worker_id
                if gpu_id is not None:
                    record["gpu_id"] = gpu_id
            except Exception as exc:  # noqa: BLE001
                record = {
                    "status": "error",
                    "sequence": job.get("sequence"),
                    "stride": job.get("stride"),
                    "source_frame": Path(job.get("source_path", "")).name,
                    "target_frame": Path(job.get("target_path", "")).name,
                    "error": str(exc),
                    "worker_id": worker_id,
                }
                if gpu_id is not None:
                    record["gpu_id"] = gpu_id
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
    return records


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
def write_summary(
    *,
    args: argparse.Namespace,
    strides: list[int],
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
        by_sequence_records.setdefault(str(record.get("sequence", "unknown")), []).append(record)

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
        "frames_root": str(args.frames_root),
        "strides": strides,
        "target_mode": args.target_mode,
        "perturb_seed": args.perturb_seed,
        "planner_vlm": str(args.planner_vlm),
        "use_orient_anything": args.use_orient_anything,
        "pre_align_min_confidence": args.pre_align_min_confidence,
        "max_pre_align_rotation": args.max_pre_align_rotation,
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


# ------------------------------------------------------------------
# Multi-GPU launcher
# ------------------------------------------------------------------
def launch_multi_gpu(args: argparse.Namespace, jobs: list[dict[str, Any]], gpu_ids: list[int]) -> None:
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
    procs: list[subprocess.Popen] = []

    print(
        f"[prealign-eval] launching {num_workers} workers on GPUs {gpu_ids} "
        f"({len(jobs)} samples)",
        flush=True,
    )
    for worker_id, (gpu_id, shard_path) in enumerate(zip(gpu_ids, shard_paths)):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable,
            str(script_path),
            "--frames_root", str(args.frames_root),
            "--output_path", str(output_path),
            "--planner_vlm", str(args.planner_vlm),
            "--vlm_device_map", "cuda:0",
            "--strides", args.strides,
            "--target_mode", args.target_mode,
            "--perturb_seed", str(args.perturb_seed),
            "--pre_align_min_confidence", str(args.pre_align_min_confidence),
            "--max_pre_align_rotation", str(args.max_pre_align_rotation),
            "--max_prealign_verify_attempts", str(args.max_prealign_verify_attempts),
            "--prealign_bruteforce_after_attempts", str(args.prealign_bruteforce_after_attempts),
            "--orient_anything_confidence_threshold", str(args.orient_anything_confidence_threshold),
            "--jsonl_path", str(shard_path),
            "--worker_id", str(worker_id),
            "--num_workers", str(num_workers),
            "--jobs_path", str(jobs_path),
            "--gpu_id", str(gpu_id),
        ]
        if args.sequences:
            cmd.extend(["--sequences", args.sequences])
        if args.no_save_images:
            cmd.append("--no_save_images")
        if args.quiet:
            cmd.append("--quiet")
        if args.use_orient_anything:
            cmd.append("--use_orient_anything")
            cmd.extend(["--orient_anything_repo", str(args.orient_anything_repo)])
            cmd.extend(["--orient_anything_model_size", str(args.orient_anything_model_size)])
            if args.orient_anything_ckpt:
                cmd.extend(["--orient_anything_ckpt", str(args.orient_anything_ckpt)])
            if args.orient_anything_cache_dir:
                cmd.extend(["--orient_anything_cache_dir", str(args.orient_anything_cache_dir)])
        print(f"[prealign-eval] start worker {worker_id} on GPU {gpu_id}", flush=True)
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
        args=args,
        strides=parse_strides(args.strides),
        jsonl_path=jsonl_path,
        output_path=output_path,
        started=started,
        num_sequences=len(sequence_dirs),
        all_records=all_records,
        gpu_ids=gpu_ids,
    )
    print(f"[prealign-eval] wrote summary: {output_path}", flush=True)
    print(f"[prealign-eval] wrote records: {jsonl_path}", flush=True)
    print(f"[prealign-eval] overall recovery_rate: {payload['overall']['recovery_rate']}", flush=True)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test pre-align recovery of a known random flip/rotation on DAVIS "
            "(source=frame_0, target=frame_S), with optional multi-GPU sharding."
        ),
    )
    parser.add_argument("--frames_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--planner_vlm", default=str(PLANNER_VLM_MODEL))
    parser.add_argument("--vlm_device_map", default="cuda:0")
    parser.add_argument("--strides", default="1,2,4,8,16")
    parser.add_argument("--sequences", default=None, help="Comma-separated names. Default: ALL.")
    parser.add_argument(
        "--target_mode",
        choices=("stride", "last", "last_strided"),
        default="stride",
        help="Target frame: 'stride'=frame_S (default), 'last'=final frame, "
        "'last_strided'=last multiple of stride.",
    )
    parser.add_argument("--perturb_seed", type=int, default=0)
    parser.add_argument("--gpus", default=DEFAULT_GPUS, help="GPU ids. Default: 0-7 (8 GPUs).")
    parser.add_argument("--jsonl_path", type=Path, default=None)
    parser.add_argument("--no_save_images", action="store_true", help="Do not save per-sample PNGs.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # Pre-align params (defaults mirror progressive_pose_edit.py).
    parser.add_argument("--pre_align_min_confidence", type=float, default=0.60)
    parser.add_argument("--max_pre_align_rotation", type=float, default=30.0)
    parser.add_argument("--max_prealign_verify_attempts", type=int, default=0)
    parser.add_argument("--prealign_bruteforce_after_attempts", type=int, default=5)

    # Optional Orient Anything.
    parser.add_argument("--use_orient_anything", action="store_true")
    parser.add_argument("--orient_anything_repo", default=str(ppe.DEFAULT_ORIENT_ANYTHING_REPO))
    parser.add_argument("--orient_anything_ckpt", default=None)
    parser.add_argument(
        "--orient_anything_model_size",
        choices=("small", "base", "large"),
        default=ppe.DEFAULT_ORIENT_ANYTHING_MODEL_SIZE,
    )
    parser.add_argument("--orient_anything_confidence_threshold", type=float, default=0.50)
    parser.add_argument("--orient_anything_cache_dir", default=None)

    # Internal worker flags.
    parser.add_argument("--worker_id", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--jobs_path", type=Path, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    ppe.VERBOSE = not args.quiet

    frames_root = args.frames_root.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    jsonl_path = (
        args.jsonl_path.expanduser().resolve()
        if args.jsonl_path is not None
        else output_path.with_suffix(".jsonl")
    )
    samples_root = output_path.parent / "samples"
    strides = parse_strides(args.strides)

    # Worker mode.
    if args.worker_id is not None:
        if args.jobs_path is None:
            raise ValueError("--jobs_path is required in worker mode")
        jobs = json.loads(args.jobs_path.read_text(encoding="utf-8"))
        jobs = shard_jobs(jobs, args.worker_id, args.num_workers)
        run_worker(
            jobs,
            args,
            jsonl_path=jsonl_path,
            samples_root=samples_root,
            worker_id=args.worker_id,
            gpu_id=args.gpu_id,
        )
        return

    sequence_dirs = list_sequence_dirs(frames_root, parse_sequences(args.sequences))
    jobs = plan_jobs(sequence_dirs, strides, args.target_mode)
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
                    "target_mode": args.target_mode,
                    "planned_samples": plan_counts,
                    "num_jobs": len(jobs),
                    "gpus": gpu_ids,
                    "num_d4_perturbations": len(D4_PERTURBATIONS),
                },
                indent=2,
            )
        )
        return

    if len(gpu_ids) > 1:
        launch_multi_gpu(args, jobs, gpu_ids)
        return

    gpu_id = gpu_ids[0] if gpu_ids else None
    if gpu_id is not None:
        args.vlm_device_map = f"cuda:{gpu_id}"

    started = time.time()
    print(f"[prealign-eval] frames_root={frames_root}", flush=True)
    print(
        f"[prealign-eval] sequences={len(sequence_dirs)}, planned_samples={plan_counts}, "
        f"gpu_ids={gpu_ids}",
        flush=True,
    )
    all_records = run_worker(
        jobs,
        args,
        jsonl_path=jsonl_path,
        samples_root=samples_root,
        worker_id=0,
        gpu_id=gpu_id,
    )
    payload = write_summary(
        args=args,
        strides=strides,
        jsonl_path=jsonl_path,
        output_path=output_path,
        started=started,
        num_sequences=len(sequence_dirs),
        all_records=all_records,
        gpu_ids=gpu_ids,
    )
    print(f"[prealign-eval] wrote summary: {output_path}", flush=True)
    print(f"[prealign-eval] wrote records: {jsonl_path}", flush=True)
    print(f"[prealign-eval] overall recovery_rate: {payload['overall']['recovery_rate']}", flush=True)


if __name__ == "__main__":
    main()
