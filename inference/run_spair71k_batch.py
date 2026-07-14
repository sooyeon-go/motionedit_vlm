#!/usr/bin/env python3
"""Run progressive pose editing on SPair-71k pair annotations.

Each PairAnnotation JSON provides src_imname / trg_imname and category.
Images live at:
  <dataset_root>/JPEGImages/<category>/<imname>

Multi-GPU usage (4 workers, one per GPU):
  CUDA_VISIBLE_DEVICES=0 python inference/run_spair71k_batch.py --worker_id 0 --num_workers 4 ...
  CUDA_VISIBLE_DEVICES=1 python inference/run_spair71k_batch.py --worker_id 1 --num_workers 4 ...

Or use the launcher:
  GPU_IDS=0,1,2,3 bash run_spair71k.sh
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from tqdm import tqdm

INFERENCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFERENCE_DIR.parent
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

import progressive_pose_edit as ppe  # noqa: E402
from spair71k_pairs import (  # noqa: E402
    SpairPair,
    interleave_pairs_by_category,
    iter_pair_annotation_files,
    load_spair_pair,
    normalize_splits,
    shard_items,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_splits(raw: str) -> tuple[str, ...]:
    return normalize_splits(raw.split(","))


def pair_output_dir(output_root: Path, pair: SpairPair) -> Path:
    return output_root / pair.split / pair.output_name


def pair_is_complete(output_dir: Path) -> bool:
    """True when a previous run finished and wrote the canonical artifacts."""
    return (output_dir / "result.json").is_file() and (output_dir / "final.png").is_file()


def partition_assigned_pairs(
    assigned: list[SpairPair],
    output_root: Path,
    skip_existing: bool,
) -> tuple[list[SpairPair], list[SpairPair]]:
    """Split worker shard into already-complete vs pending pairs."""
    if not skip_existing:
        return assigned, []

    pending: list[SpairPair] = []
    skipped: list[SpairPair] = []
    for pair in assigned:
        out_dir = pair_output_dir(output_root, pair)
        if pair_is_complete(out_dir):
            skipped.append(pair)
        else:
            pending.append(pair)
    return pending, skipped


def write_pair_meta(output_dir: Path, pair: SpairPair) -> None:
    meta = {
        "pair_id": pair.pair_id,
        "filename": pair.filename,
        "split": pair.split,
        "category": pair.category,
        "src_imname": pair.src_imname,
        "trg_imname": pair.trg_imname,
        "annotation_path": str(pair.annotation_path),
        "src_image_path": str(pair.src_image_path),
        "trg_image_path": str(pair.trg_image_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pair_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_worker_record(
    worker_log_path: Path,
    record: dict[str, Any],
) -> None:
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    with worker_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_score_summary(output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "result.json"
    if not result_path.is_file():
        return {}
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    score_summary = result.get("score_summary", {})
    return score_summary if isinstance(score_summary, dict) else {}


def aggregate_score_summaries(score_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    overall_scores: list[dict[str, float]] = []
    per_step_scores: list[dict[str, float]] = []
    trajectory_scores: list[dict[str, float]] = []

    for summary in score_summaries:
        if not summary:
            continue
        overall = summary.get("overall_mean", {})
        per_step = summary.get("per_step_mean", {})
        trajectory = summary.get("trajectory", {})
        if isinstance(overall, dict) and overall:
            overall_scores.append(overall)
        if isinstance(per_step, dict) and per_step:
            per_step_scores.append(per_step)
        if isinstance(trajectory, dict) and trajectory:
            trajectory_scores.append(trajectory)

    return {
        "scale": "0.0-5.0",
        "num_samples_with_scores": len(overall_scores),
        "overall_mean": ppe.average_score_dicts(overall_scores),
        "per_step_mean": ppe.average_score_dicts(per_step_scores),
        "trajectory_mean": ppe.average_score_dicts(trajectory_scores),
    }


def run_single_pair(
    pair: SpairPair,
    output_dir: Path,
    planner_vlm: ppe.QwenVLMClient,
    editor: ppe.MotionNFTEditor,
    flow_estimator: ppe.UniMatchFlowEstimator,
    identity_scorer: ppe.DINOv2IdentityScorer,
    orient_anything: Optional[ppe.OrientAnythingClient],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    write_pair_meta(output_dir, pair)

    if not pair.src_image_path.is_file():
        raise FileNotFoundError(f"Source image not found: {pair.src_image_path}")
    if not pair.trg_image_path.is_file():
        raise FileNotFoundError(f"Target image not found: {pair.trg_image_path}")

    source_img = Image.open(pair.src_image_path).convert("RGB")
    target_img = Image.open(pair.trg_image_path).convert("RGB")

    ppe.save_image(source_img, output_dir / "source.png")
    ppe.save_image(target_img, output_dir / "target.png")

    aligned_source_img = source_img
    pre_alignment: Optional[ppe.PreAlignDecision] = None
    if args.skip_pre_align:
        ppe.save_image(aligned_source_img, output_dir / "step_00.png")
    else:
        aligned_source_img, pre_alignment, _ = ppe.pre_align_source_until_verified(
            source_img=source_img,
            target_img=target_img,
            planner_vlm=planner_vlm,
            output_dir=output_dir,
            min_confidence=args.pre_align_min_confidence,
            max_rotation=args.max_pre_align_rotation,
            max_attempts=args.max_prealign_verify_attempts,
            bruteforce_after_attempts=args.prealign_bruteforce_after_attempts,
            orient_anything=orient_anything,
            orient_confidence_threshold=args.orient_anything_confidence_threshold,
        )
        ppe.save_image(aligned_source_img, output_dir / "aligned_source.png")
        ppe.save_image(aligned_source_img, output_dir / "step_00.png")

    if args.compare_angle_lora:
        serialized_result = ppe.progressive_pose_edit_angle_lora_comparison(
            source_img=aligned_source_img,
            target_img=target_img,
            planner_vlm=planner_vlm,
            editor=editor,
            flow_estimator=flow_estimator,
            identity_scorer=identity_scorer,
            output_dir=output_dir,
            n_steps=args.n_steps,
            max_retries=args.max_retries,
            flow_threshold=args.flow_threshold,
            skip_vlm_verify=args.skip_vlm_verify,
            skip_trajectory_vlm=args.skip_trajectory_vlm,
            trajectory_flow_ratio=args.trajectory_flow_ratio,
            pre_alignment=pre_alignment,
            skip_trajectory_repair=args.skip_trajectory_repair,
            max_trajectory_repairs=args.max_trajectory_repairs,
            max_pose_steps=args.max_pose_steps,
            max_planning_attempts=args.max_planning_attempts,
            coarse_angle=args.coarse_angle,
            max_angle_steps=args.max_angle_steps,
            orient_anything=orient_anything,
            orient_confidence_threshold=args.orient_anything_confidence_threshold,
        )
        trajectory_ok = serialized_result.get("variants", {}).get("angle_lora", {}).get("trajectory_ok")
    else:
        result = ppe.progressive_pose_edit(
            source_img=aligned_source_img,
            target_img=target_img,
            planner_vlm=planner_vlm,
            editor=editor,
            flow_estimator=flow_estimator,
            identity_scorer=identity_scorer,
            output_dir=output_dir,
            n_steps=args.n_steps,
            max_retries=args.max_retries,
            flow_threshold=args.flow_threshold,
            skip_vlm_verify=args.skip_vlm_verify,
            skip_trajectory_vlm=args.skip_trajectory_vlm,
            trajectory_flow_ratio=args.trajectory_flow_ratio,
            pre_alignment=pre_alignment,
            skip_trajectory_repair=args.skip_trajectory_repair,
            max_trajectory_repairs=args.max_trajectory_repairs,
            max_pose_steps=args.max_pose_steps,
            max_planning_attempts=args.max_planning_attempts,
            coarse_angle=args.coarse_angle,
            max_angle_steps=args.max_angle_steps,
            orient_anything=orient_anything,
            orient_confidence_threshold=args.orient_anything_confidence_threshold,
        )
        serialized_result = ppe.write_pipeline_artifacts(output_dir, result)
        trajectory_ok = (
            result.trajectory_verify.overall_ok if result.trajectory_verify is not None else None
        )

    elapsed = time.time() - started
    return {
        "status": "ok",
        "pair_id": pair.pair_id,
        "filename": pair.filename,
        "split": pair.split,
        "category": pair.category,
        "output_dir": str(output_dir),
        "elapsed_sec": round(elapsed, 2),
        "trajectory_ok": trajectory_ok,
        "score_summary": serialized_result.get("score_summary", {}),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch progressive pose editing on SPair-71k pair annotations.",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/mnt/sy/dataset/SPair-71k"),
        help="Root of extracted SPair-71k dataset.",
    )
    parser.add_argument(
        "--pair_annotation_dir",
        type=Path,
        default=None,
        help="Override PairAnnotation directory (default: <dataset_root>/PairAnnotation).",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/spair71k_progressive_pose_edit"),
        help="Root directory for per-pair outputs.",
    )
    parser.add_argument(
        "--splits",
        default="test,val,trn",
        help="Comma-separated splits to process (always run in order: test -> val -> trn).",
    )
    parser.add_argument(
        "--worker_id",
        type=int,
        default=0,
        help="This worker index in [0, num_workers).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Total number of parallel workers/GPUs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many pairs on this worker (after sharding).",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip pairs whose output dir already has result.json and final.png (default: on).",
    )
    parser.add_argument(
        "--interleave_classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Round-robin across categories within each split "
            "(aeroplane, bicycle, bus, ...) instead of class-by-class blocks (default: on)."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List assigned pairs without loading models or running the pipeline.",
    )
    parser.add_argument(
        "--worker_log",
        type=Path,
        default=None,
        help="JSONL log path for this worker (default: <output_root>/logs/worker_<id>.jsonl).",
    )

    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--editor_base_model", default=str(ppe.DEFAULT_EDITOR_BASE_MODEL))
    parser.add_argument("--motionedit_lora_path", default=str(ppe.DEFAULT_MOTIONEDIT_LORA_PATH))
    parser.add_argument("--qwen_angles_lora_path", default=str(ppe.DEFAULT_QWEN_ANGLES_LORA_PATH))
    parser.add_argument("--planner_vlm", default=str(ppe.DEFAULT_PLANNER_VLM))
    parser.add_argument("--dinov2_model", default=str(ppe.DEFAULT_DINOV2_MODEL))
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
    parser.add_argument("--unimatch_ckpt", default=str(ppe.DEFAULT_UNIMATCH_CKPT))
    parser.add_argument("--skip_path_check", action="store_true")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--editor_device_map", default=None)
    parser.add_argument("--vlm_device_map", default="auto")
    parser.add_argument("--flow_resize_to", type=int, default=None)
    parser.add_argument("--flow_threshold", type=float, default=0.5)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--skip_vlm_verify", action="store_true")
    parser.add_argument("--skip_trajectory_vlm", action="store_true")
    parser.add_argument("--trajectory_flow_ratio", type=float, default=4.0)
    parser.add_argument("--max_trajectory_repairs", type=int, default=2)
    parser.add_argument("--skip_trajectory_repair", action="store_true")
    parser.add_argument("--skip_pre_align", action="store_true")
    parser.add_argument("--pre_align_min_confidence", type=float, default=0.60)
    parser.add_argument("--max_pre_align_rotation", type=float, default=30.0)
    parser.add_argument("--max_prealign_verify_attempts", type=int, default=0)
    parser.add_argument("--prealign_bruteforce_after_attempts", type=int, default=5)
    parser.add_argument("--max_pose_steps", type=int, default=4)
    parser.add_argument("--max_planning_attempts", type=int, default=0)
    parser.add_argument("--coarse_angle", dest="coarse_angle", action="store_true", default=True)
    parser.add_argument("--fine_angle", dest="coarse_angle", action="store_false")
    parser.add_argument("--max_angle_steps", type=int, default=2)
    parser.add_argument("--compare_angle_lora", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ppe.VERBOSE = not args.quiet
    torch.manual_seed(args.seed)

    dataset_root = args.dataset_root.expanduser().resolve()
    pair_annotation_dir = (
        args.pair_annotation_dir.expanduser().resolve()
        if args.pair_annotation_dir is not None
        else dataset_root / "PairAnnotation"
    )
    output_root = args.output_root.expanduser().resolve()
    worker_log_path = (
        args.worker_log.expanduser().resolve()
        if args.worker_log is not None
        else output_root / "logs" / f"worker_{args.worker_id:02d}.jsonl"
    )
    splits = parse_splits(args.splits)

    log("========== SPair-71k Batch Progressive Pose Edit ==========")
    log(f"dataset_root         = {dataset_root}")
    log(f"pair_annotation_dir  = {pair_annotation_dir}")
    log(f"output_root          = {output_root}")
    log(f"splits               = {', '.join(splits)}")
    log(f"worker               = {args.worker_id}/{args.num_workers}")
    log(f"device               = {args.device}")
    if torch.cuda.is_available():
        log(f"cuda_visible_devices = {torch.cuda.device_count()} visible GPU(s)")

    all_pairs: list[SpairPair] = []
    for annotation_path in iter_pair_annotation_files(pair_annotation_dir, splits=splits):
        all_pairs.append(load_spair_pair(annotation_path, dataset_root))

    if args.interleave_classes:
        all_pairs = interleave_pairs_by_category(all_pairs)

    assigned = shard_items(all_pairs, worker_id=args.worker_id, num_workers=args.num_workers)
    if args.limit is not None:
        assigned = assigned[: args.limit]

    scan_started = time.time()
    pending, skipped_existing = partition_assigned_pairs(
        assigned,
        output_root=output_root,
        skip_existing=args.skip_existing,
    )
    scan_elapsed = time.time() - scan_started

    log(f"total_pairs={len(all_pairs)}, assigned_to_worker={len(assigned)}")
    log(
        f"pair_order={'class round-robin' if args.interleave_classes else 'annotation filename sort'}"
    )
    if args.skip_existing:
        log(
            f"resume scan: complete={len(skipped_existing)}, "
            f"pending={len(pending)} (missing or incomplete output), "
            f"scan_time={scan_elapsed:.2f}s"
        )
    else:
        log(f"resume scan: skip_existing=off, pending={len(pending)}")

    if args.dry_run:
        log("[dry_run] Would process pending pairs:")
        for pair in pending[:20]:
            out_dir = pair_output_dir(output_root, pair)
            log(
                f"  PENDING [{pair.split}] pair_id={pair.pair_id} "
                f"src={pair.src_image_path.name} trg={pair.trg_image_path.name} "
                f"-> {out_dir}"
            )
        if len(pending) > 20:
            log(f"  ... and {len(pending) - 20} more pending")
        if skipped_existing:
            log(f"[dry_run] Would skip {len(skipped_existing)} complete pair(s)")
        return

    if not pending:
        log("[resume] All assigned pairs already complete; nothing to do.")
        existing_score_summaries = [
            load_existing_score_summary(pair_output_dir(output_root, pair))
            for pair in skipped_existing
        ]
        summary = {
            "worker_id": args.worker_id,
            "num_workers": args.num_workers,
            "assigned": len(assigned),
            "ok": 0,
            "skipped": len(skipped_existing),
            "failed": 0,
            "pending": 0,
            "output_root": str(output_root),
            "worker_log": str(worker_log_path),
            "score_summary": aggregate_score_summaries(existing_score_summaries),
        }
        summary_path = output_root / "logs" / f"worker_{args.worker_id:02d}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    ppe.log("[deps] Checking runtime dependency versions...")
    ppe._check_runtime_dependencies()
    ppe.log("[deps] OK")

    if not args.skip_path_check:
        ppe.validate_pretrained_paths(
            editor_base_model=args.editor_base_model,
            motionedit_lora_path=args.motionedit_lora_path,
            qwen_angles_lora_path=args.qwen_angles_lora_path,
            planner_vlm=args.planner_vlm,
            dinov2_model=args.dinov2_model,
            unimatch_ckpt=args.unimatch_ckpt,
        )

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    log("\n[load] Planner VLM (Qwen3-VL)...")
    t0 = time.time()
    planner_vlm = ppe.QwenVLMClient(
        model_id=args.planner_vlm,
        device_map=args.vlm_device_map,
        torch_dtype="auto",
    )
    log(f"[load] Planner VLM ready ({time.time() - t0:.1f}s)")

    log("[load] MotionNFT editor (Qwen Image Edit + LoRA)...")
    t0 = time.time()
    editor = ppe.MotionNFTEditor(
        base_model=args.editor_base_model,
        lora_path=args.motionedit_lora_path,
        angle_lora_path=args.qwen_angles_lora_path,
        device=args.device,
        device_map=args.editor_device_map,
        dtype=dtype,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    log(f"[load] MotionNFT editor ready ({time.time() - t0:.1f}s)")

    log("[load] UniMatch optical flow...")
    t0 = time.time()
    flow_estimator = ppe.UniMatchFlowEstimator(
        ckpt_path=Path(args.unimatch_ckpt),
        device=args.device,
        resize_to=args.flow_resize_to,
    )
    log(f"[load] UniMatch ready ({time.time() - t0:.1f}s)")

    log("[load] DINOv2 identity scorer...")
    t0 = time.time()
    identity_scorer = ppe.DINOv2IdentityScorer(args.dinov2_model, args.device)
    log(f"[load] DINOv2 ready ({time.time() - t0:.1f}s)")

    orient_anything: Optional[ppe.OrientAnythingClient] = None
    if args.use_orient_anything:
        log("[load] Orient Anything geometry estimator...")
        t0 = time.time()
        orient_anything = ppe.OrientAnythingClient(
            repo_dir=Path(args.orient_anything_repo),
            checkpoint_path=(
                Path(args.orient_anything_ckpt)
                if args.orient_anything_ckpt is not None
                else None
            ),
            model_size=args.orient_anything_model_size,
            device=args.device,
            cache_dir=(
                Path(args.orient_anything_cache_dir)
                if args.orient_anything_cache_dir is not None
                else None
            ),
        )
        log(f"[load] Orient Anything ready ({time.time() - t0:.1f}s)")

    ok_count = 0
    skip_count = len(skipped_existing)
    fail_count = 0
    score_summaries: list[dict[str, Any]] = []

    if skip_count:
        log(f"[resume] Skipping {skip_count} pair(s) with existing result.json + final.png")
        for pair in skipped_existing:
            out_dir = pair_output_dir(output_root, pair)
            score_summary = load_existing_score_summary(out_dir)
            if score_summary:
                score_summaries.append(score_summary)
            append_worker_record(
                worker_log_path,
                {
                    "status": "skipped",
                    "reason": "existing_output",
                    "pair_id": pair.pair_id,
                    "filename": pair.filename,
                    "split": pair.split,
                    "output_dir": str(out_dir),
                    "score_summary": score_summary,
                },
            )

    pair_bar = tqdm(pending, desc=f"Worker {args.worker_id}", unit="pair", disable=args.quiet)

    for pair in pair_bar:
        out_dir = pair_output_dir(output_root, pair)
        pair_bar.set_postfix(split=pair.split, pair_id=pair.pair_id)

        try:
            record = run_single_pair(
                pair=pair,
                output_dir=out_dir,
                planner_vlm=planner_vlm,
                editor=editor,
                flow_estimator=flow_estimator,
                identity_scorer=identity_scorer,
                orient_anything=orient_anything,
                args=args,
            )
            ok_count += 1
            score_summary = record.get("score_summary", {})
            if isinstance(score_summary, dict) and score_summary:
                score_summaries.append(score_summary)
            append_worker_record(worker_log_path, record)
        except Exception as exc:
            fail_count += 1
            record = {
                "status": "error",
                "pair_id": pair.pair_id,
                "filename": pair.filename,
                "split": pair.split,
                "category": pair.category,
                "output_dir": str(out_dir),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_worker_record(worker_log_path, record)
            log(f"[error] {pair.filename}: {exc}")

    summary = {
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "assigned": len(assigned),
        "pending": len(pending),
        "ok": ok_count,
        "skipped": skip_count,
        "failed": fail_count,
        "skip_existing": args.skip_existing,
        "output_root": str(output_root),
        "worker_log": str(worker_log_path),
        "score_summary": aggregate_score_summaries(score_summaries),
    }
    summary_path = output_root / "logs" / f"worker_{args.worker_id:02d}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log("\n========== Worker Summary ==========")
    for key, value in summary.items():
        log(f"  {key}: {value}")
    log(f"[done] Worker {args.worker_id} finished.")


if __name__ == "__main__":
    main()
