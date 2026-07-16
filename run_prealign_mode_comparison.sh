#!/usr/bin/env bash
set -euo pipefail

# Run all four DAVIS pre-align recovery modes and collect results under one folder.
#
# Usage:
#   bash run_prealign_mode_comparison.sh
#
# Optional overrides:
#   FRAMES_ROOT=/path/to/DAVIS/JPEGImages/Full-Resolution \
#   GPUS=0,1,2,3,4,5,6,7 \
#   OUTPUT_ROOT=outputs/output_comparison \
#   bash run_prealign_mode_comparison.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

FRAMES_ROOT="${FRAMES_ROOT:-/data/shared-vilab/datasets/DAVIS/JPEGImages/Full-Resolution}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/output_comparison}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PERTURB_SEED="${PERTURB_SEED:-0}"
ORIENT_ANYTHING_REPO="${ORIENT_ANYTHING_REPO:-${REPO_ROOT}/third_party/Orient-Anything}"
ORIENT_ANYTHING_CACHE_DIR="${ORIENT_ANYTHING_CACHE_DIR:-}"
GROUNDING_DINO_CACHE_DIR="${GROUNDING_DINO_CACHE_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODES=(
  "vlm"
  "oa"
  "grounding_dino"
  "oa_grounding_dino"
)

mkdir -p "${OUTPUT_ROOT}"

echo "[compare] frames_root=${FRAMES_ROOT}"
echo "[compare] output_root=${OUTPUT_ROOT}"
echo "[compare] gpus=${GPUS}"
echo "[compare] modes=${MODES[*]}"

for mode in "${MODES[@]}"; do
  mode_dir="${OUTPUT_ROOT}/${mode}"
  summary_path="${mode_dir}/summary.json"
  mkdir -p "${mode_dir}"

  cmd=(
    "${PYTHON_BIN}" tools/evaluate_davis_prealign_recovery.py
    --frames_root "${FRAMES_ROOT}"
    --output_path "${summary_path}"
    --gpus "${GPUS}"
    --prealign_mode "${mode}"
    --perturb_seed "${PERTURB_SEED}"
    --orient_anything_repo "${ORIENT_ANYTHING_REPO}"
  )

  if [[ -n "${ORIENT_ANYTHING_CACHE_DIR}" ]]; then
    cmd+=(--orient_anything_cache_dir "${ORIENT_ANYTHING_CACHE_DIR}")
  fi
  if [[ -n "${GROUNDING_DINO_CACHE_DIR}" ]]; then
    cmd+=(--grounding_dino_cache_dir "${GROUNDING_DINO_CACHE_DIR}")
  fi

  echo
  echo "============================================================"
  echo "[compare] start mode=${mode}"
  echo "[compare] summary=${summary_path}"
  echo "============================================================"
  "${cmd[@]}"
  echo "[compare] done mode=${mode}"
done

manifest_path="${OUTPUT_ROOT}/comparison_manifest.json"
"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${manifest_path}" "${MODES[@]}" <<'PY'
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
modes = sys.argv[3:]

entries = []
for mode in modes:
    summary_path = output_root / mode / "summary.json"
    jsonl_path = output_root / mode / "summary.jsonl"
    samples_dir = output_root / mode / "samples"
    payload = {}
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    entries.append(
        {
            "mode": mode,
            "summary_path": str(summary_path),
            "jsonl_path": str(jsonl_path),
            "samples_dir": str(samples_dir),
            "overall": payload.get("overall", {}),
            "by_sequence": payload.get("by_sequence", {}),
        }
    )

manifest = {
    "output_root": str(output_root),
    "modes": modes,
    "entries": entries,
}
manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"[compare] wrote manifest: {manifest_path}")
for entry in entries:
    overall = entry.get("overall") or {}
    print(
        f"  - {entry['mode']}: recovery_rate={overall.get('recovery_rate')} "
        f"verify_apply_rate={overall.get('verify_apply_rate')} "
        f"grounding_source={overall.get('grounding_source_detection_success_rate')} "
        f"grounding_target={overall.get('grounding_target_detection_success_rate')} "
        f"grounding_both={overall.get('grounding_crop_success_rate')}"
    )
PY

echo
echo "[compare] all modes finished"
echo "[compare] results under: ${OUTPUT_ROOT}"
echo "[compare] manifest: ${manifest_path}"
