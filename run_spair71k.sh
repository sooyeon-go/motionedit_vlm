#!/usr/bin/env bash
set -euo pipefail

# Launch SPair-71k batch progressive pose editing across multiple GPUs.
#
# Examples:
#   GPU_IDS=0 bash run_spair71k.sh
#   GPU_IDS=0,1,2,3 bash run_spair71k.sh
#   GPU_IDS=0,1 SPLITS=test LIMIT=10 bash run_spair71k.sh
#   GPU_IDS=0,1 OUTPUT_ROOT=/data/outputs/spair71k bash run_spair71k.sh
#
# Each GPU runs one worker with a disjoint shard of pair annotations.
# Models are loaded once per worker process.
#
# Resume: by default (SKIP_EXISTING=1) pairs that already have
#   <OUTPUT_ROOT>/<split>/<pair_name>/result.json + final.png
# are skipped; only missing/incomplete pairs are edited.
# Force re-run everything: SKIP_EXISTING=0 bash run_spair71k.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

GPU_IDS="${GPU_IDS:-0}"
DATASET_ROOT="${DATASET_ROOT:-/data/shared-vilab/datasets/spair-71k/SPair-71k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/spair71k_progressive_pose_edit}"
# Processed in fixed order: test -> val -> trn (train), regardless of list order here.
SPLITS="${SPLITS:-test,val,trn}"
N_STEPS="${N_STEPS:-5}"
MAX_RETRIES="${MAX_RETRIES:-2}"
SEED="${SEED:-42}"
MAX_PREALIGN_VERIFY_ATTEMPTS="${MAX_PREALIGN_VERIFY_ATTEMPTS:-0}"
PREALIGN_BRUTEFORCE_AFTER_ATTEMPTS="${PREALIGN_BRUTEFORCE_AFTER_ATTEMPTS:-5}"
MAX_PLANNING_ATTEMPTS="${MAX_PLANNING_ATTEMPTS:-0}"
MAX_POSE_STEPS="${MAX_POSE_STEPS:-6}"
LIMIT="${LIMIT:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
INTERLEAVE_CLASSES="${INTERLEAVE_CLASSES:-1}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

IFS=',' read -ra GPU_ARR <<< "${GPU_IDS}"
NUM_WORKERS="${#GPU_ARR[@]}"

mkdir -p "${OUTPUT_ROOT}/logs"

echo "[run_spair71k] repo         = ${REPO_ROOT}"
echo "[run_spair71k] dataset_root = ${DATASET_ROOT}"
echo "[run_spair71k] output_root  = ${OUTPUT_ROOT}"
echo "[run_spair71k] splits       = ${SPLITS}"
echo "[run_spair71k] gpus         = ${GPU_IDS} (${NUM_WORKERS} workers)"
echo "[run_spair71k] skip_existing= ${SKIP_EXISTING} (1=resume, 0=re-run all)"
echo "[run_spair71k] interleave_classes=${INTERLEAVE_CLASSES} (1=round-robin per class, 0=class blocks)"
echo "[run_spair71k] prealign_verify_attempts = ${MAX_PREALIGN_VERIFY_ATTEMPTS} (0=use bruteforce threshold)"
echo "[run_spair71k] prealign_bruteforce_after = ${PREALIGN_BRUTEFORCE_AFTER_ATTEMPTS} (unique flip/rotate VLM pick, typically 8)"
echo "[run_spair71k] planning_attempts        = ${MAX_PLANNING_ATTEMPTS} (0=unlimited)"
echo "[run_spair71k] max_pose_steps           = ${MAX_POSE_STEPS}"
if [[ -n "${LIMIT}" ]]; then
  echo "[run_spair71k] limit/worker = ${LIMIT}"
fi

PIDS=()
for WORKER_ID in "${!GPU_ARR[@]}"; do
  GPU_ID="${GPU_ARR[$WORKER_ID]}"
  LOG_FILE="${OUTPUT_ROOT}/logs/worker_${WORKER_ID}_gpu${GPU_ID}.log"

  CMD=(
    python inference/run_spair71k_batch.py
    --dataset_root "${DATASET_ROOT}"
    --output_root "${OUTPUT_ROOT}"
    --splits "${SPLITS}"
    --worker_id "${WORKER_ID}"
    --num_workers "${NUM_WORKERS}"
    --n_steps "${N_STEPS}"
    --max_retries "${MAX_RETRIES}"
    --seed "${SEED}"
    --max_prealign_verify_attempts "${MAX_PREALIGN_VERIFY_ATTEMPTS}"
    --prealign_bruteforce_after_attempts "${PREALIGN_BRUTEFORCE_AFTER_ATTEMPTS}"
    --max_planning_attempts "${MAX_PLANNING_ATTEMPTS}"
    --max_pose_steps "${MAX_POSE_STEPS}"
    --device cuda
  )

  if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
  fi
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    CMD+=(--skip_existing)
  else
    CMD+=(--no-skip_existing)
  fi
  if [[ "${INTERLEAVE_CLASSES}" == "1" ]]; then
    CMD+=(--interleave_classes)
  else
    CMD+=(--no-interleave_classes)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    CMD+=(--dry_run)
  fi
  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARR[@]}")
  fi

  echo "[run_spair71k] starting worker ${WORKER_ID} on GPU ${GPU_ID} -> ${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for INDEX in "${!PIDS[@]}"; do
  PID="${PIDS[$INDEX]}"
  if wait "${PID}"; then
    echo "[run_spair71k] worker ${INDEX} finished (pid ${PID})"
  else
    echo "[run_spair71k] worker ${INDEX} failed (pid ${PID})" >&2
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "[run_spair71k] one or more workers failed. Check logs under ${OUTPUT_ROOT}/logs/" >&2
  exit 1
fi

echo "[run_spair71k] all workers finished."
echo "[run_spair71k] summaries: ${OUTPUT_ROOT}/logs/worker_*_summary.json"
