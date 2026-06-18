#!/usr/bin/env bash
set -euo pipefail

# Physical GPU index on this machine (0, 1, 2, ...).
# Override at runtime: GPU_ID=3 bash run.sh
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[run] Using GPU ${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"

python inference/progressive_pose_edit.py \
  --source_image /data/project-vilab/sy/qwen/VQA_edit/image_dataset/cat_1.png \
  --target_image /data/project-vilab/sy/qwen/VQA_edit/image_dataset/cat_3.png \
  --output_dir outputs/progressive_pose_edit/demo \
  --device cuda
