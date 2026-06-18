#!/usr/bin/env bash
set -euo pipefail

# Fix dependencies for progressive_pose_edit.py.
#
# Uses tools/repair_torch_stack.py to fix torch/torchvision mismatch first,
# then installs the HF inference stack.
#
# Usage:
#   bash tools/fix_dependencies.sh
#   STRATEGY=match-current bash tools/fix_dependencies.sh

ENV_NAME="${ENV_NAME:-motionedit}"
STRATEGY="${STRATEGY:-auto}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  else
    echo "conda was not found." >&2
    exit 1
  fi
fi

cd "${REPO_ROOT}"

echo "[fix] Repairing torch/torchvision + HF stack (strategy=${STRATEGY})"
conda run -n "${ENV_NAME}" python tools/repair_torch_stack.py --strategy "${STRATEGY}"

echo "[fix] Done. Re-run your script."
