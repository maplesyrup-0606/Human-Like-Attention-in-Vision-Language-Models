#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:2
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=100G
#SBATCH --cpus-per-task=14
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/output_logs/output-llmjudge.log
#SBATCH --error=slurm_logs/error_logs/error-llmjudge.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mercurymcindoe@gmail.com

module load cuda-12.4
export CUDA_HOME=/pkgs/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH

echo "CUDA_HOME = $CUDA_HOME"
echo "SLURM_JOB_ID       = ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_GPUS     = ${SLURM_JOB_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-}"
which nvcc || true
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); import torch as _t; print(_t.cuda.get_device_name(0) if _t.cuda.is_available() else 'no cuda')"

# ----------------------------------------------------------------------
# Robust: strict mode
# ----------------------------------------------------------------------
set -euo pipefail
IFS=$'\n\t'

# ----------------------------------------------------------------------
# Robust: resolve script directory and repo root (works from any CWD)
# ----------------------------------------------------------------------
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(realpath "$SCRIPT_DIR/../..")"
fi

# ----------------------------------------------------------------------
# Robust: conda activation (portable, no hardcoding)
#   1) Use shell hook if conda is on PATH
#   2) Fallback to your original source path (kept)
#   3) Try hook again after sourcing
# ----------------------------------------------------------------------
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
elif [[ -r "/scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh" ]]; then
  # original path (kept as fallback)
  # shellcheck disable=SC1091
  source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
  if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
  fi
fi

# If conda is still not available, fail fast with a clear message
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Load your conda module or ensure conda is on PATH." >&2
  exit 1
fi

# Activate the intended env
conda activate internvl

GT_CAPTIONS="$REPO_ROOT/data/generated_captions/CUB_captions/CUB_captions.json"
SAVE_DIR="$REPO_ROOT/data/generated_captions/jul18_samples/llm-judge-ratings"
gen_captions=(
    # "$REPO_ROOT/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_new_threshold.json"
    # "$REPO_ROOT/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_with_drop_out_captions.json"
    # "$REPO_ROOT/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_with_drop_out-k-4_captions.json"
    # "$REPO_ROOT/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_with_reverse_drop_out-k-4_captions.json"
    # "$REPO_ROOT/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads.json"
    # "$REPO_ROOT/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_with_zero_out_captions.json"
    # "$REPO_ROOT/data/generated_captions/jul18_samples/generated_captions/cub/plain_captions.json"
    # "$REPO_ROOT/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_relative.json"
    "$REPO_ROOT/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_relative-k-8.json"
)

# Keep original working-intent but make paths stable from REPO_ROOT
cd "$REPO_ROOT/eval/captions/llm-as-a-judge"

mkdir -p "$SAVE_DIR"

for GEN_CAPTIONS in "${gen_captions[@]}"; do
    python caption-eval.py "$GT_CAPTIONS" "$GEN_CAPTIONS" "$SAVE_DIR" || {
        echo "Failed on $GEN_CAPTIONS"
        exit 1
    }
done