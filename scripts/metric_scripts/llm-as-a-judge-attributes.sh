#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:2
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=100G
#SBATCH --cpus-per-task=14
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/output_logs/output-llmjudge-attr.log
#SBATCH --error=slurm_logs/error_logs/error-llmjudge-attr.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mercurymcindoe@gmail.com

set -euo pipefail
IFS=$'\n\t'

module load cuda-12.4
export CUDA_HOME=/pkgs/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH

echo "CUDA_HOME = $CUDA_HOME"
echo "SLURM_JOB_ID       = ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_GPUS     = ${SLURM_JOB_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-}"
which nvcc || true
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); import torch as _t; print(_t.cuda.get_device_name(0) if _t.cuda.is_available() else 'no cuda')"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$SUBMIT_DIR"
  for _ in 1 2 3 4 5 6; do
    [[ -d "$REPO_ROOT/.git" || -d "$REPO_ROOT/eval" ]] && break
    PARENT="$(dirname "$REPO_ROOT")"
    [[ "$PARENT" == "$REPO_ROOT" ]] && break
    REPO_ROOT="$PARENT"
  done
fi

if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("$CONDA_EXE" info --base 2>/dev/null)" || true
  if [[ -n "${CONDA_BASE:-}" && -r "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    eval "$(conda shell.bash hook)" || true
  fi
elif [[ -r "/scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh" ]]; then
  # original path (kept as fallback)
  # shellcheck disable=SC1091
  source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
  eval "$(conda shell.bash hook)" || true
fi
conda activate internvl

# Keep original working-intent but make paths stable from REPO_ROOT
cd "$REPO_ROOT/eval/captions/llm-as-a-judge"

# python attribute-eval.py \
#     --attributes ../../data/CUB_attributes.json \
#     --captions ../../data/generated_captions/jun26_samples/generated_captions/cub/plain_captions.json \
#     --save-dir ../../data/generated_captions/jun26_samples/attributes_eval
DEST_DIR=aug14_samples
CAPTIONS_DIR="$REPO_ROOT/data/generated_captions/${DEST_DIR}/generated_captions/cub"
SAVE_DIR="$REPO_ROOT/data/generated_captions/${DEST_DIR}/attributes_eval"
ATTRIBUTES="$REPO_ROOT/data/CUB_attributes.json"

mkdir -p "$SAVE_DIR"

for CAPTION_FILE in "$CAPTIONS_DIR"/*.json; do
    echo "Evaluating $CAPTION_FILE"
    python attribute-eval.py \
        --attributes "$ATTRIBUTES" \
        --captions "$CAPTION_FILE" \
        --save-dir "$SAVE_DIR"
done
