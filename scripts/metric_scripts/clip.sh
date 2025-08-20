#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/output_logs/output-clipscore.log
#SBATCH --error=slurm_logs/error_logs/error-clipscore.log 

set -euo pipefail
IFS=$'\n\t'

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

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Please load your conda module or ensure it is on PATH." >&2
  exit 1
fi

conda activate clip

SEMANTICS_DIR="$REPO_ROOT/eval/captions/semantics"
cd "$SEMANTICS_DIR"

DEST_DIR="jul18_samples"

python clip-eval.py \
    --images "$REPO_ROOT/data/images/CUB_200_2011/CUB_200_2011/images" \
    --captions-dir "$REPO_ROOT/data/generated_captions/${DEST_DIR}/generated_captions/cub" \
    --save-dir "$REPO_ROOT/data/generated_captions/${DEST_DIR}/semantics/cub" \
    --ground-truth-captions "$REPO_ROOT/data/generated_captions/CUB_captions/CUB_captions.json"

# python clip-eval.py \
#     --images "$REPO_ROOT/data/images/MSCOCO_images" \
#     --captions-dir "$REPO_ROOT/data/generated_captions/jul18_samples/generated_captions/mscoco" \
#     --save-dir "$REPO_ROOT/data/generated_captions/jul18_samples/semantics/mscoco" \
#     --ground-truth-captions "$REPO_ROOT/data/generated_captions/sampled_captions.json"
