#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/output_logs/output-sbert.log
#SBATCH --error=slurm_logs/error_logs/error-sbert.log 

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(realpath "$SCRIPT_DIR/../..")"
fi

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

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Load your conda module or ensure conda is on PATH." >&2
  exit 1
fi

conda activate sentence-trnsfr

cd "$REPO_ROOT/eval/captions/semantics"

DEST_DIR=aug14_samples

python sbert-eval.py \
    --ref "$REPO_ROOT/data/generated_captions/CUB_captions/CUB_captions.json" \
    --pred-dir "$REPO_ROOT/data/generated_captions/${DEST_DIR}/generated_captions/cub" \
    --save-dir "$REPO_ROOT/data/generated_captions/${DEST_DIR}/semantics/cub"
# python sbert-eval.py \
#     --ref "$REPO_ROOT/data/generated_captions/sampled_captions.json" \
#     --pred-dir "$REPO_ROOT/data/generated_captions/jul18_samples/generated_captions/mscoco" \
#     --save-dir "$REPO_ROOT/data/generated_captions/jul18_samples/semantics/mscoco"