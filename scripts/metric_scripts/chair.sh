#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/output_logs/output-chair.log
#SBATCH --error=slurm_logs/error_logs/error-chair.log 

set -euo pipefail
IFS=$'\n\t'

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  # Walk up from the submit dir until we see a repo marker
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
  echo "ERROR: conda not found. Load your conda module or ensure conda is on PATH." >&2
  exit 1
fi
conda activate chair

CHAIR_DIR="$REPO_ROOT/eval/captions/semantics-metrics/CHAIR-metric-standalone"
CHAIR_PARENT_DIR="$(dirname "$CHAIR_DIR")"

CAP_DIR="$REPO_ROOT/data/generated_captions/jul18_samples/generated_captions/mscoco"
SAVE_DIR="$REPO_ROOT/data/generated_captions/jul18_samples/semantics/mscoco"
CACHE_FILE="$CHAIR_DIR/chair.pkl"   # keep cache alongside CHAIR as before

# Ensure output directory exists
mkdir -p "$SAVE_DIR"

cd "$CHAIR_DIR"

# for cap_file in "$CAP_DIR"/*.json; do
#     base_name=$(basename "$cap_file")                          # e.g. salient_post_sm_gaussian_captions.json
#     no_suffix=${base_name%_captions.json}                      # e.g. salient_post_sm_gaussian
#     save_name="chair_${no_suffix}.json"                        # e.g. chair_salient_post_sm_gaussian.json
#     save_path="$SAVE_DIR/$save_name"
#
#     echo "▶ Running CHAIR for: $cap_file"
#     echo "  → Saving to: $save_path"
#
#     python chair.py \
#         --cap_file "$cap_file" \
#         --image_id_key image_id \
#         --caption_key caption \
#         --cache "$CACHE_FILE" \
#         --save_path "$save_path"
# done

cd "$CHAIR_PARENT_DIR"

python chair-eval.py \
    --results-dir "$SAVE_DIR" \
    --save-dir "$SAVE_DIR"