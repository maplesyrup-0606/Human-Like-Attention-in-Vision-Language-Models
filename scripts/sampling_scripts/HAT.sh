#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=40G
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mercurymcindoe@gmail.com
#SBATCH --output=slurm_logs/output_logs/output-HAT.log
#SBATCH --error=slurm_logs/error_logs/error-HAT.log

set -euo pipefail

# Prefer the directory sbatch was invoked from (SLURM-safe); fall back to PWD
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# Try to find the repo root from the submit dir; fall back to walking up
if REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$SUBMIT_DIR"
  # Walk up a few levels until we spot a repo marker (HAT/ or .git/)
  for _ in 1 2 3 4 5 6; do
    [[ -d "$REPO_ROOT/HAT" || -d "$REPO_ROOT/.git" ]] && break
    PARENT="$(dirname "$REPO_ROOT")"
    [[ "$PARENT" == "$REPO_ROOT" ]] && break
    REPO_ROOT="$PARENT"
  done
fi

IMAGE_ROOT="$(realpath "$REPO_ROOT/data/CUB_200_2011/CUB_200_2011/images")"
SCAN_ROOT="$(realpath -m "$REPO_ROOT/data/scanpaths/cub_scanpaths")"                              # *.npy
VIS_ROOT="$(realpath -m "$REPO_ROOT/data/scanpaths/cub_scanpaths/images_with_scanpaths")"         # *_output.jpg

mkdir -p "$SCAN_ROOT" "$VIS_ROOT"

if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("$CONDA_EXE" info --base 2>/dev/null)" || true
  if [[ -n "${CONDA_BASE:-}" && -r "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    eval "$(conda shell.bash hook)" || true
  fi
fi
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Please load/initialize conda first." >&2
  exit 1
fi
conda activate NSERC

cd "$REPO_ROOT/HAT"

python inference.py "$IMAGE_ROOT" "$SCAN_ROOT" "$VIS_ROOT"