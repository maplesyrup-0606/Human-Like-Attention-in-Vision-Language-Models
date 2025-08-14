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
#SBATCH --output=slurm_logs/output_logs/output-caption_sampling.log
#SBATCH --error=slurm_logs/error_logs/error-caption_sampling.log

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
  echo "ERROR: conda not found. Please load it first." >&2
  exit 1
fi
conda activate NSERC

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then :; else
  REPO_ROOT="$(realpath "$SCRIPT_DIR/../..")"
fi

cd "$REPO_ROOT/LLaVA/llava/eval"

export PYTHONPATH="$REPO_ROOT/LLaVA:${PYTHONPATH:-}"

SCANPATH_DIR="$REPO_ROOT/data/scanpaths/cub_scanpaths"
IMAGES_DIR="$REPO_ROOT/data/images/CUB_200_2011/CUB_200_2011/images"
CAPTIONS_FILE_PATH="$REPO_ROOT/data/generated_captions/CUB_captions/CUB_captions.json"
DEST_DIR=jul30_samples

runs=(
    "pd-new-prompt,non-gaussian,0,0,"
    "pdm-new-prompt,non-gaussian,1,0,"
    "pdt-new-prompt,non-gaussian,0,1,"
    "pdtm-new-prompt,non-gaussian,1,1,"
    "gaussian-new-prompt,gaussian,0,0,"
)

for run in "${runs[@]}"; do
    IFS="," read -r RUN_NAME TYPE MARGIN TRAJECTORY_MODE TARGET_LAYER <<< "$run"

    ANSWERS_FILE_PATH="$REPO_ROOT/data/generated_captions/${DEST_DIR}/generated_captions/cub/${RUN_NAME}.json"
    WEIGHTS_DIR="$REPO_ROOT/data/weights/cub/${RUN_NAME}"
    mkdir -p "$WEIGHTS_DIR"

    echo "Running $RUN_NAME (type=$TYPE margin=$MARGIN trajectory=$TRAJECTORY_MODE target_layer=$TARGET_LAYER)"

    python -m model_CUB \
        --model-path  liuhaotian/llava-v1.5-7b \
        --load-4bit \
        --temperature 0.8 \
        --scanpath      "$SCANPATH_DIR" \
        --captions-file "$CAPTIONS_FILE_PATH" \
        --images-dir    "$IMAGES_DIR" \
        --answers-file  "$ANSWERS_FILE_PATH" \
        --weights-dir   "$WEIGHTS_DIR" \
        --trajectory    "$TRAJECTORY_MODE" \
        --num-samples   1000 \
        --seed          1 \
        --type          "$TYPE" \
        --margin        "$MARGIN" \
        $( [[ -n $TARGET_LAYER ]] && echo --target-layer "$TARGET_LAYER" )
done

cd "$REPO_ROOT"