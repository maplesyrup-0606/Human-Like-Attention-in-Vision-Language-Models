#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:2
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=80G
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mercurymcindoe@gmail.com
#SBATCH --output=slurm_logs/output_logs/output-pope_mscoco2014.log
#SBATCH --error=slurm_logs/error_logs/error-pope_mscoco2014.log

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then :; else
  REPO_ROOT="$(realpath "$SCRIPT_DIR/../..")"
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
fi
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Please load/initialize conda first." >&2
  exit 1
fi
conda activate NSERC

NSERC_ROOT="$REPO_ROOT"
LLAVA_ROOT="$NSERC_ROOT/LLaVA"
HAT_ROOT="$NSERC_ROOT/HAT"

export PYTHONPATH="$NSERC_ROOT:$HAT_ROOT:$LLAVA_ROOT:${PYTHONPATH:-}"
cd "$LLAVA_ROOT"

# FIXED
SCANPATH_DIR="$NSERC_ROOT/data/scanpaths/cub_scanpaths"
IMAGES_DIR="$NSERC_ROOT/data/images/MSCOCO2014/val2014"
CAPTIONS_FILE_PATH="$NSERC_ROOT/data/generated_captions/CUB_captions/CUB_captions.json"
QUESTIONS_FILE_PATH="$NSERC_ROOT/eval/captions/hallucination/POPE/output/coco/coco_pope_random.jsonl"

runs=(
    # RUN_NAME, TYPE, MARGIN, TRAJECTORY_MODE, TARGET_LAYER
#    "baseline,,0,0,"
#    "salient-head,salient-head,0,0,"
	"baseline-13b,,0,0,"
	"salient-head-13b,salient-head,0,0,"
)

for run in "${runs[@]}"; do
    IFS="," read -r RUN_NAME TYPE MARGIN TRAJECTORY_MODE TARGET_LAYER <<< "$run"

    ANSWERS_FILE_PATH="$NSERC_ROOT/data/generated_captions/jul30_samples/generated_captions/POPE/MSCOCO2014/${RUN_NAME}_answers.jsonl"
    WEIGHTS_DIR="$NSERC_ROOT/data/weights/cub/${RUN_NAME}"
    mkdir -p "$WEIGHTS_DIR"

    echo "Running $RUN_NAME (type=$TYPE margin=$MARGIN trajectory=$TRAJECTORY_MODE target_layer=$TARGET_LAYER)"

    python -m llava.eval.model_POPE \
        --model-path liuhaotian/llava-v1.5-13b \
        --load-4bit \
        --temperature 0.8 \
        --questions-file "$QUESTIONS_FILE_PATH" \
        --images-dir    "$IMAGES_DIR" \
        --answers-file  "$ANSWERS_FILE_PATH" \
        --weights-dir   "$WEIGHTS_DIR" \
        --trajectory    "$TRAJECTORY_MODE" \
        --num-samples   3000 \
        --seed          1 \
        --type          "$TYPE" \
        --margin        "$MARGIN" \
        $( [[ -n $TARGET_LAYER ]] && echo --target-layer "$TARGET_LAYER" )
done

cd "$NSERC_ROOT"