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
  echo "ERROR: conda not found on PATH and CONDA_EXE not set. Load your conda module first." >&2
  exit 1
fi
conda activate NSERC

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$SUBMIT_DIR"
  for _ in 1 2 3 4 5; do
    [[ -d "$REPO_ROOT/LLaVA" ]] && break
    REPO_ROOT="$(dirname "$REPO_ROOT")"
  done
fi

cd "$REPO_ROOT/LLaVA/llava/eval"

export PYTHONPATH="$REPO_ROOT/LLaVA:${PYTHONPATH:-}"

# FIXED
SCANPATH_DIR="$REPO_ROOT/data/scanpaths/coco_scanpaths"
IMAGES_DIR="$REPO_ROOT/data/images/MSCOCO_images"
CAPTIONS_FILE_PATH="$REPO_ROOT/data/generated_captions/sampled_captions.json"

runs=(
    # RUN_NAME, TYPE, MARGIN, TRAJECTORY_MODE, TARGET_LAYER
    # "backup_plain,None,0,0," 
    # "pd,non-gaussian,0,0,"
    # "pdm,non-gaussian,1,0,"
    # "pdt,non-gaussian,0,1,"
    # "pdtm,non-gaussian,1,1,"
    # "gaussian,gaussian,0,0,"
    # "salient_pre_sm_gaussian,None,0,0,"
    # "salient_post_sm_gaussian,None,0,0,"
    # "salient_post_sm_gaussian_layer_19,None,0,0,19"
    # "salient_post_sm_gaussian_layer_20,None,0,0,20"
    # "salient_post_sm_gaussian_layer_21,None,0,0,21"
    # "salient_post_sm_gaussian_layer_22,None,0,0,22"
    # "salient_post_sm_gaussian_layer_23,None,0,0,23"
    # "salient_post_sm_gaussian_layer_24,None,0,0,24"
    # "salient_post_sm_gaussian_layer_25,None,0,0,25"
    # "salient_post_sm_gaussian_layer_26,None,0,0,26"
    # "salient_head,salient-head,0,0,-1"
    "salient_heads_relative-k-8,salient-head,0,0,"
)

for run in "${runs[@]}"; do
    IFS="," read -r RUN_NAME TYPE MARGIN TRAJECTORY_MODE TARGET_LAYER <<< "$run"

    ANSWERS_FILE_PATH="$REPO_ROOT/data/generated_captions/jul30_samples/generated_captions/mscoco/${RUN_NAME}_captions.json"
    WEIGHTS_DIR="$REPO_ROOT/data/weights/mscoco/${RUN_NAME}"
    mkdir -p "$WEIGHTS_DIR"

    echo "Running $RUN_NAME (type=$TYPE margin=$MARGIN trajectory=$TRAJECTORY_MODE target_layer=$TARGET_LAYER)"

    python -m model_cococaptions2017 \
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