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

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate NSERC

cd ~/NSERC/LLaVA/llava/eval

export PYTHONPATH=/fs01/home/merc0606/NSERC/LLaVA:$PYTHONPATH

# FIXED
SCANPATH_DIR=~/NSERC/data/scanpaths/cub_scanpaths
IMAGES_DIR=~/NSERC/data/images/CUB_200_2011/CUB_200_2011/images
CAPTIONS_FILE_PATH=~/NSERC/data/generated_captions/CUB_captions/CUB_captions.json

runs=(
    # RUN_NAME, TYPE, MARGIN, TRAJECTORY_MODE, TARGET_LAYER
    # "plain-old-prompt,None,0,0,45" 
    # "salient_post_sm_gaussian-old-prompt,salient-head,0,0," 
    # "pd,non-gaussian,0,0,"
    # "pdm,non-gaussian,1,0,"
    # "pdt,non-gaussian,0,1,"
    # "pdtm,non-gaussian,1,1,"
    # "gaussian,gaussian,0,0,"
    # "pd-new-prompt,non-gaussian,0,0,"
    # "pdm-new-prompt,non-gaussian,1,0,"
    # "pdt-new-prompt,non-gaussian,0,1,"
    # "pdtm-new-prompt,non-gaussian,1,1,"
    # "gaussian-new-prompt,gaussian,0,0,"
    # "salient_heads_with_zero_out,salient-head,0,0,"
    # "salient_heads_relative,salient-head,0,0,"
    "salient_heads_relative-k-8,salient-head,0,0,"
)

for run in "${runs[@]}"; do
    IFS="," read -r RUN_NAME TYPE MARGIN TRAJECTORY_MODE TARGET_LAYER <<< "$run"

    ANSWERS_FILE_PATH=~/NSERC/data/generated_captions/jul18_samples/generated_captions/cub/${RUN_NAME}.json
    WEIGHTS_DIR=~/NSERC/data/weights/cub/${RUN_NAME}
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

cd ~/NSERC
