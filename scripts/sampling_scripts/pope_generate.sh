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

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate NSERC

#export PYTHONPATH=$HOME/NSERC/HAT:$PYTHONPATH
NSERC_ROOT="$HOME/NSERC"
LLAVA_ROOT="$NSERC_ROOT/LLaVA"
HAT_ROOT="$NSERC_ROOT/HAT"

# --- make BOTH packages importable (order matters: put parents first)
export PYTHONPATH="$NSERC_ROOT:$HAT_ROOT:$LLAVA_ROOT:${PYTHONPATH:-}"
cd $HOME/NSERC/LLaVA

# FIXED
SCANPATH_DIR=~/NSERC/data/scanpaths/cub_scanpaths
IMAGES_DIR=~/NSERC/data/images/MSCOCO2014/val2014
CAPTIONS_FILE_PATH=~/NSERC/data/generated_captions/CUB_captions/CUB_captions.json
QUESTIONS_FILE_PATH=~/NSERC/eval/captions/hallucination/POPE/output/coco/coco_pope_random.jsonl

runs=(
    # RUN_NAME, TYPE, MARGIN, TRAJECTORY_MODE, TARGET_LAYER
#    "baseline,,0,0,"
#    "salient-head,salient-head,0,0,"
	"baseline-13b,,0,0,"
	"salient-head-13b,salient-head,0,0,"
)


for run in "${runs[@]}"; do
    IFS="," read -r RUN_NAME TYPE MARGIN TRAJECTORY_MODE TARGET_LAYER <<< "$run"

    ANSWERS_FILE_PATH=~/NSERC/data/generated_captions/jul30_samples/generated_captions/POPE/MSCOCO2014/${RUN_NAME}_answers.jsonl
    WEIGHTS_DIR=~/NSERC/data/weights/cub/${RUN_NAME}
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

cd ~/NSERC
