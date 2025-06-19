#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/output_logs/plain.log
#SBATCH --error=slurm_logs/error_logs/plain.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mercurymcindoe@gmail.com

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate NSERC

cd ~/NSERC/LLaVA/llava/eval

export PYTHONPATH=/fs01/home/merc0606/NSERC/LLaVA:$PYTHONPATH

# FIXED
SCANPATH_DIR=~/NSERC/data/scanpaths
IMAGES_DIR=~/NSERC/data/images
CAPTIONS_FILE_PATH=~/NSERC/data/generated_captions/sampled_captions.json

ANSWERS_FILE_PATH=~/NSERC/data/generated_captions/jun5_samples/gaussian_captions.json
WEIGHTS_DIR=~/NSERC/data/weights/gaussian

python -m model_cococaptions2017 \
 --model-path liuhaotian/llava-v1.5-7b \
 --load-4bit \
 --temperature 0.8 \
 --scanpath "$SCANPATH_DIR" \
 --captions-file "$CAPTIONS_FILE_PATH" \
 --images-dir "$IMAGES_DIR" \
 --answers-file "$ANSWERS_FILE_PATH" \
 --weights-dir "$WEIGHTS_DIR"