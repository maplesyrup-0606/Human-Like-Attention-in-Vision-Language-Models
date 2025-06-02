#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/output_logs/output-clipsore-cuda.log
#SBATCH --error=logs/error_logs/error-clipscore-cuda.log 

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate NSERC

cd ~/NSERC/LLaVA/llava/eval

# IMAGE_DIR=~/NSERC/samples/may26_samples/sampled_images_1000
# CAPTIONS_DIR=~/NSERC/samples/may26_samples/answered_captions_wordcap.json
# SAVE_PATH=~/NSERC/samples/may26_samples/unguided_wordcap_clip

# python eval_clip_with_captions.py "$IMAGE_DIR" "$CAPTIONS_DIR" "$SAVE_PATH"

# IMAGE_DIR=~/NSERC/samples/may26_samples/sampled_images_1000
# CAPTIONS_DIR=~/NSERC/samples/may26_samples/guided_answered_captions_wordcap.json
# SAVE_PATH=~/NSERC/samples/may26_samples/guided_wordcap_clip

# python eval_clip_with_captions.py "$IMAGE_DIR" "$CAPTIONS_DIR" "$SAVE_PATH"


IMAGE_DIR=~/NSERC/samples/may26_samples/sampled_images_1000
CAPTIONS_DIR=~/NSERC/samples/may26_samples/gaussian_answered_captions_wordcap.json
SAVE_PATH=~/NSERC/samples/may26_samples/guassian_wordcap_clip

python eval_clip_with_captions.py "$IMAGE_DIR" "$CAPTIONS_DIR" "$SAVE_PATH"
