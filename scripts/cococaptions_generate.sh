#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:2
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/output-captiongen_gaussian.log
#SBATCH --error=slurm_logs/error-captiongen_gaussian.log 

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate NSERC


cd ~/NSERC/LLaVA/llava/eval

export PYTHONPATH=/fs01/home/merc0606/NSERC/LLaVA:$PYTHONPATH

python -m model_cococaptions2017 \
 --model-path liuhaotian/llava-v1.5-7b \
 --load-4bit \
 --temperature 0.8 \
 --scanpath ~/NSERC/samples/may26_samples/scanpaths \
 --captions-file ~/NSERC/samples/may26_samples/sampled_captions_1000.json \
 --images-dir ~/NSERC/samples/may26_samples/sampled_images_1000 \
 --answers-file ~/NSERC/samples/may26_samples/gaussian_answered_captions_wordcap.json

cd ~/NSERC