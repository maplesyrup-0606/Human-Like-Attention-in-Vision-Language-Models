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
conda activate metrics

cd ~/NSERC/eval/captions/semantics-metrics

python gem-eval.py \
    --ref ~/NSERC/data/generated_captions/CUB_captions/CUB_captions.json \
    --pred-dir ~/NSERC/data/generated_captions/jun26_samples/generated_captions/cub \
    --save-dir ~/NSERC/data/generated_captions/jun26_samples/semantics