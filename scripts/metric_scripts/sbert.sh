#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/output_logs/output-sbert.log
#SBATCH --error=logs/error_logs/error-sbert.log 

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate sentence-trnsfr

cd ~/NSERC/eval/captions/semantics-metrics

python sbert-eval.py \
    --ref ~/NSERC/data/generated_captions/CUB_captions/CUB_captions.json \
    --pred-dir ~/NSERC/data/generated_captions/jun26_samples/generated_captions/cub \
    --save-dir ~/NSERC/data/generated_captions/jun26_samples/semantics