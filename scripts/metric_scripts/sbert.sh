#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/output_logs/output-sbert.log
#SBATCH --error=slurm_logs/error_logs/error-sbert.log 

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate sentence-trnsfr

cd ~/NSERC/eval/captions/semantics

python sbert-eval.py \
    --ref ~/NSERC/data/generated_captions/CUB_captions/CUB_captions.json \
    --pred-dir ~/NSERC/data/generated_captions/jul18_samples/generated_captions/cub \
    --save-dir ~/NSERC/data/generated_captions/jul18_samples/semantics/cub
# python sbert-eval.py \
#     --ref ~/NSERC/data/generated_captions/sampled_captions.json \
#     --pred-dir ~/NSERC/data/generated_captions/jul18_samples/generated_captions/mscoco \
#     --save-dir ~/NSERC/data/generated_captions/jul18_samples/semantics/mscoco
