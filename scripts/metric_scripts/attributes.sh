#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/output_logs/output-attributes.log
#SBATCH --error=slurm_logs/error_logs/error-attributes.log 

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate clip

cd ~/NSERC/eval/captions/semantics-metrics

python attribute-eval.py \
    --attributes ~/NSERC/eval/captions/attributes.json \
    --captions-dir ~/NSERC/data/generated_captions/jun26_samples/generated_captions/cub \
    --save-dir ~/NSERC/data/generated_captions/jun26_samples/semantics