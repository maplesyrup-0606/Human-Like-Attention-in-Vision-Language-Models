#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/output_logs/output-clipsore.log
#SBATCH --error=slurm_logs/error_logs/error-clipscore.log 

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate clip

cd ~/NSERC/eval/captions/semantics-metrics

# python clip-eval.py \
#     --images ~/NSERC/data/images/CUB_200_2011/CUB_200_2011/images \
#     --captions-dir ~/NSERC/data/generated_captions/jun26_samples/generated_captions/cub \
#     --save-dir ~/NSERC/data/generated_captions/jun26_samples/semantics \
#     --ground-truth-captions ~/NSERC/data/generated_captions/CUB_captions/CUB_captions.json
python clip-eval.py \
    --images ~/NSERC/data/images/MSCOCO_images \
    --captions-dir ~/NSERC/data/generated_captions/jul18_samples/generated_captions/mscoco \
    --save-dir ~/NSERC/data/generated_captions/jul18_samples/semantics/mscoco \
    --ground-truth-captions ~/NSERC/data/generated_captions/sampled_captions.json