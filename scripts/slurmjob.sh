#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1 
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=output-captiongen.log
#SBATCH --error=error-captiongen.log 

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate NSERC
bash scanpath_generation.sh