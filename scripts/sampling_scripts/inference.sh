#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=40G
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mercurymcindoe@gmail.com
#SBATCH --output=slurm_logs/output_logs/output-HAT.log
#SBATCH --error=slurm_logs/error_logs/error-HAT.log

set -euo pipefail

# --------------------------------------------------------------------
# Paths (convert to absolute to make prefix-stripping reliable)
# --------------------------------------------------------------------
IMAGE_ROOT=$(realpath ~/NSERC/data/CUB_200_2011/CUB_200_2011/images)
SCAN_ROOT=$(realpath -m ~/NSERC/data/scanpaths/cub_scanpaths)              # *.npy
VIS_ROOT=$(realpath -m ~/NSERC/data/scanpaths/cub_scanpaths/images_with_scanpaths)       # *_output.jpg

mkdir -p "$SCAN_ROOT" "$VIS_ROOT"

# ── activate env and cd into repo once ────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate NSERC
cd ~/NSERC/HAT

python inference.py "$IMAGE_ROOT" "$SCAN_ROOT" "$VIS_ROOT"
