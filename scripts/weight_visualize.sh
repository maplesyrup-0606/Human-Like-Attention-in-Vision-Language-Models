#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/output_logs/output-wv.log
#SBATCH --error=slurm_logs/error_logs/error-wv.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mercurymcindoe@gmail.com
source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate NSERC

cd ~/NSERC/data/scripts_for_data

IMAGE_DIR=~/NSERC/data/images
weight_paths=(
    # ~/NSERC/data/weights/patch_drop/0_weights.pt
    # ~/NSERC/data/weights/patch_drop_with_box/0_weights.pt
    # ~/NSERC/data/weights/patch_drop_with_trajectory/0_weights.pt
    # ~/NSERC/data/weights/patch_drop_with_trajectory_with_box/0_weights.pt
    # ~/NSERC/data/weights/gaussian/0_weights.pt
    # ~/NSERC/data/weights/plain/0_weights.pt
    # ~/NSERC/data/weights/pdt_later_inject/0_weights.pt
    ~/NSERC/data/weights/gaussian_later_inject/0_weights.pt
)

save_directories=(
    # ~/NSERC/data/weights/patch_drop/
    # ~/NSERC/data/weights/patch_drop_with_box/
    # ~/NSERC/data/weights/patch_drop_with_trajectory/
    # ~/NSERC/data/weights/patch_drop_with_trajectory_with_box/
    # ~/NSERC/data/weights/gaussian/
    # ~/NSERC/data/weights/plain/
    # ~/NSERC/data/weights/pdt_later_inject/
    ~/NSERC/data/weights/gaussian_later_inject/
)

for n in ${!weight_paths[@]}; do
    echo "Running on ${weight_paths[$n]} -> ${save_directories[$n]}"
    python weight_visualization.py "$IMAGE_DIR" "${weight_paths[$n]}" "${save_directories[$n]}"
done
