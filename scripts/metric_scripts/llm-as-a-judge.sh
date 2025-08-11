#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:2
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=100G
#SBATCH --cpus-per-task=14
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/output_logs/output-llmjudge.log
#SBATCH --error=slurm_logs/error_logs/error-llmjudge.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mercurymcindoe@gmail.com

module load cuda-12.4
export CUDA_HOME=/pkgs/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH

echo "CUDA_HOME = $CUDA_HOME"
echo "SLURM_JOB_ID       = $SLURM_JOB_ID"
echo "SLURM_JOB_GPUS     = $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
which nvcc
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate internvl

GT_CAPTIONS=~/NSERC/data/generated_captions/CUB_captions/CUB_captions.json
SAVE_DIR=~/NSERC/data/generated_captions/jul18_samples/llm-judge-ratings
gen_captions=(
    # ~/NSERC/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_new_threshold.json
    # ~/NSERC/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_with_drop_out_captions.json
    # ~/NSERC/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_with_drop_out-k-4_captions.json
    # ~/NSERC/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_with_reverse_drop_out-k-4_captions.json
    # ~/NSERC/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads.json
    #~/NSERC/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_with_zero_out_captions.json
    #~/NSERC/data/generated_captions/jul18_samples/generated_captions/cub/plain_captions.json
    #~/NSERC/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_relative.json
    ~/NSERC/data/generated_captions/jul18_samples/generated_captions/cub/salient_heads_relative-k-8.json
    )

cd ~/NSERC/eval/captions/llm-as-a-judge

for GEN_CAPTIONS in "${gen_captions[@]}";
do 
    python caption-eval.py "$GT_CAPTIONS" "$GEN_CAPTIONS" "$SAVE_DIR" || {
        echo "Failed on $GEN_CAPTIONS"
        exit 1
    }
done



