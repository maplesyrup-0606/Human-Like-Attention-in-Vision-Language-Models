#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:2
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=80G
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
which nvcc
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate internvl

# python gradioServer.py &
# SERVER_PID=$!

# until curl -s http://127.0.0.1:7860/; do
#     echo "Waiting for server to start"
#     sleep 2 
# done

# echo "Server launched, now running client"

# sleep 20

# python gradioClient.py "$GT_CAPTIONS" "$GEN_CAPTIONS" "$SAVE_DIR"

# kill $SERVER_PID

GT_CAPTIONS=~/NSERC/data/generated_captions/sampled_captions.json 
SAVE_DIR=~/NSERC/data/generated_captions/jun18_samples/
gen_captions=(
    # ~/NSERC/data/generated_captions/jun5_samples/plain_captions.json
    # ~/NSERC/data/generated_captions/jun5_samples/gaussian_captions.json
    # ~/NSERC/data/generated_captions/jun5_samples/patch_drop_captions.json
    # ~/NSERC/data/generated_captions/jun5_samples/patch_drop_with_trajectory_captions.json
    # ~/NSERC/data/generated_captions/jun5_samples/patch_drop_with_box_captions.json
    # ~/NSERC/data/generated_captions/jun5_samples/patch_drop_with_box_with_trajectory_captions.json
    # ~/NSERC/data/generated_captions/jun18_samples/generated_captions/pdt_later_inject_captions.json
    ~/NSERC/data/generated_captions/jun18_samples/generated_captions/gaussian_later_inject_captions.json
)

cd ~/NSERC/eval/captions/llm-as-a-judge

for GEN_CAPTIONS in "${gen_captions[@]}";
do 
    python qwen3B-judge.py "$GT_CAPTIONS" "$GEN_CAPTIONS" "$SAVE_DIR" || {
        echo "Failed on $GEN_CAPTIONS"
        exit 1
    }
done



