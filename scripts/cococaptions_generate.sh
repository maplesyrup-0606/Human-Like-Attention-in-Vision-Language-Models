#!/bin/bash

# Slurm directives
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --mem=40G
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/output_logs/output-laterinjectg.log
#SBATCH --error=slurm_logs/error_logs/error-laterinjectg.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mercurymcindoe@gmail.com

source /scratch/ssd004/scratch/merc0606/miniconda3/etc/profile.d/conda.sh
conda activate NSERC

cd ~/NSERC/LLaVA/llava/eval

export PYTHONPATH=/fs01/home/merc0606/NSERC/LLaVA:$PYTHONPATH

# FIXED
SCANPATH_DIR=~/NSERC/data/scanpaths
IMAGES_DIR=~/NSERC/data/images
CAPTIONS_FILE_PATH=~/NSERC/data/generated_captions/sampled_captions.json

# LOOP

##############1111111111#######################
# ANSWERS_FILE_PATH=~/NSERC/data/generated_captions/jun5_samples/plain_captions.json
# WEIGHTS_DIR=~/NSERC/data/weights/plain

# python -m model_cococaptions2017 \
#  --model-path liuhaotian/llava-v1.5-7b \
#  --load-4bit \
#  --temperature 0.8 \
#  --scanpath "$SCANPATH_DIR" \
#  --captions-file "$CAPTIONS_FILE_PATH" \
#  --images-dir "$IMAGES_DIR" \
#  --answers-file "$ANSWERS_FILE_PATH" \
#  --weights-dir "$WEIGHTS_DIR"

# cd ~/NSERC
##############1111111111#######################


##############2222222222#######################
# ANSWERS_FILE_PATH=~/NSERC/data/generated_captions/jun5_samples/patch_drop_with_trajectory_captions.json
# WEIGHTS_DIR=~/NSERC/data/weights/patch_drop_with_trajectory

# python -m model_cococaptions2017 \
#  --model-path liuhaotian/llava-v1.5-7b \
#  --load-4bit \
#  --temperature 0.8 \
#  --scanpath "$SCANPATH_DIR" \
#  --captions-file "$CAPTIONS_FILE_PATH" \
#  --images-dir "$IMAGES_DIR" \
#  --answers-file "$ANSWERS_FILE_PATH" \
#  --weights-dir "$WEIGHTS_DIR"


# ANSWERS_FILE_PATH=~/NSERC/data/generated_captions/jun5_samples/patch_drop_captions.json
# WEIGHTS_DIR=~/NSERC/data/weights/patch_drop

# python -m model_cococaptions2017 \
#  --model-path liuhaotian/llava-v1.5-7b \
#  --load-4bit \
#  --temperature 0.8 \
#  --scanpath "$SCANPATH_DIR" \
#  --captions-file "$CAPTIONS_FILE_PATH" \
#  --images-dir "$IMAGES_DIR" \
#  --answers-file "$ANSWERS_FILE_PATH" \
#  --weights-dir "$WEIGHTS_DIR"

# cd ~/NSERC
##############2222222222#######################



##############3333333333#######################
# ANSWERS_FILE_PATH=~/NSERC/data/generated_captions/jun5_samples/patch_drop_with_box_captions.json
# WEIGHTS_DIR=~/NSERC/data/weights/patch_drop_with_box

# python -m model_cococaptions2017 \
#  --model-path liuhaotian/llava-v1.5-7b \
#  --load-4bit \
#  --temperature 0.8 \
#  --scanpath "$SCANPATH_DIR" \
#  --captions-file "$CAPTIONS_FILE_PATH" \
#  --images-dir "$IMAGES_DIR" \
#  --answers-file "$ANSWERS_FILE_PATH" \
#  --weights-dir "$WEIGHTS_DIR"


# ANSWERS_FILE_PATH=~/NSERC/data/generated_captions/jun5_samples/patch_drop_with_box_with_trajectory_captions.json
# WEIGHTS_DIR=~/NSERC/data/weights/patch_drop_with_trajectory_with_box

# python -m model_cococaptions2017 \
#  --model-path liuhaotian/llava-v1.5-7b \
#  --load-4bit \
#  --temperature 0.8 \
#  --scanpath "$SCANPATH_DIR" \
#  --captions-file "$CAPTIONS_FILE_PATH" \
#  --images-dir "$IMAGES_DIR" \
#  --answers-file "$ANSWERS_FILE_PATH" \
#  --weights-dir "$WEIGHTS_DIR"

# cd ~/NSERC
##############3333333333#######################

##############4444444444#######################
# ANSWERS_FILE_PATH=~/NSERC/data/generated_captions/jun5_samples/gaussian_captions.json
# WEIGHTS_DIR=~/NSERC/data/weights/gaussian

# python -m model_cococaptions2017 \
#  --model-path liuhaotian/llava-v1.5-7b \
#  --load-4bit \
#  --temperature 0.8 \
#  --scanpath "$SCANPATH_DIR" \
#  --captions-file "$CAPTIONS_FILE_PATH" \
#  --images-dir "$IMAGES_DIR" \
#  --answers-file "$ANSWERS_FILE_PATH" \
#  --weights-dir "$WEIGHTS_DIR"

# cd ~/NSERC
##############4444444444#######################

ANSWERS_FILE_PATH=~/NSERC/data/generated_captions/jun18_samples/generated_captions/gaussian_later_inject.json
WEIGHTS_DIR=~/NSERC/data/weights/gaussian_later_inject
python -m model_cococaptions2017 \
 --model-path liuhaotian/llava-v1.5-7b \
 --load-4bit \
 --temperature 0.8 \
 --scanpath "$SCANPATH_DIR" \
 --captions-file "$CAPTIONS_FILE_PATH" \
 --images-dir "$IMAGES_DIR" \
 --answers-file "$ANSWERS_FILE_PATH" \
 --weights-dir "$WEIGHTS_DIR" \
 --trajectory 0

cd ~/NSERC