#!/bin/bash

cd ~/NSERC/LLaVA/llava/eval

python -m model_cococaptions2017 \
 --model-path liuhaotian/llava-v1.5-7b \
 --load-4bit \
 --temperature 0.4 \
 --scanpath ~/NSERC/samples/may22_samples/scanpaths \
 --captions-file ~/NSERC/samples/may22_samples/sampled_captions.json \
 --answers-file ~/NSERC/samples/may22_samples/answered_captions.json \
 --images-dir ~/NSERC/samples/may22_samples/sampled_images

cd ~/NSERC

# 1000 images

# cd ~/NSERC/LLaVA/llava/eval

# python -m model_cococaptions2017 \
#  --model-path liuhaotian/llava-v1.5-7b \
#  --load-4bit \
#  --temperature 0.4 \
#  --scanpath ~/NSERC/samples/may26_samples/scanpaths \
#  --captions-file ~/NSERC/samples/may26_samples/sampled_captions_1000.json \
#  --answers-file ~/NSERC/samples/may26_samples/answered_captions.json \
#  --images-dir ~/NSERC/samples/may26_samples/sampled_images_1000

# cd ~/NSERC