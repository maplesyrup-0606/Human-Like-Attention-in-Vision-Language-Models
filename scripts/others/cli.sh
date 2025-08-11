# script for the LLaVA CLI tool

cd ~/NSERC/LLaVA

python -m llava.serve.cli \
 --model-path liuhaotian/llava-v1.5-7b \
 --load-4bit \
 --image-file ~/NSERC/data/images/MSCOCO_images/000000002149.jpg\
 --scanpath ~/NSERC/data/scanpaths/coco_scanpaths/000000002149_scanpath.npy
