cd ../LLaVA

python -m llava.serve.cli \
 --model-path liuhaotian/llava-v1.5-7b \
 --load-4bit \
 --image-file ~/NSERC/data/images/000000190236.jpg\
 --scanpath ~/NSERC/data/scanpaths/000000190236_scanpath.npy