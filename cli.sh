cd LLaVA

python -m llava.serve.cli \
 --model-path liuhaotian/llava-v1.5-7b \
 --image-file ~/NSERC/room.jpg \
 --temperature 0 \
 --load-4bit \
 --scanpath ~/NSERC/scanpath.npy

cd ..