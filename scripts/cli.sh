cd ../LLaVA

python -m llava.serve.cli \
 --model-path liuhaotian/llava-v1.5-7b \
 --load-4bit \
 --image-file ~/NSERC/samples/may26_samples/sampled_images_1000/000000179765.jpg\
 --scanpath ~/NSERC/samples/may26_samples/scanpaths/000000179765_scanpath.npy
#  --image-file ~/NSERC/images_captions_for_test/COCO_train2014_000000539984.jpg \
#  --scanpath ~/NSERC/images_captions_for_test_scanned/scanpaths/COCO_train2014_000000539984_scanpath.npy
#  --image-file ~/NSERC/images_captions_for_test/COCO_train2014_000000318556.jpg \
#  --scanpath ~/NSERC/images_captions_for_test_scanned/scanpaths/COCO_train2014_000000318556_scanpath.npy

cd ..


