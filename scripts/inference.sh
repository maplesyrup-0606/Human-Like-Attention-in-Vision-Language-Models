#!/bin/bash

# IMAGE_DIR=~/NSERC/room.jpg
# IMAGE_SAVE_DIR=~/NSERC/room2.jpg
# SAVE_DIR=~/NSERC/scanpath.npy

# cd HAT
# python inference.py "$IMAGE_DIR" "$SAVE_DIR" "$IMAGE_SAVE_DIR"
# cd ..

IMAGE_DIR=~/NSERC/images_captions_for_test
SAVE_DIR=~/NSERC/scanpaths
IMAGE_SAVE_DIR=~/NSERC/images_captions_for_test_scanned

mkdir -p "$IMAGE_SAVE_DIR"
mkdir -p "$SAVE_DIR"

cd HAT

images=(
  "COCO_train2014_000000318556.jpg"
  "COCO_train2014_000000513461.jpg"
  "COCO_train2014_000000539984.jpg"
)

for filename in "${images[@]}"; do
    image_path="$IMAGE_DIR/$filename"
    image_stem="${filename%.*}"

    output_scanpath="$SAVE_DIR/${image_stem}_scanpath.npy"
    output_image="$IMAGE_SAVE_DIR/${image_stem}_output.jpg"

    python inference.py "$image_path" "$output_scanpath" "$output_image"
done

cd ..