#!/bin/bash
cd ..

# IMAGE_DIR=~/NSERC/samples/may22_samples/sampled_images
# SAVE_DIR=~/NSERC/samples/may22_samples/scanpaths
# IMAGE_SAVE_DIR=~/NSERC/samples/may22_samples/sampled_images_with_scanpaths
IMAGE_DIR=~/NSERC/samples/may26_samples/sampled_images_1000
SAVE_DIR=~/NSERC/samples/may26_samples/scanpaths
IMAGE_SAVE_DIR=~/NSERC/samples/may26_samples/sampled_images_with_scanpaths

mkdir -p "$SAVE_DIR"
mkdir -p "$IMAGE_SAVE_DIR"

cd HAT

for img in "$IMAGE_DIR"/*.jpg; do
    filename=$(basename "$img")
    image_stem="${filename%.*}"
    output_scanpath="$SAVE_DIR/${image_stem}_scanpath.npy"
    output_image="$IMAGE_SAVE_DIR/${image_stem}_output.jpg"

    python inference.py "$img" "$output_scanpath" "$output_image"
done