#!/bin/bash

IMAGE_DIR=~/NSERC/room.jpg
IMAGE_SAVE_DIR=~/NSERC/room2.jpg
SAVE_DIR=~/NSERC/scanpath.npy

cd HAT
python inference.py "$IMAGE_DIR" "$SAVE_DIR" "$IMAGE_SAVE_DIR"
cd ..