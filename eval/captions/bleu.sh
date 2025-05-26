#!/bin/bash

REF_DIR=~/NSERC/samples/may22_samples/sampled_captions.json
# PRED_DIR=~/NSERC/samples/may22_samples/answered_captions.json
PRED_DIR=~/NSERC/samples/may22_samples/guided_answered_captions.json

python -m bleu "$REF_DIR" "$PRED_DIR"