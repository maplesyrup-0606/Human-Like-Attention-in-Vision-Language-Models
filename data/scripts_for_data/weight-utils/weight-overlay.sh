python head-wise-weight-overlay.py \
  --weights ~/NSERC/data/weights/mscoco/plain \
  --save-dir ~/NSERC/vis/mosaics-norm \
  --layers 19-26 \
  --grid-size 24 \
  --token-start 5 --token-end 100 \
  --agg mean \
  --tile-size 96 \
  --sample-images 8
