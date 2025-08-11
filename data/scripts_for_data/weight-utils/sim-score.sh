
RUN_NAME=salient_heads_relative-k-8

python head-sim-score.py \
  --weights ~/NSERC/data/weights/mscoco/"$RUN_NAME" \
  --save-dir ~/NSERC/vis/mscoco/"$RUN_NAME"/head-sim-scores-heat-map \
  --layers 19-26 \
  --sigma 1.5 \
  --use-q-rule \
  --sample-images 8
