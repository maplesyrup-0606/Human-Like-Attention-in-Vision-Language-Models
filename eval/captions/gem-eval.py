import os
import sys
sys.path.append("../../../GEM-metrics")

import gem_metrics

pred_path = os.path.expanduser("~/NSERC/data/generated_captions/gaussian_answered_captions_wordcap_gem.json")
ref_path = os.path.expanduser("~/NSERC/data/generated_captions/sampled_captions_gem.json")

preds = gem_metrics.texts.Predictions(pred_path)
refs = gem_metrics.texts.References(ref_path)

result = gem_metrics.compute(preds, refs, metrics_list=['bleu', 'rouge', 'bertscore'])

print(result)