import os
import tempfile
import sys
import json
import matplotlib.pyplot as plt
sys.path.append("../../../../GEM-metrics")

import gem_metrics

from typing import Dict, Tuple

def format_for_gem(ref_captions: Dict, pred_captions: Dict) -> Tuple[Dict, Dict]:
    """
    Formats ground truth and predicted captions into GEM format.

    Args:
        ref_captions (dict): A dictionary mapping image IDs to ground truth captions (string or list of strings).
        pred_captions (dict): A dictionary mapping image IDs to a list of generated captions (strings).

    Returns:
        tuple: (formatted_references, formatted_predictions), each a dict with GEM schema.
    """
    ref_gem = {
        "values": [],
        "language": "en"
    }

    pred_gem = {
        "values": [],
        "language": "en"
    }

    for image_id, gt_caption in ref_captions.items():
        # Assume each ID has one GT caption; repeat it 4 times for comparison
        for _ in range(4):
            ref_gem["values"].append({"target": gt_caption})

    for image_id, preds in pred_captions.items():
        for pred in preds:
            pred_gem["values"].append(pred)

    return ref_gem, pred_gem

def run_gem_eval_with_tempfiles(ref_gem, pred_gem):
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as ref_temp, \
         tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as pred_temp:

        # Dump GEM-formatted data into the temporary files
        json.dump(ref_gem, ref_temp)
        json.dump(pred_gem, pred_temp)
        ref_temp.flush()
        pred_temp.flush()

        pred_gem_t = gem_metrics.texts.Predictions(pred_temp.name)
        ref_gem_t = gem_metrics.texts.References(ref_temp.name)

        result = gem_metrics.compute(pred_gem_t, ref_gem_t, metrics_list=['bleu', 'rouge', 'bertscore'])

    # Remove temp files
    os.remove(ref_temp.name)
    os.remove(pred_temp.name)

    return result

def main() :
    ref_path = os.path.expanduser("~/NSERC/data/generated_captions/sampled_captions.json")
    pred_paths = [
        os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/generated_captions/gaussian_captions.json"),
        os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/generated_captions/patch_drop_captions.json"),
        os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/generated_captions/patch_drop_with_box_captions.json"),
        os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/generated_captions/patch_drop_with_box_with_trajectory_captions.json"),
        os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/generated_captions/patch_drop_with_trajectory_captions.json"),
        os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/generated_captions/plain_captions.json"),
        os.path.expanduser("~/NSERC/data/generated_captions/jun18_samples/generated_captions/pdt_later_inject_captions.json"),
        os.path.expanduser("~/NSERC/data/generated_captions/jun18_samples/generated_captions/gaussian_later_inject_captions.json"),
    ]

    ref_dict = json.load(open(ref_path, "r"))
    save_dir = os.path.expanduser("~/NSERC/data/generated_captions/jun18_samples/semantics")
    os.makedirs(save_dir, exist_ok=True)

    ret_dict = {}
    for pred_path in pred_paths :
        pred_dict = json.load(open(pred_path, "r"))    
        base_name = os.path.basename(pred_path)
 
        file_name = os.path.splitext(base_name)[0]
        ref_gem, pred_gem = format_for_gem(ref_captions=ref_dict, pred_captions=pred_dict)

        result = run_gem_eval_with_tempfiles(ref_gem, pred_gem)
        ret_dict[f"{file_name}"] = result

    json.dump(ret_dict, open(os.path.join(save_dir, "gem_summary.json"), "w"), indent=2)
    
    # now let's visualize
    metric_keys = {
        "bleu": "BLEU",
        "bertscore_f1": "BERTScore (F1)",
        "rouge1": "ROUGE-1 (F1)",
        "rouge2": "ROUGE-2 (F1)",
        "rougeL": "ROUGE-L (F1)"
    }

    scores_by_metric = {k : {} for k in metric_keys}
    
    for method_name, result in ret_dict.items() :
        scores_by_metric["bleu"][method_name] = result["bleu"]
        scores_by_metric["bertscore_f1"][method_name] = result["bertscore"]["f1"]
        scores_by_metric["rouge1"][method_name] = result["rouge1"]["fmeasure"]
        scores_by_metric["rouge2"][method_name] = result["rouge2"]["fmeasure"]
        scores_by_metric["rougeL"][method_name] = result["rougeL"]["fmeasure"]

    for metric_key, display_name in metric_keys.items() :
        methods = list(scores_by_metric[metric_key].keys())
        values = [scores_by_metric[metric_key][m] for m in methods]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(methods, values, color="skyblue", edgecolor="black")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Score")
        plt.title(f"{display_name} Comparison Across Models")

        # Annotate scores
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        save_path = os.path.join(save_dir, f"{metric_key}_comparison.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__" :
    main()


