import os
import tempfile
import sys
import json
import matplotlib.pyplot as plt
sys.path.append("../../../../GEM-metrics")
import argparse
from pathlib import Path
import gem_metrics

from typing import Dict, Tuple, List

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

    common_ids = set(ref_captions) & set(pred_captions)
    if not common_ids:
        raise ValueError("No overlapping image IDs between reference and predictions")

    for img_id in common_ids:
        gt_list = ref_captions[img_id]         # list[str]
        preds   = pred_captions[img_id]        # list[str]
        k       = len(preds)

        # deterministically repeat / truncate the GT list to length k
        if len(gt_list) >= k:
            refs_for_id = gt_list[:k]
        else:
            repeats, remain = divmod(k, len(gt_list))
            refs_for_id = gt_list * repeats + gt_list[:remain]

        # add to GEM containers
        ref_gem["values"].extend({"target": ref} for ref in refs_for_id)
        pred_gem["values"].extend(preds)

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

        result = gem_metrics.compute(pred_gem_t, ref_gem_t, metrics_list=['bleu', 'rouge', 'bertscore', 'cider'])

    # Remove temp files
    os.remove(ref_temp.name)
    os.remove(pred_temp.name)

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",        required=True,
                        help="Ground-truth caption JSON (one caption per image‐id)")
    parser.add_argument("--pred-dir",   required=True,
                        help="Directory containing *.json prediction files to evaluate")
    parser.add_argument("--save-dir",   required=True,
                        help="Where to write gem_summary.json and plots")
    args = parser.parse_args()

    ref_path   = Path(os.path.expanduser(args.ref))
    pred_root  = Path(os.path.expanduser(args.pred_dir))
    save_root  = Path(os.path.expanduser(args.save_dir))
    save_root.mkdir(parents=True, exist_ok=True)

    # ----- load reference once ------------------------------------------------
    with ref_path.open() as f:
        ref_dict = json.load(f)

    # ----- discover prediction files -----------------------------------------
    pred_files = sorted(pred_root.glob("*.json"))
    if not pred_files:
        raise FileNotFoundError(f"No *.json caption files found in {pred_root}")

    ret_dict = {}
    for pf in pred_files:
        method = pf.stem                      # filename without .json
        print(f"→ {method}")

        with pf.open() as f:
            pred_dict = json.load(f)
        
        ref_gem, pred_gem = format_for_gem(ref_dict, pred_dict)
        result = run_gem_eval_with_tempfiles(ref_gem, pred_gem)
        ret_dict[method] = result

    # ----- dump summary -------------------------------------------------------
    with (save_root / "gem_summary.json").open("w") as f:
        json.dump(ret_dict, f, indent=2)

    # ----- plotting -----------------------------------------------------------
    metric_keys = {
        "bleu":       ("BLEU",              lambda r: r["bleu"]),
        "bertscore":  ("BERTScore (F1)",    lambda r: r["bertscore"]["f1"]),
        "rouge1":     ("ROUGE-1 (F1)",      lambda r: r["rouge1"]["fmeasure"]),
        "rouge2":     ("ROUGE-2 (F1)",      lambda r: r["rouge2"]["fmeasure"]),
        "rougeL":     ("ROUGE-L (F1)",      lambda r: r["rougeL"]["fmeasure"]),
        "cider":      ("CIDEr",             lambda r: r["CIDEr"]),
    }

    for key, (title, getter) in metric_keys.items():
        pairs = [(m, getter(ret_dict[m])) for m in ret_dict]      # list of tuples
        pairs.sort(key=lambda t: t[1], reverse=True)              # ↓  highest-to-lowest
        methods, vals = zip(*pairs)  

        plt.figure(figsize=(max(6, 0.6*len(methods)), 4))
        bars = plt.bar(methods, vals, color="skyblue", edgecolor="black")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Score");   plt.title(f"{title} comparison")

        for b, v in zip(bars, vals):
            plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.004,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout();  plt.grid(axis="y", ls="--", alpha=0.4)
        plt.savefig(save_root / f"{key}_comparison.png")
        plt.close()
        print("Saved plot:", save_root / f"{key}_comparison.png")


if __name__ == "__main__" :
    main()


