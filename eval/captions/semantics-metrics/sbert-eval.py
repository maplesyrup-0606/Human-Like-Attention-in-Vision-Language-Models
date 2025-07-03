#!/usr/bin/env python3
import argparse, os, json, gc
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


# --------------------------------------------------------------------------- #
#  Original math – kept exactly the same                                      #
# --------------------------------------------------------------------------- #
def sbert_f1_for_image(model: SentenceTransformer,
                       gt: List[str],
                       gen: List[str]) -> float:
    gt_emb = model.encode(gt)          # (m , d)
    gen_emb = model.encode(gen)        # (n , d)

    S = util.cos_sim(gt_emb, gen_emb).numpy()   # m × n
    P = S.max(axis=0).mean()                    # precision
    R = S.max(axis=1).mean()                    # recall
    return 2 * P * R / (P + R + 1e-12)          # F1


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",       required=True, help="Ground-truth caption JSON")
    parser.add_argument("--pred-dir",  required=True, help="Directory with *.json prediction files")
    parser.add_argument("--save-dir",  required=True, help="Where to store the plot + summary JSON")
    args = parser.parse_args()

    ref_path   = Path(os.path.expanduser(args.ref))
    pred_root  = Path(os.path.expanduser(args.pred_dir))
    save_root  = Path(os.path.expanduser(args.save_dir))
    save_root.mkdir(parents=True, exist_ok=True)

    print("Loading SBERT model …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- load reference captions -------------------------------------------
    ref_dict = json.load(ref_path.open())                  # {img_id: [gt1, …]}
    img_ids  = list(ref_dict.keys())

    # --- discover prediction files -----------------------------------------
    pred_files = sorted(pred_root.glob("*.json"))
    if not pred_files:
        raise FileNotFoundError(f"No *.json prediction files in {pred_root}")

    mean_f1, std_f1 = [], []
    methods         = []

    for pf in pred_files:
        method = pf.stem           # filename w/o .json
        methods.append(method)
        print(f"→ {method}")

        pred_dict = json.load(pf.open())                  # {img_id: [gen1, …]}
        common_ids = set(ref_dict) & set(pred_dict)
        if not common_ids:
            print(f"   ⚠️  no shared image-ids with {method}; skipped.")
            continue

        f1_scores = []
        for img_id in tqdm(sorted(common_ids), leave=False):
            f1 = sbert_f1_for_image(model,
                                    ref_dict[img_id],
                                    pred_dict[img_id])
            f1_scores.append(f1)
            gc.collect()

        f1_scores = np.array(f1_scores)
        mean_f1.append(f1_scores.mean())
        std_f1 .append(f1_scores.std())

    # --- save summary JSON --------------------------------------------------
    summary = {m: {"mean": float(mu), "std": float(sd)}
               for m, mu, sd in zip(methods, mean_f1, std_f1)}
    with (save_root / "sbert_f1_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    items = sorted(zip(methods, mean_f1, std_f1),
                   key=lambda t: t[1],    # sort by mean
                   reverse=True)          # highest first
    methods, mean_f1, std_f1 = zip(*items) 

    # --- bar plot -----------------------------------------------------------
    plt.figure(figsize=(max(6, 0.6 * len(methods)), 4))
    bars = plt.bar(methods, mean_f1, yerr=std_f1, capsize=5,
                   color="skyblue", edgecolor="black")

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean F1 similarity")
    plt.title("Sentence-BERT F1 comparison")
    for b, mu in zip(bars, mean_f1):
        plt.text(b.get_x()+b.get_width()/2, mu+0.003,
                 f"{mu:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout();  plt.grid(axis="y", ls="--", alpha=0.4)
    fig_path = save_root / "sbert_f1_comparison.png"
    plt.savefig(fig_path);  plt.close()
    print("Saved plot:", fig_path)


if __name__ == "__main__":
    main()