# import os 
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def extract_scores(rating_file_path) :
#     with open(os.path.expanduser(rating_file_path)) as f :
#         ratings = json.load(f)
    
#     scores = []
#     invalid_count = 0
#     for _, id_ratings in ratings.items(): 
#         max_rating = -1

#         for rating in id_ratings :
#             if rating != "INVALID" :
#                 idx = rating.find("Total rating: ")
#                 score = int(rating[idx + len("Total rating: "):])

#                 max_rating = max(max_rating, score)
#             else :
#                 invalid_count += 1
#         if max_rating != -1 :
#             scores.append(max_rating)
            
#     return np.array(scores), invalid_count

# def compare_judgements(rating_file_paths, sort_by="mean", descending=True):
#     """Draw one histogram per ratings-file, ordered by a chosen statistic."""
#     # ── 1. Collect stats for every file ──────────────────────────────────
#     entries = []
#     save_json = {}
#     for path in rating_file_paths:
#         file_name = os.path.splitext(os.path.basename(path))[0]
#         scores, invalids = extract_scores(path)

#         stats = {
#             "path": path,
#             "file_name": file_name,
#             "scores": scores,
#             "invalids": invalids,
#             "mean":   np.mean(scores),
#             "median": np.median(scores),
#             "std":    np.std(scores)
#         }
#         entries.append(stats)
#         save_json[file_name] = {
#         'mean' : float(np.mean(scores))
#         }     

#     # ── 2. Sort by the requested key  ───────────────────────────────────
#     entries.sort(key=lambda e: e[sort_by], reverse=descending)
    
#     # ── 3. Plot in the sorted order  ────────────────────────────────────
#     n = len(entries)
#     fig, axes = plt.subplots(nrows=n, ncols=1,
#                              figsize=(10, 3.5 * n), sharex=True)
#     if n == 1:
#         axes = [axes]

#     bins   = [0.5 + i for i in range(6)]
#     colors = ["blue", "orange", "green", "purple", "red", "cyan", "magenta"]

#     for idx, e in enumerate(entries):
#         ax   = axes[idx]
#         col  = colors[idx % len(colors)]

#         ax.hist(e["scores"], bins=bins, color=col,
#                 edgecolor="black", alpha=0.7, align="mid")
#         ax.axvline(e["mean"], color="red", linestyle="dashed", linewidth=1.2)

#         ax.text(0.98, 0.95, f"INVALIDs: {e['invalids']}",
#                 transform=ax.transAxes, ha="right", va="top",
#                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
#                                        edgecolor="gray", facecolor="white",
#                                        alpha=0.7))

#         ax.set_ylabel("Frequency")
#         ax.set_title(f"{e['file_name']} | "
#                      f"Mean: {e['mean']:.2f}, "
#                      f"Median: {e['median']:.2f}, "
#                      f"Std: {e['std']:.2f}")
#         ax.grid(True)

#     axes[-1].set_xlabel("Score")
#     plt.xticks(range(1, 6))

#     plt.tight_layout()
#     save_path = (Path(rating_file_paths[0]).expanduser()
#                  .parent / "all_histograms_subplot.png")
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Saved to: {save_path}")

#     json_save_path = (Path(rating_file_paths[0]).expanduser()
#                  .parent / "llm-judge_summary.json")

#     json_save_path.parent.mkdir(parents=True, exist_ok=True)

#     with open(json_save_path, "w") as f : 
#         json.dump(save_json, f, indent=2)
#     print(f"Saved Summary to : {json_save_path}")



# if __name__ == "__main__" :
#     files = Path("~/NSERC/data/generated_captions/jun26_samples/llm-judge-ratings").expanduser()

#     file_paths = sorted(
#         str(p)
#         for p in files.glob("*_ratings.json")
#         if "summary" not in p.name
#     )       

#     compare_judgements(file_paths)

import json, torch, csv
from pathlib import Path
import matplotlib.pyplot as plt
import re

PATTERN = re.compile(r'Total\s*rating:\s*(\d+)', re.IGNORECASE)

def get_scores(rating_path):
    with open(rating_path, "r", encoding="utf-8") as f:
        ratings = json.load(f)

    vals = []
    for img_id, evaluation in ratings.items():
        evaluation = evaluation[0]
        if not isinstance(evaluation, str):
            continue
        m = PATTERN.search(evaluation)
        if m:
            vals.append(int(m.group(1)))

    if not vals:
        return float("nan"), float("nan")

    t = torch.tensor(vals, dtype=torch.float32)
    return t.mean().item(), t.std(unbiased=False).item()  # population std; use unbiased=True for sample std

def main():
    root = Path("~/Human-Like-Attention-in-Vision-Language-Models/data/generated_captions/jul18_samples/llm-judge-ratings").expanduser()
    
    rows = []
    for p in root.rglob("*_ratings.json"):
        mean, std = get_scores(p)
        method = p.stem.replace("_ratings", "")
        rows.append((method, mean, std))

    # sort by mean desc
    rows.sort(key=lambda x: x[1], reverse=True)

    methods  = [r[0] for r in rows]
    means    = [r[1] for r in rows]
    stds     = [r[2] for r in rows]

    csv_path = root / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method","Mean","Std"])
        writer.writerows(rows)
    print(f"Saved CSV results to {csv_path}")
    # plot
    plt.figure(figsize=(10, 5))
    x = range(len(methods))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, methods, rotation=45, ha="right")
    plt.ylabel("Mean rating")
    plt.title("LLM-judge ratings by method (descending mean)")
    plt.tight_layout()
    plt.savefig(root / "results.png", dpi=300)

if __name__ == "__main__":
    main()
