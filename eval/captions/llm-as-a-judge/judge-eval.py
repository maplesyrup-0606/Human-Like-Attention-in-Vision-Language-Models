import os 
import json
import numpy as np
import matplotlib.pyplot as plt

def extract_scores(rating_file_path) :
    with open(os.path.expanduser(rating_file_path)) as f :
        ratings = json.load(f)
    
    scores = []
    invalid_count = 0
    for _, id_ratings in ratings.items(): 
        max_rating = -1

        for rating in id_ratings :
            if rating != "INVALID" :
                idx = rating.find("Total rating: ")
                score = int(rating[idx + len("Total rating: "):])

                max_rating = max(max_rating, score)
            else :
                invalid_count += 1
        if max_rating != -1 :
            scores.append(max_rating)
            
    return np.array(scores), invalid_count

def compare_judgements(rating_file_paths): 
    n = len(rating_file_paths)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 3.5 * n), sharex=True)
    # fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(4 * n, 4), sharey=True)

    if n == 1:
        axes = [axes]

    bins = [0.5 + i for i in range(6)]
    colors = ["blue", "orange", "green", "purple", "red", "cyan", "magenta"]

    for idx, path in enumerate(rating_file_paths) :
        file_name = os.path.splitext(os.path.basename(path))[0]
        scores, invalids = extract_scores(path)

        mean = np.mean(scores)
        median = np.median(scores)
        std = np.std(scores)

        ax = axes[idx]
        ax.hist(scores, bins=bins, color=colors[idx % len(colors)], edgecolor="black", align="mid", alpha=0.7)
        ax.axvline(mean, color='red', linestyle='dashed', linewidth=1.2)

        ax.text(
            0.98, 0.95, f"INVALIDs: {invalids}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.7)
        )

        ax.set_ylabel("Frequency")
        ax.set_title(f"{file_name} | Mean: {mean:.2f}, Median: {median:.2f}, Std: {std:.2f}")
        ax.grid(True)

    axes[-1].set_xlabel("Score")
    plt.xticks(range(1, 6))

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.expanduser(rating_file_paths[0])), "all_histograms_subplot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to: {save_path}")

if __name__ == "__main__" :
    file_paths = [
        "~/NSERC/data/generated_captions/jun18_samples/llm-judge-ratings/gaussian_later_inject_captions_ratings.json",
        "~/NSERC/data/generated_captions/jun18_samples/llm-judge-ratings/pdt_later_inject_captions_ratings.json",
        "~/NSERC/data/generated_captions/jun5_samples/llm-judge-ratings/gaussian_captions_ratings.json",
        "~/NSERC/data/generated_captions/jun5_samples/llm-judge-ratings/patch_drop_captions_ratings.json",
        "~/NSERC/data/generated_captions/jun5_samples/llm-judge-ratings/patch_drop_with_box_captions_ratings.json", 
        "~/NSERC/data/generated_captions/jun5_samples/llm-judge-ratings/patch_drop_with_box_with_trajectory_captions_ratings.json",
        "~/NSERC/data/generated_captions/jun5_samples/llm-judge-ratings/patch_drop_with_trajectory_captions_ratings.json",
        "~/NSERC/data/generated_captions/jun5_samples/llm-judge-ratings/plain_captions_ratings.json",
    ]

    compare_judgements(file_paths)