import os 
import json 
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm

print("Loading Model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
caption_file_paths = [
    os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/generated_captions/gaussian_captions.json"),
    os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/generated_captions/patch_drop_captions.json"),
    os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/generated_captions/patch_drop_with_box_captions.json"),
    os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/generated_captions/patch_drop_with_box_with_trajectory_captions.json"),
    os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/generated_captions/patch_drop_with_trajectory_captions.json"),
    os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/generated_captions/plain_captions.json"),
    os.path.expanduser("~/NSERC/data/generated_captions/jun18_samples/generated_captions/pdt_later_inject_captions.json"),
    os.path.expanduser("~/NSERC/data/generated_captions/jun18_samples/generated_captions/gaussian_later_inject_captions.json"),
]
print("Model Loaded!")

ground_truth_path = os.path.expanduser("~/NSERC/data/generated_captions/sampled_captions.json")

gt_dict = json.load(open(ground_truth_path,"r"))
image_ids = gt_dict.keys()

method_names = [os.path.splitext(os.path.basename(p))[0] for p in caption_file_paths]
means = []
stds = []
medians = []

for caption_file_path in caption_file_paths :
    gen_dict = json.load(open(caption_file_path, "r"))

    all_scores = []
    for image_id in tqdm(image_ids) :
        gt = gt_dict[image_id]
        gen = gen_dict[image_id]

        gt_embeddings = model.encode(gt)
        gen_embeddings = model.encode(gen)

        similarities = model.similarity(gt_embeddings, gen_embeddings)

        S = np.array(similarities)

        P = S.max(axis=0).mean() # Precision
        R = S.max(axis=1).mean() # Recall

        F1 = 2 * P * R / (P + R) # F1 Score
        all_scores.append(F1)

    all_scores = np.array(all_scores)
    means.append(np.mean(all_scores))
    stds.append(np.std(all_scores))
    medians.append(np.median(all_scores))

plt.figure(figsize=(10, 6))
bars = plt.bar(method_names, means, yerr=stds, capsize=5, color="skyblue", edgecolor="black")

for bar, mean in zip(bars, means):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f"{mean:.3f}", ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=30, ha="right")
plt.ylabel("Mean F1 Similarity Score")
plt.title("Sentence Transformer Similarity")
plt.tight_layout()

save_path = os.path.expanduser("~/NSERC/data/generated_captions/jun18_samples/semantics/sbert_f1_comparison.png")
plt.savefig(save_path)
plt.close()
print(f"Saved plot to: {save_path}")