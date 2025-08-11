import argparse, os, json, gc, glob, math, torch
from pathlib import Path
from typing import List
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("➍ creating metric …")
metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14-336").to(device)
metric.eval()    
print("➍ metric ready")

@torch.inference_mode()
def max_clip_for_one_image(image_path:Path,
                           cand_captions: List[str],    
                           gt_captions: List[str],
                           device: torch.device)-> float:
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.functional.pil_to_tensor(img).unsqueeze(0).to(device)
    
    best_score = float('-inf')

    for cand in cand_captions :
        img_score = metric(img_tensor, [cand]).item()

        txt_score = metric([cand] * len(gt_captions), gt_captions) \
                    .max().item()

        combined = 0.5 * (img_score + txt_score)

        best_score = max(best_score, combined)
    return best_score

def compute_clip_scores(images_dir: Path,
                        captions_json: Path,
                        device: torch.device,
                        gt_captions_json: Path) -> List[float] :
    
    with captions_json.open() as f :
        cap = json.load(f)
    
    with gt_captions_json.open() as f :
        gt_cap = json.load(f)

    scores = []
    for img_id, cand in tqdm(cap.items(), total=len(cap.items())) :
        img_file = images_dir / f"{img_id}.jpg"

        gt_cand = gt_cap[img_id]

        if not img_file.exists():
            print("Missing Image!")
            continue 
        score = max_clip_for_one_image(img_file, cand, gt_cand, device)
        scores.append(score)
        gc.collect()
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--captions-dir", required=True)
    parser.add_argument("--ground-truth-captions", required=True)
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()
    
    
    images_dir = Path(os.path.expanduser(args.images))
    captions_root = Path(os.path.expanduser(args.captions_dir))
    save_root = Path(os.path.expanduser(args.save_dir))
    gt_captions_root = Path(os.path.expanduser(args.ground_truth_captions))

    save_root.mkdir(parents=True, exist_ok=True)

    caption_files = [
        p for p in captions_root.glob("*.json")
        if not p.name.endswith("_summary.json")    # skip previous outputs
    ]
    if not caption_files :
        raise FileNotFoundError("No file")

    clip_summary = {}

    for path in tqdm(caption_files, total=len(caption_files)) :
        method = path.stem
        scores = compute_clip_scores(images_dir, path, device, gt_captions_root)
        avg = sum(scores) / len(scores) if scores else math.nan 
        clip_summary[method] = {"avg_max_score" : avg, 
                                "all_max_scores" : scores}
    
    with (save_root / "clip_summary.json").open("w") as f :
        json.dump(clip_summary, f, indent=2)
    
    pairs = sorted(
        clip_summary.items(),
        key=lambda t: t[1]["avg_max_score"],
        reverse=True          # highest to lowest
    )
    methods, values = zip(*[(m, d["avg_max_score"]) for m, d in pairs])

    plt.figure(figsize=(max(6, 0.6 * len(methods)), 6))
    bars = plt.bar(methods, values, color="skyblue", edgecolor="black")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Avg Max CLIPScore");  plt.title("CLIPScore comparison")

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout();  plt.grid(axis="y", ls="--", alpha=0.4)
    fig_path = save_root / "clipscore_comparison.png"
    plt.savefig(fig_path)
    plt.close()
    print("Saved plot:", fig_path)

if __name__ == "__main__" :
    main()
    