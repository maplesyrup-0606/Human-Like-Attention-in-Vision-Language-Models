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
                           captions: List[str],    
                           device: torch.device)-> float:
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.functional.pil_to_tensor(img).unsqueeze(0).to(device)
    
    score_tensor = metric(img_tensor.repeat(len(captions), 1, 1, 1), captions)

    return score_tensor.max().item()

def compute_clip_scores(images_dir: Path,
                        captions_json: Path,
                        device: torch.device) -> List[float] :
    
    with captions_json.open() as f :
        cap = json.load(f)
    
    scores = []
    for img_id, cand in tqdm(cap.items(), total=len(cap.items())) :
        img_file = images_dir / f"{img_id}.jpg"
    
        if not img_file.exists():
            print("Missing Image!")
            continue 
        score = max_clip_for_one_image(img_file, cand, device)
        scores.append(score)
        gc.collect()
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--captions-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()
    
    
    images_dir = Path(os.path.expanduser(args.images))
    captions_root = Path(os.path.expanduser(args.captions_dir))
    save_root = Path(os.path.expanduser(args.save_dir))

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
        scores = compute_clip_scores(images_dir, path, device)
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

    plt.figure(figsize=(max(6, 0.6 * len(methods)), 4))
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
    