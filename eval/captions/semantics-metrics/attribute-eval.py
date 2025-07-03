import argparse, os, json, glob, re 
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm 

def create_trigger_map(attr_json) :
    trig = {}
    for attr in attr_json.values() :
        for a in attr.keys() :
            key = re.sub(r"^has_[a-z]*::", "", a.lower())
            key = key.replace("(", "").replace(")", "").replace("-", " ")
            trig[a] = set(key.split())
    return trig

def compute_attribute_scores(attr_votes, trigger_map, captions_file) :

    caps = json.load(captions_file.open())
    f1_list = []

    for img, attr_dict in attr_votes.items() :
        img = img.replace(".jpg", "")
        if img not in caps:
            continue
    
        present = {a for a, lst in attr_dict.items()
                if any(v["is_present"] for v in lst)}
        breakpoint()
        words = set()
        for c in caps[img]:
            words |= set(re.findall(r"[a-z']+", c.lower()))

        tp = fp = fn = 0

        for attr, trig in trigger_map.items() :
            hit = bool(words & trig)
            if attr in present :
                tp += hit 
                fn += 0 if hit else 1
            else :
                fp += hit 
            
        if tp + fp + fn == 0 :
            continue 

        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        f1_list.append(f1)
    
    return float(sum(f1_list) / len(f1_list)) if f1_list else 0.0

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--captions-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()

    attr_path = Path(args.attributes).expanduser()
    captions_root = Path(args.captions_dir).expanduser()
    save_root = Path(args.save_dir).expanduser()

    attr_votes = json.load(attr_path.open())
    trigger_map = create_trigger_map(attr_votes)

    save_root.mkdir(parents=True, exist_ok=True)

    caption_files = [ p for p in captions_root.glob("*.json")]
    attribute_summary = {}

    for path in tqdm(caption_files, total=len(caption_files)) :
        method = path.stem
        score = compute_attribute_scores(attr_votes, trigger_map, path)    
        attribute_summary[method] = score
    
    with (save_root / "attribute_summary.json").open("w") as f :
        json.dump(attribute_summary, f, indent=2)
    
    methods, vals = zip(*sorted(attribute_summary.items(), key=lambda t: t[1], reverse=True))
    
    plt.figure(figsize=(max(6, 0.6*len(methods)), 4))
    bars = plt.bar(methods, vals, color="skyblue", edgecolor="black")
    plt.xticks(rotation=30, ha="right"); plt.ylabel("Mean F1"); plt.title("Attribute-hit F1")
    
    for b, v in zip(bars, vals):
        plt.text(b.get_x()+b.get_width()/2, v+0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout(); plt.grid(axis="y", ls="--", alpha=0.4)
    plt.savefig(save_root / "attribute_f1_comparison.png", dpi=300)
    plt.close()

if __name__ == "__main__" :
    main()