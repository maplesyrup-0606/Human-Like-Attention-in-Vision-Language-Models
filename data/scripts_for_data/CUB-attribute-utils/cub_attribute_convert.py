import os, json, re, csv
from pathlib import Path 
import matplotlib.pyplot as plt

"""
    Converts the LLM responses of Attribute Judgements into a png that computes the average F1 Score.
    For a given directory of attribute evaluations -> visualizes the set of evaluations into a png
"""

def extract_tp_fp_fn(text):
    pattern = {
        'TP': r'True Positives: (\d+|None|All)',
        'FP': r'False Positives: (\d+|None|All)',
        'FN': r'False Negatives: (\d+|None|All)'
    }

    def safe_extract(key):
        match = re.search(pattern[key], text)
        if not match:
            raise ValueError(f"{key} not found in:\n{text}")
        
        val = match.group(1)
        if val.lower() == "none":
            return 0
        elif val.lower() == "all":
            return float('inf')  # or return a large constant like 9999
        else:
            return int(val)

    TP = safe_extract('TP')
    FP = safe_extract('FP')
    FN = safe_extract('FN')

    return TP, FP, FN

def main() :
    attributes_root = Path("~/Human-Like-Attention-in-Vision-Language-Models/data/generated_captions/jul18_samples/attributes_eval").expanduser()
    save_path = Path("~/Human-Like-Attention-in-Vision-Language-Models/data/generated_captions/jul18_samples/attributes_eval/").expanduser()
    ret = {}
    for judgement_path in attributes_root.glob("*.json") :
        with open(judgement_path, "r") as f:
            judgement = json.load(f)

        method = judgement_path.stem 
        method = method.split("_caption")[0]
        cur_ret = {}

        for sample_id, evaluations in judgement.items() :
            f1 = 0
            for text in evaluations :
                try:
                    TP, FP, FN = extract_tp_fp_fn(text)
                except Exception as e:
                    print(f"\n❌ Error parsing sample: {sample_id}")
                    print(f"❌ Problematic text:\n{text}")
                    print(f"❌ Exception: {e}")
                    raise 
                p = 0 if TP == 0 and FP == 0 else TP / (TP + FP)
                r = 0 if TP == 0 and FN == 0 else TP / (TP + FN)
                
                temp_f1 = 0 if p == 0 and r == 0 else (2 * p * r) / (p + r)

                f1 = max(f1, temp_f1)
            
            cur_ret[sample_id] = f1 
        
        ret[method] = cur_ret

    method_to_avg = {
                        method: sum(f1s.values()) / len(f1s)
                        for method, f1s in ret.items()
                    }
    
    sorted_methods = sorted(method_to_avg.items(), key=lambda x: x[1], reverse=True)
    
    # Unpack into two lists for plotting
    methods, avg_f1s = zip(*sorted_methods)

    csv_path = save_path / "avg_f1_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Average_Max_F1"])
        writer.writerows(sorted_methods)
    print(f"✅ Saved CSV to {csv_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(methods, avg_f1s, color='skyblue')
    plt.ylabel("Average Max F1 Score")
    plt.xlabel("Method")
    plt.title("Average Max F1 Score per Method")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path / "avg_f1_scores.png", dpi=300)
    plt.close()
    return 

if __name__ == "__main__" :
    main()
