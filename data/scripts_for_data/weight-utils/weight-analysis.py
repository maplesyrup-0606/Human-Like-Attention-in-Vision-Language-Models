import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

"""
    This script uses attention weights to analyze :
        1. Similarity Score per image
        2. Sorted plot of the similarity scores averaged across all images
        3. heatmaps for votes on which heads satisfy the condition
"""


def parse_layers(s: str) :
    s = s.strip()
    if "-" in s:
        a, b = s.split('-', 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]

def find_knee(y):
    """
    Simple 'distance to chord' elbow/knee detector.
    y: 1D array-like sequence, assumed ordered along x = 0..len(y)-1.
    Returns index of maximum perpendicular distance from the straight line
    connecting the endpoints.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.arange(len(y))
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    line_vec = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    norm = np.linalg.norm(line_vec)
    if norm == 0:
        return 0
    line_vec /= norm

    vecs = np.stack([x - x1, y - y1], axis=1)
    proj_lens = np.dot(vecs, line_vec)
    proj = np.outer(proj_lens, line_vec)
    dists = np.linalg.norm(vecs - proj, axis=1)

    knee_idx = int(np.argmax(dists))
    return knee_idx

def topk_with_threshold(arr, k, threshold):
    arr = np.asarray(arr)
    n = arr.size 
    if k <= 0 or n == 0 :
        return np.array([], dtype=int)

    k = min(k, n)
    idx = np.argpartition(-arr, k - 1)[:k]
    return idx[arr[idx] > threshold]

def load_image_matrix_from_csv(similarity_file, layer_map, num_layers, num_heads, relative_norm):

    """
    """
    try :
        df = pd.read_csv(similarity_file, index_col=0)
    except Exception :
        return None 
    
    arr = np.full((num_layers, num_heads), np.nan, dtype=np.float32)

    for row_idx, row in df.iterrows() :
        # Expect "L19" etc.;
        s = str(row_idx).strip()
        if len(s) < 1 or s[0].upper() != "L" :
            continue
        
        try:
            raw_layer = int(s[1:])
        except Exception:
            continue

        if raw_layer not in layer_map:
            continue
    
        li = layer_map[raw_layer]
        vals = row.values.astype(np.float32)
        if vals.shape[0] != num_heads :
            return None 

        arr[li, :] = vals
    
    if np.isnan(arr).any() :
        return None 
    
    if relative_norm :
        # normalize each layer row by its max 
        row_max = np.max(arr, axis=1, keepdims=True)
        arr = arr / (row_max + 1e-9)
    
    return arr

# ELBOW thresholding for sorted sim-scores across all images
def do_elbow(root, target_layers, num_heads, relative_norm) :
    layer_map = {L : i for i, L in enumerate(target_layers)}
    num_layers = len(target_layers)
    R = num_layers * num_heads

    rank_sum = np.zeros(R, dtype=np.float14)
    rank_count = 0
    used = skipped = 0

    for fp in Path(root).rglob("*_similarity.csv") :
        arr = load_image_matrix_from_csv(fp, layer_map, num_layers, num_heads, relative_norm)
        if arr is None :
            skipped += 1
            continue
    
        vec = arr.flatten()
        order = np.argsort(vec)
        sorted_vals = vec[order]
        rank_sum += sorted_vals
        rank_count += 1
        used += 1
    
    if rank_count == 0:
        raise RuntimeError("No complete images processed for elbow")

    rank_avg = rank_sum / rank_count 
    knee_idx = find_knee(rank_avg)
    knee_value = rank_avg[knee_idx]

    # Save CSV and Image
    out_csv = Path(root) / "ranked-head-sim-avg.csv"
    pd.DataFrame({
        "rank": np.arange(1, R + 1, dtype=np.int64),
        "avg_score": rank_avg,
        "count": np.full(R, rank_count, dtype=np.int64)
    }).to_csv(out_csv, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(rank_avg, label='Average similarity by rank (ascending)')
    plt.axvline(knee_idx, linestyle='--', label=f'Elbow index = {knee_idx} (0-based)')
    plt.axhline(knee_value, linestyle=':', label=f'Threshold ≈ {knee_value:.4f}')
    plt.xlabel("Rank position (0 = smallest score)")
    plt.ylabel("Average similarity score")
    plt.title(f"Rank-averaged head similarity (L{target_layers[0]}–L{target_layers[-1]})\n"
              f"{used} images used, {skipped} skipped")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_png = Path(root) / "ranked-head-sim-avg-elbow-relative.png"
    plt.savefig(out_png)
    plt.close()

    print(f"[ELBOW] wrote {out_csv}")
    print(f"[ELBOW] wrote {out_png}")
    print(f"[ELBOW] knee idx={knee_idx}, value≈{knee_value:.6f}, used={used}, skipped={skipped}")

def do_heatmap(root, target_layers, num_heads, tau, topk, relative_norm) :
    layer_map = {L : i for i, L in enumerate(target_layers)}
    num_layers = len(layer_map)
    counts = np.zeros((num_layers, num_heads), dtype=np.int64)
    used = skipped = 0

    for fp in Path(root).rglob("*_similarity.csv") :
        arr = load_image_matrix_from_csv(fp, layer_map, num_layers, num_heads, relative_norm)
        if arr is None :
            skipped += 1
            continue 
    
        for li in range(num_layers) :
            idxs = topk_with_threshold(arr[li], topk, tau)
            if idxs.size :
                counts[li, idxs] += 1
            
        used += 1
    
    plt.figure(figsize=(12, 6))
    im = plt.imshow(counts, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='Count above τ within top-k')
    plt.xticks(range(num_heads), [f"H{h}" for h in range(num_heads)], rotation=90)
    plt.yticks(range(num_layers), [f"L{L}" for L in target_layers])
    plt.xlabel("Head index")
    plt.ylabel("Layer index")
    plt.title(f"Head votes above τ = {tau} (top-k = {topk})\n{used} images used, {skipped} skipped")
    plt.tight_layout()
    out_png = Path(root) / f"above_threshold_heatmap_top{topk}-threshold{tau}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[HEATMAP] wrote {out_png}")
    print(f"[HEATMAP] images used={used}, skipped={skipped}")

def do_per_image_curves(root, target_layers, num_heads, relative_norm, limit=None) :
    layer_map = {L : i for i, L in enumerate(target_layers)}
    num_layers = len(layer_map)
    used = skipped = 0

    out_dir = Path(root) / "sim-score-plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, fp in enumerate(Path(root).rglob("*_similarity.csv")) :
        if limit is not None and i >= limit :
            break
    
        arr = load_image_matrix_from_csv(fp, layer_map, num_layers, num_heads, relative_norm)
        if arr is None :
            skipped += 1
            continue
            
        vec = arr.flatten()
        sorted_vals = np.sort(vec)

        img_id = fp.stem.replace("_similarity", "")
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_vals, label=img_id)
        plt.xlabel("Rank position (ascending)")
        plt.ylabel("Similarity score")
        plt.title(f"Per-image rank-sorted similarities: {img_id}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_png = out_dir / f"individual_scores_{img_id}.png"
        plt.savefig(out_png)
        plt.close()
        used += 1

    print(f"[PER-IMAGE] wrote plots to {out_dir}")
    print(f"[PER-IMAGE] images used={used}, skipped={skipped}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="~/NSERC/vis/head-sim-scores-heat-map")
    p.add_argument("--layers", type=str, default="19-26",
                   help="e.g., '19-26' or '19,20,21'")
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--tau", type=float, default=0.5232)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--mode", type=str, choices=["elbow", "heatmap", "per-image", "all"],
                   default="all")
    p.add_argument("--relative-norm", action="store_true",
                   help="Per-layer max normalization (keeps your relative ordering).")
    p.add_argument("--per-image-limit", type=int, default=None,
                   help="Limit number of per-image plots (for quick tests).")
    args = p.parse_args()

    root = Path(args.root).expanduser()
    target_layers = parse_layers(args.layers)
    num_heads = args.heads

    if args.mode in ("elbow", "all"):
        do_elbow(root, target_layers, num_heads, args.relative_norm)

    if args.mode in ("heatmap", "all"):
        do_heatmap(root, target_layers, num_heads, args.tau, args.topk, args.relative_norm)

    if args.mode in ("per-image", "all"):
        do_per_image_curves(root, target_layers, num_heads, args.relative_norm, args.per_image_limit)

if __name__ == "__main__":
    main()