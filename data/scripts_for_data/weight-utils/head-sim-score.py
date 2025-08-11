import argparse, os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from cycler import cycler
import math
from scipy.ndimage import binary_dilation, label

def get_scanpath_gaussian(scanpath, grid_size=24, sigma=1.5, device='cpu', normalize=True):
    if len(scanpath) == 0:
        return torch.zeros(grid_size * grid_size, device=device)
    xs = torch.arange(grid_size, device=device).view(-1, 1).repeat(1, grid_size)
    ys = torch.arange(grid_size, device=device).view(1, -1).repeat(grid_size, 1)
    gauss = torch.zeros(grid_size, grid_size, device=device)
    two_sigma2 = 2.0 * (sigma ** 2)
    for (x, y) in scanpath:
        dx2 = (xs - x) ** 2
        dy2 = (ys - y) ** 2
        gauss += torch.exp(-(dx2 + dy2) / two_sigma2)
    if normalize and gauss.sum() > 0:
        gauss = gauss / gauss.sum()
    return gauss.view(-1)


def get_scanpath_mask(scanpath, margin=1, grid_size=24, device='cpu'):
    mask = torch.zeros(grid_size * grid_size, device=device)
    for x, y in scanpath:
        x_start = max(0, x - margin)
        x_end = min(grid_size - 1, x + margin)
        y_start = max(0, y - margin)
        y_end = min(grid_size - 1, y + margin)
        for xi in range(x_start, x_end + 1):
            for yi in range(y_start, y_end + 1):
                idx = xi * grid_size + yi
                mask[idx] = 1
    return mask.bool()

def build_decay_mask(scanpoints, grid_size=24, decay=[1.0, 0.5, 0.25]):
    base_mask = np.zeros((grid_size, grid_size), dtype=bool)

    # Place 3x3 square around each fixation
    for x, y in scanpoints:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                xi, yi = x + dx, y + dy
                if 0 <= xi < grid_size and 0 <= yi < grid_size:
                    base_mask[yi, xi] = True

    # Create the final mask with fixed decay values, no overlap summing
    final_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    current_mask = base_mask.copy()
    visited = np.zeros_like(base_mask)

    for weight in decay:
        # Assign weight where we haven’t already assigned a value
        new_layer = current_mask & (~visited)
        final_mask[new_layer] = weight
        visited |= new_layer

        # Dilate for the next layer
        current_mask = binary_dilation(current_mask)

    return torch.tensor(final_mask)

def build_gaussian_connected_decay_mask(scanpoints, grid_size=24, sigma=1.5, margin=1, device='cpu'):
    base_mask = np.zeros((grid_size, grid_size), dtype=bool)
    
    # Set 1 for those points + margin where gaze lies
    for x, y in scanpoints:
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                xi, yi = x + dx, y + dy
                if 0 <= xi < grid_size and 0 <= yi < grid_size:
                    base_mask[yi, xi] = True
    
    final_mask   = np.zeros((grid_size, grid_size), dtype=np.float32)
    current_mask = base_mask.copy()
    visited      = np.zeros_like(base_mask, dtype=bool)
    struct       = np.ones((3, 3), dtype=bool)   # 8-connected, matches conv2d 3×3 ones
    max_steps    = math.ceil(3 * sigma)

    for d in range(max_steps + 1):               # d = 0..max_steps
        decay_weight = math.exp(-(d**2) / (2 * sigma**2))
        new_layer = current_mask & (~visited)
        final_mask[new_layer] = decay_weight
        visited |= new_layer
        current_mask = binary_dilation(current_mask, structure=struct, iterations=1, border_value=0)

    return torch.tensor(final_mask, device=device)

def compute_similarity_per_image(total_weights, image_info, scanpath, batch_idx,
                                 layers_to_analyze, token_range,
                                 grid_size, sigma, normalize_gauss,
                                 use_q_rule, device):
    start = image_info['start_index']
    L     = image_info['num_patches']

    first_token = token_range.start
    sample = total_weights[first_token][layers_to_analyze.start - 1]  # [batch, head, q, k]
    n_heads = sample.shape[1]
    n_layers = len(layers_to_analyze)

    sim_sum   = torch.zeros(n_layers, n_heads, device=device)
    sim_count = torch.zeros_like(sim_sum)

    # fixation_mask = build_decay_mask(scanpath).view(-1).to(device)
    margin = 1
    fixation_mask = build_gaussian_connected_decay_mask(scanpath, margin=margin).view(-1).to(device)
    
    def save_mask_as_image(mask, path="mask.png"):
        if mask.is_cuda :
            mask = mask.cpu()
        mask = mask.view(24, 24)
        plt.imshow(mask.numpy(), cmap="plasma", interpolation='nearest')
        plt.colorbar()
        plt.title("Fixation Decay Mask")
        plt.tight_layout()
        plt.savefig(path)
        print(f"Saved visualization to {path}")
    

    for t in token_range:
        weights = total_weights[t]  # [layers x batch x head x q x k]
        for li, layer_idx in enumerate(layers_to_analyze):
            layer_w = weights[layer_idx - 1][batch_idx]  # [head x q x k]

            if use_q_rule:
                q_len = layer_w.shape[1]
                if t == first_token:
                    q_idx = -1 if q_len > 1 else 0
                else:
                    q_idx = 0
            else:
                q_idx = 0

            attn_map = layer_w[:, q_idx, start:start + L]  # [head x L]
            num = (attn_map * fixation_mask).sum(dim=1)    # [head]
            den = attn_map.sum(dim=1) + 1e-9               # [head] to avoid divide-by-zero
            sim = num / den                                # [head]
            
            sim_sum[li]   += sim
            sim_count[li] += 1

    sim_avg = sim_sum / (sim_count + 1e-9)
    return sim_avg  # [layers, heads]

def compute_ratio_per_image(total_weights, image_info, scanpath, batch_idx,
                            layers_to_analyze, token_range,
                            grid_size, sigma, normalize_gauss,
                            use_q_rule, device):
    """
    For each (layer, head), computes:
        ratio = (# of top-k patches within scanpath) / k
    """
    start = image_info['start_index']
    L = image_info['num_patches']
    k = len(scanpath)
    first_token = token_range.start
    sample = total_weights[first_token][layers_to_analyze.start - 1]
    n_heads = sample.shape[1]
    n_layers = len(layers_to_analyze)

    if k == 0:
        return torch.zeros(n_layers, n_heads, device=device)

    
    scan_mask = get_scanpath_mask(scanpath, grid_size=grid_size, device=device)

    ratio_matrix = torch.zeros(n_layers, n_heads, device=device)

    # Compute per head, per layer, similarity per patch
    for li, layer_idx in enumerate(layers_to_analyze):
        sim_per_head = torch.zeros(n_heads, L, device=device)
        for t in token_range:
            layer_w = total_weights[t][layer_idx - 1][batch_idx]  # [head x q x k]
            q_len = layer_w.shape[1]
            q_idx = -1 if (use_q_rule and t == first_token and q_len > 1) else 0
            attn_map = layer_w[:, q_idx, start:start + L]  # [head x L]
            sim_per_head += attn_map

        for h in range(n_heads):
            vals = sim_per_head[h]
            topk = torch.topk(vals, min(k, L)).indices
            hits = scan_mask[topk].sum()
            ratio_matrix[li, h] = hits.float() / k

    return ratio_matrix  # [layers, heads]

def plot_and_save_per_image(matrix, layers_to_analyze, out_png, out_csv, label="Gaussian–attention similarity (dot)"):
    mat_np = matrix.cpu().numpy()
    n_layers, n_heads = mat_np.shape

    # CSV
    df = pd.DataFrame(mat_np,
                      index=[f"L{l}" for l in layers_to_analyze],
                      columns=[f"H{h}" for h in range(n_heads)])
    df.to_csv(out_csv, index=True)

    # HEAT MAP
    plt.figure(figsize=(max(6, 0.35 * n_heads), max(3, 0.3 * n_layers)))
    im = plt.imshow(mat_np, aspect='auto', interpolation='nearest', cmap='magma', vmin=0, vmax=1 if "ratio" in label.lower() else None)
    plt.colorbar(im, label=label)

    plt.yticks(np.arange(n_layers), [f"L{l}" for l in layers_to_analyze])
    plt.xticks(np.arange(n_heads), [f"H{h}" for h in range(n_heads)], rotation=45)

    plt.xlabel("Head index")
    plt.ylabel("Layer")
    plt.title(label)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_global_head_stats_bar(mean_t, std_t, layers_to_analyze, out_png, out_csv, count_t=None):
    """
    mean_t, std_t, count_t: [n_layers, n_heads] torch tensors
    Saves a CSV and a wide bar plot with error bars (mean ± std) for each head across ALL samples.
    """
    mean_np = mean_t.detach().cpu().numpy()
    std_np  = std_t.detach().cpu().numpy()
    cnt_np  = count_t.detach().cpu().numpy() if count_t is not None else None

    n_layers, n_heads = mean_np.shape
    labels = []
    means  = []
    stds   = []
    counts = []

    layer_list = list(layers_to_analyze)
    for li, L in enumerate(layer_list):
        for h in range(n_heads):
            labels.append(f"L{L}-H{h}")
            means.append(mean_np[li, h])
            stds.append(std_np[li, h])
            counts.append(cnt_np[li, h] if cnt_np is not None else np.nan)

    # CSV
    df = pd.DataFrame({
        "layer": [lbl.split("-")[0] for lbl in labels],
        "head":  [lbl.split("-")[1] for lbl in labels],
        "mean":  means,
        "std":   stds,
        "count": counts
    })
    df.to_csv(out_csv, index=False)

    # Bar plot
    N = len(labels)
    plt.figure(figsize=(max(12, 0.12 * N), 6))  # width scales with number of bars (e.g., 256 -> ~42 in)
    x = np.arange(N)
    plt.bar(x, means, yerr=stds, capsize=2)
    plt.xticks(x, labels, rotation=90)
    plt.ylim(0, 1)  # similarity is in [0,1]
    plt.ylabel("Similarity (mean ± std)")
    plt.title("Global per-head similarity across all samples")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True,
                        help="Directory containing *.pt files")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Where to save per-image plots/CSVs")
    parser.add_argument("--layers", type=str, default="19-26",
                        help="Inclusive range, e.g. 19-26")
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--token-start", type=int, default=5)
    parser.add_argument("--token-end", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--no-norm-gauss", action="store_true",
                        help="Do NOT normalize Gaussian to sum=1")
    parser.add_argument("--use-q-rule", action="store_true",
                        help="Use your q-len rule for q_idx selection")
    parser.add_argument("--sample-images", type=int, default=0,
                        help="Limit #images per .pt (0 = all)")
    args = parser.parse_args()

    if "-" in args.layers:
        a, b = args.layers.split("-")
        layers_to_analyze = range(int(a), int(b) + 1)
    else:
        layers_to_analyze = [int(x) for x in args.layers.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dir = Path(args.weights).expanduser()
    save_dir   = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(weight_dir.glob("*.pt"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pt files found in {weight_dir}")
    
    # ── NEW: global accumulators across ALL files & images
    global_sum   = None
    global_sqsum = None
    global_count = None


    for f in tqdm(files, desc="files"):
        data = torch.load(f, map_location=device)
        total_weights = data['attn_weights']
        image_infos   = data['image_infos']
        batch_img_ids = data['all_image_ids']
        scanpaths     = data['scanpaths']

        token_end = min(args.token_end, len(total_weights))
        token_range = range(args.token_start, token_end)

        per_file_dir = save_dir / f.stem
        per_file_dir.mkdir(parents=True, exist_ok=True)

        done = 0
        for batch_idx, img_id in enumerate(batch_img_ids):
            if args.sample_images and done >= args.sample_images:
                break

            sim = compute_similarity_per_image(
                total_weights=total_weights,
                image_info=image_infos[batch_idx],
                scanpath=scanpaths[batch_idx],
                batch_idx=batch_idx,
                layers_to_analyze=layers_to_analyze,
                token_range=token_range,
                grid_size=args.grid_size,
                sigma=args.sigma,
                normalize_gauss=(not args.no_norm_gauss),
                use_q_rule=args.use_q_rule,
                device=device
            )

            if global_sum is None:
                global_sum   = torch.zeros_like(sim)
                global_sqsum = torch.zeros_like(sim)
                global_count = torch.zeros_like(sim)

            sim = sim / (sim.max(dim=1, keepdim=True).values + 1e-9)

            global_sum   += sim
            global_sqsum += sim * sim
            global_count += 1.0  # broadcast to [layers, heads]

            out_png_sim = per_file_dir / f"{img_id}_similarity.png"
            out_csv_sim = per_file_dir / f"{img_id}_similarity.csv"
            plot_and_save_per_image(sim, layers_to_analyze, out_png_sim, out_csv_sim, label="Gaussian–attention similarity (dot)")

            done += 1


        if global_sum is None:
            raise RuntimeError("No images processed; check your inputs/filters.")

        valid = global_count > 0
        global_mean = torch.zeros_like(global_sum)
        global_std  = torch.zeros_like(global_sum)

        global_mean[valid] = global_sum[valid] / global_count[valid]
        global_var = torch.zeros_like(global_sum)
        global_var[valid] = global_sqsum[valid] / global_count[valid] - global_mean[valid] ** 2
        global_var = torch.clamp(global_var, min=0.0)
        global_std[valid] = torch.sqrt(global_var[valid])

        out_stats_csv = save_dir / "GLOBAL_heads_similarity_stats.csv"
        out_stats_png = save_dir / "GLOBAL_heads_similarity_stats.png"
        save_global_head_stats_bar(global_mean, global_std, layers_to_analyze,
                                out_stats_png, out_stats_csv, count_t=global_count)


if __name__ == "__main__":
    main()