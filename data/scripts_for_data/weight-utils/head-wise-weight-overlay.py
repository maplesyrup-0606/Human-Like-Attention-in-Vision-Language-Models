import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ───────────────────────── helpers ─────────────────────────
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


def _get_image_path_from_info(img_id):
    IMG_ROOT = Path("~/NSERC/data/images/MSCOCO_images").expanduser()
    other = str(img_id) + ".jpg"
    return IMG_ROOT / other


def make_tile(img: Image.Image,
              heat: torch.Tensor,         # (gs, gs), already normalized to [0,1] for viz
              scan_mask: torch.Tensor,    # (gs*gs) bool
              grid_size: int,
              out_size: int = 128,
              alpha: float = 0.45) -> Image.Image:
    W, H = img.size

    # upsample heat to image size
    heat_np = heat.unsqueeze(0).unsqueeze(0)  # 1x1xgsxgs
    heat_np = F.interpolate(heat_np, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    heat_np = np.clip(heat_np, 0, 1)

    # colorize
    cmap = plt.get_cmap("jet")
    heat_color = (cmap(heat_np)[:, :, :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_color)

    blended = Image.blend(img.convert("RGBA"), heat_img.convert("RGBA"), alpha=alpha).convert("RGB")

    # draw scanpath boxes
    draw = ImageDraw.Draw(blended)
    cell_w = W / grid_size
    cell_h = H / grid_size

    scan_mask_2d = scan_mask.view(grid_size, grid_size).cpu().numpy().T
    
    ys, xs = np.where(scan_mask_2d)
    for x_cell, y_cell in zip(ys, xs):
        x0 = y_cell * cell_w
        y0 = x_cell * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

    return blended.resize((out_size, out_size), Image.BILINEAR)


def build_mosaic_from_LHL(per_img_layer_head_L,  # (n_layers, heads, L)
                          image_info, scanpath, img_id,
                          save_dir, layers_to_analyze, grid_size=24,
                          tile_size=128, margin=4, alpha=0.45):
    """
    per_img_layer_head_L: raw attention weights (after your chosen aggregation over tokens),
                          shaped [n_layers, n_heads, L].
    We normalize each tile by its own max *only for colormap visibility*.
    """
    start = image_info['start_index']
    L = image_info['num_patches']

    img_path = _get_image_path_from_info(img_id)
    base_img = Image.open(img_path).convert("RGB")

    scan_mask = get_scanpath_mask(scanpath, margin=1, grid_size=grid_size,
                                  device=per_img_layer_head_L.device)

    n_layers, n_heads, L_check = per_img_layer_head_L.shape
    assert L_check == L, f"L mismatch: {L_check} vs expected {L}"

    mosaic_w = n_heads * tile_size + (n_heads + 1) * margin
    mosaic_h = n_layers * tile_size + (n_layers + 1) * margin
    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(mosaic)

    # annotate axes
    try:
        font = ImageFont.load_default()
    except:
        font = None

    # column labels (heads)
    for c in range(n_heads):
        x = margin + c * (tile_size + margin)
        draw.text((x, 2), f"H{c}", fill=(255, 255, 255), font=font)

    # row labels (layers)
    for r, layer_idx in enumerate(layers_to_analyze):
        y = margin + r * (tile_size + margin)
        draw.text((2, y), f"L{layer_idx}", fill=(255, 255, 255), font=font)

    # paste tiles
    for r in range(n_layers):
        for c in range(n_heads):
            attn_vec = per_img_layer_head_L[r, c]  # (L,)
            # normalize only for visualization
            m = float(attn_vec.max().item())
            if m > 0:
                heat = (attn_vec / m).view(grid_size, grid_size)
            else:
                heat = torch.zeros(grid_size, grid_size, device=attn_vec.device)

            tile = make_tile(base_img, heat, scan_mask, grid_size,
                             out_size=tile_size, alpha=alpha)

            x = margin + c * (tile_size + margin)
            y = margin + r * (tile_size + margin)
            mosaic.paste(tile, (x, y))

            # this is for labelling tile with head and layer
            draw.text((x + 2, y + 2), f"L{layers_to_analyze[r]}-H{c}", fill=(255,255,255), font=font)
    out_path = save_dir / f"mosaic_{img_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(out_path)


def aggregate_attn_for_image(total_weights, layers_to_analyze, batch_idx, image_info,
                             token_range, agg: str, device):
    """
    Aggregate raw attention weights for one image across tokens,
    respecting your q-len rule:
      - for the first token in token_range → use q_idx = -1
      - for the rest → q_idx = 0
    (If q_len == 1, use 0 safely.)
    Returns tensor [n_layers, heads, L].
    """
    start = image_info['start_index']
    L = image_info['num_patches']

    first_token = token_range.start
    sample = total_weights[first_token][layers_to_analyze.start - 1]  # [batch, head, q, k]
    num_heads = sample.shape[1]

    n_layers = len(layers_to_analyze)
    out = torch.zeros(n_layers, num_heads, L, device=device)
    count = 0

    for t in token_range:
        weights = total_weights[t]  # layers x batch x head x q_len x k_len
        for li, layer_idx in enumerate(layers_to_analyze):
            layer_w = weights[layer_idx - 1][batch_idx]  # head x q x k
            q_len = layer_w.shape[1]

            # your rule:
            if t == first_token:
                q_idx = -1 if q_len > 1 else 0
            else:
                q_idx = 0

            attn = layer_w[:, q_idx, start:start + L]  # head x L

            if agg == "sum":
                out[li] += attn
            elif agg == "first":
                if count == 0:
                    out[li] = attn
            else:  # mean
                out[li] += attn
        count += 1
        if agg == "first":
            break

    if agg == "mean" and count > 0:
        out /= count

    return out  # [n_layers, heads, L]


# ───────────────────────── main ─────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to Weights Directory")
    parser.add_argument("--save-dir", type=str, required=True, help="Output path for mosaics")
    parser.add_argument("--layers", type=str, default="19-26", help="inclusive range, e.g. 19-26")
    parser.add_argument("--grid-size", type=int, default=24, help="ViT grid size (e.g., 24→576 patches)")
    parser.add_argument("--tile-size", type=int, default=96, help="Tile size inside the mosaic")
    parser.add_argument("--margin", type=int, default=4, help="Margin between tiles in the mosaic")
    parser.add_argument("--alpha", type=float, default=0.45, help="Blend alpha for heatmap overlay")
    parser.add_argument("--token-start", type=int, default=5)
    parser.add_argument("--token-end", type=int, default=100)
    parser.add_argument("--agg", type=str, default="mean", choices=["mean", "sum", "first"],
                        help="How to aggregate weights across tokens")
    parser.add_argument("--sample-images", type=int, default=0,
                        help="Cap number of images visualized per .pt file (0 = no cap)")
    args = parser.parse_args()

    if "-" in args.layers:
        a, b = args.layers.split("-")
        layers_to_analyze = range(int(a), int(b) + 1)
    else:
        layers_to_analyze = [int(x) for x in args.layers.split(",")]

    save_dir = Path(args.save_dir).expanduser()
    os.makedirs(save_dir, exist_ok=True)

    weight_path = Path(args.weights).expanduser()
    all_weight_files = sorted(weight_path.glob("*.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for weight_file in tqdm(all_weight_files, desc="files"):
        data = torch.load(weight_file, map_location=device)
        total_weights = data['attn_weights']     # list: token_idx -> [layers x batch x head x q x k]
        image_infos   = data['image_infos']
        batch_img_ids = data['all_image_ids']
        scanpaths     = data['scanpaths']

        token_start = args.token_start
        token_end   = min(args.token_end, len(total_weights))
        if token_start >= token_end:
            raise ValueError(f"Invalid token range [{token_start}, {token_end}).")

        token_range = range(token_start, token_end)
        per_file_dir = save_dir / weight_file.stem
        per_file_dir.mkdir(parents=True, exist_ok=True)

        images_done = 0
        for batch_idx, img_id in enumerate(batch_img_ids):
            if args.sample_images and images_done >= args.sample_images:
                break

            image_info = image_infos[batch_idx]
            L = image_info['num_patches']
            if L != args.grid_size * args.grid_size:
                raise ValueError(
                    f"num_patches ({L}) != grid_size^2 ({args.grid_size ** 2}). "
                    f"Pass the correct --grid-size."
                )

            per_img_layer_head_L = aggregate_attn_for_image(
                total_weights=total_weights,
                layers_to_analyze=layers_to_analyze,
                batch_idx=batch_idx,
                image_info=image_info,
                token_range=token_range,
                agg=args.agg,
                device=device
            )

            build_mosaic_from_LHL(
                per_img_layer_head_L=per_img_layer_head_L,
                image_info=image_info,
                scanpath=scanpaths[batch_idx],
                img_id=img_id,
                save_dir=per_file_dir,
                layers_to_analyze=layers_to_analyze,
                grid_size=args.grid_size,
                tile_size=args.tile_size,
                margin=args.margin,
                alpha=args.alpha
            )
            images_done += 1

    print("Done.")


if __name__ == "__main__":
    main()