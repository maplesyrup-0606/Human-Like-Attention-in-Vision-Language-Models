import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
import sys
import math
from PIL import Image, ImageDraw, ImageFont    # ← NEW
from tqdm import tqdm
from matplotlib.colors import PowerNorm
from tqdm import tqdm

"""
    Code which visualizes question-to-image attention at the moment.
"""

def expand2square(pil_img, background_color) :
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def get_global_max_prob(weight_path):
    """
    Scan the weight file once and return the largest probability mass
    that any image patch gets after:
      – averaging heads and output tokens, then
      – normalising so each map sums to 1.

    Works for attn_weights shaped: tuple(token) → tuple(layer) → tensor
    with per-layer tensors shaped either (B, H, S) or (B, H, q_len, S).
    """
    data = torch.load(os.path.expanduser(weight_path))
    attn_weights, image_infos = data["attn_weights"], data["image_infos"]

    global_max = 0.0
    num_layers = len(attn_weights[0])          # 32

    for batch_idx, info in tqdm(enumerate(image_infos)):
        img_start = info["start_index"]
        L         = info["num_patches"]
        img_slice = slice(img_start, img_start + L)

        N = len(attn_weights)                  # number of output tokens

        for layer in range(num_layers):
            summed = None
            for tok in range(N):
                # ---- pull tensor & unify to (H, 1, S) ----
                t = attn_weights[tok][layer][batch_idx]      # tensor
                if t.dim() == 2:            # (H, S)   – first layer
                    t = t.unsqueeze(1)      # -> (H, 1, S)
                else:                       # (H, q, S)
                    t = t[:, -1:, :]        # last query -> (H, 1, S)

                part   = t[..., img_slice]  # H × 1 × L
                summed = part if summed is None else summed + part

            avg   = (summed / N).mean(0).squeeze(0)  # L
            probs = avg / avg.sum()                  # L, Σ=1
            global_max = max(global_max, probs.max().item())

    return global_max

def save_layer_grid(layer_imgs, image_id, save_dir, cols=8):
    """
    layer_imgs : list[ PIL.Image ] length = #layers (32)
    cols       : how many columns in the final grid
    Makes a single image called f"{image_id}_all_layers.png"
    """
    if not layer_imgs:
        return

    w, h   = layer_imgs[0].size          # each tile size (e.g. 336×336)
    n      = len(layer_imgs)
    rows   = math.ceil(n / cols)
    grid   = Image.new("RGB", (cols * w, rows * h), color=(0, 0, 0))

    # Optional: nice small font for labels
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = None  # fallback if font missing

    for idx, tile in enumerate(layer_imgs):
        r, c = divmod(idx, cols)
        grid.paste(tile, (c * w, r * h))

        # draw tiny “L00” etc. in the corner
        if font:
            draw = ImageDraw.Draw(grid)
            txt  = f"L{idx:02d}"
            draw.text((c * w + 5, r * h + 5), txt, fill=(255, 255, 255), font=font)

    out_path = os.path.join(save_dir, f"{image_id}_all_layers.png")
    grid.save(out_path, dpi=(300, 300))
    print(f"✅ saved {out_path}")

def output_to_image(image_dir, weight_path, save_dir, global_max) :
    save_dir = os.path.join(save_dir, "out2img")
    os.makedirs(save_dir, exist_ok=True)

    num_layers = 32

    # get data for corresponding batch
    data = torch.load(os.path.expanduser(weight_path))
    
    # attn weights : output_tokens x layer x batch x head x 1 x q_len (for output)
    attn_weights, image_infos, all_image_ids = data['attn_weights'], data['image_infos'], data['all_image_ids']

    # get all images associated with batch and process them to 336 x 336
    images = []
    for image_id in all_image_ids :
        image_path = os.path.join(os.path.expanduser(image_dir), image_id + ".jpg")
        image = Image.open(image_path)
        image = expand2square(image, background_color=(0, 0, 0))
        images.append(image)

    for batch_idx, image in tqdm(enumerate(images)) :
        image_id = all_image_ids[batch_idx]
        image_info = image_infos[batch_idx]
        image_start = image_info['start_index']
        L = image_info['num_patches']

        side = int(math.sqrt(L))
        assert side * side == L

        tiles = []
        for layer_idx in range(num_layers) :
            summed = None 
            N = len(attn_weights)
            for token_idx in range(N) :
                w = attn_weights[token_idx][layer_idx][batch_idx] 
                w = w[:, -1, :].unsqueeze(1) if token_idx == 0 else w # head x 1 x 676

                output_to_img = w[..., image_start : image_start + L]

                summed = output_to_img.clone() if summed is None else (summed + output_to_img)

            # average across output tokens and head
            avg = (summed / N).mean(dim=0).squeeze(0)

            probs = avg / avg.sum()

            # reshape → 2D grid, then upsample to full image size
            grid = probs.reshape(side, side).cpu().numpy()
            patch_size = 14
            full = side * patch_size
            vis = np.repeat(np.repeat(grid, patch_size, axis=0),
                            patch_size, axis=1)

            # resize original image
            img_resized = image.resize((full, full), Image.BILINEAR)

            # Convert image to float in [0,1]
            image_array = np.array(img_resized).astype(np.float32) / 255.0  # shape: (H, W, 3)

            mask2d = np.repeat(np.repeat(grid, patch_size, axis=0),
                               patch_size, axis=1)            # (full, full)

            gamma      = 0.4
            attn_mask  = ((mask2d / global_max).clip(0, 1) ** gamma)[..., None]

            base_dark = image_array * 0.3
            blended = base_dark * (1 - attn_mask) + image_array * attn_mask
            blended = (blended * 255).clip(0, 255).astype(np.uint8)

            tiles.append(Image.fromarray(blended)) 

            # # Plot the masked image
            # fig, ax = plt.subplots(figsize=(5, 5))
            # ax.imshow(blended)
            # ax.axis("off")
            # ax.set_title(f"{image_id} — Layer {layer_idx}", fontsize=10)

            # save_path = os.path.join(save_dir, f"{image_id}_{layer_idx}_out2img.png")
            # plt.tight_layout()
            # plt.savefig(save_path, dpi=300)
            # plt.close()
            # # print(f"Saved Figure {image_id}")
        save_layer_grid(tiles, image_id, save_dir, cols=8)

def question_to_image(image_dir, weight_path, save_dir): 
    save_dir = os.path.join(save_dir, "q2img")
    os.makedirs(save_dir, exist_ok=True)
    
    # get data for corresponding batch
    data = torch.load(os.path.expanduser(weight_path))
    attn_weights, image_infos, all_image_ids = data['attn_weights'][0], data['image_infos'], data['all_image_ids']

    # get all images associated with batch and process them to 336 x 336
    images = []
    for image_id in all_image_ids :
        image_path = os.path.join(os.path.expanduser(image_dir), image_id + ".jpg")
        image = Image.open(image_path)
        image = expand2square(image, background_color=(0, 0, 0))
        images.append(image)
    
    # collect question-to-image weights for each image
    for batch_idx, image in enumerate(images) :
        image_id = all_image_ids[batch_idx]

        for layer_idx in range(len(attn_weights)) :
            all_heads_attn = attn_weights[layer_idx][batch_idx]
            sq_len = all_heads_attn.shape[-1]

            image_start = image_infos[batch_idx]['start_index']
            image_len = image_infos[batch_idx]['num_patches']
            image_end = image_start + image_len

            image_indices = list(range(image_start, image_end))

            question_span_1 = list(range(0, image_start))
            question_span_2 = list(range(image_end, sq_len))
            question_indices = question_span_1 + question_span_2

            # question - to - image token
            attn_q_to_img_all_heads = all_heads_attn[:, question_indices, :][:, :, image_indices] # H x Q x I
            avg_attn = attn_q_to_img_all_heads.mean(dim=(0, 1)) # I
            
            side = int(image_len ** 0.5)
            assert side * side == image_len 
            # —————— PLOT & SAVE THE RAW PATCH‐LEVEL HEATMAP ——————
            attn_map = avg_attn.reshape(side, side).cpu().numpy()  # shape (side, side)

            patch_size = 14

            # 2. Resize the square image so that its size = (side * patch_size, side * patch_size).
            full_size = side * patch_size  # e.g. 24 * 14 = 336
            image_resized = image.resize((full_size, full_size), resample=Image.BILINEAR)

            # 3. Normalize attn_map to sum to 1 (probability)
            attn_sum = attn_map.sum()
            if attn_sum > 1e-6:
                attn_norm = attn_map / attn_sum
            else:
                attn_norm = np.zeros_like(attn_map)

            # === REFACTORED VISUALIZATION: patch-wise upscaling via np.repeat ===
            patch_size = 14
            side = attn_norm.shape[0]
            full_size = side * patch_size  # e.g. 24 * 14 = 336

            # 4. Upscale each patch into a patch_size×patch_size block
            attn_patch_vis = np.repeat(
                np.repeat(attn_norm, patch_size, axis=0),
                patch_size, axis=1
            )  # shape: (full_size, full_size)

            # 5. Resize original image to match full_size
            image_resized = image.resize((full_size, full_size), Image.BILINEAR)

            image_array = np.array(image_resized).astype(np.float32) / 255.0  # (336, 336, 3)
            
            gamma = 0.3
            attn_boosted = attn_patch_vis ** gamma
            attn_mask = attn_boosted[..., None]

            base_dark = image_array * 0.3
            blended = base_dark * (1 - attn_mask) + image_array * attn_mask 
            blended = (blended * 255).clip(0, 255).astype(np.uint8)


            # Plot and save
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(blended)
            ax.axis("off")
            ax.set_title(f"{image_id} — Layer {layer_idx}", fontsize=10)

            save_path = os.path.join(save_dir, f"{image_id}_{layer_idx}_q2img.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved Figure {image_id}")
            

def main() :
    image_dir = sys.argv[1]
    weight_path = sys.argv[2]
    save_dir = sys.argv[3]
    os.makedirs(save_dir, exist_ok=True)

    gmax = get_global_max_prob(weight_path)  

    output_to_image(image_dir, weight_path, save_dir, gmax)
    # question_to_image(image_dir, weight_path, save_dir)
                
    print("✅ Done saving weight overlay!")


if __name__ == "__main__" :
    main()