import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image
from matplotlib.patches import Rectangle

import wget
import os

# HAT Model
from ..HAT.hat.models import HumanAttnTransformer
from ..HAT.common.config import JsonConfig
from ..HAT.hat.evaluation import scanpath_decode

# Helpers
from ..HAT.inference import actions2scanpaths, plot_scanpath

def get_JsonConfigs(viewmode) :

    if viewmode == 'TP':
        # Target-present visual search (TP)
        hparams = JsonConfig('configs/coco_search18_dense_SSL_TP.json')
    elif viewmode == 'TA':
        # Target-absent visual search (TA)
        hparams = JsonConfig('configs/coco_search18_dense_SSL_TA.json')
    elif viewmode == 'FV': 
        # Free viewing (FV)
        hparams = JsonConfig('configs/coco_freeview_dense_SSL.json')
    else:
        raise NotImplementedError

def load_HAT_pretrained(viewmode) :
    HAT_path = 'HAT'
    if not os.path.exists(f"./{HAT_path}/checkpoints/HAT_{TAP}.pt"):
        if not os.path.exists("./{HAT_path}/checkpoints/"):
            os.mkdir('./{HAT_path}/checkpoints')

        print('downloading model checkpoint...')
        url = f"http://vision.cs.stonybrook.edu/~cvlab_download/HAT/HAT_{TAP}.pt"
        wget.download(url, 'checkpoints/')

    if not os.path.exists(f"./{HAT_path}/pretrained_models/M2F_R50_MSDeformAttnPixelDecoder.pkl"):
        if not os.path.exists("./{HAT_path}/pretrained_models/"):
            os.mkdir('./{HAT_path}/pretrained_models')

        print('downloading pretrained model weights...')
        url = f"http://vision.cs.stonybrook.edu/~cvlab_download/HAT/pretrained_models/M2F_R50_MSDeformAttnPixelDecoder.pkl"
        wget.download(url, 'pretrained_models/')
        url = f"http://vision.cs.stonybrook.edu/~cvlab_download/HAT/pretrained_models/M2F_R50.pkl"
        wget.download(url, 'pretrained_models/')


def main() :
    # view mode
    TAP = 'FV' # TP, TA

    # prepare model configs 
    hparams = get_JsonConfigs(TAP)
    load_HAT_pretrained(TAP)

    # HAT model
    model_HAT = HumanAttnTransformer(
        hparams.Data,
        num_decoder_layers=hparams.Model.n_dec_layers,
        hidden_dim=hparams.Model.embedding_dim,
        nhead=hparams.Model.n_heads,
        ntask=1 if hparams.Data.TAP == 'FV' else 18,
        tgt_vocab_size=hparams.Data.patch_count + len(hparams.Data.special_symbols),
        num_output_layers=hparams.Model.num_output_layers,
        separate_fix_arch=hparams.Model.separate_fix_arch,
        train_encoder=hparams.Train.train_backbone,
        train_pixel_decoder=hparams.Train.train_pixel_decoder,
        use_dino=hparams.Train.use_dino_pretrained_model,
        dropout=hparams.Train.dropout,
        dim_feedforward=hparams.Model.hidden_dim,
        parallel_arch=hparams.Model.parallel_arch,
        dorsal_source=hparams.Model.dorsal_source,
        num_encoder_layers=hparams.Model.n_enc_layers,
        output_centermap="centermap_pred" in hparams.Train.losses,
        output_saliency="saliency_pred" in hparams.Train.losses,
        output_target_map="target_map_pred" in hparams.Train.losses,
        transfer_learning_setting=hparams.Train.transfer_learn,
        project_queries=hparams.Train.project_queries,
        is_pretraining=False,
        output_feature_map_name=hparams.Model.output_feature_map_name)

    # load checkpoint

    checkpoint_paths = {
        'TP': "./checkpoints/HAT_TP.pt", # target present
        'TA': "./checkpoints/HAT_TA.pt", # target absent
        'FV': "./checkpoints/HAT_FV.pt" # free viewing
    }

    ckpt = torch.load(checkpoint_paths[hparams.Data.TAP], map_location='cpu')
    bb_weights = ckpt['model']
    bb_weights_new = bb_weights.copy()
    for k, v in bb_weights.items() :
        if "stages." in k :
            new_k = k.replace("stages.", "")
            bb_weights_new[new_k] = v 
            bb_weights_new.pop(k)

    model_HAT.load_state_dict(bb_weights_new)

        # load test image
    orig_img = Image.open('../room.jpg')

    original_resolution = orig_img.size
    X_ratio = original_resolution[0] / 512 
    Y_ratio = original_resolution[1] / 320 

    img = orig_img.resize((512, 320))

    plt.imshow(orig_img)
    plt.axis('off')

    # preprocess
    size = (hparams.Data.im_h, hparams.Data.im_w)
    transform = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = torch.unsqueeze(transform(img), 0)

    # Load preset task name and ids
    preset_tasks = np.load("./HAT/demo/all_task_ids.npy", allow_pickle=True).item()

    # NOTE: Not really useful at the moment since we use FV
    task = 'bottle'
    sample_action = False
    if task in preset_tasks and TAP != 'FV':
        task_id = torch.tensor([preset_tasks[task]], dtype=torch.long)
    else:
        task_id = torch.tensor([0], dtype=torch.long)
    normalized_sp, _ = scanpath_decode(model, img_tensor, task_id, hparams.Data, sample_action=sample_action, center_initial=True)
    scanpath = actions2scanpaths(normalized_sp, hparams.Data.im_h, hparams.Data.im_w)[0]
    
    scanpath['X'] = scanpath['X'] * X_ratio
    scanpath['Y'] = scanpath['Y'] * Y_ratio

    plot_scanpath(orig_img, scanpath['X'], scanpath['Y'])

if __name__ == "__main__" :
    main()