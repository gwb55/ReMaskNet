import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import math
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from mvtec import *
from train import *


def load_models(args, class_name):
    print(f"Loading models for class: {class_name}...")
    RMN_embed = ReMaskNet_embed(layers_to_extract=args.feature_layers, image_size=args.image_size,
                                target_embed_dimension=args.target_embed_dim).to(args.device)
    RMN_proj = Projection(in_planes=args.target_embed_dim, out_planes=args.target_embed_dim).to(args.device)
    feature_h, feature_w = RMN_embed.feature_shapes[0]
    RMN_rec = ReMaskNet_reconstruct(in_planes=args.target_embed_dim, latent_dim=250).to(args.device)
    RMN_inp = Adpative_MAE_k_center(img_size=feature_h, in_chans=args.target_embed_dim,
                                    patch_size=3, depth=8, center_num=8, sigma=0.5, clu_depth=1).to(args.device)

    layer_str = "_".join(str(l) for l in args.feature_layers)
    base_name = f"{class_name}_layers{layer_str}"

    try:
        RMN_embed.load_state_dict(
            torch.load(os.path.join(args.save_root, f"{base_name}_embed_best.pth"), map_location=args.device,
                       weights_only=True))
        RMN_proj.load_state_dict(
            torch.load(os.path.join(args.save_root, f"{base_name}_proj_best.pth"), map_location=args.device,
                       weights_only=True))
        RMN_rec.load_state_dict(
            torch.load(os.path.join(args.save_root, f"{base_name}_rec_best.pth"), map_location=args.device,
                       weights_only=True))
        RMN_inp.load_state_dict(
            torch.load(os.path.join(args.save_root, f"{base_name}_inp_best.pth"), map_location=args.device, weights_only=True))
        print(">>> All weights loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Weights not found. Check path: {args.save_root}")
        raise e

    return RMN_embed, RMN_proj, RMN_rec, RMN_inp


def test_class(args, class_name):
    viz_dir = os.path.join(args.result_root, 'visualization', class_name)
    os.makedirs(viz_dir, exist_ok=True)

    RMN_embed, RMN_proj, RMN_rec, RMN_inp = load_models(args, class_name)

    _, test_transform = get_transforms(args.resize, args.image_size)
    test_dataset = MVTecDataset(args.data_root, class_name, 'test', test_transform, args.image_size, resize_param=args.resize)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    anomaly_maps, gt_labels, gt_masks, data_cache = predict_step(
        RMN_embed, RMN_proj, RMN_rec, RMN_inp, test_loader, args.device, args.image_size, num_iterations=args.eval_iteration
    )

    metrics = evaluate_step(anomaly_maps, gt_labels, gt_masks)

    print(f"[{class_name}] Generating visualizations...")

    image_paths = []
    if hasattr(test_dataset, 'x'):
        image_paths = test_dataset.x
    elif hasattr(test_dataset, 'image_paths'):
        image_paths = test_dataset.image_paths
    else:
        print("Warning: Could not find image paths in dataset, using indices for naming.")
        image_paths = [f"unknown/img_{i}.png" for i in range(len(test_dataset))]

    f_min = anomaly_maps.min()
    f_max = anomaly_maps.max()
    f_denom = f_max - f_min if f_max != f_min else 1e-5

    global_idx = 0
    for batch_data in tqdm(data_cache, desc="Viz"):
        orig_imgs = batch_data['orig_img']
        gt_masks = batch_data['masks']

        batch_size = orig_imgs.shape[0]
        for b in range(batch_size):

            img_tensor = orig_imgs[b]
            gt_mask_tensor = gt_masks[b]

            if global_idx >= len(image_paths):
                break

            full_path = image_paths[global_idx]
            current_anomaly_map = anomaly_maps[global_idx]

            parent_dir = os.path.basename(os.path.dirname(full_path))
            file_name = os.path.splitext(os.path.basename(full_path))[0]
            save_name = f"{parent_dir}_{file_name}.png"
            save_path = os.path.join(viz_dir, save_name)

            map_norm = (current_anomaly_map - f_min) / f_denom
            binary_res = (map_norm > 0.5).astype(int)

            visualize_single_sample(
                save_path,
                img_tensor,
                gt_mask_tensor.numpy(),
                map_norm,
                binary_res
            )

            global_idx += 1

    metrics['class'] = class_name
    return metrics
