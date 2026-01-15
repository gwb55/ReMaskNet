import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import loss_ssim, restore_feature_map


def predict_step_phase1(embed_model, proj_model, rec_model, test_loader, device, image_size):

    embed_model.eval()
    proj_model.eval()
    rec_model.eval()

    anomaly_maps = []
    gt_labels = []
    gt_masks = []

    with torch.no_grad():
        for idx, (images, val1, val2) in enumerate(tqdm(test_loader, desc="Eval Phase 1", leave=False)):
            images = images.to(device)
            if val1.dim() >= 3:
                true_masks_tensor, true_labels_tensor = val1, val2
            else:
                true_labels_tensor, true_masks_tensor = val1, val2

            if true_labels_tensor.dim() == 2: true_labels_tensor = true_labels_tensor.squeeze(1)
            if true_masks_tensor.dim() == 4: true_masks_tensor = true_masks_tensor.squeeze(1)

            true_feats, patch_shapes = embed_model(images)
            true_feats = proj_model(true_feats)
            true_feats = restore_feature_map(true_feats, patch_shapes[0])

            reconstructed, _ = rec_model(true_feats)

            error_map = torch.mean((reconstructed - true_feats) ** 2, dim=1, keepdim=True)

            error_map = F.interpolate(error_map, size=(image_size, image_size), mode='bilinear', align_corners=True)

            batch_maps = error_map.squeeze(1).cpu().numpy()

            for k in range(batch_maps.shape[0]):
                batch_maps[k] = gaussian_filter(batch_maps[k], sigma=4)

            anomaly_maps.append(batch_maps)
            gt_labels.extend(true_labels_tensor.cpu().numpy().astype(int))
            gt_masks.append(true_masks_tensor.cpu().numpy())

    anomaly_maps = np.concatenate(anomaly_maps, axis=0)
    gt_labels = np.array(gt_labels)
    gt_masks = np.concatenate(gt_masks, axis=0)

    gt_masks = (gt_masks > 0.5).astype(int)

    return anomaly_maps, gt_labels, gt_masks, None


def predict_step(embed_model, proj_model, rec_model, inp_model, test_loader, device, image_size, num_iterations=3):

    embed_model.eval()
    proj_model.eval()
    rec_model.eval()
    inp_model.eval()

    data_cache = []

    with torch.no_grad():
        for idx, (images, val1, val2) in enumerate(tqdm(test_loader, desc="Init Cache", leave=False)):
            images = images.to(device)
            if val1.dim() >= 3:
                true_masks_tensor, true_labels_tensor = val1, val2
            else:
                true_labels_tensor, true_masks_tensor = val1, val2

            if true_labels_tensor.dim() == 2: true_labels_tensor = true_labels_tensor.squeeze(1)
            if true_masks_tensor.dim() == 4: true_masks_tensor = true_masks_tensor.squeeze(1)

            true_feats, patch_shapes = embed_model(images)
            true_feats = proj_model(true_feats)
            true_feats = restore_feature_map(true_feats, patch_shapes[0])

            reconstructed, _ = rec_model(true_feats)
            initial_error_map = torch.mean((reconstructed - true_feats) ** 2, dim=1, keepdim=True)

            data_cache.append({
                'idx': idx,
                'true_feats': true_feats.detach().cpu(),
                'current_error': initial_error_map.detach().cpu(),
                'labels': true_labels_tensor.detach().cpu(),
                'masks': true_masks_tensor.detach().cpu(),
                'orig_img': images.detach().cpu()
            })
            del images, true_feats, reconstructed, initial_error_map

    for i in range(num_iterations):
        global_min = float('inf')
        global_max = float('-inf')
        for batch_data in data_cache:
            b_min = batch_data['current_error'].min().item()
            b_max = batch_data['current_error'].max().item()
            global_min = min(global_min, b_min)
            global_max = max(global_max, b_max)

        denominator = global_max - global_min if global_max != global_min else 1e-5

        for batch_data in tqdm(data_cache, desc=f"Refine {i + 1}", leave=False):
            true_feats = batch_data['true_feats'].to(device)
            current_error = batch_data['current_error'].to(device)

            norm_map = (current_error - global_min) / denominator
            input_mask = torch.where(norm_map > 0.3, 1.0, 0.0)

            masked_feat = true_feats * (1 - input_mask)
            _, inpainted_feats = inp_model(masked_feat, true_feats, 'test')
            inpainted_feats = inp_model.unpatchify(inpainted_feats)

            cos_sim = torch.nn.CosineSimilarity(dim=1)(true_feats, inpainted_feats)
            dir_map = 1 - cos_sim

            _, ssim_diff_map = loss_ssim(inpainted_feats, true_feats)
            ssim_diff_map = ssim_diff_map.mean(1)

            dir_map_un = dir_map.unsqueeze(1)
            ssim_diff_map_un = ssim_diff_map.unsqueeze(1)

            feat_h, feat_w = true_feats.shape[2], true_feats.shape[3]

            dir_map_large = F.interpolate(dir_map_un, size=(image_size, image_size),
                                          mode='bilinear', align_corners=True)
            ssim_diff_map_large = F.interpolate(ssim_diff_map_un, size=(image_size, image_size),
                                                mode='bilinear', align_corners=True)

            combined_large = dir_map_large * ssim_diff_map_large

            new_error_map = F.interpolate(combined_large, size=(feat_h, feat_w),
                                          mode='bilinear', align_corners=True)

            batch_data['current_error'] = new_error_map.detach().cpu()

            batch_data['dir_map'] = dir_map_un.detach().cpu()
            batch_data['ssim_diff_map'] = ssim_diff_map_un.detach().cpu()

            del true_feats, current_error, norm_map, input_mask, masked_feat, inpainted_feats, cos_sim, dir_map, ssim_diff_map
            del dir_map_un, ssim_diff_map_un, dir_map_large, ssim_diff_map_large, combined_large, new_error_map
            torch.cuda.empty_cache()

    anomaly_maps = []
    gt_labels = []
    gt_masks = []

    with torch.no_grad():
        for batch_data in tqdm(data_cache, desc="Finalize", leave=False):
            # final_error = batch_data['current_error'].to(device)
            # diff_map_resized = F.interpolate(final_error, size=(image_size, image_size), mode='bilinear',
            #                                  align_corners=True)
            # batch_maps = diff_map_resized.squeeze(1).cpu().numpy()
            # del final_error, diff_map_resized
            # torch.cuda.empty_cache()

            dir_map = batch_data['dir_map'].to(device)
            dir_map = F.interpolate(dir_map, size=(image_size, image_size), mode='bilinear',
                                    align_corners=True)
            dir_map_l = dir_map.squeeze(1).cpu().numpy()
            del dir_map
            torch.cuda.empty_cache()

            ssim_diff_map = batch_data['ssim_diff_map'].to(device)
            ssim_diff_map = F.interpolate(ssim_diff_map, size=(image_size, image_size), mode='bilinear',
                                          align_corners=True)
            ssim_diff_map_l = ssim_diff_map.squeeze(1).cpu().numpy()
            del ssim_diff_map
            torch.cuda.empty_cache()

            # for k in range(dir_map_l.shape[0]):
            #     dir_map_l[k] = gaussian_filter(dir_map_l[k], sigma=4)
            # for k in range(ssim_diff_map_l.shape[0]):
            #     ssim_diff_map_l[k] = gaussian_filter(ssim_diff_map_l[k], sigma=4)
            # batch_maps = dir_map_l * ssim_diff_map_l

            batch_maps = dir_map_l * ssim_diff_map_l

            # Gaussian Smoothing
            # for k in range(batch_maps.shape[0]):
            #     batch_maps[k] = gaussian_filter(batch_maps[k], sigma=4)

            anomaly_maps.append(batch_maps)
            gt_labels.extend(batch_data['labels'].numpy().astype(int))
            gt_masks.append(batch_data['masks'].numpy())

    anomaly_maps = np.concatenate(anomaly_maps, axis=0)
    gt_labels = np.array(gt_labels)
    gt_masks = np.concatenate(gt_masks, axis=0)
    gt_masks = (gt_masks > 0.5).astype(int)

    return anomaly_maps, gt_labels, gt_masks, data_cache


def evaluate_step(anomaly_maps, gt_labels, gt_masks):
    img_scores = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(axis=1)
    img_scores = (img_scores - img_scores.min()) / (img_scores.max() - img_scores.min())
    try:
        img_auroc = roc_auc_score(gt_labels, img_scores)
        img_ap = average_precision_score(gt_labels, img_scores)
    except:
        img_auroc, img_ap = 0.0, 0.0
    min_score = anomaly_maps.min()
    max_score = anomaly_maps.max()
    if max_score != min_score:
        anomaly_maps = (anomaly_maps - min_score) / (max_score - min_score)
    else:
        anomaly_maps = np.zeros_like(anomaly_maps)

    try:
        pix_auroc = roc_auc_score(gt_masks.flatten(), anomaly_maps.flatten())
        pix_ap = average_precision_score(gt_masks.flatten(), anomaly_maps.flatten())
    except:
        pix_auroc, pix_ap = 0.0, 0.0

    return {"img_auroc": img_auroc, "img_ap": img_ap, "pix_auroc": pix_auroc, "pix_ap": pix_ap}


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def visualize_single_sample(save_path, image_tensor, gt_mask, heatmap, binary_mask):
    orig_img = denormalize(image_tensor)
    if gt_mask.ndim == 3: gt_mask = gt_mask.squeeze()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(orig_img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    im3 = axes[2].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title("Heatmap")
    axes[2].axis('off')

    axes[3].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title("Segmentation")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)