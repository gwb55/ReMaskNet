import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import random
import numpy as np
import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import time
import copy
from mvtec import *
from ReMaskNet import *
from utils import *
from utils_eval import *


def get_transforms(resize, image_size):
    train_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform


def create_models(args):
    print(">>> Initializing Models...")
    RMN_embed = ReMaskNet_embed(layers_to_extract=args.feature_layers, image_size=args.image_size,
                                target_embed_dimension=args.target_embed_dim).to(args.device)

    RMN_proj = Projection(in_planes=args.target_embed_dim, out_planes=args.target_embed_dim).to(args.device)

    opt_emb = optim.AdamW(RMN_proj.parameters(), lr=0.0001)
    sch_emb = optim.lr_scheduler.CosineAnnealingLR(opt_emb, args.epochs, args.lr * 0.01, verbose=False)

    feature_h, feature_w = RMN_embed.feature_shapes[0]

    RMN_rec = ReMaskNet_reconstruct(in_planes=args.target_embed_dim, latent_dim=250).to(args.device)
    opt_rec = optim.Adam(RMN_rec.parameters(), lr=args.lr, weight_decay=1e-5)
    sch_rec = optim.lr_scheduler.CosineAnnealingLR(opt_rec, args.epochs, args.lr * 0.4, verbose=False)

    RMN_inp = Adpative_MAE_k_center(img_size=feature_h, in_chans=args.target_embed_dim,
                                    patch_size=3, depth=8, center_num=8, sigma=0.5, clu_depth=1).to(args.device)
    opt_inp = optim.AdamW(RMN_inp.parameters(), lr=args.lr, betas=(0.9, 0.95))
    sch_inp = optim.lr_scheduler.CosineAnnealingLR(opt_inp, args.epochs, args.lr * 0.4, verbose=False)

    anomaly_generator = AnomalyGenerator(
        noise_mean=0, noise_std=0.015,
        feature_h=feature_h, feature_w=feature_w,
        input_shape=[3, args.image_size, args.image_size]
    ).to(args.device)

    return (RMN_embed, RMN_proj, opt_emb, sch_emb), (RMN_rec, opt_rec, sch_rec), (RMN_inp, opt_inp, sch_inp), anomaly_generator


def train_class(args, class_name):
    print(f"\n{'=' * 50}\nTraining class: {class_name}\n{'=' * 50}")

    (RMN_embed, RMN_proj, opt_emb, sch_emb), \
        (RMN_rec, opt_rec, sch_rec), \
        (RMN_inp, opt_inp, sch_inp), \
        anomaly_generator = create_models(args)

    train_transform, test_transform = get_transforms(args.resize, args.image_size)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        MVTecDataset(args.data_root, class_name, 'train', train_transform, args.image_size, resize_param=args.resize),
        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False,
        worker_init_fn=seed_worker, generator=g
    )
    test_loader = DataLoader(
        MVTecDataset(args.data_root, class_name, 'test', test_transform, args.image_size, resize_param=args.resize),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
        worker_init_fn=seed_worker, generator=g
    )

    log_dir = os.path.join(args.save_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{class_name}_training_log.csv")
    log_file = open(log_file_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        'Epoch', 'Loss_Rec', 'Loss_Inp',
        'Img_AUROC', 'Pix_AUROC', 'Img_AP', 'Pix_AP',
        'Best_Img_AUROC', 'Best_Pix_AUROC'
    ])
    log_file.flush()

    best_score = 0.0
    best_img_auroc = 0.0
    best_pix_auroc = 0.0

    layer_str = "_".join(str(l) for l in args.feature_layers)
    base_name = f"{class_name}_layers{layer_str}"

    aug_scores_cache, fake_scores_cache, true_scores_cache = [], [], []
    aug_feats_cache, fake_feats_cache, true_feats_cache = [], [], []

    print(f"\n>>> Starting Combined Training (Epoch 1 - {args.epochs})")

    for epoch in range(args.epochs):
        start_time = time.time()

        RMN_embed.eval()
        RMN_proj.train()
        RMN_rec.train()
        RMN_inp.eval()

        step_a_loss = 0.0

        aug_scores_cache.clear()
        fake_scores_cache.clear()
        true_scores_cache.clear()
        aug_feats_cache.clear()
        fake_feats_cache.clear()
        true_feats_cache.clear()

        with tqdm(train_loader, desc=f"Ep {epoch + 1} [A: Rec Train & Gen]", leave=False) as pbar:
            for images, _ in pbar:
                images = images.to(args.device)
                opt_emb.zero_grad()
                opt_rec.zero_grad()
                with torch.no_grad():
                    true_feats, patch_shapes = RMN_embed(images)
                true_feats = RMN_proj(true_feats)
                true_feats = restore_feature_map(true_feats, patch_shapes[0])

                anomaly_paths = sorted(glob.glob(args.dtd_path + "/*/*.jpg"))
                aug_imgs, fake_feats, _, _ = anomaly_generator(images, true_feats, anomaly_paths)
                with torch.no_grad():
                    aug_feats, _ = RMN_embed(aug_imgs)
                aug_feats = RMN_proj(aug_feats)
                aug_feats = restore_feature_map(aug_feats, patch_shapes[0])

                true_rec, t_loss_ssm = RMN_rec(true_feats)
                aug_rec, a_loss_ssm = RMN_rec(aug_feats)
                fake_rec, f_loss_ssm = RMN_rec(fake_feats)

                l_feat_t = torch.mean((true_rec - true_feats) ** 2)
                l_feat_a = torch.mean((aug_rec - true_feats) ** 2)
                l_feat_f = torch.mean((fake_rec - true_feats) ** 2)

                loss_rec = l_feat_t + l_feat_a + l_feat_f + \
                           t_loss_ssm + a_loss_ssm + f_loss_ssm

                loss_rec.backward()
                opt_rec.step()
                opt_emb.step()

                step_a_loss += loss_rec.item() * images.size(0)

                with torch.no_grad():
                    t_score = torch.mean((true_rec.detach() - true_feats.detach()) ** 2, dim=1, keepdim=True)
                    a_score = torch.mean((aug_rec.detach() - aug_feats.detach()) ** 2, dim=1, keepdim=True)
                    f_score = torch.mean((fake_rec.detach() - fake_feats.detach()) ** 2, dim=1, keepdim=True)

                    true_scores_cache.append(t_score.cpu())
                    aug_scores_cache.append(a_score.cpu())
                    fake_scores_cache.append(f_score.cpu())

                    true_feats_cache.append(true_feats.detach().cpu())
                    aug_feats_cache.append(aug_feats.detach().cpu())
                    fake_feats_cache.append(fake_feats.detach().cpu())

        RMN_rec.eval()
        RMN_proj.eval()
        RMN_inp.train()
        RMN_embed.eval()

        step_b_loss = 0.0

        for iteration in range(args.iteration):
            current_min = float('inf')
            current_max = float('-inf')
            for score_list in [aug_scores_cache, fake_scores_cache, true_scores_cache]:
                for s in score_list:
                    s_min, s_max = s.min().item(), s.max().item()
                    if s_min < current_min: current_min = s_min
                    if s_max > current_max: current_max = s_max
            denom = current_max - current_min if current_max != current_min else 1e-5
            num_batches = len(aug_scores_cache)

            with tqdm(range(num_batches), desc=f"Ep {epoch + 1} [B: Inp Train Iter {iteration + 1}]", leave=False) as pbar_b:
                for i in pbar_b:
                    am_score = aug_scores_cache[i].to(args.device)
                    fm_score = fake_scores_cache[i].to(args.device)
                    tm_score = true_scores_cache[i].to(args.device)

                    af = aug_feats_cache[i].to(args.device)
                    ff = fake_feats_cache[i].to(args.device)
                    tf = true_feats_cache[i].to(args.device)

                    # opt_emb.zero_grad()
                    opt_inp.zero_grad()

                    def get_mask(score_map):
                        norm = (score_map - current_min) / denom
                        return torch.where(norm > 0.3, 1.0, 0.0)

                    am_mask = get_mask(am_score)
                    fm_mask = get_mask(fm_score)
                    tm_mask = get_mask(tm_score)

                    m_tf = tf * (1 - tm_mask)
                    m_ff = ff * (1 - fm_mask)
                    m_af = af * (1 - am_mask)

                    _, t_inp = RMN_inp(m_tf, tf, 'train')
                    _, f_inp = RMN_inp(m_ff, tf, 'train')
                    _, a_inp = RMN_inp(m_af, tf, 'train')

                    t_inp = RMN_inp.unpatchify(t_inp)
                    f_inp = RMN_inp.unpatchify(f_inp)
                    a_inp = RMN_inp.unpatchify(a_inp)

                    def calc_inp_loss(inp, target):
                        loss_cos = 1 - torch.nn.CosineSimilarity()(target, inp)
                        _, loss_ssim_map = loss_ssim(inp, target)
                        loss_ssim_map = loss_ssim_map.mean(1)
                        return (loss_cos * loss_ssim_map).mean()

                    loss_t = calc_inp_loss(t_inp, tf)
                    loss_f = calc_inp_loss(f_inp, tf)
                    loss_a = calc_inp_loss(a_inp, tf)

                    loss_inp = loss_t + loss_f + loss_a

                    loss_inp.backward()
                    # opt_emb.step()
                    opt_inp.step()

                    step_b_loss += loss_inp.item() * tf.size(0)

                    with torch.no_grad():
                        def calc_new_score(inp, target):
                            cos_map = 1 - torch.nn.CosineSimilarity(dim=1)(target, inp)
                            _, ssim_diff = loss_ssim(inp, target)
                            ssim_map = ssim_diff.mean(1)
                            return (cos_map * ssim_map).unsqueeze(1)

                        aug_scores_cache[i] = calc_new_score(a_inp, af).cpu()
                        fake_scores_cache[i] = calc_new_score(f_inp, ff).cpu()
                        true_scores_cache[i] = calc_new_score(t_inp, tf).cpu()

        avg_loss_rec = step_a_loss / len(train_loader.dataset)
        avg_loss_inp = step_b_loss / (len(train_loader.dataset) * args.iteration)

        sch_rec.step()
        sch_inp.step()

        curr_img_auroc, curr_pix_auroc = "", ""
        curr_img_ap, curr_pix_ap = "", ""

        if (epoch + 1) % args.eval_epoch == 0:
            print(f"\n   ---> Running Evaluation at Epoch {epoch + 1}...")

            pred_maps, gt_lbls, gt_msks, _ = predict_step(
                RMN_embed, RMN_proj, RMN_rec, RMN_inp, test_loader, args.device, args.image_size, num_iterations=args.eval_iteration
            )
            metrics = evaluate_step(pred_maps, gt_lbls, gt_msks)

            curr_img_auroc = metrics['img_auroc']
            curr_pix_auroc = metrics['pix_auroc']
            curr_img_ap = metrics['img_ap']
            curr_pix_ap = metrics['pix_ap']

            print(f"   [Image] AUROC: {curr_img_auroc:.4f} | AP: {curr_img_ap:.4f}")
            print(f"   [Pixel] AUROC: {curr_pix_auroc:.4f} | AP: {curr_pix_ap:.4f}")

            current_score = curr_img_auroc + curr_pix_auroc

            if current_score > best_score:
                best_score = current_score
                best_img_auroc = curr_img_auroc
                best_pix_auroc = curr_pix_auroc

                torch.save(RMN_embed.state_dict(), os.path.join(args.save_root, f"{base_name}_embed_best.pth"))
                torch.save(RMN_proj.state_dict(), os.path.join(args.save_root, f"{base_name}_proj_best.pth"))
                torch.save(RMN_rec.state_dict(), os.path.join(args.save_root, f"{base_name}_rec_best.pth"))
                torch.save(RMN_inp.state_dict(), os.path.join(args.save_root, f"{base_name}_inp_best.pth"))

                print(f"   >>> New Best Score: {best_score:.4f} (Saved)")

        log_writer.writerow([
            epoch + 1,
            f"{avg_loss_rec:.6f}",
            f"{avg_loss_inp:.6f}",
            f"{curr_img_auroc:.4f}" if curr_img_auroc != "" else "",
            f"{curr_pix_auroc:.4f}" if curr_pix_auroc != "" else "",
            f"{curr_img_ap:.4f}" if curr_img_ap != "" else "",
            f"{curr_pix_ap:.4f}" if curr_pix_ap != "" else "",
            f"{best_img_auroc:.4f}",
            f"{best_pix_auroc:.4f}"
        ])
        log_file.flush()

        print(
            f"   Epoch {epoch + 1} Done. Time: {time.time() - start_time:.2f}s | L_Rec: {avg_loss_rec:.4f} | L_Inp: {avg_loss_inp:.4f}")

    log_file.close()
    print(f"Training Complete. Best Img: {best_img_auroc:.4f}, Best Pix: {best_pix_auroc:.4f}")
    return RMN_embed
