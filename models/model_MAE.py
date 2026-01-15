import random

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from models.utils import get_2d_sincos_pos_embed


class Adpative_MAE_k_center(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=960,
                 embed_dim=768, depth=8, num_heads=12, clu_depth=1,
                 # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, center_num=8, sigma=2):
        super(Adpative_MAE_k_center, self).__init__()
        self.in_chans = in_chans
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim, requires_grad=False))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.inpainting_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def replace_mask(self, x, mask):
        masked_x = torch.where(mask.unsqueeze(-1) == 0., self.mask_token, x)
        return masked_x

    def add_jitter(self, feature_tokens, scale, prob):
        # feature_tokens = self.my_patchify(feature_tokens, p=1)
        if random.uniform(0, 1) <= prob:
            batch_size, num_tokens, dim_channel = feature_tokens.shape
            feature_norms = (
                    feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((batch_size, num_tokens, dim_channel), device=torch.device(feature_tokens.device))
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def a_map(self, deep_feature, recon_feature):
        # recon_feature = self.Roncon_model.unpatchify(pre_feature)
        batch_size = recon_feature.shape[0]
        dis_map = torch.mean((deep_feature - recon_feature) ** 2, dim=1, keepdim=True)
        dis_map = nn.functional.interpolate(dis_map, size=(288, 288), mode="bilinear", align_corners=True).squeeze(1)
        dis_map = dis_map.clone().cpu().detach().numpy()

        dir_map = 1 - torch.nn.CosineSimilarity()(deep_feature, recon_feature)
        dir_map = dir_map.reshape(batch_size, 1, deep_feature.shape[-2], deep_feature.shape[-1])
        dir_map = nn.functional.interpolate(dir_map, size=(288, 288), mode="bilinear", align_corners=True).squeeze(1)
        dir_map = dir_map.clone().cpu().detach().numpy()
        # print(deep_feature.permute(1, 0, 2, 3).shape)
        # ssim_map = torch.mean(ssim(deep_feature.permute(1, 0, 2, 3), recon_feature.permute(1, 0, 2, 3)), dim=0, keepdim=True)
        # # print(ssim_map.shape)
        # ssim_map = nn.functional.interpolate(ssim_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        # ssim_map = ssim_map.clone().squeeze(0).cpu().detach().numpy()
        return dis_map, dir_map

    def forward_encoder(self, x, stage):

        # masked_x = self.replace_mask(x, mask)
        masked_x = self.patch_embed(x)

        if stage == 'train':
            masked_x = self.add_jitter(masked_x, 20, 1)
        masked_x = masked_x + self.pos_embed
        if stage=='train':
            for blk in self.blocks:
                # print(masked_x.shape)
                masked_x = blk(masked_x)
            masked_x = self.norm(masked_x)
            masked_x = self.inpainting_pred(masked_x)
            return masked_x, None
        elif stage=='test':
            for blk in self.blocks:
                masked_x = blk(masked_x)
            masked_x = self.norm(masked_x)
            masked_x = self.inpainting_pred(masked_x)
            return masked_x, None


    def forward_loss(self, imgs, pred, pred_nor):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        # pred = self.unpatchify(pred)
        N, L, _ = target.shape
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        dis_loss = (pred - target) ** 2
        dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        dir_loss = 1 - torch.nn.CosineSimilarity(-1)(pred, target)
        # loss_g = dir_loss.mean() + dis_loss.mean()
        loss_g = dir_loss.mean()
        return loss_g

    def forward(self, masked, imgs, stage):
        pred_mask, pred_normal = self.forward_encoder(masked, stage)
        if stage == "train":
            loss = self.forward_loss(imgs, pred_mask, pred_normal)
        else:
            loss = 0.
        return loss, pred_mask


