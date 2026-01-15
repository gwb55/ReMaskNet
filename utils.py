import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import math
import cv2
import imgaug.augmenters as iaa
from typing import Tuple


class AnomalyGenerator(torch.nn.Module):
    def __init__(
            self,
            noise_mean: float,
            noise_std: float,
            feature_h: int,
            feature_w: int,
            input_shape: list,
            perlin_range: Tuple[int, int] = (0, 6),
    ):
        super().__init__()
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.min_perlin_scale = perlin_range[0]
        self.max_perlin_scale = perlin_range[1]
        self.height = feature_h
        self.width = feature_w
        self.input_shape = input_shape

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.perlin_height = self.next_power_2(self.height)
        self.perlin_width = self.next_power_2(self.width)

        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

    @staticmethod
    def next_power_2(num):
        return 1 << (num - 1).bit_length()

    def generate_perlin(self, batches, device) -> Tuple[Tensor, Tensor]:
        perlin_list, i_perlin_list = [], []
        for _ in range(batches):
            px = 2 ** torch.randint(self.min_perlin_scale, self.max_perlin_scale, (1,)).item()
            py = 2 ** torch.randint(self.min_perlin_scale, self.max_perlin_scale, (1,)).item()

            perlin_noise = rand_perlin_2d((self.perlin_height, self.perlin_width), (px, py))

            f_mask = F.interpolate(
                perlin_noise.view(1, 1, self.perlin_height, self.perlin_width),
                size=(self.height, self.width), mode="bilinear", align_corners=False
            )
            i_mask = F.interpolate(
                perlin_noise.view(1, 1, self.perlin_height, self.perlin_width),
                size=(self.input_shape[-2], self.input_shape[-1]), mode="bilinear", align_corners=False
            )

            threshold = 0.5
            perlin_list.append(torch.where(f_mask > threshold, 1.0, 0.0))
            i_perlin_list.append(torch.where(i_mask > threshold, 1.0, 0.0))

        return torch.cat(perlin_list).to(device), torch.cat(i_perlin_list).to(device)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        return iaa.Sequential([self.augmenters[i] for i in aug_ind])

    def forward(self, image: Tensor, features: Tensor, anomaly_source_paths: list) -> Tuple[
        Tensor, Tensor, Tensor, Tensor]:

        device = image.device
        b, c, f_h, f_w = features.shape

        idx = torch.randint(0, len(anomaly_source_paths), (1,)).item()
        src_img = cv2.imread(anomaly_source_paths[idx])
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        src_img = cv2.resize(src_img, (self.input_shape[-1], self.input_shape[-2]))


        aug = self.randAugmenter()
        aug_src = aug(image=src_img).astype(np.float32) / 255.0

        aug_src_t = torch.from_numpy(aug_src).permute(2, 0, 1).unsqueeze(0).to(device)
        aug_src_t = (aug_src_t - self.mean) / self.std

        f_perlin, i_perlin = self.generate_perlin(b, device)

        beta = torch.rand(1, device=device) * 0.8
        aug_image = (1 - i_perlin) * image + i_perlin * ((1 - beta) * aug_src_t + beta * image)

        noise = torch.normal(self.noise_mean, self.noise_std, size=features.shape, device=device, requires_grad=False)
        perturbed_features = features + noise * f_perlin

        return aug_image, perturbed_features, i_perlin, f_perlin


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])
            ),
            dim=-1,
        )
        % 1
    )
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = (
        lambda slice1, slice2: gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )
    dot = lambda grad, shift: (
        torch.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            dim=-1,
        )
        * grad[: shape[0], : shape[1]]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]
    )


def restore_feature_map(feats, spatial_size):

    feats = feats.reshape(-1, *spatial_size, feats.shape[-1])
    feats = feats.permute(0, -1, 1, 2)

    return feats.contiguous()


def compute_rec_map(reconstructed, original):

    scores = torch.mean((reconstructed - original) ** 2, dim=1)

    scores = scores.unsqueeze(1)

    return scores.detach()


def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_calc(img1, img2, window_size=11, size_average=True):

    if torch.max(img1) > 128:
        max_val = 255
    else:
        max_val = 1

    if torch.min(img1) < -0.5:
        min_val = -1
    else:
        min_val = 0
    l = max_val - min_val
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * l) ** 2
    C2 = (0.03 * l) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean(), ssim_map
    else:
        return ssim_map.mean(1).mean(1).mean(1), ssim_map

def loss_ssim(img1, img2):
    score, ssim_map = ssim_calc(img1, img2)
    return 1 - score, 1 - ssim_map