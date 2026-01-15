import os
import copy
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import math
import torch.nn.functional as F
from PIL import Image
from models.model_MAE import *
from models.torch_ssmctb import *
from models.feat_cae import *


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class FeatureExtractor(nn.Module):
    def __init__(self, layers_to_extract):
        super().__init__()

        self.wide_resnet = models.wide_resnet50_2(weights=None)
        weight_path = './wide_resnet50_2-95faca4d.pth'
        if os.path.exists(weight_path):
            print(f"Loading pretrained weights from local file: {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu')
            self.wide_resnet.load_state_dict(state_dict)
        else:
            print(f"Warning: Local weight file not found at {weight_path}. Trying to download...")
            self.wide_resnet = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)

        self.features = nn.Sequential(
            *list(self.wide_resnet.children())[:-2]
        )

        for param in self.features.parameters():
            param.requires_grad = False

        self.layers_to_extract = layers_to_extract

        self.layer_outputs = {}

        self._register_hooks()

    def _register_hooks(self):
        layer_mapping = {
            1: self.features[4],
            2: self.features[5],
            3: self.features[6],
            4: self.features[7]
        }

        for layer_idx in self.layers_to_extract:
            if layer_idx not in layer_mapping:
                raise ValueError(f"Invalid layer index: {layer_idx}. Must be between 1-4.")

            layer = layer_mapping[layer_idx]
            layer.register_forward_hook(self._get_hook(layer_idx))

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            self.layer_outputs[layer_idx] = output
        return hook

    def forward(self, x):
        self.layer_outputs = {}
        _ = self.features(x)

        return [self.layer_outputs[idx] for idx in sorted(self.layers_to_extract)]

    def feature_dimensions(self, input_shape):
        dummy_input = torch.ones(1, *input_shape).to(next(self.parameters()).device)
        with torch.no_grad():
            features = self.forward(dummy_input)
        channels_list = [feature.shape[1] for feature in features]
        spatial_dims_list = [feature.shape[-2:] for feature in features]

        return channels_list, spatial_dims_list


class ReMaskNet_embed(nn.Module):
    def __init__(self, layers_to_extract=None, image_size=288, target_embed_dimension=1536):
        super().__init__()
        if layers_to_extract is None:
            layers_to_extract = [2, 3]
        self.feature_extractor = FeatureExtractor(layers_to_extract)
        self.patch_maker = PatchMaker(patchsize=3, stride=1)
        self.feature_dimensions, self.feature_shapes = self.feature_extractor.feature_dimensions([3, image_size, image_size])
        self.preprocessing = Preprocessing(self.feature_dimensions, target_embed_dimension)
        self.aggregator = Aggregator(target_embed_dimension)
        # self.pre_projection = Projection(target_embed_dimension)

    def forward(self, x):
        features = self.feature_extractor(x)

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        features = self.preprocessing(features)
        features = self.aggregator(features)
        # features = self.pre_projection(features)


        return features, patch_shapes


class ReMaskNet_reconstruct(nn.Module):
    def __init__(self, in_planes=1536, latent_dim=400):
        super().__init__()
        self.attition = SSMCTB(in_planes)
        self.reconstruct = FeatCAE(in_channels=in_planes, latent_dim=latent_dim)

        self.apply(init_weight)

    def forward(self, fake_feats):
        fake_feats_att, loss_ssmctb = self.attition(fake_feats)
        reconstructed = self.reconstruct(fake_feats)

        return reconstructed, loss_ssmctb


class ReMaskNet_inpainting(nn.Module):
    def __init__(self, image_size=36, in_chans=1536):
        super().__init__()

        self.inpainting_model = Adpative_MAE_k_center(img_size=image_size, in_chans=in_chans, patch_size=3, depth=8, center_num=8, sigma=0.5, clu_depth=1)

    def forward(self, masked, imgs, stage):
        loss_inp, img_inp = self.inpainting_model(masked, imgs, stage)

        return loss_inp, img_inp


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):

        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                                s + 2 * padding - 1 * (self.patchsize - 1) - 1
                        ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class Projection(torch.nn.Module):

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc",
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn",
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):

        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x