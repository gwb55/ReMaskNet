import os
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import numpy as np
from torchvision import transforms


class MVTecDataset(Dataset):
    def __init__(self, root_dir, class_name, phase='train', transform=None, image_size=288, resize_param=None):
        self.root_dir = os.path.join(root_dir, class_name)
        self.transform = transform
        self.phase = phase
        self.class_name = class_name
        self.image_size = image_size
        self.resize_param = resize_param if resize_param is not None else image_size
        if phase == 'train':
            self.image_dir = os.path.join(self.root_dir, 'train', 'good')
            self.image_paths = [os.path.join(self.image_dir, f)
                                for f in os.listdir(self.image_dir)
                                if f.endswith(('.png', '.jpg'))]
            self.labels = [0] * len(self.image_paths)
        else:  # test
            self.image_paths = []
            self.gt_paths = []
            self.labels = []
            test_dir = os.path.join(self.root_dir, 'test')

            for defect_type in os.listdir(test_dir):
                defect_dir = os.path.join(test_dir, defect_type)

                if not os.path.isdir(defect_dir):
                    continue

                for f in os.listdir(defect_dir):
                    if f.endswith(('.png', '.jpg')):
                        img_path = os.path.join(defect_dir, f)
                        self.image_paths.append(img_path)

                        if defect_type == 'good':
                            self.labels.append(0)
                            self.gt_paths.append(None)
                        else:
                            self.labels.append(1)
                            gt_file = f.split('.')[0] + '_mask.png'
                            gt_dir = os.path.join(self.root_dir, 'ground_truth', defect_type)
                            gt_path = os.path.join(gt_dir, gt_file)
                            self.gt_paths.append(gt_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.phase == 'train':
            return image, torch.tensor(0, dtype=torch.long)  # 训练时返回伪标签

        else:  # test
            label = self.labels[idx]
            gt_path = self.gt_paths[idx]

            if gt_path is None or not os.path.exists(gt_path):
                gt = torch.zeros(1, self.image_size, self.image_size)
            else:
                gt = Image.open(gt_path).convert('L')
                mask_transform = transforms.Compose([
                    transforms.Resize(self.resize_param, interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor()
                ])
                gt = mask_transform(gt)

                gt = (gt > 0).float()

            return image, gt, torch.tensor(label, dtype=torch.long)