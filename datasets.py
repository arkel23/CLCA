import torch
from torchvision.datasets import ImageNet
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
#from timm.data.transforms import _pil_interp
from dataloaders.nabirds import NABirds
from dataloaders.nus_wide import NUSWide

from aug_factory import CutoutPIL

import os
import json
import yaml

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch.utils.data as data


ImageFile.LOAD_TRUNCATED_IMAGES = True


MEANS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.485, 0.456, 0.406),
    'cifar10': (0.4914, 0.4822, 0.4465)
}

STDS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.229, 0.224, 0.225),
    'cifar10': (0.2470, 0.2435, 0.2616)
}


class DatasetImgTarget(data.Dataset):
    def __init__(self, args, split, transform=None):
        self.root = os.path.abspath(args.dataset_root_path)
        self.transform = transform
        self.dataset_name = args.dataset_name

        if split == 'train':
            if args.train_trainval:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_trainval
            else:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_train
        elif split == 'val':
            if args.train_trainval:
                self.images_folder = args.folder_test
                self.df_file_name = args.df_test
            else:
                self.images_folder = args.folder_val
                self.df_file_name = args.df_val
        else:
            self.images_folder = args.folder_test
            self.df_file_name = args.df_test

        assert os.path.isfile(os.path.join(self.root, self.df_file_name)), \
            f'{os.path.join(self.root, self.df_file_name)} is not a file.'

        self.df = pd.read_csv(os.path.join(self.root, self.df_file_name), sep=',')
        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()

        self.num_classes = len(np.unique(self.targets))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir, target = self.data[idx], self.targets[idx]
        full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
        img = Image.open(full_img_dir)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset_name == 'CIFAR':
        dataset = datasets.CIFAR100(args.dataset_root_path, train=is_train, transform=transform)
        nb_classes = 100
    else:
        dataset = DatasetImgTarget(args, split='train' if is_train else 'val', transform=transform)
        nb_classes = dataset.num_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    input_size = args.input_size
    resize_size = args.resize_size if args.resize_size else int(args.input_size / 0.875)
    test_resize_size = resize_size

    mean = MEANS['imagenet']
    std = STDS['imagenet']
    if args.custom_mean_std:
        mean = MEANS[args.dataset_name] if args.dataset_name in MEANS.keys() else MEANS['05']
        std = STDS[args.dataset_name] if args.dataset_name in STDS.keys() else STDS['05']

    t = []

    if is_train:
        t.append(transforms.Resize(
            (resize_size, resize_size),
            interpolation=transforms.InterpolationMode.BICUBIC))
        t.append(transforms.RandomCrop(input_size))
        t.append(transforms.RandomHorizontalFlip())

        if args.trivial_aug:
            t.append(transforms.TrivialAugmentWide())
    else:
        t.append(transforms.Resize(
            (test_resize_size, test_resize_size),
            interpolation=transforms.InterpolationMode.BICUBIC))
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean, std=std))

    if is_train and args.re_prob > 0:
        # multiple random erasing blobs
        for _ in range(args.re_mult):
            t.append(
                transforms.RandomErasing(
                    p=args.re_prob, scale=(args.re_size_min, args.re_size_max), ratio=(args.re_r1, 3.3)
                )
            )

    transform = transforms.Compose(t)
    print(transform)
    return transform

