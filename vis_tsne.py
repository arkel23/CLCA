import os
import argparse
from typing import List
from pathlib import Path
from functools import partial

from tqdm import tqdm
import numpy as np
import wandb
import matplotlib.pyplot as plt
from timm.models import create_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models
from datasets import build_dataset
from train import get_args_parser, adjust_config


class FeatureExtractor:
    def __init__(self,
                 model: nn.Module,
                 model_layers: List[str] = None,
                 device: str ='cpu'):
        """
        :param model: (nn.Module) Neural Network 1
        :param model_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model = model

        self.device = device

        self.model_info = {}

        self.model_info['Layers'] = []

        self.model_features = {}

        self.model_layers = model_layers

        self._insert_hooks()
        self.model = self.model.to(self.device)

        self.model.eval()

        print(self.model_info)


    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model":
            self.model_features[name] = out
        else:
            raise RuntimeError("Unknown model name for _log_layer.")


    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model.named_modules():
            if self.model_layers is not None:
                if name in self.model_layers:
                    self.model_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model", name))
            else:
                self.model_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model", name))

    def extract_features(self,
                dataloader: DataLoader, num_images) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader: (DataLoader)
        """

        num_batches = len(dataloader)

        images, _ = next(iter(dataloader))
        images = images.to(self.device)
        _ = self.model(images)
        feat = list(self.model_features.keys())[-1]
        b, s, d = feat.shape

        features = torch.zeros(num_images, d, device=self.device)
        labels = torch.zeros(num_images, device=self.device)

        i = 0
        for (images, targets) in tqdm(dataloader, desc="| Comparing features |", total=num_batches):

            self.model_features = {}
            images = images.to(self.device)
            _ = self.model(images)

            feat = list(self.model_features.keys())[-1]

            for j in range(feat.shape[0]):
                features[i] = feat[j]
                labels[i] = targets[j]
                i += 1

        print(features.shape, labels.shape)
        return features, labels


def vis_tsne(features, labels):
    # https://github.com/CannyLab/tsne-cuda/blob/main/examples/cifar.py    tsne = TSNE(n_iter=5000, verbose=1, perplexity=10000, num_neighbors=128)
    from tsnecuda import TSNE

    tsne = TSNE(n_iter=5000, verbose=1, perplexity=10000, num_neighbors=128)
    tsne_results = tsne.fit_transform(features)

    print(tsne_results.shape)

    # Create the figure
    fig = plt.figure( figsize=(8,8) )
    ax = fig.add_subplot(1, 1, 1, title='T-SNE')

    # Create the scatter
    ax.scatter(
        x=tsne_results[:,0],
        y=tsne_results[:,1],
        c=labels,
        cmap=plt.cm.get_cmap('Paired'),
        alpha=0.4,
        s=0.5)
    plt.show()

    return 0


def main():

    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    adjust_config(args)

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_train, args.num_classes = build_dataset(is_train=True, args=args)
    dataset_test, _ = build_dataset(is_train=False, args=args)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    args.keep_rate_single = args.keep_rate[0] if args.keep_rate else None
    kr = f'_{args.keep_rate_single}' if args.keep_rate_single else ''

    args.run_name = '{}_{}{}_{}'.format(args.dataset_name, args.model, kr, args.serial)

    args.output_dir = os.path.join(args.output_dir, args.run_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.debugging:
        wandb.init(project=args.wandb_project, entity=args.wandb_group, settings=wandb.Settings(start_method="fork"), config=args)
        wandb.run.name = args.run_name

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
        args = args
    )
    if args.dataset_name.lower() != "imagenet":
        model.reset_classifier(args.num_classes)
    model.to(args.device)

    model.eval()

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)

    # only works for deit/vit base
    if 'deit' in args.model or 'vit' in args.model:
        layers = []
        for name, _ in model.named_modules():
            if name == 'norm':
                layers.append(name)
    else:
        raise NotImplementedError

    extractor = FeatureExtractor(model, layers, args.device)

    with torch.no_grad():
        features_train, labels_train = extractor.extract_features(loader_train, len(dataset_train))
        vis_tsne(features_train, labels_train)

        features_test, labels_test = extractor.extract_features(loader_test, len(dataset_test))
        vis_tsne(features_test, labels_test)


    return 0


if __name__ == "__main__":
    main()
