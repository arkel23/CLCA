import os
import random
import argparse
import json
from warnings import warn
from typing import List, Dict
from pathlib import Path
from functools import partial
from textwrap import wrap
from contextlib import suppress
from statistics import mean, stdev

import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from einops import rearrange, reduce
from timm.models import create_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models
from datasets import build_dataset
from train import get_args_parser, adjust_config, set_seed, set_run_name


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 15})


MODELS_DIC = {
    'vit_base_patch16_224.orig_in21k': 'ViT',
    'deit_base_patch16_224.fb_in1k': 'DeiT',
    'vit_base_patch16_224_miil.in21k': 'IN21k-P',
    'vit_base_patch16_224.in1k_mocov3': 'MoCo v3',
    'vit_base_patch16_224.dino': 'DINO',
    'vit_base_patch16_224.mae': 'MAE',
    'deit3_base_patch16_224.fb_in22k_ft_in1k': 'DeiT 3 (IN21k)',
    'deit3_base_patch16_224.fb_in1k': 'DeiT 3 (IN1k)',
    'vit_base_patch16_clip_224.laion2b': 'CLIP',
}


DATASETS_DIC = {
    'aircraft': 'Aircraft',
    'cars': 'Cars',
    'cotton': 'Cotton',
    'cub': 'CUB',
    'dafb': 'DAFB',
    'dogs': 'Dogs',
    'flowers': 'Flowers',
    'food': 'Food',
    'inat17': 'iNat17',
    'moe': 'Moe',
    'nabirds': 'NABirds',
    'pets': 'Pets',
    'soyageing': 'SoyAgeing',
    'soyageingr1': 'SoyAgeingR1',
    'soyageingr3': 'SoyAgeingR3',
    'soyageingr4': 'SoyAgeingR4',
    'soyageingr5': 'SoyAgeingR5',
    'soyageingr6': 'SoyAgeingR6',
    'soygene': 'SoyGene',
    'soyglobal': 'SoyGlobal',
    'soylocal': 'SoyLocal',
    'vegfru': 'VegFru',
}

SETTINGS_DIC = {
    'fz_bl': 'Frozen Baseline',
    'ft_bl': 'Baseline',
    'ft_cla': 'CLA',
    'ft_clca': 'CLCA',
}

def add_colorbar(im, aspect=10, pad_fraction=0.2, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


class FeatureMetrics:
    def __init__(self,
                 model: nn.Module,
                 model_name: str = None,
                 model_layers: List[str] = None,
                 device: str ='cpu',
                 image_size: int = 224,
                 setting: str = 'fz',
                 compute_attention_average: bool = False,
                 debugging: bool = False):
        """

        :param model: (nn.Module) Neural Network 1
        :param model_name: (str) Name of model 1
        :param model_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model = model

        self.device = device

        self.model_info = {}

        self.model_info['Setting'] = SETTINGS_DIC.get(setting, setting)

        if model_name is None:
            self.model_info['Name_og'] = model.__repr__().split('(')[0]
        else:
            self.model_info['Name_og'] = model_name
        self.model_info['Name'] = MODELS_DIC.get(self.model_info['Name_og'], self.model_info['Name_og'])

        self.model_info['Layers'] = []

        self.model_features = {}

        if len(list(model.modules())) > 150 and model_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model_layers' parameter. Your CPU/GPU will thank you :)")

        self.model_layers = model_layers

        self._insert_hooks()
        self.model = self.model.to(self.device)

        self.model.eval()

        self._check_shape(image_size)

        self.compute_attention_average = compute_attention_average

        self.debugging = debugging

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

    def _check_shape(self, image_size):
        with torch.no_grad():
            x = torch.rand(2, 3, image_size, image_size).to(self.device)
            _ = self.model(x)

            # -1 in certain cases corresponds to classification layer
            last = self.model_info['Layers'][-2]
            feat_out = self.model_features[last]

            if len(feat_out.shape) == 4:
                b, c, h, w = feat_out.shape
                if h == w:
                    self.bchw = True
                    h = feat_out.shape[-1]
                else:
                    self.bchw = False
                    h = feat_out.shape[1]
            elif len(feat_out.shape) == 3:
                h = int(feat_out.shape[1] ** 0.5)
                self.cls = False if h ** 2 == feat_out.shape[1] else True
            else:
                pass

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def _pool_features(self, feat, pool=True):
        pooled = feat.flatten(1)
 
        return pooled

    def process_attention(self, x):
        # only process attention of cls token to others
        x = x[:, :, 0, 1:]
        mean = reduce(x, 'b h s2 -> 1', 'mean').squeeze()
        std = torch.std(x)
        return mean, std

    def compare(self,
                dataloader1: DataLoader) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        """

        self.model_info['Dataset_og'] = dataloader1.dataset.dataset_name
        self.model_info['Dataset'] = DATASETS_DIC.get(self.model_info['Dataset_og'], self.model_info['Dataset_og'])

        layers = self.model_layers if self.model_layers is not None else list(self.model.modules())

        if self.compute_attention_average:
            N = len([layer for layer in layers if 'attn' in layer])
            self.attn_mean = torch.zeros(N, device=self.device)
            self.attn_std = torch.zeros(N, device=self.device)
        else:
            # N = len([layer for layer in layers if 'attn' not in layer])
            N = len(layers)
            self.hsic_matrix = torch.zeros(N, N, 3)
            self.dist_cum = torch.zeros(N, device=self.device)
            self.dist_cum_norm = torch.zeros(N, device=self.device)
            self.l2_norm = torch.zeros(N, device=self.device)

        num_batches = len(dataloader1)

        for (x1, *_) in tqdm(dataloader1, desc="| Comparing features |", total=num_batches):

            self.model_features = {}
            x1 = x1.to(self.device)
            _ = self.model(x1)

            if self.compute_attention_average:
                self.compare_attn_mean_std(num_batches)
            else:
                self.compare_cka_l2_dist(num_batches)

        if not self.compute_attention_average:
            self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                            self.hsic_matrix[:, :, 2].sqrt())

    def compare_attn_mean_std(self, num_batches):
        for i, (name1, feat1) in enumerate(self.model_features.items()):
            attn_mean, attn_std = self.process_attention(feat1)
            self.attn_mean[i] += attn_mean / num_batches
            self.attn_std[i] += attn_std / num_batches
        return 0

    def compare_cka_l2_dist(self, num_batches):
        for i, (name1, feat1) in enumerate(self.model_features.items()):
            X = feat1.flatten(1)

            # frobenius norm
            self.l2_norm[i] += torch.norm(X, p='fro', dim=-1).mean() / num_batches

            dist = torch.cdist(X, X, p=2.0)

            dist_avg = (torch.sum(dist) / torch.nonzero(dist).size(0))
            self.dist_cum[i] += dist_avg / num_batches

            dist = (dist - dist.min()) / (dist.max() - dist.min())
            dist_avg_norm = (torch.sum(dist) / torch.nonzero(dist).size(0))
            self.dist_cum_norm[i] += dist_avg_norm / num_batches

            K = X @ X.t()
            K.fill_diagonal_(0.0)
            self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

            for j, (name2, feat2) in enumerate(self.model_features.items()):
                Y = feat2.flatten(1)

                L = Y @ Y.t()
                L.fill_diagonal_(0)

                assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        if self.compute_attention_average:
            return {
                "model_name": self.model_info['Name'],
                "model_name_og": self.model_info['Name_og'],
                "model_layers": self.model_info['Layers'],
                "dataset_name": self.model_info['Dataset'],
                "dataset_name_og": self.model_info['Dataset_og'],
                "setting": self.model_info['Setting'],
                'attn_mean': self.attn_mean,
                'attn_std': self.attn_std,
            }

        return {
            "model_name": self.model_info['Name'],
            "model_name_og": self.model_info['Name_og'],
            "model_layers": self.model_info['Layers'],
            "dataset_name": self.model_info['Dataset'],
            "dataset_name_og": self.model_info['Dataset_og'],
            "setting": self.model_info['Setting'],
            'l2_norm': self.l2_norm,
            "CKA": self.hsic_matrix,
            "dist": self.dist_cum,
            "dist_norm": self.dist_cum_norm,
        }

    def plot_cka(self,
                 save_path: str = None,
                 title: str = None,
                 show: bool = False):
        fig, ax = plt.subplots(figsize=(6, 5.25))
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')

        ax.set_xlabel(f"Layers", fontsize=16)
        ax.set_ylabel(f"Layers", fontsize=16)

        labels = range(self.hsic_matrix.shape[0])
        ax.set_xticks(labels)
        ax.set_yticks(labels)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            title = f"CKA on {self.model_info['Dataset']} for {self.model_info['Name']}\n {self.model_info['Setting']}"
            ax.set_title(title, fontsize=18)

        add_colorbar(im)
        plt.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if not self.debugging:
            fn = os.path.splitext(os.path.split(save_path)[-1])[0]
            wandb.log({fn: wandb.Image(fig)})

        if show:
            plt.show()

    def plot_metrics(self,
                     metric: str = 'norms',
                     save_path: str = None,
                     title: str = None,
                     show: bool = False):
        fig, ax = plt.subplots()

        if metric == 'norms':
            labels = range(self.l2_norm.shape[0])
            ax.bar(labels, self.l2_norm.cpu())
            y_label = 'L2-Norm'
        elif metric == 'dist':
            labels = range(self.dist_cum.shape[0])
            ax.bar(labels, self.dist_cum.cpu())
            y_label = 'L2-Distance'
        elif metric == 'dist_norm':
            labels = range(self.dist_cum_norm.shape[0])
            ax.bar(labels, self.dist_cum_norm.cpu())
            y_label = 'Normalized L2-Distance'
        elif metric == 'attn_mean':
            labels = range(self.attn_mean.shape[0])
            ax.bar(labels, self.attn_mean.cpu())
            y_label = 'Attention Mean'
        elif metric == 'attn_std':
            labels = range(self.attn_std.shape[0])
            ax.bar(labels, self.attn_std.cpu())
            y_label = 'Attention Std.'

        ax.set_xlabel("Layer", fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_xticks(labels)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            title = f"{y_label} per Layer on {self.model_info['Dataset']}\n for {self.model_info['Name']} {self.model_info['Setting']}"
            ax.set_title(title, fontsize=18)

        plt.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if not self.debugging:
            fn = os.path.splitext(os.path.split(save_path)[-1])[0]
            wandb.log({fn: wandb.Image(fig)})

        if show:
            plt.show()


def calc_cka(results, split='train'):
    name = 'CKA'

    ckas = {}

    results = results[name]


    if torch.isnan(results).any():
        print(split, name, results)
    else:
        for i, cka in enumerate(results):
            # 1st order derivative of cka with respect to layers
            cka = cka.tolist()
            layer_change = [abs(cka[l + 1] - cka[l]) for l in range(0, len(cka) - 1)]
            layer_change_mean = mean(layer_change)
            layer_change_std = stdev(layer_change)

            # 2nd order derivative of cka with respect to layers (change of layer change)
            layer_change_2nd = [abs(layer_change[i + 1] - layer_change[i]) for i in range(0, len(layer_change) - 1)]
            layer_change_2nd_mean = mean(layer_change_2nd)
            layer_change_2nd_std = stdev(layer_change_2nd)

            ckas.update({
                f'{name.lower()}_change_mean_{i}_{split}': layer_change_mean,
                f'{name.lower()}_change_std_{i}_{split}': layer_change_std,
                f'{name.lower()}_change2_mean_{i}_{split}': layer_change_2nd_mean,
                f'{name.lower()}_change2_std_{i}_{split}': layer_change_2nd_std,
                })

    results = results.fill_diagonal_(0)

    for i, cka in enumerate(results):
        layer_mean = (torch.sum(cka) / torch.nonzero(cka).size(0)).item()
        ckas.update({f'{name.lower()}_{i}_{split}': layer_mean})

    overall_mean = (torch.sum(results) / torch.nonzero(results).size(0)).item()
    ckas.update({f'{name.lower()}_avg_{split}': overall_mean})

    return ckas


def calc_distances(results, split='train'):
    dists = {}
    for i, (dist, dist_norm) in enumerate(zip(results['dist'], results['dist_norm'])):
        dists.update({f'dist_{i}_{split}': dist.item(), f'dist_norm_{i}_{split}': dist_norm.item()})

    dists.update({f'dist_avg_{split}': torch.mean(results['dist']).item(),
                  f'dist_norm_avg_{split}': torch.mean(results['dist_norm']).item()})
    return dists


def calc_l2_norm(results, split='train'):
    norms = {}
    for i, norm in enumerate(results['l2_norm']):
        norms.update({f'l2_norm_{i}_{split}': norm.item()})

    norms.update({f'l2_norm_avg_{split}': torch.mean(results['l2_norm']).item()})
    return norms


def calc_attn_mean_std(results, split='train'):
    attns = {}
    for i, (attn_mean, attn_std) in enumerate(zip(results['attn_mean'], results['attn_std'])):
        attns.update({f'attn_mean_{i}_{split}': attn_mean.item()})

        attns.update({f'attn_std_{i}_{split}': attn_std.item()})

    attns.update({f'attn_mean_avg_{split}': torch.mean(results['attn_mean']).item()})
    return attns


def save_results_to_json(args, results_train, results_test):
    # needs to convert tensors (l2_norm, dist, dist_norm, CKA) to list
    if args.compute_attention_average:
        results_train['attn_mean'] = results_train['attn_mean'].tolist()
        results_test['attn_mean'] = results_test['attn_mean'].tolist()
        results_train['attn_std'] = results_train['attn_std'].tolist()
        results_test['attn_std'] = results_test['attn_std'].tolist()
    else:
        results_train['l2_norm'] = results_train['l2_norm'].tolist()
        results_train['dist'] = results_train['dist'].tolist()
        results_train['dist_norm'] = results_train['dist_norm'].tolist()
        results_train['CKA'] = results_train['CKA'].tolist()

        results_test['l2_norm'] = results_test['l2_norm'].tolist()
        results_test['dist'] = results_test['dist'].tolist()
        results_test['dist_norm'] = results_test['dist_norm'].tolist()
        results_test['CKA'] = results_test['CKA'].tolist()

    data = {'train': results_train, 'test': results_test} 

    fp = os.path.join(args.output_dir, 'feature_metrics.json')
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4)

    return 0


def setup_environment(args):
    # fix the seed for reproducibility
    set_seed(args.seed)

    dataset_train, args.num_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(args.dataset_name, args.num_classes, len(dataset_train), len(dataset_val))

    if args.finetune and args.ifa_head and args.clc:
        args.setting = 'ft_clca'
    elif args.finetune and args.ifa_head:
        args.setting = 'ft_cla'
    elif args.finetune:
        args.setting = 'ft_bl'
    else:
        args.setting = 'fz_bl'

    set_run_name(args, setting=True)

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
    if args.num_clr:
        model.add_clr(args.num_clr)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)

    model.to(args.device)
    model.eval()

    layers = []
    for name, _ in model.named_modules():
        # print(name)
        if args.compute_attention_average:
            if ('vit_b16' in args.model) and (any([kw in name for kw in ['attn.drop']])):
                layers.append(name)
            elif ('deit' in args.model or 'vit' in args.model) and (any(
                [kw in name for kw in ['attn_drop']])):
                layers.append(name)
        elif args.compute_attention_cka:
            if ('vit_b16' in args.model) and (any([kw in name for kw in ['attn.drop']])):
                layers.append(name)
            elif ('deit' in args.model or 'vit' in args.model) and (any(
                [kw in name for kw in ['attn_drop']])):
                layers.append(name)
        else:
            if ('vit_b16' in args.model) and (any([kw in name for kw in ('norm2', 'encoder_norm')])):
                layers.append(name)
            elif ('deit' in args.model or 'vit' in args.model) and (any(
                [kw in name for kw in ('norm2', 'model.norm')])):
                layers.append(name)

    if args.keep_rate:
        tr, model_name = args.model.split('_', 1)
    else:
        model_name = args.model

    feature_metrics = FeatureMetrics(model, model_name, layers, args.device,
                          args.input_size, args.setting, debugging=args.debugging,
                          compute_attention_average=args.compute_attention_average)

    return loader_train, loader_test, feature_metrics


def main():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--compute_attention_average', action='store_true')
    parser.add_argument('--compute_attention_cka', action='store_true')
    parser.set_defaults(output_dir='results_inference')
    args = parser.parse_args()
    adjust_config(args)

    loader_train, loader_test, feature_metrics = setup_environment(args)

    amp_autocast = torch.cuda.amp.autocast if args.use_amp else suppress

    with torch.no_grad():
        with amp_autocast():
            feature_metrics.compare(loader_train)

            results_train = feature_metrics.export()
            if args.compute_attention_average:
                attn_train = calc_attn_mean_std(results_train, split='train')
                feature_metrics.plot_metrics('attn_mean', os.path.join(args.output_dir, 'attn_mean_train.png'))
                feature_metrics.plot_metrics('attn_std', os.path.join(args.output_dir, 'attn_std_train.png'))
            else:
                feature_metrics.plot_cka(os.path.join(args.output_dir, 'cka_train.png'))
                feature_metrics.plot_metrics('norms', os.path.join(args.output_dir, 'norms_train.png'))
                feature_metrics.plot_metrics('dist', os.path.join(args.output_dir, 'dist_train.png'))
                feature_metrics.plot_metrics('dist_norm', os.path.join(args.output_dir, 'dist_norm_train.png'))

                cka_train = calc_cka(results_train, split='train')
                dists_train = calc_distances(results_train, split='train')
                norms_train = calc_l2_norm(results_train, split='train')

            feature_metrics.compare(loader_test)

            results_test = feature_metrics.export()
            if args.compute_attention_average:
                feature_metrics.plot_metrics('attn_mean', os.path.join(args.output_dir, 'attn_mean_test.png'))
                feature_metrics.plot_metrics('attn_std', os.path.join(args.output_dir, 'attn_std_test.png'))
                attn_test = calc_attn_mean_std(results_test, split='test')
            else:
                feature_metrics.plot_cka(os.path.join(args.output_dir, 'cka_test.png'))
                feature_metrics.plot_metrics('norms', os.path.join(args.output_dir, 'norms_test.png'))
                feature_metrics.plot_metrics('dist', os.path.join(args.output_dir, 'dist_test.png'))
                feature_metrics.plot_metrics('dist_norm', os.path.join(args.output_dir, 'dist_norm_test.png'))

                cka_test = calc_cka(results_test, split='test')
                dists_test = calc_distances(results_test, split='test')
                norms_test = calc_l2_norm(results_test, split='test')

    log_dic = {'setting': args.setting}
    if args.compute_attention_average:
        log_dic.update(attn_train)
        log_dic.update(attn_test)
    else:
        log_dic.update(cka_train)
        log_dic.update(dists_train)
        log_dic.update(norms_train)
        log_dic.update(cka_test)
        log_dic.update(dists_test)
        log_dic.update(norms_test)

    if not args.debugging:
        wandb.log(log_dic)
        wandb.finish()
    else:
        print(log_dic)

    save_results_to_json(args, results_train, results_test)

    return 0


if __name__ == "__main__":
    main()
