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

import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from einops import rearrange
from timm.models import create_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models
from datasets import build_dataset
from train import get_args_parser, adjust_config



def setup_environment(args):
    # fix the seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_train, args.num_classes = build_dataset(is_train=True, args=args)

    args.keep_rate_single = args.keep_rate[0] if args.keep_rate else None
    kr = f'_{args.keep_rate_single}' if args.keep_rate_single else ''
    if args.ifa_head and args.ifa_dws_conv_groups:
        head = '_cla'
    elif args.ifa_head:
        head = '_ifa'
    else:
        head = ''
    clc = '_clc' if args.clc else ''
    clr = f'_{args.num_clr}' if args.num_clr else ''

    args.run_name = '{}_{}{}{}{}{}_{}'.format(args.dataset_name, args.model, kr, head, clc, clr, args.serial)

    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, args.run_name)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # if not args.debugging:
        # wandb.init(project=args.wandb_project, entity=args.wandb_group, settings=wandb.Settings(start_method="fork"), config=args)
        # wandb.run.name = args.run_name

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
    model.to(args.device)

    model.eval()

    inputs = torch.rand(args.batch_size, 3, args.input_size, args.input_size).to(args.device)

    return model, inputs


def main():
    # https://pytorch.org/docs/stable/profiler.html
    # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--profile_cpu', action='store_false')
    parser.add_argument('--profile_cuda', action='store_false')
    parser.add_argument('--record_shapes', action='store_false')
    parser.add_argument('--profile_memory', action='store_false')
    parser.add_argument('--with_stack', action='store_true')
    parser.add_argument('--with_flops', action='store_false')
    parser.add_argument('--with_modules', action='store_false')
    parser.add_argument('--row_limit', type=int, default=50)
    args = parser.parse_args()
    adjust_config(args)

    model, inputs = setup_environment(args)

    activities = []
    if args.profile_cpu:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if args.profile_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    assert args.profile_cpu or args.profile_cuda, 'Needs to profile either C/GPU'

    # amp_autocast = torch.cuda.amp.autocast if args.use_amp else suppress
    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        with_flops=args.with_flops,
        with_modules=args.with_modules,
    ) as prof:
        model(inputs)

    fp = os.path.join(args.output_dir, 'log_profiler.txt')
    with open(fp, 'w') as f:
        # Print the profiling results to console
        stats = ('cpu_time_total', 'cuda_time_total', 'cpu_memory_usage', 'cuda_memory_usage')
        for stat in stats:
            print(f'{stat}, group_by_input_shape=False', file=f)
            print(prof.key_averages(group_by_input_shape=False).table(
                sort_by=stat, row_limit=args.row_limit), file=f)

            print(f'{stat}, group_by_input_shape=True', file=f)
            print(prof.key_averages(group_by_input_shape=True).table(
                sort_by=stat, row_limit=args.row_limit), file=f)

            # group_by_stack_n=INT if need to visualize stack (file)

    fp = os.path.join(args.output_dir, 'trace.json')
    prof.export_chrome_trace(fp)

    # if not args.debugging:
        # wandb.finish()

    return 0


if __name__ == "__main__":
    main()

