# https://github.com/zhijian-liu/torchprofile
import time
import argparse
from contextlib import suppress

import wandb
import torch
from timm.models import create_model
import torch.backends.cudnn as cudnn

import models
import utils
from datasets import build_dataset
from train import get_args_parser, adjust_config, count_params, set_seed, set_run_name


def setup_environment(args):
    set_seed(args.seed)
    cudnn.benchmark = True

    dataset_val, args.num_classes = build_dataset(is_train=False, args=args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    set_run_name(args)

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
    model.to(args.device)

    model.eval()

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)

    return model, data_loader_val


def measure_tp(model, loader, device='cuda', multiple=1, amp=True):
    torch.cuda.synchronize()
    start = time.time()

    amp_autocast = torch.cuda.amp.autocast if amp else suppress

    for m in range(multiple):
        for i,(x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                with amp_autocast():
                    model(x)

            if i % 100 == 0:
                print(f'{m} / {multiple}: {i} / {len(loader)}')

    torch.cuda.synchronize()
    time_total = time.time() - start

    num_images = len(loader.dataset) * multiple
    throughput = round((num_images / time_total), 4)
    time_total = round((time_total / 60), 4)
    return throughput, time_total


def main():

    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--test_multiple', type=int, default=5,
                        help='test multiple loops (to reduce model loading time influence)')
    args = parser.parse_args()
    adjust_config(args)

    model, test_loader = setup_environment(args)

    tp, time_total = measure_tp(model, test_loader, args.device, args.test_multiple, args.use_amp)

    profiler = 'thop' if 'ats' in args.model else 'torchprofile'
    flops = utils.count_flops(model, args.input_size, args.device, profiler=profiler)
    flops = round(flops / 1e9, 4)

    max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
    max_memory = round(max_memory, 4)

    no_params = count_params(model)
    no_params = round(no_params / (1e6), 4)  # millions of parameters

    no_params_trainable = count_params(model, trainable=True)
    no_params_trainable = round(no_params_trainable / (1e6), 4)  # millions of parameters

    if not args.debugging:
        wandb.run.summary['throughput'] = tp
        wandb.run.summary['time_total'] = time_total
        wandb.run.summary['flops'] = flops
        wandb.run.summary['max_memory'] = max_memory
        wandb.run.summary['no_params'] = no_params
        wandb.run.summary['no_params_trainable'] = no_params_trainable
        wandb.finish()

    print('run_name,tp,time_total,flops,max_memory,no_params,no_params_trainable')
    line = f'{args.run_name},{tp},{time_total},{flops},{max_memory},{no_params},{no_params_trainable}'
    print(line)
    return 0


if __name__ == "__main__":
    main()

