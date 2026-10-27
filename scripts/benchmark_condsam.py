#!/usr/bin/env python3
"""Short SAM/CondSAM timing with the vision config, loader, DDP and SyncBN.

Run with torchrun, e.g.:
  CUDA_VISIBLE_DEVICES=0 SAM_DATASET_ROOT=$PWD/datasets env/bin/python \
    -m torch.distributed.run --standalone --nproc_per_node=1 \
    scripts/benchmark_condsam.py --ifile configs/CIFAR10_ResNet50.ini

Compilation/warm-up is excluded. No checkpoints, W&B runs or weight updates.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

import main as training
from optimizer.condsam import CondSAM
from optimizer.sam import SAM
from utility.args import Args


def main():
    timing_parser = argparse.ArgumentParser(add_help=False)
    timing_parser.add_argument('--steps', type=int, default=20)
    timing_parser.add_argument('--warmup', type=int, default=5)
    timing_parser.add_argument('--rounds', type=int, default=2)
    timing_parser.add_argument('--compile-sam', action=argparse.BooleanOptionalAction, default=True)
    timing_parser.add_argument('--condsam-channels-last', action=argparse.BooleanOptionalAction, default=True)
    timing_parser.add_argument('--output', type=Path)
    timing, remaining = timing_parser.parse_known_args()
    if min(timing.steps, timing.warmup, timing.rounds) < 1:
        timing_parser.error('steps, warmup and rounds must be positive')
    sys.argv = [sys.argv[0], *remaining]
    Args.parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    try:
        training.initialize()
        torch.set_float32_matmul_precision('high')
        dataset = training.DataLoader()
        results = {'SAM': [], 'CondSAM': []}
        models, optimizers, channels_last = {}, {}, {}
        initial = None
        for name, optimizer_class in [('SAM', SAM), ('CondSAM', CondSAM)]:
            model = training.getModel()(num_classes=dataset.numClasses)
            if initial is None:
                initial = {key: value.clone() for key, value in model.state_dict().items()}
            else:
                model.load_state_dict(initial)
            model = model.cuda(local_rank)
            channels_last[name] = name == 'CondSAM' and timing.condsam_channels_last and any(
                isinstance(module, torch.nn.Conv2d) for module in model.modules())
            if channels_last[name]:
                model = model.to(memory_format=torch.channels_last)
            if name == 'SAM' and timing.compile_sam and Args.model != 'ViT':
                model = torch.compile(model)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model)
            extra = dict(beta=Args.condsamBeta, eps=Args.condsamEps, power_iters=Args.condsamPowerIters) if name == 'CondSAM' else {}
            rho = Args.condsamRho if Args.condsamRho >= 0 else Args.rho
            optimizers[name] = optimizer_class(
                model.parameters(), lr=0., momentum=Args.momentum,
                weight_decay=Args.weightDecay, nesterov=Args.nesterov, rho=rho, **extra)
            models[name] = model.train()
        del initial

        for round_index in range(timing.rounds):
            # Alternate order to reduce timing bias on a shared GPU.
            order = ['SAM', 'CondSAM'] if round_index % 2 == 0 else ['CondSAM', 'SAM']
            for name in order:
                dataset.train.sampler.set_epoch(round_index)
                batches = iter(dataset.train)
                for step in range(timing.warmup + timing.steps):
                    if step == timing.warmup:
                        torch.cuda.synchronize(local_rank)
                        dist.barrier()
                        start = time.perf_counter()
                    try:
                        batch = next(batches)
                    except StopIteration:
                        batches = iter(dataset.train)
                        batch = next(batches)
                    inputs, targets = (value.cuda(local_rank) for value in batch)
                    if channels_last[name] and inputs.ndim == 4:
                        inputs = inputs.contiguous(memory_format=torch.channels_last)
                    loss, predictions = optimizers[name].step(models[name], inputs, targets)
                    del loss, predictions
                torch.cuda.synchronize(local_rank)
                elapsed = torch.tensor(time.perf_counter() - start, device=local_rank)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                seconds = elapsed.item() / timing.steps
                results[name].append(seconds)
                if dist.get_rank() == 0:
                    print(f'[TIMING] round={round_index + 1} {name} seconds/step={seconds:.6f}', flush=True)
                del batches
        if dist.get_rank() == 0:
            median = {name: statistics.median(values) for name, values in results.items()}
            report = {
                'seconds_per_step': results, 'median_seconds_per_step': median,
                'condsam_over_sam': median['CondSAM'] / median['SAM'],
                'within_2x_sam': median['CondSAM'] <= 2 * median['SAM'],
                'sam_compiled': timing.compile_sam and Args.model != 'ViT',
                'condsam_channels_last': channels_last['CondSAM'],
                'gpu': torch.cuda.get_device_name(local_rank),
                'torch_version': torch.__version__,
                'dataset': Args.dataset, 'model': Args.model, 'depth': Args.depth,
                'stats_chunk_bytes': CondSAM.STATS_CHUNK_BYTES,
                'condsam_power_iters': Args.condsamPowerIters,
                'includes': 'real loader, host-to-device copy, DDP, SyncBN, optimizer step',
                'excludes': 'compilation, warm-up, logging, evaluation, checkpoint saving',
                'batch_size': Args.batchSize, 'world_size': dist.get_world_size(),
                'steps_per_round': timing.steps, 'warmup': timing.warmup,
            }
            print(json.dumps(report, indent=2), flush=True)
            if timing.output:
                timing.output.parent.mkdir(parents=True, exist_ok=True)
                timing.output.write_text(json.dumps(report, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
