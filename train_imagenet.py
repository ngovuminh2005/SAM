import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from torchvision import models
import torchmetrics
import numpy as np
from tqdm import tqdm

import os
import time
import json
from contextlib import nullcontext
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

from optimizer import POLICIES, apply_perturbations, freeze_batchnorm_statistics, \
    select_perturbation_parameters

Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__())), default='resnet18'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True)
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd', 'adamw', 'adam'])), 'The optimizer', default='sgd'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0),
    precision=Param(OneOf(['fp16', 'bf16']), 'autocast dtype (SAM policies need bf16)', default='fp16'),
    grad_clip=Param(float, 'max grad norm for the descent step (0 = off)', default=0.),
    seed=Param(int, 'random seed (-1 = unseeded)', default=-1)
)

Section('sam', 'perturbation policy, independent of the base optimizer').params(
    method=Param(OneOf(['none'] + list(POLICIES)), 'perturbation policy', default='none'),
    rho=Param(float, 'perturbation radius', default=0.01),
    include=Param(str, 'regex of parameter names to perturb', default=''),
    exclude=Param(str, 'regex of parameter names not to perturb', default=''),
    friendlysam_sigma=Param(float, 'FriendlySAM sigma', default=1.0),
    friendlysam_lambda=Param(float, 'FriendlySAM EMA lambda', default=0.9),
    fishersam_eta=Param(float, 'FisherSAM eta', default=1.0),
    condsam_beta=Param(float, 'CondSAM factor EMA beta', default=0.95),
    condsam_eps=Param(float, 'CondSAM eps', default=1e-8),
    condsam_power_iters=Param(int, 'CondSAM power iterations', default=1),
    condsam_stats_every=Param(int, 'CondSAM: refresh Kronecker factors every N steps', default=1),
    condsam_sam_rho=Param(float, 'CondSAM2: radius of the added SAM perturbation', default=0.05),
    local_perturbation=Param(int, 'under DDP, perturb each rank with its own batch gradient '
                             '(SAM paper m-sharpness) instead of the all-reduced one (1/0)', default=1)
)

Section('wandb', 'Weights & Biases tracking').params(
    mode=Param(OneOf(['online', 'offline', 'disabled']), default='disabled'),
    project=Param(str, 'project name', default='ffcv-imagenet'),
    entity=Param(str, 'entity (team/user)', default=''),
    group=Param(str, 'run group', default=''),
    name=Param(str, 'run name (default: built from the config)', default=''),
    log_interval=Param(int, 'log train metrics every N iterations', default=50)
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]

class NormalizeImageTorch(ch.nn.Module):
    """GPU NormalizeImage without CuPy: ffcv's same uint8 -> float lookup
    table, gathered with torch instead of a CuPy elementwise kernel."""
    def __init__(self, mean, std, dtype):
        super().__init__()
        table = (np.arange(256)[:, None] - mean[None, :]) / std[None, :]
        self.register_buffer('table', ch.from_numpy(table.astype(dtype)).view(-1))
        self.register_buffer('offsets', ch.arange(len(mean), dtype=ch.int32))

    def forward(self, images):
        # ffcv calls pipeline modules directly, without moving them to a device
        if self.table.device != images.device:
            self.to(images.device)
        assert images.is_contiguous(memory_format=ch.channels_last), 'Images need to be in channel last'
        B, C, H, W = images.shape
        pixels = images.permute(0, 2, 3, 1).reshape(-1, C)
        result = self.table[pixels.int() * C + self.offsets]
        return result.view(B, H, W, C).permute(0, 3, 1, 2)

class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv(blurred)   # through __call__, so module hooks (CondSAM) run

class ImageNetTrainer:
    @param('training.distributed')
    @param('training.seed')
    @param('training.precision')
    @param('training.grad_clip')
    def __init__(self, gpu, distributed, seed, precision, grad_clip):
        self.all_params = get_current_config()
        self.gpu = gpu
        self.seed = None if seed < 0 else seed
        self.amp_dtype = {'fp16': ch.float16, 'bf16': ch.bfloat16}[precision]
        self.grad_clip = grad_clip
        self.step = 0

        self.uid = str(uuid4())

        if self.seed is not None:
            ch.manual_seed(self.seed + gpu)
            np.random.seed(self.seed + gpu)

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()
        self.initialize_logger()
        

    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing):
        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]

        # lr is overwritten every iteration by the schedule
        if optimizer == 'sgd':
            self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        else:
            opt_cls = {'adamw': ch.optim.AdamW, 'adam': ch.optim.Adam}[optimizer]
            self.optimizer = opt_cls(param_groups, lr=1)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.policy = self.create_policy()

    @param('sam.method')
    @param('sam.rho')
    @param('sam.include')
    @param('sam.exclude')
    @param('sam.friendlysam_sigma')
    @param('sam.friendlysam_lambda')
    @param('sam.fishersam_eta')
    @param('sam.condsam_beta')
    @param('sam.condsam_eps')
    @param('sam.condsam_power_iters')
    @param('sam.condsam_stats_every')
    @param('sam.condsam_sam_rho')
    @param('sam.local_perturbation')
    @param('training.precision')
    def create_policy(self, method, rho, include, exclude, friendlysam_sigma,
                      friendlysam_lambda, fishersam_eta, condsam_beta,
                      condsam_eps, condsam_power_iters, condsam_stats_every, condsam_sam_rho, local_perturbation,
                      precision):
        self.local_perturbation = bool(local_perturbation)
        if method == 'none':
            return None
        # GradScaler would scale the first-pass gradients (and CondSAM's
        # hooked output gradients) that the perturbation is computed from.
        if precision != 'bf16':
            raise ValueError('SAM policies require --training.precision=bf16')

        selected = select_perturbation_parameters(self.model, include, exclude)
        options = dict(rho=rho)
        if method in ('condsam', 'condsam1', 'condsam2'):
            options.update(beta=condsam_beta, eps=condsam_eps,
                           power_iters=condsam_power_iters, stats_every=condsam_stats_every)
        if method == 'condsam2':
            options.update(sam_rho=condsam_sam_rho)
        elif method == 'friendlysam':
            options.update(sigma=friendlysam_sigma, lmbda=friendlysam_lambda)
        elif method == 'fishersam':
            options.update(eta=fishersam_eta)
        return POLICIES[method]([p for _, p in selected], **options)

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImageTorch(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        seed=self.seed,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImageTorch(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    @param('training.epochs')
    @param('logging.log_level')
    def train(self, epochs, log_level):
        for epoch in range(epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            start_train = time.time()
            train_loss, num_images = self.train_loop(epoch)
            train_time = time.time() - start_train

            extra_dict = {
                'train_loss': train_loss,
                'epoch': epoch,
                'res': res,
                'train_time': train_time,
                'imgs_per_sec_per_gpu': num_images / train_time
            }
            if log_level > 0:
                self.eval_and_log(extra_dict)
            else:
                self.wandb_log_epoch(extra_dict)

        self.eval_and_log({'epoch': epoch}, final=True)
        if self.gpu == 0:
            ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')

    def eval_and_log(self, extra_dict={}, final=False):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'top_1': stats['top_1'],
                'top_5': stats['top_5'],
                'val_loss': stats['loss'],
                'val_time': val_time
            }, **extra_dict))
            prefix = 'final' if final else 'val'
            self.wandb_log_epoch(extra_dict, {
                f'{prefix}/top_1': stats['top_1'],
                f'{prefix}/top_5': stats['top_5'],
                f'{prefix}/loss': stats['loss'],
                f'{prefix}/time': val_time
            })

        return stats

    def wandb_log_epoch(self, extra_dict, metrics={}):
        if self.wandb_run is None:
            return
        metrics = dict(metrics, **{f'epoch/{k}': v for k, v in extra_dict.items()
                                   if k != 'epoch'})
        # number of completed epochs, so the final eval lands at x = epochs
        metrics['epoch'] = extra_dict['epoch'] + 1
        self.wandb_run.log(metrics)

    @param('model.arch')
    @param('model.pretrained')
    @param('training.distributed')
    @param('training.use_blurpool')
    def create_model_and_scaler(self, arch, pretrained, distributed, use_blurpool):
        # bf16 has fp32's exponent range, so it needs no loss scaling
        scaler = GradScaler(enabled=self.amp_dtype == ch.float16)
        model = getattr(models, arch)(pretrained=pretrained)
        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler

    def all_ranks(self, flag):
        value = ch.tensor(int(flag), device=next(self.model.parameters()).device)
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return bool(value.item())

    def average_gradients(self):
        """What DDP's backward would have done for the no_sync first pass."""
        world_size = dist.get_world_size()
        for p in self.model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad.div_(world_size)

    def train_step(self, images, target):
        policy = self.policy
        use_sam = policy is not None and policy.options['rho'] > 0
        # SAM paper (m-sharpness): under DDP each rank perturbs with its own
        # batch gradient, and only the descent gradient is averaged.
        local = (use_sam and self.local_perturbation and
                 isinstance(self.model, ch.nn.parallel.DistributedDataParallel))

        self.optimizer.zero_grad(set_to_none=True)
        # CondSAM hooks the first pass to collect its Kronecker factors; the
        # unwrapped module keeps them per rank in local mode.
        observed = self.model.module if local else self.model
        observation = policy.observe(observed, target.shape[0]) if use_sam else nullcontext()
        with observation, self.model.no_sync() if local else nullcontext():
            with autocast(dtype=self.amp_dtype):
                output = self.model(images)
                loss_train = self.loss(output, target)
            self.scaler.scale(loss_train).backward()

        if use_sam:
            deltas = policy.perturbations()
            # Every rank must take the second (all-reducing) pass, or none
            if local and not self.all_ranks(bool(deltas)):
                deltas = {}
            if deltas:
                # Descend with the gradient taken at the perturbed weights
                self.optimizer.zero_grad(set_to_none=True)
                with apply_perturbations(deltas), freeze_batchnorm_statistics(self.model):
                    with autocast(dtype=self.amp_dtype):
                        loss_perturbed = self.loss(self.model(images), target)
                    loss_perturbed.backward()
            elif local:
                self.average_gradients()

        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            ch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss_train.detach()

    @param('logging.log_level')
    @param('wandb.log_interval')
    def train_loop(self, epoch, log_level, log_interval):
        model = self.model
        model.train()
        losses = []
        num_images = 0

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]

            loss_train = self.train_step(images, target)
            losses.append(loss_train)
            num_images += target.shape[0]
            self.step += 1
            ### Training end

            ### Logging start
            if self.wandb_run is not None and self.step % log_interval == 0:
                self.wandb_run.log({
                    'train/step': self.step,
                    'train/epoch': epoch + ix / iters,
                    'train/loss': loss_train.item(),
                    'train/lr': lrs[ix]
                })

            if log_level > 0:
                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{loss_train.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)
            ### Logging end

        return ch.stack(losses).mean().item(), num_images

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast(dtype=self.amp_dtype):
                for images, target in tqdm(self.val_loader):
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in ['top_1', 'top_5']:
                        self.val_meters[k].update(output, target)

                    loss_val = self.loss(output, target)
                    self.val_meters['loss'].update(loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric().to(self.gpu)
        }

        self.wandb_run = None
        if self.gpu == 0:
            folder = (Path(folder) / str(self.uid)).absolute()
            folder.mkdir(parents=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)

            self.wandb_run = self.initialize_wandb(params)

    @param('wandb.mode')
    @param('wandb.project')
    @param('wandb.entity')
    @param('wandb.group')
    @param('wandb.name')
    def initialize_wandb(self, params, mode, project, entity, group, name):
        if mode == 'disabled':
            return None
        import wandb

        run = wandb.init(project=project, entity=entity or None,
                         group=group or None, name=name or self.run_name(params),
                         mode=mode, dir=str(self.log_folder), config=params,
                         job_type='train')
        run.define_metric('train/step')
        run.define_metric('train/*', step_metric='train/step')
        run.define_metric('epoch')
        for prefix in ['val', 'final', 'epoch']:
            run.define_metric(f'{prefix}/*', step_metric='epoch')
        run.define_metric('val/top_1', summary='max')
        run.summary['log_folder'] = str(self.log_folder)
        return run

    @staticmethod
    def run_name(params):
        method = params['sam.method']
        parts = [params['model.arch'], f"{params['training.epochs']}ep",
                 method, params['training.optimizer']]
        if method != 'none':
            parts.append(f"rho{params['sam.rho']:g}")
        extras = {
            'condsam': [('beta', 'condsam_beta'), ('pi', 'condsam_power_iters'), ('se', 'condsam_stats_every')],
            'condsam1': [('beta', 'condsam_beta'), ('pi', 'condsam_power_iters'), ('se', 'condsam_stats_every')],
            'condsam2': [('beta', 'condsam_beta'), ('pi', 'condsam_power_iters'), ('se', 'condsam_stats_every'),
                         ('samrho', 'condsam_sam_rho')],
            'friendlysam': [('sigma', 'friendlysam_sigma'), ('lambda', 'friendlysam_lambda')],
            'fishersam': [('eta', 'fishersam_eta')],
        }
        for label, key in extras.get(method, []):
            parts.append(f"{label}{params['sam.' + key]:g}")
        parts.append(f"lr{params['lr.lr']:g}")
        return '-'.join(parts)

    def log(self, content):
        print(f'=> Log: {content}')
        if self.gpu != 0: return
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.eval_and_log({'epoch': 0}, final=True)
        else:
            trainer.train()

        if trainer.wandb_run is not None:
            trainer.wandb_run.finish()

        if distributed:
            trainer.cleanup_distributed()

# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == "__main__":
    make_config()
    ImageNetTrainer.launch_from_args()
