import os
import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional

from llava.optimizer import POLICIES, select_perturbation_parameters


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


def patch_zero3_linear_bias_only():
    """Fix DeepSpeed's ZeRO-3 linear backward for a trainable bias on a frozen weight.

    It sets `dim` only when the weight needs a gradient, so BitFit (`bias`)
    raised UnboundLocalError (still the case in DeepSpeed 0.16.9).
    """
    from deepspeed.runtime.zero.linear import LinearFunctionForZeroStage3
    original = LinearFunctionForZeroStage3.backward
    if getattr(original, 'bias_only_fix', False):
        return

    def backward(ctx, grad_output):
        if ctx.needs_input_grad[1] or not ctx.needs_input_grad[2]:
            return original(ctx, grad_output)
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_output.matmul(weight) if ctx.needs_input_grad[0] else None
        return grad_input, None, grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

    backward.bias_only_fix = True
    LinearFunctionForZeroStage3.backward = staticmethod(backward)


def release_zero3_parameters(model):
    """Release every gathered ZeRO-3 parameter so GatheredParameters can repartition.

    Modules whose backward is not run (the ascent pass, frozen layers) leave
    parameters marked active until the next forward resets them; this is what
    engine.empty_partition_cache does, without its gc/empty_cache.
    """
    offload = getattr(getattr(model, 'optimizer', None), 'parameter_offload', None)
    if offload is not None:
        offload.partition_all_parameters()


class LLaVATrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_policy = None
        self.sam_stats = {'clean_loss': [], 'grad_norm': [], 'delta_norm': []}
        patch_zero3_linear_bias_only()
        if self.args.sam_method != 'none':
            self.keep_vision_tower_unchecked()

    def keep_vision_tower_unchecked(self):
        """Leave the vision tower out of gradient checkpointing for SAM runs.

        SAM needs non-reentrant checkpointing, and its recomputation inside CLIP's
        encoder sees frozen ZeRO-3 weights already released (shape 0). CoVFT's tower
        calls its layers directly and is never checkpointed either.
        """
        enable = getattr(self.model, 'gradient_checkpointing_enable', None)
        if enable is None:
            return

        def gradient_checkpointing_enable(*args, **kwargs):
            enable(*args, **kwargs)
            get_model = getattr(self.model, 'get_model', None)
            vision_tower = get_model().get_vision_tower() if get_model else None
            for module in vision_tower.modules() if vision_tower is not None else []:
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = False

        self.model.gradient_checkpointing_enable = gradient_checkpointing_enable

    def create_sam_policy(self):
        args = self.args
        if args.sam_method not in POLICIES:
            raise ValueError(f'Unknown sam_method {args.sam_method}; use none or one of {sorted(POLICIES)}')
        if not args.bf16:
            raise ValueError('SAM policies require --bf16 True')
        if args.sam_local_perturbation and is_deepspeed_zero3_enabled():
            raise ValueError('ZeRO-3 shares parameter partitions across ranks, so every rank must use the '
                             'same perturbation: set --sam_local_perturbation False or use ZeRO-2')
        selected = select_perturbation_parameters(self.model, args.sam_include, args.sam_exclude)
        options = dict(rho=args.sam_rho)
        if args.sam_method == 'condsam':
            options.update(beta=args.sam_condsam_beta, eps=args.sam_condsam_eps,
                           power_iters=args.sam_condsam_power_iters,
                           update_interval=args.sam_condsam_update_interval)
        elif args.sam_method == 'friendlysam':
            options.update(sigma=args.sam_friendlysam_sigma, lmbda=args.sam_friendlysam_lambda)
        elif args.sam_method == 'fishersam':
            options.update(eta=args.sam_fishersam_eta)
        numel = sum(getattr(p, 'ds_numel', p.numel()) for _, p in selected)
        logger.info(f'SAM {args.sam_method} {options}: perturbing {len(selected)} tensors '
                    f'({numel / 1e6:.1f}M parameters) matching include={args.sam_include!r} '
                    f'exclude={args.sam_exclude!r}')
        return POLICIES[args.sam_method]([p for _, p in selected], **options)

    def get_batch_samples(self, epoch_iterator, num_batches):
        # transformers 4.47 counts the label tokens of the accumulated batches even when the model
        # cannot use them (LLaVA's forward takes no **kwargs), and training_step then skips the
        # division by gradient_accumulation_steps: the logged loss came out x GA (gradients were
        # fine, DeepSpeed/accelerate scale them). Without the count the logged loss is the mean.
        batch_samples, num_items_in_batch = super().get_batch_samples(epoch_iterator, num_batches)
        if not self.model_accepts_loss_kwargs:
            num_items_in_batch = None
        return batch_samples, num_items_in_batch

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.args.sam_method == 'none':
            return super().training_step(model, inputs, num_items_in_batch)
        if self.sam_policy is None:
            self.sam_policy = self.create_sam_policy()
        policy = self.sam_policy
        if policy.options['rho'] == 0:
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()
        inputs = self._prepare_inputs(inputs)
        params = list(policy.params)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        shared = world_size > 1 and not self.args.sam_local_perturbation

        # Ascent pass. torch.autograd.grad returns the selected gradients without
        # running DeepSpeed's gradient hooks, so the accumulated gradients of this
        # optimizer step stay untouched. With a shared perturbation CondSAM pools
        # its factors over all ranks.
        with policy.observe(self.model, inputs['input_ids'].shape[0], distributed=shared):
            with self.compute_loss_context_manager():
                clean_loss = self.compute_loss(model, inputs)
            grads = list(torch.autograd.grad(clean_loss, params, allow_unused=True))
        clean_loss = clean_loss.detach()
        release_zero3_parameters(model)
        # A parameter outside this rank's graph still joins the all-reduce. Under ZeRO-3 p is a
        # shape-[0] partition here, so the zeros take the full shape (ds_shape).
        grads = [torch.zeros(getattr(p, 'ds_shape', p.shape), dtype=p.dtype, device=clean_loss.device)
                 if g is None and shared else g for p, g in zip(params, grads)]
        if shared:
            # Perturb every rank with the all-reduced gradient (m = global batch).
            flat = torch.cat([g.float().flatten() for g in grads])
            dist.all_reduce(flat)
            flat /= world_size
            grads = [chunk.view_as(g).to(g.dtype)
                     for chunk, g in zip(flat.split([g.numel() for g in grads]), grads)]

        # The policy reads p.grad and full weights; under ZeRO-3 they exist only
        # while gathered. modifier_rank=0 writes rank 0's perturbed weights back
        # into every partition; without ZeRO-3 these contexts do nothing.
        originals = {}
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            try:
                for p, g in zip(params, grads):
                    p.grad = g
                deltas = policy.perturbations()
            finally:
                for p in params:
                    p.grad = None
            with torch.no_grad():
                for p, delta in deltas.items():
                    originals[p] = p.data.clone()
                    p.data.add_(delta)
        grad_norm = policy._norm([g for g in grads if g is not None]).float()
        delta_norm = policy._norm(deltas.values()).float() if deltas else grad_norm.new_zeros(())
        del grads, deltas

        # Descent pass at the perturbed weights: the usual step, so DeepSpeed
        # accumulates and averages these gradients across ranks as without SAM.
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        finally:
            if originals:
                release_zero3_parameters(model)
                with deepspeed.zero.GatheredParameters(list(originals), modifier_rank=0):
                    with torch.no_grad():
                        for p, original in originals.items():
                            p.data.copy_(original)
        self.sam_stats['clean_loss'].append(clean_loss)
        self.sam_stats['grad_norm'].append(grad_norm)
        self.sam_stats['delta_norm'].append(delta_norm)
        return loss

    def log(self, logs, start_time=None):
        # Called on every rank at the same steps, so the stats can be averaged here.
        if 'loss' in logs and self.sam_stats['clean_loss']:
            names = list(self.sam_stats)
            values = torch.stack([torch.stack(self.sam_stats[n]).float().mean() for n in names])
            if dist.is_initialized():
                dist.all_reduce(values)
                values /= dist.get_world_size()
            for name, value in zip(names, values.tolist()):
                logs[f'sam/{name}'] = round(value, 6)
            for name in names:
                self.sam_stats[name].clear()
        super().log(logs, start_time)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif self.args.decrease_vision_parameter_lr is not None:
                print("Use different lr for vision tower")
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "vision_tower" not in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": [
                            n for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "vision_tower" not in n)
                        ],
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad  and "vision_tower" not in n)
                        ],
                        "weight_decay": 0.0,
                        "name": [
                            n for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "vision_tower" not in n)
                        ],
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "vision_tower" in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.decrease_vision_parameter_lr,
                        "name": [
                            n for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "vision_tower" in n)
                        ],
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "vision_tower" in n)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.decrease_vision_parameter_lr,
                        "name": [
                            n for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "vision_tower" in n)
                        ],

                    },

                ]
                for group in optimizer_grouped_parameters:
                    print(f"Group: name {group.get('name', '-')}, lr {group.get('lr', self.args.learning_rate)}")
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial)
            #super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            record = self.model.generation_config.do_sample
            self.model.generation_config.do_sample = True
            super(LLaVATrainer, self)._save(output_dir, state_dict)
            self.model.generation_config.do_sample = record
