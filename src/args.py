import os
import argparse

import torch

from src.sam_config import DEFAULT_SAM_CONFIG, load_sam_defaults
from src.models.eval_suite import DEFAULT_ALPHAS, DEFAULT_EVAL_DATASETS


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=str(DEFAULT_SAM_CONFIG.parent.parent / 'datasets'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=DEFAULT_EVAL_DATASETS.copy(),
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
             " Note that same model used for all datasets, so much have same classnames"
             "for zero shot.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Which prompt template is used. Leave as None for linear probe, etc.",
    )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )
    parser.add_argument(
        "--alpha",
        default=DEFAULT_ALPHAS.copy(),
        nargs='*',
        type=float,
        help=(
            'WiSE-FT interpolation coefficients in [0, 1]. '
            'Default: 0, 0.1, ..., 1. Final evaluation always includes both endpoints.'
        )
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00003,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument('--optimizer', type=str.lower, default='adamw',
                        choices=['adamw', 'adam', 'sgd'], help='Base descent optimizer.')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum.')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov SGD.')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                        help='Descent gradient norm limit; zero disables clipping.')
    parser.add_argument('--sam', type=str.lower, default='none',
                        choices=['none', 'sam', 'asam', 'condsam', 'friendlysam',
                                 'fishersam', 'mongesam'],
                        help='Perturbation policy, independent of --optimizer.')
    parser.add_argument('--sam-config', default=str(DEFAULT_SAM_CONFIG),
                        help='INI defaults for supported SAM policies; CLI flags take precedence.')
    parser.add_argument('--sam-rho', type=float, default=0.05)
    parser.add_argument('--sam-include', default=None,
                        help='Regex selecting named parameters for perturbation; default: all trainable.')
    parser.add_argument('--sam-exclude', default=None,
                        help='Regex excluding named parameters from perturbation, not from training.')
    parser.add_argument('--condsam-beta', type=float, default=0.95)
    parser.add_argument('--condsam-eps', type=float, default=1e-8)
    parser.add_argument('--condsam-power-iters', type=int, default=1)
    parser.add_argument('--friendlysam-sigma', type=float, default=1.0)
    parser.add_argument('--friendlysam-lambda', type=float, default=0.95)
    parser.add_argument('--fishersam-eta', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for Python, NumPy and PyTorch/CUDA RNG streams.')
    parser.add_argument('--save-interpolated', action='store_true',
                        help='Save each interpolated checkpoint in addition to evaluating it.')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'],
                        default=os.environ.get('WANDB_MODE') or
                        ('online' if os.environ.get('WANDB_API_KEY') else 'offline'))
    parser.add_argument('--wandb-project', default='wise-ft')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-group', default=None)
    parser.add_argument('--wandb-dir', default=None)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--fisher",
        type=lambda x: x.split(","),
        default=None,
        help="TODO",
    )
    parser.add_argument(
        "--fisher_floor",
        type=float,
        default=1e-8,
        help="TODO",
    )
    config_args, _ = parser.parse_known_args(argv)
    try:
        parser.set_defaults(**load_sam_defaults(config_args.sam_config, config_args.sam))
    except ValueError as error:
        parser.error(str(error))
    parsed_args = parser.parse_args(argv)
    if not 0 <= parsed_args.seed < 2**32:
        parser.error('--seed must be in [0, 2**32)')
    if parsed_args.log_every < 1 or parsed_args.num_workers < 0:
        parser.error('--log-every must be positive and --num-workers must be nonnegative')
    if parsed_args.epochs < 1:
        parser.error('--epochs must be positive')
    if not parsed_args.alpha or any(not 0 <= value <= 1 for value in parsed_args.alpha):
        parser.error('--alpha requires values in [0, 1]')
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
