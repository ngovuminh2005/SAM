"""Experiment configuration on top of HfArgumentParser.

Precedence, highest first:
  1. command-line flags
  2. YAML/JSON files given with --config (later files win)
  3. the SAM INI (--sam_config, default llava/optimizer/defaults.ini)
  4. dataclass defaults

YAML/JSON keys are argument names. A top-level mapping can group them into
sections; a section name is an optional prefix, so `sam: {method: sam}` sets
`sam_method` and `training: {learning_rate: 2e-5}` sets `learning_rate`.
"""

import argparse
import configparser
import dataclasses
import json
import math
import os
from pathlib import Path

DEFAULT_SAM_CONFIG = str(Path(__file__).resolve().parents[1] / 'optimizer' / 'defaults.ini')

# All radii are declared in [SAM]; only the selected policy's radius is applied.
_RADII = {name + 'Rho': name for name in
          ('sam', 'asam', 'condsam', 'friendlysam', 'fishersam', 'mongesam')}

# INI section/key -> argument name and value type.
_SAM_OPTIONS = {
    'SAM': {key: ('sam_rho', float) for key in _RADII},
    'FriendlySAM': {'sigma': ('sam_friendlysam_sigma', float),
                    'lmbda': ('sam_friendlysam_lambda', float)},
    'FisherSAM': {'eta': ('sam_fishersam_eta', float)},
    'CondSAM': {'beta': ('sam_condsam_beta', float),
                'eps': ('sam_condsam_eps', float),
                'power_iters': ('sam_condsam_power_iters', int)},
}


def load_sam_defaults(path, method='none'):
    """Return the INI's values for `method`; reject typos and invalid values."""
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    try:
        with Path(path).expanduser().open() as stream:
            config.read_file(stream)
    except (OSError, configparser.Error) as error:
        raise ValueError(f'Cannot read SAM config {path}: {error}') from error
    if config.defaults():
        raise ValueError('Use explicit SAM sections instead of [DEFAULT]')
    defaults = {}
    for section in config.sections():
        if section not in _SAM_OPTIONS:
            raise ValueError(f'Unknown SAM config section [{section}] in {path}')
        for key, raw in config.items(section):
            if key not in _SAM_OPTIONS[section]:
                raise ValueError(f'Unknown SAM option [{section}] {key} in {path}')
            target, convert = _SAM_OPTIONS[section][key]
            try:
                value = convert(raw)
            except ValueError as error:
                raise ValueError(f'Invalid [{section}] {key}: {raw}') from error
            if not math.isfinite(value):
                raise ValueError(f'[{section}] {key} must be finite')
            if key in ('beta', 'lmbda'):
                valid, expected = 0 <= value < 1, 'in [0, 1)'
            elif key in ('eps', 'power_iters'):
                valid, expected = value > 0, 'positive'
            else:
                valid, expected = value >= 0, 'nonnegative'
            if not valid:
                raise ValueError(f'[{section}] {key} must be {expected}')
            if section != 'SAM' or _RADII[key] == method.lower():
                defaults[target] = value
    return defaults


def _read_mapping(path):
    path = Path(path).expanduser()
    try:
        text = path.read_text()
    except OSError as error:
        raise ValueError(f'Cannot read config {path}: {error}') from error
    if path.suffix in ('.yaml', '.yml'):
        import yaml
        data = yaml.safe_load(text) or {}
    elif path.suffix == '.json':
        data = json.loads(text)
    else:
        raise ValueError(f'Config {path} must be .yaml, .yml or .json')
    if not isinstance(data, dict):
        raise ValueError(f'Config {path} must contain a mapping')
    return data


def load_config_files(paths, known):
    """Merge YAML/JSON files into {argument: value}; unknown keys are errors."""
    merged = {}
    for path in paths:
        for key, value in _read_mapping(path).items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    prefixed = f'{key}_{sub_key}'
                    if prefixed in known:
                        merged[prefixed] = sub_value
                    elif sub_key in known:
                        merged[sub_key] = sub_value
                    else:
                        raise ValueError(f'Unknown option {key}.{sub_key} in {path}')
            elif key in known:
                merged[key] = value
            else:
                raise ValueError(f'Unknown option {key} in {path}')
    return merged


def parse_args(parser, args=None):
    """Parse dataclasses from config files, the SAM INI and the command line."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument('--config', action='append', default=[])
    pre.add_argument('--sam_config', default=None)
    pre.add_argument('--sam_method', default=None)
    pre_args, remaining = pre.parse_known_args(args)
    # --sam_config / --sam_method stay in `remaining` for the dataclass parser.
    if pre_args.sam_config is not None:
        remaining += ['--sam_config', pre_args.sam_config]
    if pre_args.sam_method is not None:
        remaining += ['--sam_method', pre_args.sam_method]

    actions = {a.dest: a for a in parser._actions if a.dest != 'help'}
    from_files = load_config_files(pre_args.config, set(actions))
    method = pre_args.sam_method or from_files.get('sam_method') or actions['sam_method'].default
    sam_config = pre_args.sam_config or from_files.get('sam_config') or actions['sam_config'].default
    defaults = dict(load_sam_defaults(sam_config, method), **from_files)

    for dest, value in defaults.items():
        # A value from a file satisfies required arguments such as output_dir.
        actions[dest].required = False
        # argparse converts string defaults with the action's type, e.g. YAML's "2e-5".
        actions[dest].default = value
    parsed = parser.parse_args_into_dataclasses(args=remaining)
    return (*parsed, pre_args.config)


def dump_params(path, config_files, **sections):
    """Write the resolved arguments, one section per dataclass, as JSON."""
    params = {'config_files': list(config_files)}
    for name, value in sections.items():
        if hasattr(value, 'to_dict'):
            params[name] = value.to_dict()
        else:
            params[name] = dataclasses.asdict(value)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(params, handle, indent=2, default=str)


def wandb_run_name(model_args, training_args):
    """Name a run after what varies across a sweep (like ffcv-imagenet)."""
    method = training_args.sam_method
    parts = [model_args.vfm_tuning_type, method]
    if method != 'none':
        parts.append(f'rho{training_args.sam_rho:g}')
    extras = {
        'condsam': [('beta', 'sam_condsam_beta'), ('pi', 'sam_condsam_power_iters')],
        'friendlysam': [('sigma', 'sam_friendlysam_sigma'), ('lambda', 'sam_friendlysam_lambda')],
        'fishersam': [('eta', 'sam_fishersam_eta')],
    }
    for label, key in extras.get(method, []):
        parts.append(f'{label}{getattr(training_args, key):g}')
    parts += [f'lr{training_args.learning_rate:g}', f'seed{training_args.seed}']
    return '-'.join(parts)


def setup_wandb(model_args, training_args):
    """Point HF's WandbCallback at the configured project/group before Trainer init."""
    if 'wandb' not in (training_args.report_to or []):
        return
    for env, value in [('WANDB_PROJECT', training_args.wandb_project),
                       ('WANDB_ENTITY', training_args.wandb_entity),
                       ('WANDB_RUN_GROUP', training_args.wandb_group),
                       ('WANDB_MODE', training_args.wandb_mode)]:
        if value:
            os.environ[env] = value
    # Keep the W&B files next to the checkpoints of the run.
    os.environ.setdefault('WANDB_DIR', training_args.output_dir)
    os.makedirs(training_args.output_dir, exist_ok=True)
    # HF sets run_name = output_dir when it is not given.
    if training_args.run_name in (None, training_args.output_dir):
        training_args.run_name = wandb_run_name(model_args, training_args)
