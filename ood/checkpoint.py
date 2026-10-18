"""Restore a vision checkpoint with the original architecture and class count."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch


def normalized_state_dict(state):
    output = {}
    for name, tensor in state.items():
        key = name
        while key.startswith(('module.', '_orig_mod.')):
            key = key.split('.', 1)[1]
        if key in output:
            raise ValueError(f'Checkpoint key collision after prefix removal: {key}')
        output[key] = tensor
    return output


def load_vision_checkpoint(run_dir, checkpoint_name, id_dataset, device):
    # Import registrations; the main training function is not executed.
    import main  # noqa: F401
    from models import getModel
    from utility.args import Args
    directory = Path(run_dir).expanduser().resolve()
    params_path = directory / 'params.json'
    checkpoint_path = directory / checkpoint_name
    if not params_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f'Expected {params_path} and {checkpoint_path}; set RUN_DIR to a trained vision run')
    params = json.loads(params_path.read_text())
    if params.get('dataset') != id_dataset:
        raise ValueError(f'Checkpoint trained on {params.get("dataset")}, task requires {id_dataset}')
    previous = Args.data
    try:
        Args.data = SimpleNamespace(**params)
        model = getModel()(num_classes={'CIFAR10': 10, 'CIFAR100': 100}[id_dataset])
    finally:
        Args.data = previous
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'modelState' not in checkpoint:
        raise ValueError('Expected a vision checkpoint containing modelState')
    model.load_state_dict(normalized_state_dict(checkpoint['modelState']), strict=True)
    return model.to(device), params, checkpoint_path
