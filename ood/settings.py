"""Typed settings shared by all OOD tasks."""
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRUPTIONS = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression')


@dataclass(frozen=True)
class OODSetting:
    name: str
    kind: str
    id_dataset: str
    ood_dataset: str = ''
    data_root: str = 'datasets'
    batch_size: int = 256
    num_workers: int = 4
    severities: tuple = (1, 2, 3, 4, 5)
    corruptions: tuple = CORRUPTIONS
    score_methods: tuple = ('msp', 'energy')
    temperature: float = 1.0
    max_examples: int = 0 # 0 means full dataset; positive values mark a diagnostic subset

    def __post_init__(self):
        if self.kind not in ('corruption', 'detection'):
            raise ValueError('OOD kind must be corruption or detection')
        if self.id_dataset not in ('CIFAR10', 'CIFAR100'):
            raise ValueError('OOD ID dataset must be CIFAR10 or CIFAR100')
        if any(type(value) is not int for value in (self.batch_size, self.num_workers, self.max_examples)):
            raise ValueError('OOD batch size, worker count and sample limit must be integers')
        if self.batch_size < 1 or self.num_workers < 0 or self.max_examples < 0:
            raise ValueError('Invalid OOD batch size, worker count or sample limit')
        if self.kind == 'detection' and (self.ood_dataset not in ('CIFAR10', 'CIFAR100', 'SVHN') or self.ood_dataset == self.id_dataset):
            raise ValueError('Choose a different OOD dataset: CIFAR10, CIFAR100 or SVHN')
        if not self.severities or any(type(s) is not int or s not in range(1, 6) for s in self.severities):
            raise ValueError('Corruption severities must be integers in [1, 5]')
        if not self.corruptions or any(c not in CORRUPTIONS for c in self.corruptions):
            raise ValueError('Unknown corruption; use one of the 15 benchmark corruptions')
        if len(set(self.severities)) != len(self.severities) or len(set(self.corruptions)) != len(self.corruptions):
            raise ValueError('Duplicate corruption/severity entries')
        if len(set(self.score_methods)) != len(self.score_methods):
            raise ValueError('Duplicate OOD score methods')
        if not self.score_methods or any(s not in ('msp', 'energy') for s in self.score_methods):
            raise ValueError('Supported OOD scores are msp and energy')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Energy temperature must be finite and positive')

    @property
    def root(self):
        path = Path(self.data_root).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @classmethod
    def from_json(cls, filename, **overrides):
        values = json.loads(Path(filename).read_text())
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown OOD settings: {sorted(unknown)}')
        values.update({key: value for key, value in overrides.items() if value is not None})
        for key in ('corruptions', 'severities', 'score_methods'):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)
