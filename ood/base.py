"""Abstract OOD task contract and shared evaluation lifecycle."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .settings import OODSetting


@dataclass
class OODCase:
    name: str
    dataset: Dataset
    metadata: dict = field(default_factory=dict)


class AbstractOODTask(ABC):
    def __init__(self, setting: OODSetting, transform):
        self.setting = setting
        self.transform = transform

    def loader(self, dataset):
        if self.setting.max_examples:
            dataset = Subset(dataset, range(min(len(dataset), self.setting.max_examples)))
        if len(dataset) == 0:
            raise ValueError('Cannot evaluate an empty OOD dataset')
        return DataLoader(dataset, batch_size=self.setting.batch_size, shuffle=False,
                          num_workers=self.setting.num_workers)

    @abstractmethod
    def cases(self):
        """Yield OODCase instances in deterministic reporting order."""

    @abstractmethod
    def evaluate_case(self, model, case, device):
        """Return numeric metrics for one case without updating model weights."""

    def aggregate(self, cases):
        return {}

    def run(self, model, device, on_case=None):
        training = {module: module.training for module in model.modules()}
        results = []
        try:
            model.eval()
            with torch.inference_mode():
                for case in self.cases():
                    result = {'case': case.name, **case.metadata,
                              'metrics': self.evaluate_case(model, case, device)}
                    results.append(result)
                    if on_case is not None:
                        on_case(result)
            if not results:
                raise ValueError('No OOD cases selected')
            return {'task': self.setting.name, 'kind': self.setting.kind,
                    'diagnostic_subset': bool(self.setting.max_examples),
                    'cases': results, 'aggregate': self.aggregate(results)}
        finally:
            for module, mode in training.items():
                module.training = mode
