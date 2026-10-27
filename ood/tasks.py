import numpy as np
import torch
import torch.nn.functional as F
from .base import AbstractOODTask, OODCase
from .data import CorruptedCIFAR, clean_dataset
from .metrics import detection_metrics, ood_scores


class CorruptionTask(AbstractOODTask):

    def cases(self):
        directory = self.setting.root / {'CIFAR10': 'CIFAR-10-C', 'CIFAR100': 'CIFAR-100-C'}[self.setting.id_dataset]
        # Fail before evaluation if any selected corruption is absent.
        for name in (*self.setting.corruptions, 'labels'):
            if not (directory / f'{name}.npy').is_file():
                raise FileNotFoundError(f'Missing {directory}/{name}.npy; see scripts/ood/prepare_datasets.sh')
        yield OODCase("clean", clean_dataset(self.setting.id_dataset, self.setting.root, self.transform), {"is_clean": True})
        for corruption in self.setting.corruptions:
            for severity in self.setting.severities:
                yield OODCase(f'{corruption}/severity_{severity}',
                              CorruptedCIFAR(directory, corruption, severity, self.transform),
                              {'is_clean': False, 'corruption': corruption, 'severity': severity})

    def evaluate_case(self, model, case, device):
        total, correct, loss = 0, 0, 0.
        for images, targets in self.loader(case.dataset):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            if not torch.isfinite(logits).all():
                raise ValueError('Non-finite classification logits')
            loss += F.cross_entropy(logits.float(), targets, reduction='sum').item()
            correct += (logits.argmax(-1) == targets).sum().item()
            total += len(targets)
        return {'accuracy': correct/total, 'error': 1-correct/total, 'loss': loss/total, 'num_examples': total}

    def aggregate(self, cases):
        corrupt = [case for case in cases if not case.get('is_clean')]
        clean = next(case['metrics'] for case in cases if case.get('is_clean'))
        result = {f'mean_{key}': float(np.mean([c['metrics'][key] for c in corrupt]))
                  for key in ('accuracy', 'error', 'loss')}
        for severity in self.setting.severities:
            rows = [case['metrics'] for case in corrupt if case.get('severity') == severity]
            if rows:
                result[f'severity_{severity}/accuracy'] = float(np.mean([r['accuracy'] for r in rows]))
        for corruption in self.setting.corruptions:
            rows = [case['metrics'] for case in corrupt if case.get('corruption') == corruption]
            if rows:
                result[f'{corruption}/accuracy'] = float(np.mean([r['accuracy'] for r in rows]))
        result.update(clean_accuracy=clean['accuracy'], clean_loss=clean['loss'],
                      accuracy_drop=clean['accuracy']-result['mean_accuracy'])
        return result


class DetectionTask(AbstractOODTask):

    def cases(self):
        yield OODCase(self.setting.ood_dataset,
                      clean_dataset(self.setting.ood_dataset, self.setting.root, self.transform),
                      {'id_dataset': self.setting.id_dataset, 'ood_dataset': self.setting.ood_dataset})

    def _scores(self, model, dataset, device):
        values = {method: [] for method in self.setting.score_methods}
        for images, _ in self.loader(dataset):
            logits = model(images.to(device))
            for method, scores in ood_scores(logits, self.setting.score_methods, self.setting.temperature).items():
                values[method].append(scores.cpu().numpy())
        return {method: np.concatenate(parts) for method, parts in values.items()}

    def evaluate_case(self, model, case, device):
        inside = self._scores(model, clean_dataset(self.setting.id_dataset, self.setting.root, self.transform), device)
        outside = self._scores(model, case.dataset, device)
        metrics = {}
        for method in self.setting.score_methods:
            metrics.update({f'{method}/{name}': value for name, value in detection_metrics(inside[method], outside[method]).items()})
        metrics['num_id'] = len(next(iter(inside.values())))
        metrics['num_ood'] = len(next(iter(outside.values())))
        return metrics

    def aggregate(self, cases):
        return cases[0]['metrics'].copy()
