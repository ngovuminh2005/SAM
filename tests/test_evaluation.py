import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

from src import tracking
from src.args import parse_arguments
from src.datasets.imagenetv2 import ImageNetV2DatasetWithPaths
from src.models.eval import evaluate
from src.models.eval_suite import DEFAULT_EVAL_DATASETS, evaluate_all_settings
from src.models.interpolation import merge_weights
from src.models.utils import torch_save


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classification_head = torch.nn.Linear(2, 2)
        self.val_preprocess = None
        self.train_preprocess = None
        self.process_images = True

    def forward(self, x):
        return self.classification_head(x)

    def save(self, path):
        torch_save(self, path)


class TinyDataset:
    constructions = []

    def __init__(self, preprocess, **kwargs):
        self.constructions.append(type(self).__name__)
        data = torch.utils.data.TensorDataset(torch.tensor([[1., 0.], [0., 1.]]),
                                             torch.tensor([0, 1]))
        self.train_loader = self.test_loader = torch.utils.data.DataLoader(data, batch_size=2)


def fake_datasets():
    return {name: type(name, (TinyDataset,), {}) for name in DEFAULT_EVAL_DATASETS}


class EvaluationTests(unittest.TestCase):
    def args(self, directory, **overrides):
        args = parse_arguments(['--save', directory, '--wandb-mode', 'disabled',
                                '--num-workers', '0'])
        args.device = 'cpu'
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_final_suite_evaluates_55_combinations_and_restores_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, results_db=str(Path(directory) / 'results.jsonl'))
            model = TinyClassifier()
            final = copy.deepcopy(model.state_dict())
            initial = {key: value + 0.3 for key, value in final.items()}
            TinyDataset.constructions.clear()
            with patch.multiple('src.datasets', **fake_datasets()), contextlib.redirect_stdout(io.StringIO()):
                report = evaluate_all_settings(model, args, reference_state=initial)
            self.assertTrue(report['complete'])
            self.assertEqual(len(report['settings']), 11)
            self.assertEqual(len(TinyDataset.constructions), 55)
            self.assertEqual([row['alpha'] for row in report['settings']], [i / 10 for i in range(11)])
            self.assertTrue(all(len(row['metrics']) == 5 for row in report['settings']))
            self.assertEqual(len(Path(args.results_db).read_text().splitlines()), 11)
            saved = json.loads((Path(directory) / 'evaluation_summary.json').read_text())
            self.assertTrue(saved['complete'])
            self.assertIsInstance(args.alpha, list)
            self.assertFalse(any(':' in key for key in vars(args)))
            self.assertTrue(model.training)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, final[key]))

    def test_partial_failure_keeps_report_incomplete_and_restores_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            model = TinyClassifier()
            final = copy.deepcopy(model.state_dict())
            initial = {key: value + 0.3 for key, value in final.items()}
            with patch('src.models.eval.evaluate', side_effect=[{'ImageNet:top1': 0.5}, RuntimeError('failed')]):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'failed'):
                    evaluate_all_settings(model, args, reference_state=initial)
            report = json.loads((Path(directory) / 'evaluation_summary.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual(len(report['settings']), 1)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, final[key]))

    def test_finetune_automatically_runs_final_suite(self):
        from src.models.finetune import finetune
        with tempfile.TemporaryDirectory() as directory:
            reference = str(Path(directory) / 'initial.pt')
            TinyClassifier().save(reference)
            args = self.args(str(Path(directory) / 'run'), load=reference,
                             train_dataset='ImageNet', epochs=2, warmup_length=0)
            TinyDataset.constructions.clear()
            with patch.multiple('src.datasets', **fake_datasets()), contextlib.redirect_stdout(io.StringIO()):
                checkpoint = finetune(args)
            self.assertTrue(Path(checkpoint).exists())
            self.assertEqual(len(TinyDataset.constructions), 1 + 2 * 5 + 55)
            self.assertTrue((Path(args.save) / 'evaluation_summary.json').exists())

    def test_wise_ft_owns_one_final_suite_and_does_not_mutate_args(self):
        from src.wise_ft import wise_ft
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, load=['initial.pt', 'final.pt'])
            classifier = TinyClassifier()
            with patch('src.wise_ft.ImageClassifier.load', return_value=classifier), \
                    patch('src.wise_ft.evaluate_all_settings', return_value={'complete': True}) as suite:
                self.assertEqual(wise_ft(args), {'complete': True})
            suite.assert_called_once_with(classifier, args, reference_checkpoint='initial.pt')
            self.assertEqual(args.load, ['initial.pt', 'final.pt'])

    def test_interpolation_and_fisher_endpoints(self):
        zero = dict(weight=torch.tensor([1., 2.]), count=torch.tensor(3))
        final = dict(weight=torch.tensor([3., 4.]), count=torch.tensor(7))
        mid = merge_weights(0.25, zero, final)
        torch.testing.assert_close(mid['weight'], torch.tensor([1.5, 2.5]))
        self.assertEqual(mid['count'].dtype, torch.int64)
        self.assertEqual(mid['count'].item(), 3)
        fishers = [dict(weight=torch.tensor([2., 2.])), dict(weight=torch.tensor([4., 4.]))]
        torch.testing.assert_close(merge_weights(0.5, zero, final, fishers)['weight'],
                                   torch.tensor([7/3, 10/3]))
        self.assertTrue(torch.equal(merge_weights(0, zero, final, fishers)['weight'], zero['weight']))
        self.assertTrue(torch.equal(merge_weights(1, zero, final, fishers)['weight'], final['weight']))
        with self.assertRaises(ValueError):
            merge_weights(1.2, zero, final)

    def test_v2_uses_numeric_labels_not_lexicographic_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'imagenetv2-matched-frequency-format-val'
            for label in (0, 2, 10):
                path = root / str(label)
                path.mkdir(parents=True)
                Image.new('RGB', (4, 4)).save(path / 'image.jpeg')
            data = ImageNetV2DatasetWithPaths(location=directory)
            labels = {int(Path(row['image_paths']).parent.name): row['labels'] for row in data}
            self.assertEqual(labels, {0: 0, 2: 2, 10: 10})

    def test_save_does_not_move_live_model_or_replace_parameters(self):
        model = TinyClassifier()
        params = list(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            torch_save(model, str(Path(directory) / 'model.pt'))
        self.assertTrue(all(a is b for a, b in zip(params, model.parameters())))
        self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))


class TrackingTests(unittest.TestCase):
    def test_name_contains_effective_cli_and_ini_parameters(self):
        args = parse_arguments(['--sam', 'condsam', '--sam-rho', '0.5', '--lr', '1e-5',
                                '--condsam-beta', '0.9', '--condsam-power-iters', '4',
                                '--exp_name', 'trial'])
        name = tracking.parameter_run_name(args)
        for part in ['condsam-adamw-rho0p5', 'lr0p00001', 'wd0p1', 'beta0p9',
                     'eps0p00000001', 'poweriters4']:
            self.assertIn(part, name)
        self.assertTrue(name.endswith('-trial'))
        friendly = tracking.parameter_run_name(parse_arguments(['--sam', 'friendlysam']))
        self.assertIn('lambda0p9', friendly)
        baseline = tracking.parameter_run_name(parse_arguments(['--sam', 'none']))
        self.assertNotIn('rho', baseline)

    def test_nested_training_and_evaluation_share_one_run(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(save=directory, wandb_mode='offline', exp_name='test')
            run = Mock()
            run.summary = {}
            wandb = SimpleNamespace(init=Mock(return_value=run), Table=Mock())
            with patch.dict(sys.modules, {'wandb': wandb}):
                with tracking.experiment_run(args):
                    with tracking.experiment_run(args):
                        tracking.log({'train/loss': 1.0})
                    run.finish.assert_not_called()
                    tracking.log_final_results([dict(alpha=0.5, metrics={'ImageNet:top1': 0.7})], 'report.json')
            wandb.init.assert_called_once()
            run.finish.assert_called_once_with(exit_code=0)
            self.assertIsNone(tracking._active_run)

    def test_failed_run_finishes_with_error_status(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(save=directory, wandb_mode='offline', exp_name='test')
            run = Mock()
            with patch.dict(sys.modules, {'wandb': SimpleNamespace(init=Mock(return_value=run))}):
                with self.assertRaises(RuntimeError):
                    with tracking.experiment_run(args):
                        raise RuntimeError('failed')
            run.finish.assert_called_once_with(exit_code=1)
            self.assertIsNone(tracking._active_run)


if __name__ == '__main__':
    unittest.main()
