import copy
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from optimizer.condsam import CondSAM
from optimizer.condsam1 import CondSam1
from utility.args import Args
from utility.loss import smooth_crossentropy


def ddp_metrics_worker(rank, path):
    Args.data = SimpleNamespace(label_smoothing=0.1, grad_clip_norm=0.0)
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=2)
    try:
        torch.manual_seed(17)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(3, 2))
        opt = CondSam1(model.parameters(), rho=0.1)
        inputs, targets = torch.randn(4, 3) + rank, torch.zeros(4, dtype=torch.long)
        opt.step(model, inputs, targets)
        metrics = opt.pop_metrics()
        assert bool(metrics) == (rank == 0)
        if rank == 0:
            assert all(math.isfinite(value) for value in metrics.values())
            assert len(metrics) == 74
        assert not model._forward_hooks
        assert not opt._direction_cache
        for p in model.parameters():
            reference = p.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(p, reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class CondSam1Tests(unittest.TestCase):
    def setUp(self):
        Args.data = SimpleNamespace(label_smoothing=0.1, grad_clip_norm=0.0)
        torch.manual_seed(17)

    def test_matches_standard_condsam_trajectory(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4),
                                    torch.nn.ReLU(), torch.nn.Linear(4, 2))
        reference = copy.deepcopy(model)
        opt = CondSam1(model.parameters(), rho=0.1)
        baseline = CondSAM(reference.parameters(), rho=0.1)
        for _ in range(3):
            inputs, targets = torch.randn(6, 3), torch.randint(2, (6,))
            rng = torch.get_rng_state()
            loss, predictions = opt.step(model, inputs, targets)
            torch.set_rng_state(rng)
            expected_loss, expected_predictions = baseline.step(reference, inputs, targets)
            torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
            torch.testing.assert_close(predictions, expected_predictions, rtol=0, atol=0)
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for key in ('A', 'B', 'x', 'y', 'momentum_buffer'):
                    if key in opt.state[actual]:
                        torch.testing.assert_close(opt.state[actual][key], baseline.state[expected][key], rtol=0, atol=0)
            metrics = opt.pop_metrics()
            self.assertEqual(len(metrics), 74)
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
            self.assertAlmostEqual(metrics['cs/loss_original'], loss.item())
            self.assertAlmostEqual(metrics['cs/loss_gap'],
                                   metrics['cs/loss_perturbed'] - metrics['cs/loss_original'])
            self.assertEqual(opt._direction_cache, {})

        self.assertEqual(opt.pop_metrics(), {})
        self.assertFalse(hasattr(baseline, 'pop_metrics'))
        self.assertFalse(hasattr(baseline, '_observe_directions'))
        self.assertFalse(hasattr(baseline, '_observe_losses'))
        self.assertFalse(any(module._forward_hooks for module in model.modules()))

    def test_losses_match_existing_two_forwards_classification_and_regression(self):
        for regression in (False, True):
            model = torch.nn.Linear(3, 1 if regression else 2)
            opt = CondSam1(model.parameters())
            inputs = torch.randn(5, 3)
            targets = torch.randn(5) if regression else torch.randint(2, (5,))
            outputs = []
            handle = model.register_forward_hook(lambda m, args, output: outputs.append(output.detach().clone()))
            try:
                opt.step(model, inputs, targets, regression=regression)
                metrics = opt.pop_metrics()
                self.assertEqual(len(outputs), 2)
                for prediction, label in zip(outputs, ('original', 'perturbed')):
                    expected = (F.mse_loss(prediction.squeeze(-1), targets.float()) if regression
                                else smooth_crossentropy(prediction, targets).mean())
                    self.assertEqual(metrics[f'cs/loss_{label}'], expected.item())
                self.assertEqual(len(model._forward_hooks), 1)
            finally:
                handle.remove()

    def test_interval_and_nonzero_rank(self):
        model = torch.nn.Linear(3, 2)
        opt = CondSam1(model.parameters(), metrics_every=2)
        inputs, targets = torch.randn(4, 3), torch.zeros(4, dtype=torch.long)
        opt.step(model, inputs, targets)
        self.assertEqual(opt.pop_metrics(), {})
        opt.step(model, inputs, targets)
        self.assertTrue(opt.pop_metrics())
        opt.metrics_every = 1
        with patch('optimizer.condsam1.dist.is_initialized', return_value=True), \
             patch('optimizer.condsam1.dist.get_rank', return_value=1):
            opt.step(model, inputs, targets)
        self.assertEqual(opt.pop_metrics(), {})
        self.assertEqual(opt._direction_cache, {})
        self.assertFalse(model._forward_hooks)

    def test_exception_restores_parameters_and_cleans_diagnostics(self):
        class Broken(torch.nn.Sequential):
            calls = 0

            def forward(self, inputs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError('intentional')
                return super().forward(inputs)

        model = Broken(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
        opt = CondSam1(model.parameters())
        before = [p.detach().clone() for p in model.parameters()]
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            opt.step(model, torch.randn(4, 3), torch.zeros(4, dtype=torch.long))
        for actual, expected in zip(model.parameters(), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(opt.pop_metrics(), {})
        self.assertEqual(opt._direction_cache, {})
        self.assertFalse(opt._collect_metrics)
        self.assertTrue(model[1].track_running_stats)
        self.assertFalse(any(m._forward_hooks for m in model.modules()))

    def test_decomposition_matches_matrix_formula(self):
        p = torch.nn.Parameter(torch.zeros(2, 2))
        opt = CondSam1([p])
        p.grad = torch.tensor([[1., 2.], [3., 4.]])
        a, b = torch.diag(torch.tensor([1., 3.])), torch.diag(torch.tensor([2., 4.]))
        x, y = torch.tensor([0., 1.]), torch.tensor([0., 1.])
        opt.state[p].update(A=a, B=b, x=x, y=y)
        g = p.grad
        projection = y @ g @ x
        numerator = ((x @ a @ x) * (y @ b @ y)).square()
        alpha = numerator / (2 * (projection + 1e-8))
        beta = a.square().sum() * b.square().sum() / (2 * g.numel() * (g.square().sum() + 1e-8))
        s, t = alpha.double() * torch.outer(y.double(), x.double()), beta.double() * g.double()
        opt._collect_metrics = True
        direction = opt._direction(g, opt.state[p], 1e-8)
        norms = direction.double().norm().reshape(1)
        opt._observe_directions([(p, direction, 0.1)], norms, norms.norm())
        metrics = opt.pop_metrics()
        expected = {
            'spectral_norm': s.norm(), 'gradient_term_norm': t.norm(),
            'spectral_gradient_ratio': s.norm() / t.norm(),
            'cancellation_ratio': (s - t).norm() / (s.norm() + t.norm()),
            'spectral_cosine': (s * g.double()).sum() / (s.norm() * g.double().norm()),
            'component_cosine': (s * t).sum() / (s.norm() * t.norm()),
            'grad_dot_spectral': (g.double() * s).sum(),
            'grad_dot_gradient_term': (g.double() * t).sum(),
        }
        for name, value in expected.items():
            for label in ('min', 'median', 'max'):
                self.assertAlmostEqual(metrics[f'cs/{name}/{label}'], value.item(), places=6)

    def test_distributed_metrics_only_on_rank_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_metrics_worker, args=(os.path.join(directory, 'init'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
