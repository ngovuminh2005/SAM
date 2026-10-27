import copy
import math
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from optimizer.condsam2 import CondSam2
from optimizer.condsam import CondSAM
from utility.args import Args


def ddp_worker(rank, path):
    Args.data = SimpleNamespace(label_smoothing=0.1, grad_clip_norm=0.)
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=2)
    try:
        torch.manual_seed(17)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(3, 2))
        opt = CondSam2(model.parameters(), rho=0.1)
        opt.step(model, torch.randn(4, 3) + rank, torch.zeros(4, dtype=torch.long))
        weight = model.module.weight
        for value in [*model.parameters(), *[opt.state[weight][key] for key in ('A', 'B', 'x', 'y')]]:
            reference = value.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class CondSam2Tests(unittest.TestCase):
    def setUp(self):
        Args.data = SimpleNamespace(label_smoothing=0.1, grad_clip_norm=0.)
        torch.manual_seed(17)

    def test_log_dimension_direction_and_sam_fallback(self):
        state = {'A': torch.diag(torch.tensor([1., 2., 3.])),
                 'B': torch.diag(torch.tensor([2., 4.])),
                 'x': torch.tensor([0., 0., 1.]), 'y': torch.tensor([0., 1.])}
        g = torch.tensor([[1., 2., 3.], [4., 5., -6.]])
        outer = torch.outer(state['y'], state['x'])
        divisor = math.log10(7)
        expected = 144 / (2 * (-6 - 1e-8)) * outer - 280 / (2 * divisor * (91 + 1e-8)) * g
        torch.testing.assert_close(CondSam2._direction(g, state, 1e-8), expected)
        baseline = CondSAM._direction(g, state, 1e-8)
        original_gradient_term = 280 / (2 * 6 * (91 + 1e-8)) * g
        torch.testing.assert_close(expected - baseline, -(6 / divisor - 1) * original_gradient_term)
        flipped = dict(state, x=-state['x'])
        torch.testing.assert_close(CondSam2._direction(g, flipped, 1e-8), expected)
        bias_grad = torch.tensor([1., 2.], dtype=torch.float16)
        fallback = CondSam2._direction(bias_grad, {}, 1e-8)
        self.assertEqual(fallback.dtype, torch.float32)
        torch.testing.assert_close(fallback, bias_grad.float())
        self.assertNotEqual(fallback.data_ptr(), bias_grad.data_ptr())
        conv_state = dict(state, A=torch.eye(4), B=torch.eye(2),
                          x=torch.tensor([0., 1., 0., 0.]))
        conv_gradient = torch.arange(1., 9.).reshape(2, 1, 2, 2)
        conv = CondSam2._direction(conv_gradient, conv_state, 1e-8)
        expected_conv = (torch.outer(conv_state['y'], conv_state['x']) / (2 * (6 + 1e-8))
                         - 8 / (2 * math.log10(9) * (204 + 1e-8)) * conv_gradient.reshape(2, 4))
        torch.testing.assert_close(conv.reshape(2, 4), expected_conv)

    def test_scalar_and_zero_gradient_are_finite(self):
        scalar = torch.tensor([[2.]])
        scalar_state = {key: torch.ones(1, 1) for key in ('A', 'B')}
        scalar_state.update(x=torch.ones(1), y=torch.ones(1))
        expected_scalar = torch.tensor([[1 / (2 * (2 + 1e-8))
                                         - 2 / (2 * math.log10(2) * (4 + 1e-8))]])
        self.assertGreater(math.log10(2), 0.)
        actual_scalar = CondSam2._direction(scalar, scalar_state, 1e-8)
        torch.testing.assert_close(actual_scalar, expected_scalar)
        self.assertFalse(torch.equal(actual_scalar, scalar))
        state = {'A': torch.eye(2), 'B': torch.eye(2),
                 'x': torch.tensor([1., 0.]), 'y': torch.tensor([1., 0.])}
        self.assertTrue(torch.isfinite(CondSam2._direction(torch.zeros(2, 2), state, 1e-8)).all())

    def test_two_pass_perturbation_and_sgd_update(self):
        model = torch.nn.Linear(3, 2)
        opt = CondSam2(model.parameters(), lr=0.02, rho=0.1, weight_decay=0.03)
        inputs, targets = torch.randn(5, 3), torch.randint(2, (5,))
        before = [p.detach().clone() for p in model.parameters()]
        second_grads = []
        calls = []
        handle = model.register_forward_hook(lambda m, args, output: calls.append(1))
        original_move_up, original_descent = opt.move_up, opt.descent_step

        def check_move_up():
            state = opt.state[model.weight]
            g = model.weight.grad.detach().clone()
            a, b, x, y = (state[key] for key in ('A', 'B', 'x', 'y'))
            projection = y @ g @ x
            denominator = projection + torch.where(projection >= 0, 1e-8, -1e-8)
            expected_weight = (((x @ a @ x) * (y @ b @ y)).square() / (2 * denominator)
                               * torch.outer(y, x)
                               - a.square().sum() * b.square().sum()
                               / (2 * math.log10(7) * (g.square().sum() + 1e-8)) * g)
            expected_bias = model.bias.grad.detach().clone()
            norm = torch.sqrt(expected_weight.double().square().sum() + expected_bias.double().square().sum())
            original_move_up()
            for p, old, direction in zip(model.parameters(), before, (expected_weight, expected_bias)):
                torch.testing.assert_close(p, old + direction * (0.1 / norm.item()))

        def capture_descent():
            for p, old in zip(model.parameters(), before):
                torch.testing.assert_close(p, old, rtol=0, atol=0)
                second_grads.append(p.grad.detach().clone())
            original_descent()

        opt.move_up, opt.descent_step = check_move_up, capture_descent
        try:
            opt.step(model, inputs, targets)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 2)
        for p, old, grad in zip(model.parameters(), before, second_grads):
            torch.testing.assert_close(p, old - 0.02 * (grad + 0.03 * old))
        torch.testing.assert_close(opt.state[model.weight]['A'],
                                   0.95 * torch.eye(3) + 0.05 * inputs.T @ inputs / len(inputs))
        self.assertFalse(hasattr(opt, 'pop_metrics'))

    def test_checkpoint_resume(self):
        model = torch.nn.Linear(3, 2)
        opt = CondSam2(model.parameters(), rho=0.1)
        inputs, targets = torch.randn(5, 3), torch.randint(2, (5,))
        opt.step(model, inputs, targets)
        reference = copy.deepcopy(model)
        restored = CondSam2(reference.parameters())
        restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        for _ in range(2):
            opt.step(model, inputs, targets)
            restored.step(reference, inputs, targets)
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_config_scheduler_and_entrypoints(self):
        script = '''
import sys, torch
import main
from utility.args import Args
from optimizer import getOptimizer
from optimizer.condsam2 import CondSam2
from utility.rhoScheduler import getRhoScheduler
sys.argv = ['check', '--ifile', 'configs/default.ini', '--optimizer', 'CondSam2', '--condsamRho', '0.123']
Args.parse_args()
opt = getOptimizer(torch.nn.Linear(3, 2).parameters())
assert isinstance(opt, CondSam2)
assert getRhoScheduler(opt).base_rhos == [0.123]
assert main._resolve_optimizer_rho() == '0.123'
from nlp.run_glue_bert import SAM_SGD_VARIANTS, SAM_REGRESSION_STEP_VARIANTS
assert 'CondSam2' in SAM_SGD_VARIANTS & SAM_REGRESSION_STEP_VARIANTS
'''
        subprocess.run([sys.executable, '-c', script], check=True)

    def test_distributed(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker, args=(os.path.join(directory, 'init'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
