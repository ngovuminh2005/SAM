"""CondSAM factor refresh interval (run: python -m unittest tests.test_condsam)."""
import os
import sys
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from optimizer.condsam import CondSAM


def model_and_batches(steps):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.Flatten(), torch.nn.Linear(4 * 6 * 6, 5))
    batches = [(torch.randn(8, 3, 6, 6), torch.randint(0, 5, (8,))) for _ in range(steps)]
    return model, batches


def factors_per_step(stats_every, steps=7):
    model, batches = model_and_batches(steps)
    policy = CondSAM(list(model.parameters()), rho=0.05, beta=0.9, power_iters=2, stats_every=stats_every)
    history = []
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1)
        with policy.observe(model, x.shape[0]):
            torch.nn.functional.cross_entropy(model(x), y).backward()
        policy.perturbations()
        history.append([{k: v.clone() for k, v in policy.state[p].items()}
                        for p in (model[0].weight, model[3].weight)])
    return history


class CondSAMStatsEveryTest(unittest.TestCase):
    def test_factors_refresh_only_every_n_steps(self):
        history = factors_per_step(stats_every=3)
        for step in range(1, len(history)):
            changed = any(not torch.equal(a[k], b[k]) for a, b in zip(history[step - 1], history[step])
                          for k in ('A', 'B', 'x', 'y'))
            self.assertEqual(changed, step % 3 == 0, f'step {step}')

    def test_refresh_steps_match_an_exact_run_on_those_batches(self):
        every = factors_per_step(stats_every=1, steps=1)
        sparse = factors_per_step(stats_every=3, steps=3)
        for a, b in zip(every[0], sparse[2]):
            for k in ('A', 'B', 'x', 'y'):
                torch.testing.assert_close(a[k], b[k])

    def test_rejects_invalid_interval(self):
        model, _ = model_and_batches(0)
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                CondSAM(list(model.parameters()), stats_every=value)


if __name__ == '__main__':
    unittest.main()
