import copy
import unittest
from types import SimpleNamespace

import torch

from optimizer import ASAM, CondSAM, FisherSAM, FriendlySAM, MongeSAM, POLICIES, SAM
from src.models.training import (apply_perturbations, build_optimizer_and_policy,
                                 select_perturbation_parameters, train_batch)


class PerturbationTests(unittest.TestCase):
    def test_reference_directions_and_no_weight_or_gradient_mutation(self):
        for cls in (SAM, ASAM, FisherSAM, MongeSAM):
            with self.subTest(policy=cls.__name__):
                p = torch.nn.Parameter(torch.tensor([3., 4.], dtype=torch.float64))
                p.grad = torch.tensor([1., -2.], dtype=torch.float64)
                g, w = p.grad.clone(), p.detach().clone()
                policy = cls([p], rho=0.2)
                if cls is ASAM:
                    direction, norm = w.square() * g, (w.abs() * g).norm()
                elif cls is FisherSAM:
                    direction = g / (1 + g.square())
                    norm = (g * direction).sum().sqrt()
                elif cls is MongeSAM:
                    direction, norm = g, g.norm() * (1 + g.square().sum()).sqrt()
                else:
                    direction, norm = g, g.norm()
                torch.testing.assert_close(policy.perturbations()[p], 0.2 * direction / (norm + 1e-12))
                self.assertTrue(torch.equal(p, w))
                self.assertTrue(torch.equal(p.grad, g))
                self.assertNotIsInstance(policy, torch.optim.Optimizer)

    def test_friendly_state_is_separate_from_base_optimizer(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        p.grad = torch.tensor([3., 4.])
        policy = FriendlySAM([p], rho=0.2, sigma=0.5, lmbda=0.8)
        policy.perturbations()
        p.grad = torch.tensor([1., -2.])
        gradient = p.grad.clone()
        direction = gradient - 0.5 * torch.tensor([3., 4.])
        torch.testing.assert_close(policy.perturbations()[p], 0.2 * direction / direction.norm())
        torch.testing.assert_close(policy.state[p]['ema'], torch.tensor([2.6, 2.8]))
        self.assertTrue(torch.equal(p.grad, gradient))

    def test_empty_zero_and_nonfinite_gradients(self):
        p = torch.nn.Parameter(torch.ones(2))
        policy = SAM([p])
        self.assertEqual(policy.perturbations(), {})
        p.grad = torch.zeros_like(p)
        self.assertEqual(policy.perturbations(), {})
        p.grad[0] = float('nan')
        with self.assertRaises(FloatingPointError):
            policy.perturbations()
        self.assertTrue(torch.equal(p, torch.ones_like(p)))

    def test_selection_controls_perturbation_not_descent(self):
        model = torch.nn.Linear(2, 1)
        args = SimpleNamespace(lr=0.01, wd=0.1, optimizer='adamw', sam='sam',
                               sam_include='weight', sam_exclude=None)
        opt, policy, names = build_optimizer_and_policy(model, args)
        self.assertEqual(names, ['weight'])
        self.assertEqual([id(p) for p in policy.params], [id(model.weight)])
        self.assertEqual({id(p) for g in opt.param_groups for p in g['params']},
                         {id(p) for p in model.parameters()})
        model.weight.grad = torch.ones_like(model.weight)
        model.bias.grad = torch.full_like(model.bias, 1e6)
        torch.testing.assert_close(policy.perturbations()[model.weight].norm(), torch.tensor(0.05))
        bias = model.bias.detach().clone()
        train_batch(model, torch.ones(4, 2), torch.zeros(4, 1), torch.nn.MSELoss(), opt, policy)
        self.assertFalse(torch.equal(model.bias, bias))
        with self.assertRaises(ValueError):
            select_perturbation_parameters(model, include='does-not-exist')
        selected = select_perturbation_parameters(torch.nn.DataParallel(model), exclude='bias')
        self.assertEqual([name for name, _ in selected], ['weight'])

    def test_every_policy_composes_with_every_base_optimizer(self):
        for method in POLICIES:
            for base in ('adamw', 'adam', 'sgd'):
                with self.subTest(policy=method, optimizer=base):
                    torch.manual_seed(4)
                    model = torch.nn.Linear(2, 2)
                    args = SimpleNamespace(lr=0.01, wd=0.1, optimizer=base, sam=method)
                    opt, policy, _ = build_optimizer_and_policy(model, args)
                    loss = train_batch(model, torch.randn(3, 2), torch.tensor([0, 1, 0]),
                                       torch.nn.CrossEntropyLoss(), opt, policy)
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_policy_checkpoint_roundtrip(self):
        for cls in (FriendlySAM, CondSAM):
            with self.subTest(policy=cls.__name__):
                model = torch.nn.Linear(2, 2)
                policy = cls(model.parameters())
                with policy.observe(model, 3):
                    model(torch.randn(3, 2)).square().mean().backward()
                policy.perturbations()
                loaded = cls(model.parameters(), rho=0.3)
                loaded.load_state_dict(policy.state_dict())
                first, second = policy.perturbations(), loaded.perturbations()
                for p in first:
                    torch.testing.assert_close(first[p], second[p])
                self.assertEqual(loaded.options, policy.options)
                with self.assertRaises(ValueError):
                    SAM(model.parameters()).load_state_dict(policy.state_dict())


class TrainingTests(unittest.TestCase):
    def test_sam_matches_explicit_two_pass_update(self):
        for base in (torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD):
            with self.subTest(optimizer=base.__name__):
                torch.manual_seed(10)
                model = torch.nn.Linear(2, 1).double()
                reference = copy.deepcopy(model)
                opt = base(model.parameters(), lr=0.03, weight_decay=0.1)
                ref_opt = base(reference.parameters(), lr=0.03, weight_decay=0.1)
                x, y = torch.randn(4, 2).double(), torch.randn(4, 1).double()
                loss_fn = torch.nn.MSELoss()
                ref_opt.zero_grad()
                loss_fn(reference(x), y).backward()
                norm = torch.cat([p.grad.flatten() for p in reference.parameters()]).norm()
                originals = [p.detach().clone() for p in reference.parameters()]
                with torch.no_grad():
                    for p in reference.parameters():
                        p.add_(0.07 * p.grad / (norm + 1e-12))
                ref_opt.zero_grad()
                loss_fn(reference(x), y).backward()
                with torch.no_grad():
                    for p, original in zip(reference.parameters(), originals):
                        p.copy_(original)
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
                ref_opt.step()
                train_batch(model, x, y, loss_fn, opt, SAM(model.parameters(), rho=0.07))
                for p, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(p, expected, rtol=1e-10, atol=1e-12)

    def test_disabled_sam_matches_single_pass(self):
        model = torch.nn.Linear(2, 1)
        reference = copy.deepcopy(model)
        x, y = torch.randn(4, 2), torch.randn(4, 1)
        fn = torch.nn.MSELoss()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        ref_opt = torch.optim.AdamW(reference.parameters(), lr=0.01)
        fn(reference(x), y).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        ref_opt.step()
        train_batch(model, x, y, fn, opt, SAM(model.parameters(), rho=0))
        for p, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, expected, rtol=0, atol=0)

    def test_restore_is_exact_without_second_gradient(self):
        p = torch.nn.Parameter(torch.tensor([1e8, 1e-8]))
        original = p.detach().clone()
        with apply_perturbations({p: torch.tensor([1., 2.])}):
            p.grad = None
        self.assertTrue(torch.equal(p, original))

    def test_failure_restores_weights_batchnorm_and_hooks(self):
        for cls in (SAM, CondSAM):
            with self.subTest(policy=cls.__name__):
                model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3),
                                            torch.nn.Linear(3, 2))
                originals = [p.detach().clone() for p in model.parameters()]
                opt = torch.optim.SGD(model.parameters(), lr=0.1)
                calls = []

                def loss_fn(logits, labels):
                    calls.append(1)
                    if len(calls) == 2:
                        raise RuntimeError('second pass failed')
                    return torch.nn.functional.cross_entropy(logits, labels)

                with self.assertRaisesRegex(RuntimeError, 'second pass failed'):
                    train_batch(model, torch.randn(4, 2), torch.tensor([0, 1, 1, 0]),
                                loss_fn, opt, cls(model.parameters()))
                for p, original in zip(model.parameters(), originals):
                    self.assertTrue(torch.equal(p, original))
                self.assertTrue(model[1].track_running_stats)
                self.assertEqual(model[1].num_batches_tracked.item(), 1)
                self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_batchnorm_statistics_are_updated_once(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3),
                                    torch.nn.Linear(3, 2))
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        train_batch(model, torch.randn(4, 2), torch.tensor([0, 1, 1, 0]),
                    torch.nn.CrossEntropyLoss(), opt, SAM(model.parameters()))
        self.assertEqual(model[1].num_batches_tracked.item(), 1)
        self.assertTrue(model[1].track_running_stats)


class ConditioningTests(unittest.TestCase):
    def test_conv_factors_match_unfold_covariances(self):
        torch.manual_seed(1)
        model = torch.nn.Conv2d(2, 3, kernel_size=2, padding=1)
        policy = CondSAM(model.parameters(), beta=0)
        x = torch.randn(3, 2, 4, 4)
        with policy.observe(model, len(x)):
            output = model(x)
            # Average over images; sum spatial/output dimensions for an
            # analytically known unaveraged derivative equal to output.
            (output.square().sum() / (2 * len(x))).backward()
        patches = torch.nn.functional.unfold(x, model.kernel_size, padding=model.padding)
        a = patches.transpose(1, 2).reshape(-1, patches.shape[1])
        b = output.detach().flatten(2).transpose(1, 2).reshape(-1, 3)
        state = policy.state[model.weight]
        torch.testing.assert_close(state['A'], a.T @ a / len(a))
        torch.testing.assert_close(state['B'], b.T @ b / len(b))

    def test_shared_linear_weight_pools_observations(self):
        class Shared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(2, 2, bias=False)
                self.second = torch.nn.Linear(2, 2, bias=False)
                self.second.weight = self.first.weight

            def forward(self, x):
                return self.first(x) + self.second(2 * x)

        model = Shared()
        policy = CondSAM(model.parameters(), beta=0)
        x = torch.randn(3, 2)
        with policy.observe(model, len(x)):
            model(x).sum().div(len(x)).backward()
        expected = (x.T @ x + (2 * x).T @ (2 * x)) / (2 * len(x))
        torch.testing.assert_close(policy.state[model.first.weight]['A'], expected)

    def test_linear_factors_match_empirical_covariances(self):
        torch.manual_seed(4)
        model = torch.nn.Linear(3, 2)
        policy = CondSAM(model.parameters(), beta=0)
        x, y = torch.randn(5, 3), torch.tensor([0, 1, 1, 0, 1])
        with policy.observe(model, len(y)):
            logits = model(x)
            residual = logits.detach().softmax(-1) - torch.nn.functional.one_hot(y, 2)
            torch.nn.functional.cross_entropy(logits, y).backward()
        state = policy.state[model.weight]
        torch.testing.assert_close(state['A'], x.T @ x / len(y))
        torch.testing.assert_close(state['B'], residual.T @ residual / len(y))
        self.assertNotIn('A', policy.state[model.bias])
        self.assertFalse(model._forward_hooks)

    def test_conditioned_direction_matches_imported_formula(self):
        g = torch.tensor([[1., 2.], [-2., 0.5]])
        a, b = torch.diag(torch.tensor([2., 3.])), torch.diag(torch.tensor([4., 5.]))
        x, y = torch.tensor([0.6, 0.8]), torch.tensor([0.8, 0.6])
        eps = 1e-8
        s = y @ g @ x
        expected = (((x @ a @ x) * (y @ b @ y)).square() / (2 * (s + eps))) * torch.outer(y, x)
        expected -= a.square().sum() * b.square().sum() / (2 * g.numel() * (g.square().sum() + eps)) * g
        actual = CondSAM._direction(g, dict(A=a, B=b, x=x, y=y), eps)
        torch.testing.assert_close(actual, expected)

    def test_unused_and_functional_layers_fall_back_to_sam(self):
        model = torch.nn.Linear(2, 2)
        policy = CondSAM(model.parameters())
        with policy.observe(model, 3):
            torch.nn.functional.linear(torch.ones(3, 2), model.weight, model.bias).sum().backward()
        expected = SAM(model.parameters()).perturbations()
        actual = policy.perturbations()
        for p in expected:
            torch.testing.assert_close(actual[p], expected[p])
        self.assertNotIn('A', policy.state[model.weight])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
