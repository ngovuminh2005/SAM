"""CondSAM factor refresh interval (run: python -m unittest tests.test_condsam)."""
import os
import sys
import unittest

if not os.environ.get('CONDSAM_GPU_TESTS'):
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


class Bypass(torch.nn.Module):
    """Calls the conv's forward directly, as ffcv's BlurPoolConv2d used to."""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x):
        return self.conv.forward(x)


def observe_once(model, policy, x, y):
    model.zero_grad(set_to_none=True)
    with policy.observe(model, x.shape[0]):
        torch.nn.functional.cross_entropy(model(x), y).backward()
    return policy.perturbations()


class CondSAMStateTest(unittest.TestCase):
    def test_checkpoint_round_trip_with_grouped_factors(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
                                    torch.nn.Conv2d(4, 4, 3, padding=1), torch.nn.Flatten(),
                                    torch.nn.Linear(4 * 6 * 6, 5))
        x, y = torch.randn(8, 3, 6, 6), torch.randint(0, 5, (8,))
        policy = CondSAM(list(model.parameters()), rho=0.05, beta=0.9, power_iters=2)
        for _ in range(2):
            observe_once(model, policy, x, y)
        checkpoint = policy.state_dict()
        for p, state in zip(model.parameters(), checkpoint['state']):
            for k, v in state.items():
                if torch.is_tensor(v):   # a per-layer copy, not the shared group tensor
                    self.assertEqual(v.untyped_storage().nbytes(), v.numel() * v.element_size(), k)
        restored = CondSAM(list(model.parameters()), rho=0.05, beta=0.9, power_iters=2)
        restored.load_state_dict(checkpoint)
        torch.manual_seed(5)
        expected = observe_once(model, policy, x, y)
        torch.manual_seed(5)
        got = observe_once(model, restored, x, y)
        for p in model.parameters():
            torch.testing.assert_close(got[p], expected[p], rtol=0, atol=0)


class CondSAMHooksTest(unittest.TestCase):
    def test_blurpool_layers_are_conditioned(self):
        from train_imagenet import BlurPoolConv2d
        torch.manual_seed(0)
        model = torch.nn.Sequential(BlurPoolConv2d(torch.nn.Conv2d(16, 8, 3, stride=2, padding=1)),
                                    torch.nn.Flatten(), torch.nn.Linear(8 * 4 * 4, 5))
        policy = CondSAM(list(model.parameters()), rho=0.05)
        observe_once(model, policy, torch.randn(4, 16, 8, 8), torch.randint(0, 5, (4,)))
        self.assertIn('A', policy.state[model[0].conv.weight])

    def test_warns_when_a_layer_bypasses_hooks(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(Bypass(), torch.nn.Flatten(), torch.nn.Linear(4 * 6 * 6, 5))
        policy = CondSAM(list(model.parameters()), rho=0.05)
        with self.assertWarns(UserWarning):
            observe_once(model, policy, torch.randn(4, 3, 6, 6), torch.randint(0, 5, (4,)))
        self.assertNotIn('A', policy.state[model[0].conv.weight])

    def test_in_place_input_change_is_an_error(self):
        class Modify(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

            def forward(self, x):
                x = x.clone()
                out = self.conv(x)
                x.mul_(2)   # changes the saved input before backward
                return out + x.mean()

        torch.manual_seed(0)
        model = torch.nn.Sequential(Modify(), torch.nn.Flatten(), torch.nn.Linear(4 * 6 * 6, 5))
        policy = CondSAM(list(model.parameters()), rho=0.05)
        with self.assertRaises(RuntimeError):
            observe_once(model, policy, torch.randn(4, 3, 6, 6), torch.randint(0, 5, (4,)))


@unittest.skipUnless(torch.cuda.is_available() or os.environ.get('CONDSAM_GPU_TESTS'), 'needs CUDA')
class CondSAMGPUTest(unittest.TestCase):
    """Run with CONDSAM_GPU_TESTS=1 (the module hides GPUs by default)."""

    def test_triton_gram_matches_fp64(self):
        from optimizer.gram import gram_accumulate
        for d, n in [(64, 300000), (256, 100000), (576, 50000), (2304, 20000)]:
            x = (torch.randn(n, d, device='cuda') * 3).to(torch.bfloat16)
            ref = x.double().T @ x.double()
            out = gram_accumulate(x, torch.zeros(d, d, device='cuda'), 4.)
            self.assertTrue(torch.equal(out, out.T))
            self.assertLess(((out.double() - 4 * ref).norm() / (4 * ref).norm()).item(), 1e-5, d)

    def test_capture_tolerates_cuda_work_in_other_threads(self):
        # ffcv's loader threads run GPU transforms while training captures
        import threading
        stop = threading.Event()

        def loader():
            while not stop.is_set():
                torch.ones(64, 64, device='cuda').mul(2).sum()   # no CUDA RNG, like ffcv's loader

        thread = threading.Thread(target=loader, daemon=True)
        thread.start()
        try:
            torch.manual_seed(0)
            model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.Flatten(),
                                        torch.nn.Linear(8 * 8 * 8, 5)).cuda()
            policy = CondSAM(list(model.parameters()), rho=0.05)
            for _ in range(4):
                x, y = torch.randn(32, 3, 8, 8, device='cuda'), torch.randint(0, 5, (32,), device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    observe_once(model, policy, x, y)
            self.assertIsNotNone(policy._update_graph)
            self.assertIsNotNone(policy._perturb_graph)
        finally:
            stop.set()
            thread.join()

    def test_fast_paths_match_eager_bitwise(self):
        from train_imagenet import BlurPoolConv2d
        results = []
        for fast in (False, True):   # eager on one stream vs CUDA graphs + side stream
            CondSAM.CUDA_GRAPHS = CondSAM.SIDE_STREAM = fast
            torch.manual_seed(0)
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.BatchNorm2d(16), torch.nn.ReLU(),
                BlurPoolConv2d(torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(32, 10)
            ).cuda().to(memory_format=torch.channels_last)
            policy = CondSAM(list(model.parameters()), rho=0.05, beta=0.9, power_iters=3)
            deltas = []
            for step in range(4):
                torch.manual_seed(1 + step)
                x = torch.randn(256, 3, 16, 16, device='cuda').to(memory_format=torch.channels_last)
                y = torch.randint(0, 10, (256,), device='cuda')
                model.zero_grad(set_to_none=True)
                with policy.observe(model, x.shape[0]):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        torch.nn.functional.cross_entropy(model(x), y).backward()
                d = policy.perturbations()
                deltas.append([d[p].clone() for p in model.parameters()])
            results.append((deltas, [{k: v.clone() for k, v in policy.state[p].items()}
                                     for p in model.parameters()], policy._update_graph is not None))
        CondSAM.CUDA_GRAPHS = CondSAM.SIDE_STREAM = True
        (d0, s0, _), (d1, s1, captured) = results
        self.assertTrue(captured)
        for a, b in zip(d0, d1):
            for u, v in zip(a, b):
                self.assertTrue(torch.equal(u, v))
        for a, b in zip(s0, s1):
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]), k)


if __name__ == '__main__':
    unittest.main()


class CondSAM1Test(unittest.TestCase):
    def test_layers_share_rho_as_in_sam_with_condsam_directions(self):
        from optimizer.condsam1 import CondSAM1
        model, batches = model_and_batches(3)
        policy = CondSAM1(list(model.parameters()), rho=0.7, beta=0.9, power_iters=3)
        for x, y in batches:
            model.zero_grad(set_to_none=True)
            with policy.observe(model, x.shape[0]):
                torch.nn.functional.cross_entropy(model(x), y).backward()
            deltas = policy.perturbations()
        grads = {p: p.grad for p in model.parameters()}
        total = torch.sqrt(sum(g.square().sum() for g in grads.values()))
        for p in model.parameters():
            # SAM's share of rho per tensor...
            torch.testing.assert_close(deltas[p].norm(), 0.7 * grads[p].norm() / total, rtol=1e-4, atol=0)
        for p in (model[0].weight, model[3].weight):
            # ...along CondSAM's direction (same factors and power vectors) inside each layer
            original = CondSAM._direction(grads[p], policy.state[p], policy.options['eps'])
            cosine = torch.nn.functional.cosine_similarity(deltas[p].flatten(), original.flatten(), 0)
            torch.testing.assert_close(cosine, torch.tensor(1.), rtol=0, atol=1e-5)


class CondSAM2Test(unittest.TestCase):
    def test_adds_sam_perturbation_to_condsam1(self):
        from optimizer.condsam2 import CondSAM2
        model, batches = model_and_batches(3)
        policy = CondSAM2(list(model.parameters()), rho=0.3, sam_rho=0.05, beta=0.9, power_iters=3)
        for x, y in batches:
            model.zero_grad(set_to_none=True)
            with policy.observe(model, x.shape[0]):
                torch.nn.functional.cross_entropy(model(x), y).backward()
            deltas = policy.perturbations()
        grads = {p: p.grad for p in model.parameters()}
        total = torch.sqrt(sum(g.square().sum() for g in grads.values()))
        condsam1 = CondSAM.perturbations.__get__(policy)()     # the CondSAM1 part alone
        for p in model.parameters():
            torch.testing.assert_close(deltas[p], condsam1[p] + 0.05 * grads[p] / total, rtol=1e-5, atol=1e-7)
