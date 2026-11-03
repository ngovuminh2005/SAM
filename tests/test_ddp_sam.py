"""CPU/gloo checks of the SAM step under DDP (run: python -m unittest tests.test_ddp_sam)."""
import copy
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings

warnings.filterwarnings('ignore')
from train_imagenet import ImageNetTrainer
from optimizer import POLICIES, apply_perturbations, freeze_batchnorm_statistics

WORLD, BATCH, LR = 2, 8, 0.1
OPTIONS = {'sam': {}, 'asam': {}, 'friendlysam': {}, 'fishersam': {}, 'mongesam': {},
           'condsam': dict(beta=0.99, eps=1e-8, power_iters=5)}


def make_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 5))


def make_batch(rank):
    g = torch.Generator().manual_seed(100 + rank)
    return torch.randn(BATCH, 3, 6, 6, generator=g), torch.randint(0, 5, (BATCH,), generator=g)


def make_policy(method, model, rho):
    return POLICIES[method](list(model.parameters()), rho=rho, **OPTIONS[method])


def gradient(model, x, y, deltas=None, policy=None):
    model.zero_grad(set_to_none=True)
    observe = policy.observe(model, BATCH) if policy is not None else None
    if observe is not None:
        with observe:
            torch.nn.functional.cross_entropy(model(x), y).backward()
    elif deltas:
        with apply_perturbations(deltas), freeze_batchnorm_statistics(model):
            torch.nn.functional.cross_entropy(model(x), y).backward()
    else:
        torch.nn.functional.cross_entropy(model(x), y).backward()
    return [p.grad.clone() for p in model.parameters()]


def reference(method, rho, local):
    """One SGD step of SAM by hand: per-rank (local) or all-reduced (global) perturbation."""
    model = make_model()
    batches = [make_batch(r) for r in range(WORLD)]
    descent = [torch.zeros_like(p) for p in model.parameters()]
    if local:
        for x, y in batches:
            replica = copy.deepcopy(model)
            policy = make_policy(method, replica, rho)
            torch.manual_seed(7)  # CondSAM draws its power-iteration start vectors
            gradient(replica, x, y, policy=policy)
            deltas = policy.perturbations()
            for d, g in zip(descent, gradient(replica, x, y, deltas)):
                d += g / WORLD
    else:
        # Global: the perturbation comes from the rank-averaged gradient
        policy = make_policy(method, model, rho)
        replicas = [copy.deepcopy(model) for _ in batches]
        grads = [gradient(r, x, y) for r, (x, y) in zip(replicas, batches)]
        for p, *gs in zip(model.parameters(), *grads):
            p.grad = sum(gs) / WORLD
        deltas = policy.perturbations()
        deltas = {id_p: d for id_p, d in deltas.items()}
        for (x, y), replica in zip(batches, replicas):
            replica_deltas = {rp: deltas[p] for p, rp in zip(model.parameters(), replica.parameters())
                              if p in deltas}
            for d, g in zip(descent, gradient(replica, x, y, replica_deltas)):
                d += g / WORLD
    with torch.no_grad():
        return [p - LR * d for p, d in zip(model.parameters(), descent)]


def worker(rank, init_file, method, rho, local, drop_rank, out):
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=WORLD)
    model = torch.nn.parallel.DistributedDataParallel(make_model())
    policy = make_policy(method, model.module, rho)
    if rank == drop_rank:
        policy.perturbations = lambda: {}
    trainer = SimpleNamespace(
        model=model, policy=policy, local_perturbation=local,
        loss=torch.nn.CrossEntropyLoss(), amp_dtype=torch.bfloat16, grad_clip=0.,
        optimizer=torch.optim.SGD(model.parameters(), lr=LR),
        scaler=torch.cuda.amp.GradScaler(enabled=False))
    trainer.all_ranks = lambda flag: ImageNetTrainer.all_ranks(trainer, flag)
    trainer.average_gradients = lambda: ImageNetTrainer.average_gradients(trainer)
    torch.manual_seed(7)
    ImageNetTrainer.train_step(trainer, *make_batch(rank))
    torch.save([p.detach() for p in model.parameters()], f'{out}.{rank}')
    dist.destroy_process_group()


def run_ddp(method, rho, local, drop_rank=-1):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'params')
        mp.spawn(worker, args=(os.path.join(tmp, 'init'), method, rho, local, drop_rank, out),
                 nprocs=WORLD, join=True)
        return [torch.load(f'{out}.{r}', weights_only=True) for r in range(WORLD)]


class DDPSAMTest(unittest.TestCase):
    def check(self, got, expected):
        for a, b in zip(got, expected):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)

    def test_local_matches_per_rank_perturbation(self):
        for method in OPTIONS:
            with self.subTest(method=method):
                ranks = run_ddp(method, 0.05, local=True)
                self.check(ranks[1], ranks[0])
                self.check(ranks[0], reference(method, 0.05, local=True))

    def test_global_matches_all_reduced_perturbation(self):
        ranks = run_ddp('sam', 0.05, local=False)
        self.check(ranks[1], ranks[0])
        self.check(ranks[0], reference('sam', 0.05, local=False))

    def test_local_differs_from_global(self):
        local, global_ = run_ddp('sam', 0.5, local=True)[0], run_ddp('sam', 0.5, local=False)[0]
        self.assertFalse(all(torch.allclose(a, b) for a, b in zip(local, global_)))

    def test_rank_without_perturbation_falls_back_in_step(self):
        ranks = run_ddp('sam', 0.05, local=True, drop_rank=1)
        self.check(ranks[1], ranks[0])
        # Every rank skips the second pass: plain SGD on the averaged first-pass gradient
        self.check(ranks[0], reference('sam', 0.0, local=True))


if __name__ == '__main__':
    unittest.main()
