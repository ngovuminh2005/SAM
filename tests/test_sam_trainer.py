"""LLaVATrainer's SAM step against a hand-written reference (run: python -m unittest tests.test_sam_trainer)."""
import copy
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')

import torch

from llava.optimizer import POLICIES
from llava.train.llava_trainer import LLaVATrainer
from llava.train.train import TrainingArguments

OPTIONS = {'sam': {}, 'asam': {}, 'mongesam': {}, 'fishersam': dict(sam_fishersam_eta=1.0),
           'friendlysam': dict(sam_friendlysam_sigma=1.0, sam_friendlysam_lambda=0.9)}
RHO = 0.05


class Output(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, key) if isinstance(key, str) else list(self.values())[key]


class TinyModel(torch.nn.Module):
    """Parameter names mimic LLaVA: only `vision_tower` matches the default sam_include."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.vision_tower = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8))
        self.lm_head = torch.nn.Linear(8, 5)

    def forward(self, input_ids, labels, **kwargs):
        logits = self.lm_head(torch.tanh(self.vision_tower(input_ids)))
        return Output(loss=torch.nn.functional.cross_entropy(logits, labels))


def make_inputs(device):
    g = torch.Generator().manual_seed(1)
    return {'input_ids': torch.randn(4, 6, generator=g).to(device),
            'labels': torch.randint(0, 5, (4,), generator=g).to(device)}


def reference(method, inputs, device):
    """Gradients at w + delta, with delta from the policy applied to grad L(w) of vision_tower only."""
    model = TinyModel().to(device)
    vision = list(model.vision_tower.parameters())
    model(**inputs)['loss'].backward()
    options = {k.split('_', 2)[2]: v for k, v in OPTIONS[method].items()}
    if method == 'friendlysam':
        options['lmbda'] = options.pop('lambda')
    deltas = POLICIES[method](vision, rho=RHO, **options).perturbations()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for p, d in deltas.items():
            p.add_(d)
    model(**inputs)['loss'].backward()
    return [p.grad.clone() for p in model.parameters()]


class SAMTrainerTest(unittest.TestCase):
    def make_trainer(self, method, tmp, **extra):
        args = TrainingArguments(output_dir=tmp, bf16=True, report_to=[], sam_method=method, sam_rho=RHO,
                                 per_device_train_batch_size=4, gradient_accumulation_steps=1,
                                 **OPTIONS.get(method, {}), **extra)
        return LLaVATrainer(model=TinyModel(), args=args)

    def test_step_matches_reference_and_restores_weights(self):
        for method in OPTIONS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                trainer = self.make_trainer(method, tmp)
                model = trainer.model
                before = copy.deepcopy(model.state_dict())
                inputs = make_inputs(trainer.args.device)
                trainer.training_step(model, inputs)
                for (name, value), original in zip(model.state_dict().items(), before.values()):
                    self.assertTrue(torch.equal(value, original), f'{name} not restored')
                expected = reference(method, inputs, trainer.args.device)
                for (name, p), grad in zip(model.named_parameters(), expected):
                    torch.testing.assert_close(p.grad, grad, msg=f'{method} {name}')
                self.assertEqual(len(trainer.sam_stats['clean_loss']), 1)

    def test_condsam_runs_and_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer('condsam', tmp, sam_condsam_power_iters=3)
            before = copy.deepcopy(trainer.model.state_dict())
            for _ in range(3):
                trainer.model.zero_grad(set_to_none=True)
                trainer.training_step(trainer.model, make_inputs(trainer.args.device))
            for value, original in zip(trainer.model.state_dict().values(), before.values()):
                self.assertTrue(torch.equal(value, original))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters()))

    def test_zero3_linear_bias_only_patch(self):
        from deepspeed.runtime.zero.linear import LinearFunctionForZeroStage3
        from llava.train.llava_trainer import patch_zero3_linear_bias_only
        patch_zero3_linear_bias_only()
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4, requires_grad=True)
        weight, bias = torch.randn(5, 4), torch.randn(5, requires_grad=True)
        LinearFunctionForZeroStage3.apply(x, weight, bias).square().sum().backward()
        x_ref, bias_ref = x.detach().clone().requires_grad_(), bias.detach().clone().requires_grad_()
        torch.nn.functional.linear(x_ref, weight, bias_ref).square().sum().backward()
        torch.testing.assert_close(bias.grad, bias_ref.grad)
        torch.testing.assert_close(x.grad, x_ref.grad)

    def test_no_sam_and_empty_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer('none', tmp)
            trainer.training_step(trainer.model, make_inputs(trainer.args.device))
            self.assertIsNone(trainer.sam_policy)
            trainer = self.make_trainer('sam', tmp, sam_include='no_such_module')
            with self.assertRaisesRegex(ValueError, 'matched no trainable parameters'):
                trainer.training_step(trainer.model, make_inputs(trainer.args.device))


if __name__ == '__main__':
    unittest.main()
