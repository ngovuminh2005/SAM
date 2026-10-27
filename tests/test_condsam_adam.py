import copy
import unittest
from types import SimpleNamespace

import torch

from optimizer.condsam import CondSAM
from optimizer.condsam_adam import CondSAM_adam
from utility.args import Args
from utility.rhoScheduler import ConstRho


class CondSAMAdamTests(unittest.TestCase):
    def setUp(self):
        Args.data = SimpleNamespace(label_smoothing=0.0, grad_clip_norm=0.0,
                                    optimizer='CondSAM_adam', rho=0.5)
        torch.manual_seed(17)

    def test_matches_torch_adam_after_condsam_perturbation(self):
        for rho in (0.0, 0.01):
            model = torch.nn.Linear(3, 2)
            reference = copy.deepcopy(model)
            opt = CondSAM_adam(model.parameters(), rho=rho, power_iters=2)
            perturbation = CondSAM(reference.parameters(), rho=rho, power_iters=2)
            adam = torch.optim.Adam(reference.parameters(), lr=1e-3, weight_decay=5e-4)
            perturbation.descent_step = adam.step
            for _ in range(3):
                x, y = torch.randn(5, 3), torch.randint(2, (5,))
                rng_state = torch.get_rng_state()
                opt.step(model, x, y)
                torch.set_rng_state(rng_state)
                perturbation.step(reference, x, y)
                for p, q in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(p, q)
                    for key in ('exp_avg', 'exp_avg_sq'):
                        torch.testing.assert_close(opt.state[p][key], adam.state[q][key])

    def test_checkpoint_resume_and_rho_scheduler(self):
        model = torch.nn.Linear(3, 2)
        opt = CondSAM_adam(model.parameters(), rho=0.01, power_iters=2)
        x, y = torch.randn(5, 3), torch.randint(2, (5,))
        opt.step(model, x, y)
        resumed = copy.deepcopy(model)
        restored = CondSAM_adam(resumed.parameters(), power_iters=2)
        restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        ConstRho(restored).step()
        self.assertEqual(restored.param_groups[0]['rho'], 0.01)
        for _ in range(2):
            opt.step(model, x, y)
            restored.step(resumed, x, y)
        for p, q in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(p, q)
            self.assertEqual(opt.state[p]['step'], restored.state[q]['step'])
            for key in ('exp_avg', 'exp_avg_sq', 'A', 'B', 'x', 'y'):
                if key in opt.state[p]:
                    torch.testing.assert_close(opt.state[p][key], restored.state[q][key])


if __name__ == '__main__':
    unittest.main()
