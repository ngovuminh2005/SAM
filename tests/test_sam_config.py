import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest

import torch

from src.args import parse_arguments
from src.models.training import build_optimizer_and_policy
from src.sam_config import DEFAULT_SAM_CONFIG, load_sam_defaults


class SAMConfigTests(unittest.TestCase):
    def write_config(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'experiment.ini'
        path.write_text(text)
        return str(path)

    def test_per_policy_radii_reach_the_selected_policy(self):
        names = ('sam', 'asam', 'condsam', 'friendlysam', 'fishersam', 'mongesam')
        path = self.write_config('[SAM]\n' + '\n'.join(
            '{}Rho = {}'.format(name, i / 100) for i, name in enumerate(names, 1)))
        for i, name in enumerate(names, 1):
            with self.subTest(method=name):
                args = parse_arguments(['--sam-config', path, '--sam', name])
                self.assertEqual(args.sam_rho, i / 100)
                _, policy, _ = build_optimizer_and_policy(torch.nn.Linear(2, 2), args)
                self.assertEqual(policy.options['rho'], i / 100)

    def test_cli_overrides_ini_independently_of_argument_order(self):
        path = self.write_config('[SAM]\ncondsamRho = 0.12\n[CondSAM]\nbeta = 0.8\n')
        for flags in (['--sam-rho', '0.3', '--sam-config', path, '--sam', 'condsam'],
                      ['--sam-config', path, '--sam', 'condsam', '--sam-rho', '0.3']):
            args = parse_arguments(flags)
            self.assertEqual(args.sam_rho, 0.3)
            self.assertEqual(args.condsam_beta, 0.8)

    def test_partial_config_uses_code_defaults_and_zero_disables_perturbation(self):
        path = self.write_config('[SAM]\nsamRho = 0\n')
        args = parse_arguments(['--sam-config', path, '--sam', 'sam'])
        self.assertEqual(args.sam_rho, 0)
        self.assertEqual(args.friendlysam_lambda, 0.95)
        args = parse_arguments(['--sam-config', path, '--sam', 'asam'])
        self.assertEqual(args.sam_rho, 0.05)

    def test_default_config_is_independent_of_working_directory(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                args = parse_arguments(['--sam', 'FriendlySAM'])
            finally:
                os.chdir(original)
        self.assertEqual(args.sam_config, str(DEFAULT_SAM_CONFIG))
        self.assertEqual(args.sam_rho, 0.01)
        self.assertEqual(args.friendlysam_lambda, 0.9)
        self.assertEqual(args.optimizer, 'adamw')
        self.assertEqual(parse_arguments([]).sam, 'none')

    def test_method_options_are_read_and_cli_can_override_them(self):
        path = self.write_config('''[FriendlySAM]
sigma = 0.5
lmbda = 0.7
[FisherSAM]
eta = 2
[CondSAM]
beta = 0.85
eps = 1e-6
power_iters = 3
''')
        args = parse_arguments(['--sam-config', path, '--condsam-power-iters', '4'])
        self.assertEqual(args.friendlysam_sigma, 0.5)
        self.assertEqual(args.friendlysam_lambda, 0.7)
        self.assertEqual(args.fishersam_eta, 2)
        self.assertEqual(args.condsam_beta, 0.85)
        self.assertEqual(args.condsam_eps, 1e-6)
        self.assertEqual(args.condsam_power_iters, 4)

    def test_invalid_unknown_and_duplicate_settings_are_rejected(self):
        for text in ('[SAM]\ncondsamRho = -1\n',
                     '[SAM]\nsamRho = nan\n',
                     '[SAM]\nrho = 0.1\n',
                     '[CondSAM]\ncondsamRho = 0.1\n',
                     '[CondSAM]\npower_iters = 1.5\n',
                     '[CondSAM]\nbeta = 1\n',
                     '[FriendlySAM]\nlmbda = -0.1\n',
                     '[SAM]\nsamRho = 0.1\nsamRho = 0.2\n',
                     '[ESAM]\nbeta = 0.5\n'):
            with self.subTest(config=text):
                path = self.write_config(text)
                with self.assertRaises(ValueError):
                    load_sam_defaults(path, 'sam')

    def test_missing_custom_file_is_an_argparse_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'missing.ini')
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_arguments(['--sam-config', path])
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
