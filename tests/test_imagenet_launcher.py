import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


class ImageNetLauncherTests(unittest.TestCase):
    def commands(self, replacements=None, extra=()):
        source = Path('scripts/run_imagenet.sh').read_text()
        for original, updated in (replacements or {}).items():
            self.assertIn(original, source)
            source = source.replace(original, updated, 1)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', dir='scripts') as script:
            script.write(source)
            script.flush()
            result = subprocess.run(
                ['bash', script.name, *extra],
                env={**os.environ, 'DRY_RUN': '1', 'PYTHON': sys.executable},
                capture_output=True, text=True, check=True,
            )
        return [shlex.split(line) for line in result.stdout.splitlines()]

    def test_default_cartesian_sweep(self):
        commands = self.commands()
        pairs = [(float(cmd[cmd.index('--condsamRho') + 1]),
                  float(cmd[cmd.index('--condsamBeta') + 1])) for cmd in commands]
        expected = {(rho, beta) for rho in (.1, .15, .5, 1., 1.5) for beta in (.99, .995)}
        self.assertEqual(len(commands), 10)
        self.assertEqual(set(pairs), expected)

    def test_sam_ignores_beta_and_sgd_runs_once(self):
        sam = self.commands({'OPTIMIZER="CondSAM"': 'OPTIMIZER="SAM"'})
        self.assertEqual(len(sam), 5)
        self.assertEqual([float(cmd[cmd.index('--samRho') + 1]) for cmd in sam], [.1, .15, .5, 1., 1.5])
        self.assertTrue(all('--condsamBeta' not in cmd for cmd in sam))
        sgd = self.commands({'OPTIMIZER="CondSAM"': 'OPTIMIZER="SGD"'})
        self.assertEqual(len(sgd), 1)
        self.assertNotIn('--condsamRho', sgd[0])
        self.assertNotIn('--condsamBeta', sgd[0])

    def test_empty_lists_use_config_and_singletons_run_once(self):
        empty = self.commands({
            'RHO_LIST=(0.1 0.15 0.5 1 1.5)': 'RHO_LIST=()',
            'CONDSAM_BETA_LIST=(0.99 0.995)': 'CONDSAM_BETA_LIST=()',
        })
        self.assertEqual(len(empty), 1)
        self.assertNotIn('--condsamRho', empty[0])
        self.assertNotIn('--condsamBeta', empty[0])
        single = self.commands({
            'RHO_LIST=(0.1 0.15 0.5 1 1.5)': 'RHO_LIST=(0.2)',
            'CONDSAM_BETA_LIST=(0.99 0.995)': 'CONDSAM_BETA_LIST=(0.9)',
        }, extra=('--topK', '1', '5', '20'))
        self.assertEqual(len(single), 1)
        self.assertEqual(single[0][-4:], ['--topK', '1', '5', '20'])
        self.assertEqual(single[0][single[0].index('--condsamRho') + 1], '0.2')
        self.assertEqual(single[0][single[0].index('--condsamBeta') + 1], '0.9')


if __name__ == '__main__':
    unittest.main()
