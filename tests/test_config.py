"""YAML/JSON/INI config resolution (run: python -m unittest tests.test_config)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transformers

from llava.train.config import load_sam_defaults, parse_args, wandb_run_name
from llava.train.train import DataArguments, ModelArguments, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]


def parse(*args):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    return parse_args(parser, list(args))


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text)
        return str(path)

    def test_stacked_yaml_sets_recipe_and_output_dir(self):
        model, data, training, files = parse(
            '--config', str(ROOT / 'configs/base.yaml'), '--config', str(ROOT / 'configs/vfm/covft.yaml'),
            '--config', str(ROOT / 'configs/sam/sam.yaml'), '--output_dir', str(self.dir),
            '--tf32', 'False')    # tf32 needs a GPU; also checks a CLI bool over YAML
        self.assertEqual(model.vfm_tuning_type, 'context_moe_layernorm')
        self.assertEqual(model.moe_start_layer, 11)
        self.assertEqual(model.version, 'v1_w_context')
        self.assertEqual(data.image_aspect_ratio, 'pad')
        self.assertAlmostEqual(training.learning_rate, 2e-5)
        self.assertTrue(training.bf16 and training.gradient_checkpointing)
        self.assertFalse(training.tf32)
        self.assertEqual(training.report_to, ['wandb'])
        self.assertEqual(training.sam_method, 'sam')
        self.assertEqual(training.sam_rho, 0.01)          # from defaults.ini
        self.assertEqual(len(files), 3)

    def test_precedence_cli_over_yaml_over_ini(self):
        ini = self.write('custom.ini', '[SAM]\nsamRho = 0.2\nasamRho = 0.3\n[CondSAM]\nbeta = 0.5\n')
        override = self.write('override.yaml', 'sam:\n  rho: 0.4\ntraining:\n  learning_rate: 1e-4\n')
        *_, training, _ = parse('--output_dir', str(self.dir), '--sam_method', 'asam', '--sam_config', ini)
        self.assertEqual(training.sam_rho, 0.3)           # the selected method's INI radius
        self.assertEqual(training.sam_condsam_beta, 0.5)
        *_, training, _ = parse('--output_dir', str(self.dir), '--sam_config', ini,
                                '--config', str(ROOT / 'configs/sam/sam.yaml'), '--config', override)
        self.assertEqual(training.sam_rho, 0.4)           # YAML over INI
        self.assertAlmostEqual(training.learning_rate, 1e-4)   # YAML "1e-4" string converted
        *_, training, _ = parse('--output_dir', str(self.dir), '--config', override, '--sam_rho', '0.7')
        self.assertEqual(training.sam_rho, 0.7)           # CLI over YAML

    def test_json_config_and_output_dir_from_file(self):
        path = self.write('run.json', json.dumps({'output_dir': str(self.dir), 'sam_method': 'condsam',
                                                  'sam': {'condsam_power_iters': 3}}))
        *_, training, _ = parse('--config', path)
        self.assertEqual(training.output_dir, str(self.dir))
        self.assertEqual(training.sam_condsam_power_iters, 3)

    def test_unknown_keys_are_errors(self):
        with self.assertRaisesRegex(ValueError, 'Unknown option'):
            parse('--output_dir', str(self.dir), '--config', self.write('typo.yaml', 'sam:\n  rhoo: 1\n'))
        with self.assertRaisesRegex(ValueError, 'Unknown SAM option'):
            load_sam_defaults(self.write('typo.ini', '[SAM]\nrho = 0.1\n'))
        with self.assertRaisesRegex(ValueError, 'must be in'):
            load_sam_defaults(self.write('bad.ini', '[CondSAM]\nbeta = 1.5\n'))

    def test_bundled_ini_and_run_name(self):
        defaults = load_sam_defaults(ROOT / 'llava/optimizer/defaults.ini', 'condsam')
        self.assertEqual(defaults['sam_rho'], 0.01)
        self.assertEqual(defaults['sam_condsam_power_iters'], 1)
        model, _, training, _ = parse('--config', str(ROOT / 'configs/vfm/covft.yaml'),
                                      '--output_dir', str(self.dir), '--sam_method', 'condsam')
        self.assertEqual(wandb_run_name(model, training),
                         'context_moe_layernorm-condsam-rho0.01-beta0.95-pi1-lr5e-05-seed42')


if __name__ == '__main__':
    unittest.main()
