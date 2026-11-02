import contextlib
import io
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.args import parse_arguments
from src.randomness import seed_everything


class SeedTests(unittest.TestCase):
    def test_default_and_cli_override(self):
        self.assertEqual(parse_arguments([]).seed, 42)
        self.assertEqual(parse_arguments(['--seed', '123']).seed, 123)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_arguments(['--seed', '-1'])

    def test_reseeding_reproduces_python_numpy_and_torch(self):
        def sample(seed):
            seed_everything(seed)
            return random.random(), np.random.rand(4), torch.rand(4), torch.randperm(12)
        first = sample(42)
        second = sample(42)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertTrue(torch.equal(first[3], second[3]))
        self.assertFalse(torch.equal(first[2], sample(43)[2]))

    def test_cuda_rngs_are_seeded(self):
        with patch('torch.cuda.manual_seed_all') as cuda:
            seed_everything(42)
        cuda.assert_called_with(42)


if __name__ == '__main__':
    unittest.main()
