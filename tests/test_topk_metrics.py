import argparse
import configparser
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

from utility.args import Args
from utility.dataLogger import DataLogger
from utility.metrics import MetricAccuracy, MetricTopKAccuracy


class TopKTests(unittest.TestCase):
    def test_accuracy_and_invalid_k(self):
        state = {
            'predictions': torch.tensor([[9., 8., 7., 6.], [6., 7., 8., 9.], [9., 8., 7., 6.]]),
            'targets': torch.tensor([0, 2, 3]),
        }
        np.testing.assert_array_equal(MetricTopKAccuracy(1).calcMetric(state), MetricAccuracy().calcMetric(state))
        np.testing.assert_array_equal(MetricTopKAccuracy(2).calcMetric(state), [1., 1., 0.])
        np.testing.assert_array_equal(MetricTopKAccuracy(4).calcMetric(state), [1., 1., 1.])
        for k in (0, -1):
            with self.assertRaisesRegex(ValueError, 'positive'):
                MetricTopKAccuracy(k)
        with self.assertRaisesRegex(ValueError, 'number of classes'):
            MetricTopKAccuracy(5).calcMetric(state)

    def test_config_and_cli_lists(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--topK', type=int, nargs='*')
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.ini'
            config.write_text('[general]\ntopK = 1  5\n')
            with patch.object(Args, 'parser', parser), patch.object(Args, 'data', None):
                for extra, expected in (([], [1, 5]), (['--topK', '2', '3'], [2, 3]), (['--topK'], [])):
                    with patch('sys.argv', ['test', '--ifile', str(config), *extra]):
                        Args.parse_args()
                    self.assertEqual(Args.topK, expected)

    def test_multiple_metrics_logged_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(verbose=False, logEach=1, metrics=['accuracy'], topK=[1, 2, 2],
                                   logDir=directory, logSubDir='', contin=False, truncate=True)
            with patch.object(Args, 'data', args), \
                 patch('utility.dataLogger.torch.distributed.get_rank', return_value=0):
                logger = DataLogger()
                self.assertEqual([m.name for m in logger.metrics], ['accuracy', 'accuracy_top1', 'accuracy_top2'])
                for mode in ('train', 'test'):
                    logger.train = mode == 'train'
                    logger.epoch = 1
                    logger.printTerminal = lambda: None
                    for metric in logger.metrics:
                        metric.buffer = [np.array([1., 0.])]
                    with patch('utility.dataLogger.tracking.log') as log:
                        logger.flush()
                    for name in ('accuracy_top1', 'accuracy_top2'):
                        self.assertEqual(log.call_args.args[0][f'{mode}/{name}'], .5)
                        with h5py.File(logger.filePath, 'r') as file:
                            self.assertEqual(file[f'{mode}/{name}'][:].tolist(), [.5])
                args.contin = True
                args.topK = [1, 2, 3]
                resumed = DataLogger()
                with h5py.File(resumed.filePath, 'r') as file:
                    self.assertTrue(np.isnan(file['test/accuracy_top3'][0]))

    def test_vision_configs_define_topk(self):
        for path in Path('configs').glob('*.ini'):
            config = configparser.ConfigParser()
            config.read(path)
            values = {key: value for section in config.sections() for key, value in config[section].items()}
            if 'metrics' in values:
                self.assertIn('topk', values, str(path))


if __name__ == '__main__':
    unittest.main()
