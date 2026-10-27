import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
from utility import tracking
from utility.dataLogger import DataLogger
from utility.metrics import MetricAccuracy


class TrackingTests(unittest.TestCase):
    def tearDown(self):
        tracking._run = None

    def test_rank_zero_and_missing_login(self):
        fake = SimpleNamespace(login=MagicMock(return_value=False), init=MagicMock(), Settings=MagicMock())
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', wandb=fake), \
             patch.dict(os.environ, {'WANDB_MODE': 'online', 'WANDB_PROJECT': 'CondSAM'}), \
             patch('utility.tracking.dist.is_initialized', return_value=True), \
             patch('utility.tracking.dist.get_rank', return_value=1):
            tracking.start({}, directory)
            fake.init.assert_not_called()
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', wandb=fake), \
             patch.dict(os.environ, {'WANDB_MODE': 'online', 'WANDB_PROJECT': 'CondSAM'}), \
             patch('utility.tracking.dist.is_initialized', return_value=False):
            tracking.start({'optimizer': 'CondSAM'}, directory)
            self.assertEqual(fake.init.call_args.kwargs['project'], 'CondSAM')
            self.assertEqual(fake.init.call_args.kwargs['mode'], 'offline')
            fake.init.return_value.define_metric.assert_any_call('cs/*', step_metric='train_step')

    def test_exception_finishes_failed_run(self):
        run = MagicMock()
        tracking._run = run
        def fail():
            raise ValueError('test failure')
        with self.assertRaises(ValueError):
            tracking.execute(fail)
        run.finish.assert_called_once_with(exit_code=1)
        self.assertIsNone(tracking._run)

    def test_metric_logged_before_buffer_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = DataLogger.__new__(DataLogger)
            logger.filePath = directory + '/metrics.hdf5'
            logger.epoch = 4
            logger.train = False
            logger.printTerminal = MagicMock()
            logger._print_and_store = MagicMock()
            metric = MetricAccuracy()
            metric.buffer = [np.array([1., 1., 0.])]
            logger.metrics = [metric]
            with h5py.File(logger.filePath, 'w') as file:
                file.create_group('train'); file.create_group('test')
                metric.createDatasets(file)
            with patch('utility.dataLogger.torch.distributed.get_rank', return_value=0), \
                 patch('utility.dataLogger.tracking.log') as log:
                logger.flush()
            self.assertAlmostEqual(log.call_args.args[0]['test/accuracy'], 2/3)
            self.assertEqual(log.call_args.args[0]['epoch'], 4)
            self.assertEqual(metric.buffer, [])
            with h5py.File(logger.filePath, 'r') as stream:
                self.assertEqual(stream['test/epoch'][:].tolist(), [4])


if __name__ == '__main__':
    unittest.main()
