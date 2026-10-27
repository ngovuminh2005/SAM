import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

from ood.base import AbstractOODTask, OODCase
from ood.checkpoint import normalized_state_dict
from ood.data import CorruptedCIFAR, evaluation_transform
from ood.metrics import detection_metrics, ood_scores
from ood.settings import OODSetting
from ood.tasks import CorruptionTask, DetectionTask
from utility.evaluation import should_evaluate


class EvaluationScheduleTests(unittest.TestCase):
    def test_schedule_and_final(self):
        for total, expected in [(200,list(range(20,201,20))), (45,[20,40,45]), (3,[3])]:
            self.assertEqual([e for e in range(1,total+1) if should_evaluate(e,total,20)],expected)
        self.assertEqual([e for e in range(21,46) if should_evaluate(e,45,20)],[40,45])
        with self.assertRaises(ValueError):should_evaluate(1,20,0)


class OODTests(unittest.TestCase):
    def test_detection_metric_orientation_and_ties(self):
        result=detection_metrics([0,.1],[.9,1])
        self.assertEqual(result,dict(auroc=1.,aupr_out=1.,aupr_in=1.,fpr95=0.))
        self.assertEqual(detection_metrics([1,1],[0,0])['auroc'],0.)
        tied=detection_metrics([1,1],[1,1])
        self.assertEqual(tied['auroc'],.5);self.assertEqual(tied['fpr95'],1.)
        for args in [([], [0]), ([np.nan], [0])]:
            with self.assertRaises(ValueError):detection_metrics(*args)

    def test_score_formulas(self):
        x=torch.tensor([[3.,0.],[1.,1.]])
        values=ood_scores(x,temperature=2.)
        torch.testing.assert_close(values['msp'],1-x.softmax(-1).max(-1).values)
        torch.testing.assert_close(values['energy'],-2*torch.logsumexp(x/2,dim=-1))
        with self.assertRaises(ValueError):ood_scores(torch.tensor([[float('inf'),0.]]))

    def test_severity_slicing_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            images=np.stack([np.full((32,32,3),i,dtype=np.uint8) for i in range(10)])
            np.save(Path(tmp)/'gaussian_noise.npy',images)
            for labels in (np.array([0,1]),np.tile([0,1],5)):
                np.save(Path(tmp)/'labels.npy',labels)
                ds=CorruptedCIFAR(tmp,'gaussian_noise',3,transforms.ToTensor(),samples_per_severity=2)
                self.assertEqual(len(ds),2)
                self.assertAlmostEqual(float(ds[0][0][0,0,0]),4/255,places=6)
                self.assertEqual(ds[1][1],1)
                self.assertIsNone(ds.__getstate__()['_images'])
            with self.assertRaises(ValueError):CorruptedCIFAR(tmp,'gaussian_noise',0,transforms.ToTensor(),samples_per_severity=2)
            with self.assertRaises(ValueError):CorruptedCIFAR(tmp,'gaussian_noise',1,transforms.ToTensor())

    def test_id_preprocessing_for_out_dataset(self):
        transform=evaluation_transform('CIFAR10',{'normalize':'min_max','imageSize':0})
        value=transform(transforms.functional.to_pil_image(np.zeros((32,32,3),dtype=np.uint8)))
        torch.testing.assert_close(value,torch.full_like(value,-1))

    def test_abstract_contract_restore_and_classification(self):
        with self.assertRaises(TypeError):AbstractOODTask(OODSetting('x','corruption','CIFAR10'),None)
        ds=TensorDataset(torch.tensor([[3.,0.],[0.,3.]]),torch.tensor([0,1]))
        class ToyTask(CorruptionTask):
            def cases(self):
                yield OODCase('clean',ds,{'is_clean':True})
                yield OODCase('toy/1',ds,{'is_clean':False})
        task=ToyTask(OODSetting('toy','corruption','CIFAR10',num_workers=0),None)
        model=torch.nn.Identity();model.train()
        result=task.run(model,'cpu')
        self.assertTrue(model.training)
        self.assertEqual(result['aggregate']['mean_accuracy'],1.)
        self.assertEqual(result['aggregate']['accuracy_drop'],0.)
        with self.assertRaisesRegex(RuntimeError,'callback'):
            task.run(model,'cpu',on_case=lambda _: (_ for _ in ()).throw(RuntimeError('callback')))
        self.assertTrue(model.training)

    def test_detection_ignores_out_labels(self):
        inside=TensorDataset(torch.tensor([[8.,0.],[8.,0.]]),torch.tensor([0,0]))
        outside=TensorDataset(torch.tensor([[0.,0.],[0.,0.]]),torch.tensor([999,999]))
        task=DetectionTask(OODSetting('toy','detection','CIFAR10','CIFAR100',num_workers=0),None)
        with patch('ood.tasks.clean_dataset',side_effect=lambda name,*args: inside if name=='CIFAR10' else outside):
            report=task.run(torch.nn.Identity(),'cpu')
        self.assertEqual(report['aggregate']['msp/auroc'],1.)
        self.assertEqual(report['aggregate']['energy/fpr95'],0.)

    def test_configs_and_checkpoint_prefixes(self):
        for path in Path('configs/ood').glob('*.json'):OODSetting.from_json(path)
        with self.assertRaises(ValueError):OODSetting('x','detection','CIFAR10','CIFAR10')
        state=normalized_state_dict({'module._orig_mod.fc.weight':torch.zeros(1)})
        self.assertIn('fc.weight',state)
        with self.assertRaises(ValueError):normalized_state_dict({'module.x':0,'x':1})

    def test_real_checkpoint_cli_end_to_end(self):
        from models.resnet import ResNet
        from utility.args import Args
        from types import SimpleNamespace
        params=dict(dataset='CIFAR10',model='resnet',depth=18,imageSize=0,normalize='std')
        previous=Args.data
        try:
            Args.data=SimpleNamespace(**params)
            model=ResNet(num_classes=10)
        finally:Args.data=previous
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();data=root/'datasets'
            (run/'params.json').write_text(json.dumps(params))
            torch.save({'modelState':{'module._orig_mod.'+k:v for k,v in model.state_dict().items()}},run/'final.model')
            for name,classes in [('cifar10',10),('cifar100',100)]:
                folder=data/name;folder.mkdir(parents=True)
                np.savez(folder/'test.npz',images=np.zeros((2,32,32,3),dtype=np.uint8),labels=np.zeros(2,dtype=np.int64))
                (folder/'metadata.json').write_text(json.dumps({'classes':[str(i) for i in range(classes)]}))
            setting=root/'setting.json'
            setting.write_text(json.dumps(dict(name='smoke',kind='detection',id_dataset='CIFAR10',ood_dataset='CIFAR100',data_root=str(data),num_workers=0)))
            env={**os.environ,'WANDB_MODE':'disabled','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
            env.pop('SAM_DATASET_ROOT',None)
            subprocess.run([sys.executable,'scripts/evaluate_ood.py','--setting',str(setting),'--run-dir',str(run),
                            '--device','cpu','--max-examples','2','--output-dir',str(root/'out')],env=env,check=True,capture_output=True,text=True)
            report=json.loads((root/'out/results.json').read_text())
            self.assertTrue(report['diagnostic_subset'])
            self.assertEqual(report['aggregate']['num_id'],2)
            self.assertTrue((root/'out/results.csv').is_file())
