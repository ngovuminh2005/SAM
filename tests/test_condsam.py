import copy
import os
import tempfile
import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from optimizer.condsam import CondSAM
from utility.args import Args


def setup_args():
    Args.data = SimpleNamespace(label_smoothing=0.1, grad_clip_norm=0.0)


def ddp_worker(rank, path):
    setup_args()
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=2)
    torch.manual_seed(17)
    model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(3, 2))
    opt = CondSAM(model.parameters())
    x = torch.randn(3 + rank, 3) + rank
    expected_a=x.T@x
    count=torch.tensor(float(len(x)))
    dist.all_reduce(expected_a); dist.all_reduce(count)
    expected_a=.95*torch.eye(3)+.05*expected_a/count
    original_move_up=opt.move_up
    def capture_move_up():
        before=[p.detach().clone() for p in model.parameters()]
        original_move_up()
        for p,old in zip(model.parameters(),before):
            delta=p.detach()-old
            other=delta.clone();dist.broadcast(other,src=0)
            torch.testing.assert_close(delta,other)
    opt.move_up=capture_move_up
    opt.step(model, x, torch.zeros(3 + rank, dtype=torch.long))
    torch.testing.assert_close(opt.state[model.module.weight]['A'], expected_a)
    for value in [*model.parameters(), *[opt.state[model.module.weight][k] for k in ('A','B','x','y')]]:
        other = value.detach().clone()
        dist.broadcast(other, src=0)
        torch.testing.assert_close(value, other)
    # A layer with no local observations still joins every collective.
    p=model.module.weight
    obs={p:(torch.eye(3) if rank==0 else torch.zeros(3,3),
            torch.eye(2) if rank==0 else torch.zeros(2,2),torch.tensor(1. if rank==0 else 0.))}
    previous=opt.state[p]['A'].clone()
    opt._update_factors(obs)
    torch.testing.assert_close(opt.state[p]['A'],.95*previous+.05*torch.eye(3))
    dist.destroy_process_group()


class CondSAMTests(unittest.TestCase):
    def setUp(self):
        setup_args()
        torch.manual_seed(4)

    def test_factor_linear_tokens(self):
        model = torch.nn.Linear(3, 2)
        opt = CondSAM(model.parameters(), beta=0)
        obs = {model.weight: (torch.zeros(3,3), torch.zeros(2,2), torch.zeros(()))}
        handles=[]
        h=model.register_forward_hook(opt._hook(model.weight, obs, handles, 2))
        x=torch.randn(2,4,3)
        model(x).sum().div(2).backward()
        h.remove()
        for handle in handles: handle.remove()
        opt._update_factors(obs)
        torch.testing.assert_close(opt.state[model.weight]['A'], x.flatten(0,1).T @ x.flatten(0,1)/8)
        torch.testing.assert_close(opt.state[model.weight]['B'], torch.ones(2,2))

    def test_conv_factors(self):
        for padding_mode, padding in [('zeros', 1), ('reflect', 1), ('zeros', 'same')]:
            model=torch.nn.Conv2d(2,3,3,padding=padding,padding_mode=padding_mode,dilation=2)
            opt=CondSAM(model.parameters(),beta=0)
            obs={model.weight:(torch.zeros(18,18),torch.zeros(3,3),torch.zeros(()))}
            handles=[]
            h=model.register_forward_hook(opt._hook(model.weight,obs,handles,2))
            x=torch.randn(2,2,7,7)
            model(x).sum().div(2).backward()
            h.remove()
            for handle in handles: handle.remove()
            xx=x
            pp=padding
            if padding_mode!='zeros' or isinstance(padding,str):
                xx=torch.nn.functional.pad(x,model._reversed_padding_repeated_twice,mode=padding_mode if padding_mode!='zeros' else 'constant');pp=0
            patches=torch.nn.functional.unfold(xx,3,dilation=2,padding=pp).transpose(1,2).reshape(-1,18)
            opt._update_factors(obs)
            torch.testing.assert_close(opt.state[model.weight]['A'],patches.T@patches/len(patches))
            torch.testing.assert_close(opt.state[model.weight]['B'],torch.ones(3,3))

    def test_batched_covariance_random_derivatives(self):
        for kernel, stride, padding in [(1,2,0),(3,2,1),(3,1,'same')]:
            module=torch.nn.Conv2d(3,4,kernel,stride=stride,padding=padding,bias=False)
            opt=CondSAM(module.parameters());opt.STATS_CHUNK_BYTES=1024
            activation=torch.randn(7,3,9,9)
            derivative=torch.randn_like(module(activation))
            dim=3*kernel*kernel
            obs=(torch.zeros(dim,dim),torch.zeros(4,4),torch.zeros(()))
            opt._accumulate(module,activation,derivative,7,obs)
            image=activation
            pad=module.padding
            if isinstance(pad,str):
                image=torch.nn.functional.pad(image,module._reversed_padding_repeated_twice);pad=0
            patches=torch.nn.functional.unfold(image,kernel,stride=stride,padding=pad)
            rows=patches.transpose(1,2).reshape(-1,dim)
            gradients=derivative.detach().flatten(2).transpose(1,2).reshape(-1,4)*7
            # Chunk accumulation and a single GEMM have different FP32 rounding.
            # Compare against FP64 sums, including cancellation near zero.
            torch.testing.assert_close(obs[0],(rows.double().T@rows.double()).float(),rtol=1e-5,atol=1e-4)
            torch.testing.assert_close(obs[1],(gradients.double().T@gradients.double()).float(),rtol=1e-5,atol=1e-3)
            self.assertEqual(obs[2].item(),rows.shape[0])

    def test_statistic_buffers_reused_and_not_checkpointed(self):
        model=torch.nn.Linear(3,2);opt=CondSAM(model.parameters(),beta=0)
        x=torch.randn(8,3);targets=torch.zeros(8,dtype=torch.long)
        opt.step(model,x,targets)
        pointers=[value.data_ptr() for value in opt._stats_buffers[model.weight]]
        opt.step(model,x,targets)
        self.assertEqual(pointers,[value.data_ptr() for value in opt._stats_buffers[model.weight]])
        torch.testing.assert_close(opt.state[model.weight]['A'],x.T@x/8)
        self.assertEqual(set(opt.state_dict()),{'state','param_groups'})
        opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        self.assertEqual(opt._stats_buffers,{})

    def test_channels_last_statistics(self):
        module=torch.nn.Conv2d(3,4,(2,3),stride=(2,1),padding=(1,2),dilation=(2,2))
        module=module.to(memory_format=torch.channels_last)
        activation=torch.randn(5,3,9,11).to(memory_format=torch.channels_last)
        derivative=torch.randn_like(module(activation))
        opt=CondSAM(module.parameters());opt.STATS_CHUNK_BYTES=1024
        obs=(torch.zeros(18,18),torch.zeros(4,4),torch.zeros(()))
        opt._accumulate(module,activation,derivative,5,obs)
        patches=torch.nn.functional.unfold(activation,(2,3),dilation=(2,2),padding=(1,2),stride=(2,1))
        rows=patches.transpose(1,2).reshape(-1,18).double()
        gradients=derivative.detach().flatten(2).transpose(1,2).reshape(-1,4).double()*5
        torch.testing.assert_close(obs[0],(rows.T@rows).float(),rtol=1e-5,atol=1e-4)
        torch.testing.assert_close(obs[1],(gradients.T@gradients).float(),rtol=1e-5,atol=1e-3)
        self.assertEqual(obs[2].item(),len(rows))

    def test_extreme_global_norm_scaling(self):
        for magnitude in [1e-40,3e38]:
            p=torch.nn.Parameter(torch.zeros(3))
            opt=CondSAM([p],rho=.1)
            p.grad=torch.full_like(p,magnitude)
            opt.move_up()
            torch.testing.assert_close(p.double().norm(),torch.tensor(.1,dtype=torch.float64))
            opt.move_back()
            self.assertTrue(torch.equal(p,torch.zeros_like(p)))

    def test_formula_and_norm(self):
        p=torch.nn.Parameter(torch.randn(2,3)); q=torch.nn.Parameter(torch.randn(2))
        opt=CondSAM([p,q],rho=.2)
        st=opt.state[p];st.update(A=torch.diag(torch.tensor([1.,2.,3.])),B=torch.diag(torch.tensor([2.,4.])),x=torch.tensor([0.,0.,1.]),y=torch.tensor([0.,1.]))
        p.grad=torch.randn_like(p);q.grad=torch.ones_like(q)
        g=p.grad; s=g[1,2];expected=144/(2*(s+s.sign()*1e-8))*torch.outer(st['y'],st['x'])-280/(12*(g.square().sum()+1e-8))*g
        torch.testing.assert_close(opt._direction(g,st,1e-8),expected)
        before=[v.clone() for v in (p,q)];opt.move_up()
        torch.testing.assert_close(sum((v-b).square().sum() for v,b in zip((p,q),before)).sqrt(),torch.tensor(.2))
        opt.move_back()
        for v,b in zip((p,q),before): self.assertTrue(torch.equal(v,b))
        p.grad.zero_();q.grad=None
        self.assertTrue(torch.isfinite(opt._direction(p.grad,st,1e-8)).all())
        opt.param_groups[0]['rho']=0;opt.move_up();opt.move_back()
        self.assertTrue(torch.equal(p,before[0]))

    def test_step_resume_and_bn(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4),torch.nn.ReLU(inplace=True),torch.nn.Linear(4,2))
        opt=CondSAM(model.parameters());x=torch.randn(5,3);y=torch.zeros(5,dtype=torch.long)
        opt.step(model,x,y)
        self.assertEqual(model[1].num_batches_tracked.item(),1)
        other=copy.deepcopy(model); oo=CondSAM(other.parameters());oo.load_state_dict(copy.deepcopy(opt.state_dict()))
        opt.step(model,x,y);oo.step(other,x,y)
        for a,b in zip(model.parameters(),other.parameters()):torch.testing.assert_close(a,b)
        self.assertFalse(any(m._forward_hooks for m in model.modules()))
        reg=torch.nn.Linear(3,1);CondSAM(reg.parameters()).step(reg,x,torch.randn(5),regression=True)

    def test_exception_restore(self):
        class Broken(torch.nn.Sequential):
            calls=0
            def forward(self,x):
                self.calls+=1
                if self.calls==2: raise RuntimeError('intentional')
                return super().forward(x)
        model=Broken(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3));opt=CondSAM(model.parameters())
        before=[p.clone() for p in model.parameters()]
        with self.assertRaisesRegex(RuntimeError,'intentional'):opt.step(model,torch.randn(4,3),torch.zeros(4,dtype=torch.long))
        for p,b in zip(model.parameters(),before):self.assertTrue(torch.equal(p,b))
        self.assertTrue(model[1].track_running_stats)
        self.assertFalse(any(m._forward_hooks for m in model.modules()))

    def test_zero_and_nonfinite(self):
        p=torch.nn.Parameter(torch.ones(3));opt=CondSAM([p])
        p.grad=torch.zeros_like(p);opt.move_up()
        self.assertTrue(torch.equal(p,torch.ones(3)));opt.move_back()
        p.grad[0]=float('nan')
        with self.assertRaises(FloatingPointError):opt.move_up()
        self.assertTrue(torch.equal(p,torch.ones(3)))
        p.grad=None;opt.move_up();opt.move_back()

    def test_config_integration(self):
        script = """
import sys, torch
from pathlib import Path
import main
from utility.args import Args
from optimizer import getOptimizer
from utility.rhoScheduler import getRhoScheduler
for p in Path('configs').glob('*.ini'):
 if p.name.startswith('NLP_'):continue
 sys.argv=['check','--ifile',str(p),'--optimizer','CondSAM','--condsamRho','0.123']
 Args.parse_args()
 opt=getOptimizer(torch.nn.Linear(3,2).parameters())
 assert getRhoScheduler(opt).base_rhos == [0.123]
old=vars(Args.data).copy()
for k in ('condsamRho','condsamBeta','condsamEps','evalEvery'):old.pop(k)
sys.argv=['check'];Args.parse_args_contin(old)
assert (Args.condsamRho,Args.condsamBeta,Args.condsamEps)==(-1,.95,1e-8)
assert Args.evalEvery==20
"""
        subprocess.run([sys.executable,'-c',script],check=True)

    def test_glue_integration(self):
        script = """
import tempfile,sys
from pathlib import Path
from nlp.run_glue_bert import SAM_SGD_VARIANTS,SAM_REGRESSION_STEP_VARIANTS,_apply_glue_sam_rho_overrides
from utility.args import Args
assert 'CondSAM' in SAM_SGD_VARIANTS & SAM_REGRESSION_STEP_VARIANTS
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'glue.ini'
 p.write_text(Path('configs/default.ini').read_text()+'\\n[GLUE]\\nglue_task = stsb\\nglue_output_root = logs/glue\\n')
 sys.argv=['check','--ifile',str(p),'--optimizer','CondSAM']
 Args.parse_args()
 rho=_apply_glue_sam_rho_overrides('stsb')
 assert Args.condsamRho==rho
"""
        subprocess.run([sys.executable,'-c',script],check=True)

    def test_distributed(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(ddp_worker,args=(os.path.join(tmp,'init'),),nprocs=2,join=True)

if __name__=='__main__':unittest.main()
