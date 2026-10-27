from typing import List
from torch import Tensor
import torch
import math

from utility.args import Args
from utility.loss import smooth_crossentropy


class Adam_FriendlySAM(torch.optim.Optimizer):
    """
    FriendlySAM perturbation (two-step) with Adam descent update.

    Descent step matches Adam (L2-style weight decay on gradient), unlike AdamW_FriendlySAM
    which uses decoupled weight decay.
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            maximize: bool = False,
            rho=1,
            sigma=1.0,
            lmbda=0.95,
            ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            rho=rho,
            sigma=sigma,
            lmbda=lmbda,
        )
        super(Adam_FriendlySAM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(p).detach()
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p).detach()

    @torch.no_grad()
    def move_up(self):
        norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue

                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w.detach().clone()

    @torch.no_grad()
    def move_back(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.sub_(self.state[p]["e_w"])

    @torch.no_grad()
    def update_friendly_gradient(self):
        for group in self.param_groups:
            sigma = group["sigma"]
            lmbda = group["lmbda"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach().clone()
                state = self.state[p]

                if "friendly_momentum" not in state:
                    state["friendly_momentum"] = grad
                else:
                    p.grad.sub_(state["friendly_momentum"], alpha=sigma)
                    state["friendly_momentum"].mul_(lmbda).add_(grad, alpha=(1.0 - lmbda))

    def step(self, model, inputs, targets):
        predictions = model(inputs)

        loss = smooth_crossentropy(predictions, targets)
        loss.mean().backward()

        self.update_friendly_gradient()

        self.move_up()
        self.zero_grad()

        model.module.setBatchNormTracking(False)
        loss_intermediate = smooth_crossentropy(model(inputs), targets)
        model.module.setBatchNormTracking(True)
        loss_intermediate.mean().backward()

        self.move_back()

        if Args.grad_clip_norm != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)

        self.inner_step()
        self.zero_grad()

        return loss, predictions

    def _grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for group in self.param_groups for p in group["params"] if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    @torch.no_grad()
    def inner_step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                state["step"] += 1
                state_steps.append(state["step"])

            _adam_l2_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
            )

        return loss


def _adam_l2_step(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)
