import torch

from utility.loss import smooth_crossentropy
from utility.args import Args


class GSAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-1,
            momentum: float = 0.9,
            weight_decay: float = 1e-2,
            nesterov: bool = False,
            rho: float = 0.3,
            alpha: float = 0.01,
            adaptive: bool = False,
            perturb_eps: float = 1e-12,
            ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            rho=rho,
            alpha=alpha,
            adaptive=adaptive,
            perturb_eps=perturb_eps,
        )
        super(GSAM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p).detach()

    @torch.no_grad()
    def perturb_weights(self):
        grad_norm = self._grad_norm(weight_adaptive=self.param_groups[0]["adaptive"])

        for group in self.param_groups:
            rho = group["rho"]
            eps = group["perturb_eps"]
            scale = rho / (grad_norm + eps)

            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["old_g"] = p.grad.detach().clone()

                e_w = p.grad * scale.to(p)
                if group["adaptive"]:
                    e_w = e_w * torch.pow(p, 2)

                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if "old_p" in self.state[p]:
                    p.data.copy_(self.state[p]["old_p"])

    @torch.no_grad()
    def gradient_decompose(self):
        alpha = self.param_groups[0]["alpha"]
        eps = self.param_groups[0]["perturb_eps"]

        inner_prod = torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                inner_prod = inner_prod + torch.sum(self.state[p]["old_g"] * p.grad)

        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by="old_g")
        cosine = inner_prod / (new_grad_norm * old_grad_norm + eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                vertical = self.state[p]["old_g"] - cosine * old_grad_norm * p.grad / (new_grad_norm + eps)
                p.grad.add_(vertical, alpha=-alpha)

    def step(self, model, inputs, targets):
        predictions = model(inputs)
        loss = smooth_crossentropy(predictions, targets)
        loss.mean().backward()

        self.perturb_weights()
        self.zero_grad()

        model.module.setBatchNormTracking(False)
        loss_intermediate = smooth_crossentropy(model(inputs), targets)
        model.module.setBatchNormTracking(True)
        loss_intermediate.mean().backward()

        self.gradient_decompose()
        self.unperturb()

        if Args.grad_clip_norm != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)
        self.SGD_step()
        self.zero_grad()

        return loss, predictions

    def _grad_norm(self, by=None, weight_adaptive: bool = False):
        if by is None:
            norm = torch.norm(
                        torch.stack([
                            (((torch.abs(p) if weight_adaptive else 1.0) * p.grad).norm(p=2))
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                   )
        else:
            norm = torch.norm(
                        torch.stack([
                            (((torch.abs(p) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2))
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                   )
        return norm

    @torch.no_grad()
    def SGD_step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            sgd(params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    lr=lr,
                    nesterov=nesterov,
                    )

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def sgd(params,
        d_p_list,
        momentum_buffer_list,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        nesterov: bool):
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)
