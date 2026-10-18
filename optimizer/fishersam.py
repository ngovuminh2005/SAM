import torch

from utility.loss import smooth_crossentropy
from utility.args import Args


# Fisher SAM using the diagonal gradient-magnitude approximation from:
# Kim et al. (ICML 2022), with inverse diagonal entries 1 / (1 + eta * f_i).
class FisherSAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-1,
            momentum: float = 0.9,
            weight_decay: float = 1e-2,
            nesterov: bool = False,
            rho: float = 0.3,
            eta: float = 1.0,
            ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            rho=rho,
            eta=eta,
        )
        super(FisherSAM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p).detach()

    @torch.no_grad()
    def move_up(self):
        fisher_norm = self._fisher_direction_norm()

        for group in self.param_groups:
            rho = group["rho"]
            eta = group["eta"]
            scale = rho / (fisher_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad_sq = p.grad * p.grad
                inv_fisher_diag = 1.0 / (1.0 + eta * grad_sq)
                fisher_direction = inv_fisher_diag * p.grad

                e_w = fisher_direction * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w.detach().clone()

    @torch.no_grad()
    def move_back(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.sub_(self.state[p]["e_w"])

    def step(self, model, inputs, targets):
        predictions = model(inputs)

        loss = smooth_crossentropy(predictions, targets)
        loss.mean().backward()

        self.move_up()
        self.zero_grad()

        # second forward-backward step
        model.module.setBatchNormTracking(False)
        loss_intermediate = smooth_crossentropy(model(inputs), targets)
        model.module.setBatchNormTracking(True)
        loss_intermediate.mean().backward()

        self.move_back()

        if Args.grad_clip_norm != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)
        self.SGD_step()
        self.zero_grad()

        return loss, predictions

    def _fisher_direction_norm(self):
        fisher_quad_terms = []

        for group in self.param_groups:
            eta = group["eta"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad_sq = p.grad * p.grad
                inv_fisher_diag = 1.0 / (1.0 + eta * grad_sq)
                fisher_quad_terms.append((grad_sq * inv_fisher_diag).sum())

        return torch.sqrt(torch.stack(fisher_quad_terms).sum())

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
