import torch

from utility.loss import smooth_crossentropy
from utility.args import Args


def _index_batch_inputs(inputs, indices: torch.Tensor):
    """Support vision tensor batches and NLP dict batches (e.g. GLUE)."""
    if isinstance(inputs, dict):
        return {k: (v[indices] if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    return inputs[indices]

Args.add_argument("--ESAMbeta", type=float, help="")
Args.add_argument("--ESAMgamma", type=float, help="")


# ESAM aligned with the unified trusted implementation:
# - sparse perturbation by beta mask in first step
# - top-gamma instance selection by sharpness for second backward
class ESAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-1,
            momentum: float = 0.9,
            weight_decay: float = 1e-2,
            nesterov: bool = False,
            rho: float = 0.3,
            beta: float = 0.5,
            gamma: float = 0.5,
            ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            rho=rho,
            beta=beta,
            gamma=gamma,
        )
        super(ESAM, self).__init__(params, defaults)

    @torch.no_grad()
    def move_up(self):
        norm = self._grad_norm()

        for group in self.param_groups:
            rho = group["rho"]
            beta = group["beta"]
            scale = rho / (norm + 1e-12)
            sparse_scale = scale / max(beta, 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["old_p"] = p.data.clone()

                if beta < 1.0:
                    mask = (torch.rand_like(p) < beta).to(p.grad.dtype)
                else:
                    mask = 1.0

                e_w = mask * sparse_scale.to(p) * p.grad
                p.add_(e_w)

    @torch.no_grad()
    def move_back(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.data.copy_(self.state[p]["old_p"])

    def step(self, model, inputs, targets):
        predictions = model(inputs)

        loss = smooth_crossentropy(predictions, targets)
        l_before = loss.detach().clone()
        loss.mean().backward()

        self.move_up()
        self.zero_grad()

        with torch.no_grad():
            l_after = smooth_crossentropy(model(inputs), targets)
            instance_sharpness = l_after - l_before

            gamma = self.param_groups[0]["gamma"]
            batch_size = targets.size(0)
            if gamma >= 1.0:
                indices = torch.arange(batch_size, device=targets.device)
            else:
                k = max(1, int(batch_size * gamma))
                _, indices = torch.topk(instance_sharpness, k)

        self.zero_grad()

        # second forward-backward step on selected instances
        model.module.setBatchNormTracking(False)
        loss_intermediate = smooth_crossentropy(
            model(_index_batch_inputs(inputs, indices)), targets[indices]
        )
        model.module.setBatchNormTracking(True)
        loss_intermediate.mean().backward()

        self.move_back()
        if Args.grad_clip_norm != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)
        self.SGD_step()
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
    def SGD_step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    lr=lr,
                    nesterov=nesterov,
                    )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

def sgd(params,
        d_p_list,
        momentum_buffer_list,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        nesterov: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0: #@TODO decouple weight decay from momentum?
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

