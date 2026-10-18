import torch

from utility.args import Args
from utility.loss import smooth_crossentropy

Args.add_argument("--lookSAMk", type=int, help="")
Args.add_argument("--lookSAMalpha", type=float, help="")
Args.add_argument("--lookSAMLayerwise", type=bool, help="Use layer-wise LookSAM perturbation.")


class LookSAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-1,
            momentum: float = 0.9,
            weight_decay: float = 1e-2,
            nesterov: bool = False,
            rho: float = 0.3,
            alpha: float = 0.7,
            k: int = 5,
            eps: float = 1e-12,
            layerwise: bool = False,
            ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            rho=rho,
            alpha=alpha,
            k=k,
            eps=eps,
            layerwise=layerwise,
        )
        super(LookSAM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p).detach()

        self._step_count = 0

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        if self.param_groups[0]["layerwise"]:
            self._perturb_layerwise()
        else:
            self._perturb_global()

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" in self.state[p]:
                    p.data.copy_(self.state[p]["old_p"])

        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, model, inputs, targets):
        def closure():
            self.zero_grad(set_to_none=True)
            preds = model(inputs)
            loss = smooth_crossentropy(preds, targets).mean()
            loss.backward()
            return preds, loss

        self._step_count += 1
        k = self.param_groups[0]["k"]
        refresh = (self._step_count % k == 0)

        if refresh:
            predictions, loss = closure()
            self._save_current_grads_as("g_ref")

            self.first_step(zero_grad=True)

            model.module.setBatchNormTracking(False)
            closure()
            model.module.setBatchNormTracking(True)

            self._compute_and_store_gv_from_saved_g("g_ref")

            self.second_step(zero_grad=False)
            if Args.grad_clip_norm != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)
            self.SGD_step()
            self.zero_grad(set_to_none=True)
        else:
            predictions, loss = closure()
            self._apply_looksam_reuse_update()

        return loss, predictions

    @torch.no_grad()
    def _perturb_global(self):
        grad_norm = self._grad_norm_global()

        for group in self.param_groups:
            rho = group["rho"]
            eps = group["eps"]
            scale = rho / (grad_norm + eps)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad, alpha=scale.to(p).item())

    @torch.no_grad()
    def _perturb_layerwise(self):
        for group in self.param_groups:
            rho = group["rho"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["old_p"] = p.data.clone()

                param_norm = p.data.norm(p=2)
                grad_norm = p.grad.norm(p=2)
                adaptive_rate = param_norm / (grad_norm + eps)

                grad_normalized = p.grad / (grad_norm + eps)
                perturbation = rho * adaptive_rate * grad_normalized
                p.add_(perturbation)

    def _iter_params_with_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    yield p

    @torch.no_grad()
    def _save_current_grads_as(self, name: str):
        for p in self._iter_params_with_grad():
            self.state[p][name] = p.grad.detach().clone()

    @torch.no_grad()
    def _compute_and_store_gv_from_saved_g(self, g_name: str):
        eps = self.param_groups[0]["eps"]
        device = self.param_groups[0]["params"][0].device

        dot = torch.zeros((), device=device)
        g_norm_sq = torch.zeros((), device=device)
        gs_norm_sq = torch.zeros((), device=device)

        for p in self._iter_params_with_grad():
            g = self.state[p].get(g_name, None)
            if g is None:
                continue
            gs = p.grad.detach()

            dot = dot + torch.sum(g * gs)
            g_norm_sq = g_norm_sq + torch.sum(g * g)
            gs_norm_sq = gs_norm_sq + torch.sum(gs * gs)

        g_norm = torch.sqrt(g_norm_sq + eps)
        gs_norm = torch.sqrt(gs_norm_sq + eps)
        cos_theta = dot / (g_norm * gs_norm + eps)

        gv_norm_sq = torch.zeros((), device=device)
        for p in self._iter_params_with_grad():
            g = self.state[p].get(g_name, None)
            gs = p.grad.detach()

            if g is None:
                gv = gs.clone()
            else:
                proj_magnitude = gs_norm * cos_theta
                g_normalized = g / (g_norm + eps)
                gv = gs - proj_magnitude.to(gs) * g_normalized

            self.state[p]["gv"] = gv
            gv_norm_sq = gv_norm_sq + torch.sum(gv * gv)

        self.state["gv_norm"] = torch.sqrt(gv_norm_sq + eps)

    @torch.no_grad()
    def _apply_looksam_reuse_update(self):
        alpha = self.param_groups[0]["alpha"]
        eps = self.param_groups[0]["eps"]

        gv_norm = self.state.get("gv_norm", None)
        if gv_norm is None or not torch.isfinite(gv_norm) or gv_norm.item() < eps:
            if Args.grad_clip_norm != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)
            self.SGD_step()
            self.zero_grad(set_to_none=True)
            return

        g_norm = self._grad_norm_global()
        scale = alpha * (g_norm / (gv_norm + eps))

        for p in self._iter_params_with_grad():
            gv = self.state[p].get("gv", None)
            if gv is None:
                continue
            p.grad.add_(gv.to(p.grad), alpha=scale.to(p.grad).item())

        if Args.grad_clip_norm != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)
        self.SGD_step()
        self.zero_grad(set_to_none=True)

    def _grad_norm_global(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        norm_sq = torch.zeros((), device=device)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                norm_sq = norm_sq + torch.sum(p.grad.detach() ** 2)

        return torch.sqrt(norm_sq)

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

            sgd(
                params_with_grad,
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


def sgd(
        params,
        d_p_list,
        momentum_buffer_list,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        nesterov: bool,
):
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
