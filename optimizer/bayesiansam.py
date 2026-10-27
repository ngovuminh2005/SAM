import torch

from utility.loss import smooth_crossentropy


class BayesianSAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-1,
            weight_decay: float = 0.0,
            rho: float = 0.01,
            msharpness: int = 8,
            delta: float = 10.0,
            beta1: float = 0.9,
            beta2: float = 0.999,
            s_init: float = 1.0,
            damping: float = 1e-8,
            eps: float = 1e-12,
            n_data: int = 0,
            ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            rho=rho,
            msharpness=msharpness,
            delta=delta,
            beta1=beta1,
            beta2=beta2,
            s_init=s_init,
            damping=damping,
            eps=eps,
            n_data=n_data,
        )
        super(BayesianSAM, self).__init__(params, defaults)

        for group in self.param_groups:
            init_scale = group["s_init"]
            for p in group["params"]:
                if p.requires_grad:
                    st = self.state[p]
                    st["gm"] = torch.zeros_like(p)
                    st["s"] = torch.ones_like(p) * init_scale
                    st["old_p"] = torch.empty_like(p)

    @torch.no_grad()
    def _stash_params(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["old_p"].copy_(p.data)

    @torch.no_grad()
    def _restore_params(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    p.data.copy_(self.state[p]["old_p"])

    @torch.no_grad()
    def _add_noise_sample(self):
        for group in self.param_groups:
            eps = group["eps"]
            n_data = max(int(group["n_data"]), 1)
            for p in group["params"]:
                if not p.requires_grad:
                    continue

                s = self.state[p]["s"]
                scale = torch.sqrt(1.0 / (n_data * s + eps))
                p.add_(torch.randn_like(p) * scale)

    @torch.no_grad()
    def _add_perturbation(self, g_snapshot):
        for group in self.param_groups:
            rho = group["rho"]
            eps = group["eps"]
            for p in group["params"]:
                if p not in g_snapshot:
                    continue

                s = self.state[p]["s"]
                p.add_(rho * g_snapshot[p] / (s + eps))

    def step(self, model, inputs, targets):
        self._stash_params()

        if hasattr(model, "module") and hasattr(model.module, "setBatchNormTracking"):
            model.module.setBatchNormTracking(True)
            with torch.no_grad():
                _ = model(inputs)
            model.module.setBatchNormTracking(False)

        batch_size = inputs.size(0)
        m = max(1, min(int(self.param_groups[0]["msharpness"]), batch_size))
        xs = torch.chunk(inputs, m, dim=0)
        ys = torch.chunk(targets, m, dim=0)
        real_m = len(xs)

        g_eps_sum = {}
        gs_sum = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    g_eps_sum[p] = torch.zeros_like(p)
                    gs_sum[p] = torch.zeros_like(p)

        logits_lists = []
        loss_lists = []

        for xk, yk in zip(xs, ys):
            self._restore_params()
            self.zero_grad(set_to_none=True)
            self._add_noise_sample()

            logits = model(xk)
            loss1 = smooth_crossentropy(logits, yk)
            loss1.mean().backward()

            logits_lists.append(logits.detach())
            loss_lists.append(loss1.mean().detach())

            g_snapshot = {}
            eps = self.param_groups[0]["eps"]
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.detach().clone()
                    g_snapshot[p] = g
                    s = self.state[p]["s"]
                    gs_sum[p].add_(torch.sqrt(s * (g * g) + eps))

            self._restore_params()
            self.zero_grad(set_to_none=True)
            self._add_perturbation(g_snapshot)

            loss2 = smooth_crossentropy(model(xk), yk)
            loss2.mean().backward()

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g_eps_sum[p].add_(p.grad.detach())

        self._restore_params()

        for p in g_eps_sum:
            g_eps_sum[p].div_(real_m)
            gs_sum[p].div_(real_m)

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            damping = group["damping"]
            eps = group["eps"]
            n_data = max(int(group["n_data"]), 1)
            prior_wdecay = group["delta"] / n_data if group["delta"] > 0 else group["weight_decay"]

            for p in group["params"]:
                if not p.requires_grad:
                    continue

                st = self.state[p]
                gm = st["gm"]
                s = st["s"]

                g_eps = g_eps_sum[p]
                gs = gs_sum[p]

                gm.mul_(beta1).add_(g_eps + prior_wdecay * p.data, alpha=(1.0 - beta1))
                p.data.add_(-lr * gm / (s + eps))

                s.mul_(beta2).add_(gs + damping + prior_wdecay, alpha=(1.0 - beta2))

        self.zero_grad(set_to_none=True)

        if hasattr(model, "module") and hasattr(model.module, "setBatchNormTracking"):
            model.module.setBatchNormTracking(True)

        predictions = torch.cat(logits_lists, dim=0)
        loss = torch.stack(loss_lists).mean()
        return loss, predictions
