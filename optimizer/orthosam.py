import torch
import torch.nn.functional as F

from utility.loss import smooth_crossentropy
from utility.args import Args


def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Orthogonalize a 2D tensor with quintic Newton-Schulz iterations.
    """
    assert matrix.ndim == 2

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = matrix.bfloat16()

    transposed = False
    if matrix.size(0) > matrix.size(1):
        x = x.t()
        transposed = True

    x = x / (x.norm() + 1e-7)

    for _ in range(steps):
        a_mat = x @ x.t()
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x

    if transposed:
        x = x.t()

    return x


def _ns_project(tensor: torch.Tensor, steps: int, apply_matrix_scale: bool = False) -> torch.Tensor:
    if tensor.ndim < 2:
        return tensor

    original_shape = tensor.shape
    if tensor.ndim == 2:
        matrix = tensor
    else:
        matrix = tensor.reshape(tensor.size(0), -1)

    projected = zeropower_via_newtonschulz5(matrix, steps=steps).to(dtype=tensor.dtype)
    if apply_matrix_scale:
        projected = projected * (max(1.0, projected.size(0) / projected.size(1)) ** 0.5)
    return projected.reshape(original_shape)


def _is_embedding_or_output_name(param_name: str) -> bool:
    if not param_name:
        return False

    lowered_name = param_name.lower()
    if "embed_tokens" in lowered_name or "embedding" in lowered_name or "embeddings" in lowered_name or "lm_head" in lowered_name:
        return True

    return lowered_name.endswith(".head.weight") or lowered_name.endswith(".classifier.weight") or lowered_name.endswith(".fc.weight")


def _should_use_orthosam(param: torch.Tensor, param_name: str) -> bool:
    return param.ndim >= 2 and not _is_embedding_or_output_name(param_name)


class OrthoSAM(torch.optim.Optimizer):
    """
    Matrix-aware OrthoSAM optimizer:

    1) Matrix branch (ndim >= 2): SAM perturbation + Newton-Schulz update.
    2) Non-matrix branch: vanilla SAM perturbation + vanilla SAM update.
    """

    def __init__(
            self,
            params,
            lr: float = 1e-1,
            momentum: float = 0.95,
            weight_decay: float = 1e-2,
            nesterov: bool = True,
            rho: float = 0.3,
            ns_steps: int = 5,
            eps: float = 1e-12,
            ):
        parameter_entries = list(params)
        if len(parameter_entries) == 0:
            raise ValueError("OrthoSAM received an empty parameter list.")

        named_entries = []
        for entry in parameter_entries:
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str):
                name, param = entry
                named_entries.append((name, param))
            else:
                named_entries.append(("", entry))

        parameters = [param for _, param in named_entries]
        parameter_name_map = {id(param): name for name, param in named_entries}

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            rho=rho,
            ns_steps=ns_steps,
            eps=eps,
        )
        super(OrthoSAM, self).__init__(parameters, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["param_name"] = parameter_name_map.get(id(p), "")
                    state["use_orthosam"] = _should_use_orthosam(p, state["param_name"])
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p).detach()

    @torch.no_grad()
    def move_up(self):
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            rho = group["rho"]
            denom = grad_norm + group["eps"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                normalized_grad = p.grad / denom
                if self.state[p]["use_orthosam"]:
                    direction = _ns_project(normalized_grad, steps=ns_steps, apply_matrix_scale=False)
                else:
                    direction = normalized_grad

                e_w = rho * direction
                p.add_(e_w)
                self.state[p]["e_w"] = e_w.detach().clone()

    def _grad_norm(self):
        grads = [
            p.grad.norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]

        if not grads:
            first_param = self.param_groups[0]["params"][0]
            return torch.zeros((), device=first_param.device)

        return torch.norm(torch.stack(grads), p=2)

    @torch.no_grad()
    def move_back(self):
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])
                    del self.state[p]["e_w"]

    def step(self, model, inputs, targets, regression: bool = False):
        predictions = model(inputs)

        if regression:
            loss = F.mse_loss(predictions.squeeze(-1), targets.float())
        else:
            loss = smooth_crossentropy(predictions, targets).mean()
        loss.backward()

        self.move_up()
        self.zero_grad()

        model.module.setBatchNormTracking(False)
        out2 = model(inputs)
        if regression:
            loss_intermediate = F.mse_loss(out2.squeeze(-1), targets.float())
        else:
            loss_intermediate = smooth_crossentropy(out2, targets).mean()
        model.module.setBatchNormTracking(True)
        loss_intermediate.backward()

        self.move_back()
        if Args.grad_clip_norm != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)
        self._orthosam_update_step()
        self.zero_grad()

        return loss, predictions

    @torch.no_grad()
    def _orthosam_update_step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if self.state[p]["use_orthosam"]:
                    update = _ns_project(p.grad, steps=ns_steps, apply_matrix_scale=False)
                else:
                    update = p.grad

                if wd != 0:
                    update = update.add(p, alpha=wd)

                if momentum != 0:
                    buf = self.state[p]["momentum_buffer"]
                    buf.mul_(momentum).add_(update)
                    update = update.add(buf, alpha=momentum) if nesterov else buf

                p.add_(update, alpha=-lr)
