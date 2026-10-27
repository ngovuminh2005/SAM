"""CondSAM analysis variant; all diagnostic work stays in this subclass."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .condsam import CondSAM
from utility.loss import smooth_crossentropy


class CondSam1(CondSAM):
    """Match CondSAM updates and collect rank-zero diagnostics each step.

    Tensor metrics are reduced to min/median/max. Directions are cached while
    the inherited move_up runs, so diagnostics do not recompute directions.
    A temporary root forward hook observes only the second loss. All caches
    and hooks are cleared even when a step fails. CondSAM itself is unchanged.
    """

    def __init__(self, *args, metrics_every=1, **kwargs):
        if isinstance(metrics_every, bool) or not isinstance(metrics_every, int) or metrics_every < 1:
            raise ValueError('metrics_every must be a positive integer')
        super().__init__(*args, **kwargs)
        self.metrics_every = metrics_every
        self._metric_step = 0
        self._collect_metrics = False
        self._metrics = {}
        self._direction_cache = {}

    def step(self, model, inputs, targets, regression=False):
        self._metric_step += 1
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        self._collect_metrics = rank_zero and self._metric_step % self.metrics_every == 0
        self._metrics.clear()
        losses = []
        forward_count = 0

        @torch.no_grad()
        def capture_loss(module, args, output):
            nonlocal forward_count
            forward_count += 1
            if forward_count == 2:
                prediction = output.detach()
                loss = (F.mse_loss(prediction.squeeze(-1), targets.float()) if regression
                        else smooth_crossentropy(prediction, targets).mean())
                losses.append(loss)

        handle = None
        try:
            if self._collect_metrics:
                handle = model.register_forward_hook(capture_loss)
            loss, predictions = super().step(model, inputs, targets, regression=regression)
            if losses:
                self._observe_losses(loss, losses[0])
            return loss, predictions
        except BaseException:
            self._metrics.clear()
            raise
        finally:
            if handle is not None:
                handle.remove()
            self._direction_cache.clear()
            self._collect_metrics = False

    def _direction(self, gradient, state, eps):
        direction = CondSAM._direction(gradient, state, eps)
        if self._collect_metrics:
            self._direction_cache[id(state)] = direction
        return direction

    @torch.no_grad()
    def move_up(self):
        try:
            super().move_up()
            if self._collect_metrics:
                entries = [(p, self._direction_cache[id(self.state[p])], group['rho'])
                           for group in self.param_groups for p in group['params'] if p.grad is not None]
                if entries:
                    norms = torch.stack([torch.linalg.vector_norm(d, dtype=torch.float64) for _, d, _ in entries])
                    self._observe_directions(entries, norms, torch.linalg.vector_norm(norms))
        finally:
            self._direction_cache.clear()

    @torch.no_grad()
    def _observe_directions(self, entries, direction_norms, norm):
        if not self._collect_metrics:
            return
        tiny = torch.finfo(torch.float64).tiny
        rhos = norm.new_tensor([rho for _, _, rho in entries])
        delta_norms = direction_norms / norm.clamp_min(tiny) * rhos
        delta_total = torch.linalg.vector_norm(delta_norms)
        shares = (delta_norms / delta_total.clamp_min(tiny)).square()
        summaries, grad_norms, grad_dots = {}, [], []
        cond_grad_norms, cond_direction_norms, cond_grad_dots = [], [], []
        fallback_shares = []
        eps_by_param = {p: group['eps'] for group in self.param_groups for p in group['params']}
        self._metrics['cs/direction_norm'] = norm
        self._metrics['cs/perturb_norm'] = delta_total
        for (p, direction, _), d_norm, share in zip(entries, direction_norms, shares):
            g = p.grad.detach().float()
            g_scale = g.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
            d_scale = direction.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
            g_scaled, d_scaled = g / g_scale, direction / d_scale
            g_scaled_norm = torch.linalg.vector_norm(g_scaled, dtype=torch.float64)
            g_norm = g_scaled_norm * g_scale.double()
            grad_norms.append(g_norm)
            cosine = ((g_scaled * d_scaled).sum(dtype=torch.float64) / (
                g_scaled_norm * (d_norm / d_scale.double())).clamp_min(tiny)).clamp(-1, 1)
            grad_dot = cosine * g_norm * d_norm
            grad_dots.append(grad_dot)
            values = {'direction_cosine': cosine, 'perturb_share': share,
                      'grad_norm_layer': g_norm, 'direction_norm_layer': d_norm}
            state = self.state[p]
            if 'A' in state:
                cond_grad_norms.append(g_norm)
                cond_direction_norms.append(d_norm)
                cond_grad_dots.append(grad_dot)
                values['cond_direction_cosine'] = cosine
                matrix_g = g.reshape(g.shape[0], -1)
                a, b, x, y = (state[key] for key in ('A', 'B', 'x', 'y'))
                eps = eps_by_param[p]
                projected = y @ matrix_g @ x
                denominator = projected + torch.where(projected >= 0, eps, -eps)
                numerator = ((x @ a @ x) * (y @ b @ y)).square()
                alpha = (numerator / (2 * denominator)).double()
                beta = (a.square().sum() * b.square().sum() /
                        (2 * matrix_g.numel() * (matrix_g.square().sum() + eps))).double()
                outer_norm = torch.linalg.vector_norm(x.double()) * torch.linalg.vector_norm(y.double())
                s_norm, t_norm = alpha.abs() * outer_norm, beta * g_norm
                dot_gs = alpha * projected.double()
                dot_gt = beta * g_norm.square()
                dot_st = beta * dot_gs
                component_total = (s_norm + t_norm).clamp_min(tiny)
                values.update({
                    'abs_yTGx': projected.double().abs(),
                    'alignment': (projected.double().abs() / g_norm.clamp_min(tiny)).clamp(0, 1),
                    'spectral_numerator': numerator.double(),
                    'spectral_norm': s_norm,
                    'gradient_term_norm': t_norm,
                    'spectral_gradient_ratio': torch.where(t_norm > 0, s_norm / t_norm.clamp_min(tiny), 0.),
                    'spectral_cosine': (dot_gs / (g_norm * s_norm).clamp_min(tiny)).clamp(-1, 1),
                    'negative_gradient_cosine': torch.where(t_norm > 0, -1., 0.).double(),
                    'component_cosine': (dot_st / (s_norm * t_norm).clamp_min(tiny)).clamp(-1, 1),
                    'cancellation_ratio': ((s_norm / component_total).square()
                                           + (t_norm / component_total).square()
                                           - 2 * (dot_st / component_total) / component_total).clamp(0, 1).sqrt(),
                    'grad_dot_spectral': dot_gs,
                    'grad_dot_gradient_term': dot_gt,
                    'grad_dot_direction': grad_dot,
                })
                for factor, vector, label in ((a, x, 'A'), (b, y, 'B')):
                    product = factor @ vector
                    eigenvalue = vector.double() @ product.double()
                    residual = product.double() - eigenvalue * vector.double()
                    values[f'power_residual_{label}'] = torch.linalg.vector_norm(residual) / (
                        torch.linalg.vector_norm(product.double()).clamp_min(tiny))
            else:
                fallback_shares.append(share)
            for key, value in values.items():
                summaries.setdefault(key, []).append(value.detach().double())
        total_grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms))
        self._metrics['cs/grad_norm'] = total_grad_norm
        self._metrics['cs/direction_cosine/global'] = (torch.stack(grad_dots).sum() / (
            total_grad_norm * norm).clamp_min(tiny)).clamp(-1, 1)
        if cond_grad_norms:
            self._metrics['cs/direction_cosine/condsam_global'] = (torch.stack(cond_grad_dots).sum() / (
                torch.linalg.vector_norm(torch.stack(cond_grad_norms))
                * torch.linalg.vector_norm(torch.stack(cond_direction_norms))).clamp_min(tiny)).clamp(-1, 1)
        self._metrics['cs/fallback_fraction'] = norm.new_tensor(len(fallback_shares) / len(entries))
        self._metrics['cs/fallback_perturb_share'] = (
            torch.stack(fallback_shares).sum() if fallback_shares else norm.new_zeros(()))
        cosines = torch.stack(summaries['direction_cosine'])
        # Ties select the first tensor; exclude exactly one from "others".
        order = torch.argsort(shares, descending=True, stable=True)
        dominant_cosine = cosines.index_select(0, order[:1]).squeeze(0)
        self._metrics['cs/direction_cosine/dominant'] = torch.where(
            delta_total > 0, dominant_cosine, torch.zeros_like(dominant_cosine))
        if len(entries) > 1:
            others = cosines.index_select(0, order[1:])
            self._metrics['cs/direction_cosine/others_min'] = others.min()
            self._metrics['cs/direction_cosine/others_median'] = torch.quantile(others, 0.5)
            self._metrics['cs/direction_cosine/others_max'] = others.max()
        quantiles = norm.new_tensor([0., 0.5, 1.])
        for key, values in summaries.items():
            quantile_values = torch.quantile(torch.stack(values), quantiles)
            for label, value in zip(('min', 'median', 'max'), quantile_values.unbind()):
                self._metrics[f'cs/{key}/{label}'] = value

    @torch.no_grad()
    def _observe_losses(self, loss, perturbed_loss):
        if self._collect_metrics:
            self._metrics['cs/loss_original'] = loss.detach().double()
            self._metrics['cs/loss_perturbed'] = perturbed_loss.detach().double()
            self._metrics['cs/loss_gap'] = perturbed_loss.detach().double() - loss.detach().double()

    def pop_metrics(self):
        """Consume scalar metrics with one device-to-host transfer."""
        if not self._metrics:
            return {}
        keys = list(self._metrics)
        values = torch.stack([self._metrics[key] for key in keys]).cpu().tolist()
        self._metrics.clear()
        return dict(zip(keys, values))
