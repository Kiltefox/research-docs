"""Losses and evaluation metrics."""

from __future__ import annotations

import math
from typing import Any

import torch


def student_t_nll(pred: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    mu = torch.nan_to_num(pred["mu"].float(), nan=0.0, posinf=1e4, neginf=-1e4)
    sigma = torch.nan_to_num(pred["sigma"].float(), nan=1.0, posinf=1e4, neginf=1e-5).clamp_min(1e-5)
    nu = torch.nan_to_num(pred["nu"].float(), nan=2.0, posinf=1e4, neginf=2.0).clamp_min(2.0)
    dist = torch.distributions.StudentT(df=nu, loc=mu, scale=sigma, validate_args=False)
    return -dist.log_prob(target).mean()


@torch.no_grad()
def regression_metrics(
    pred: dict[str, torch.Tensor],
    target_scaled: torch.Tensor,
    target_raw: torch.Tensor,
    normalizer: Any,
    coverage: float = 0.9,
) -> dict[str, float]:
    mu_raw = normalizer.inverse_rho(pred["mu"].detach().cpu())
    target_raw = target_raw.detach().cpu()
    nll = float(student_t_nll({k: v.detach().cpu() for k, v in pred.items()}, target_scaled.detach().cpu()).item())
    err = mu_raw - target_raw
    interval = prediction_interval(pred, normalizer, coverage=coverage)
    covered = (target_raw >= interval[0]) & (target_raw <= interval[1])
    return {
        "nll": nll,
        "mae": float(err.abs().mean().item()),
        "rmse": float(err.square().mean().sqrt().item()),
        "coverage": float(covered.float().mean().item()),
        "mu_mean": float(mu_raw.mean().item()),
        "target_mean": float(target_raw.mean().item()),
    }


@torch.no_grad()
def prediction_interval(
    pred: dict[str, torch.Tensor],
    normalizer: Any,
    coverage: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = 1.0 - coverage
    q = _student_t_quantile(1.0 - alpha / 2.0, pred["nu"].detach().cpu())
    lower = pred["mu"].detach().cpu() - q * pred["sigma"].detach().cpu()
    upper = pred["mu"].detach().cpu() + q * pred["sigma"].detach().cpu()
    return normalizer.inverse_rho(lower), normalizer.inverse_rho(upper)


def _student_t_quantile(prob: float, df: torch.Tensor) -> torch.Tensor:
    try:
        from scipy.stats import t as student_t

        values = student_t.ppf(prob, df.numpy())
        return torch.as_tensor(values, dtype=torch.float32)
    except Exception:
        # Normal 90% two-sided quantile. Used only if scipy is unavailable.
        if abs(prob - 0.95) < 1e-6:
            value = 1.6448536269514722
        else:
            value = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * prob - 1.0)).item()
        return torch.full_like(df, float(value), dtype=torch.float32)
