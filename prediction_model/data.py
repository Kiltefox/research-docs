"""Dataset, preprocessing, and batching utilities for rho prediction."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

STATS_DIM = 37
REWARD_STAT_SAMPLE_LIMIT = 1024
REWARD_VALUES_PER_SAMPLE_LIMIT = 2048


@dataclass
class NormalizerState:
    tau_mean: list[float]
    tau_std: list[float]
    reward_median: float
    reward_iqr: float
    rho_mean: float
    rho_std: float


class Normalizer:
    """Feature normalizer fitted on the training split only."""

    def __init__(self, state: NormalizerState) -> None:
        self.state = state

    @classmethod
    def fit(cls, dataset: "RhoTensorDataset", indices: list[int]) -> "Normalizer":
        rng = random.Random(0)
        tau_sum: torch.Tensor | None = None
        tau_sq_sum: torch.Tensor | None = None
        tau_count: torch.Tensor | None = None
        rewards: list[torch.Tensor] = []
        rhos: list[float] = []

        reward_indices = indices
        if len(indices) > REWARD_STAT_SAMPLE_LIMIT:
            reward_indices = rng.sample(indices, REWARD_STAT_SAMPLE_LIMIT)
        reward_index_set = set(reward_indices)

        for idx in indices:
            sample = dataset[idx]
            tau = sample["tau"].float()
            reward = sample["rewards"].float()
            mask = finite_mask(tau, reward)
            tau_mask = torch.isfinite(tau) & mask.unsqueeze(-1)
            clean_tau = torch.where(tau_mask, tau, torch.zeros_like(tau))
            if tau_sum is None:
                tau_sum = clean_tau.sum(dim=(0, 1))
                tau_sq_sum = (clean_tau * clean_tau).sum(dim=(0, 1))
                tau_count = tau_mask.sum(dim=(0, 1)).clamp_min(1)
            else:
                tau_sum += clean_tau.sum(dim=(0, 1))
                tau_sq_sum += (clean_tau * clean_tau).sum(dim=(0, 1))
                tau_count += tau_mask.sum(dim=(0, 1))

            valid_rewards = reward[mask & torch.isfinite(reward)]
            if valid_rewards.numel() > 0:
                if idx in reward_index_set:
                    if valid_rewards.numel() > REWARD_VALUES_PER_SAMPLE_LIMIT:
                        sample_ids = torch.randint(
                            valid_rewards.numel(),
                            (REWARD_VALUES_PER_SAMPLE_LIMIT,),
                            device=valid_rewards.device,
                        )
                        valid_rewards = valid_rewards[sample_ids]
                    rewards.append(valid_rewards.detach().cpu())
            rhos.append(float(sample["rho"]))

        if tau_sum is None or tau_sq_sum is None or tau_count is None:
            raise ValueError("Cannot fit normalizer on an empty split.")

        tau_count = tau_count.clamp_min(1)
        tau_mean = tau_sum / tau_count
        tau_var = (tau_sq_sum / tau_count - tau_mean.square()).clamp_min(1e-12)
        tau_std = tau_var.sqrt().clamp_min(1e-6)

        reward_all = torch.cat(rewards) if rewards else torch.zeros(1)
        reward_q = torch.quantile(reward_all, torch.tensor([0.25, 0.5, 0.75]))
        reward_iqr = float((reward_q[2] - reward_q[0]).abs().clamp_min(1e-6))

        rho_tensor = torch.tensor(rhos, dtype=torch.float32)
        rho_std = float(rho_tensor.std(unbiased=False).clamp_min(1e-6))
        state = NormalizerState(
            tau_mean=tau_mean.tolist(),
            tau_std=tau_std.tolist(),
            reward_median=float(reward_q[1]),
            reward_iqr=reward_iqr,
            rho_mean=float(rho_tensor.mean()),
            rho_std=rho_std,
        )
        return cls(state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Normalizer":
        return cls(NormalizerState(**data))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.state)

    def transform_tau(self, tau: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.state.tau_mean, dtype=tau.dtype, device=tau.device)
        std = torch.tensor(self.state.tau_std, dtype=tau.dtype, device=tau.device)
        return (tau - mean) / std

    def transform_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        return (rewards - self.state.reward_median) / self.state.reward_iqr

    def transform_rho(self, rho: torch.Tensor) -> torch.Tensor:
        return (rho - self.state.rho_mean) / self.state.rho_std

    def inverse_rho(self, rho: torch.Tensor) -> torch.Tensor:
        return rho * self.state.rho_std + self.state.rho_mean


class RhoTensorDataset(Dataset):
    """Dataset backed by a compact torch file produced by prepare_dataset.py."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a dict payload in {self.path}")
        for key in ("tau", "rewards", "rho"):
            if key not in payload:
                raise KeyError(f"Dataset cache is missing key: {key}")
        self.tau = payload["tau"]
        self.rewards = payload["rewards"]
        self.rho = torch.as_tensor(payload["rho"], dtype=torch.float32)
        self.groups = payload.get("groups")
        self.metadata = payload.get("metadata", {})

    def __len__(self) -> int:
        return int(len(self.rho))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tau = self.tau[idx] if isinstance(self.tau, list) else self.tau[idx]
        rewards = self.rewards[idx] if isinstance(self.rewards, list) else self.rewards[idx]
        return {
            "tau": torch.as_tensor(tau, dtype=torch.float32),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32),
            "rho": torch.as_tensor(self.rho[idx], dtype=torch.float32),
            "group": None if self.groups is None else self.groups[idx],
            "index": idx,
        }

    @property
    def sample_shape(self) -> tuple[int, int, int]:
        first = self[0]["tau"]
        if first.dim() != 3:
            raise ValueError(f"tau must have shape [N_vehicle, N_road, D_tau], got {tuple(first.shape)}")
        return tuple(int(v) for v in first.shape)


class SubsetDataset(Dataset):
    def __init__(self, dataset: RhoTensorDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.dataset[self.indices[idx]]


def finite_mask(tau: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(rewards) & torch.isfinite(tau).all(dim=-1)


def make_collate_fn(normalizer: Normalizer, augment: bool = False, tau_noise_std: float = 0.0):
    def collate(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_vehicle = max(int(s["tau"].shape[0]) for s in samples)
        max_road = max(int(s["tau"].shape[1]) for s in samples)
        tau_dim = int(samples[0]["tau"].shape[-1])
        batch = len(samples)

        tau = torch.zeros(batch, max_vehicle, max_road, tau_dim, dtype=torch.float32)
        rewards = torch.zeros(batch, max_vehicle, max_road, dtype=torch.float32)
        mask = torch.zeros(batch, max_vehicle, max_road, dtype=torch.bool)
        rho = torch.zeros(batch, dtype=torch.float32)
        indices = torch.zeros(batch, dtype=torch.long)

        for row, sample in enumerate(samples):
            raw_tau = sample["tau"].float()
            raw_rewards = sample["rewards"].float()
            n_vehicle, n_road, _ = raw_tau.shape
            valid = finite_mask(raw_tau, raw_rewards)
            clean_tau = torch.nan_to_num(raw_tau, nan=0.0, posinf=0.0, neginf=0.0)
            clean_rewards = torch.nan_to_num(raw_rewards, nan=0.0, posinf=0.0, neginf=0.0)

            if augment and n_vehicle > 1:
                perm = torch.randperm(n_vehicle)
                clean_tau = clean_tau[perm]
                clean_rewards = clean_rewards[perm]
                valid = valid[perm]
            if augment and tau_noise_std > 0:
                clean_tau = clean_tau + torch.randn_like(clean_tau) * tau_noise_std

            tau[row, :n_vehicle, :n_road] = normalizer.transform_tau(clean_tau)
            rewards[row, :n_vehicle, :n_road] = normalizer.transform_rewards(clean_rewards)
            mask[row, :n_vehicle, :n_road] = valid
            rho[row] = sample["rho"]
            indices[row] = int(sample["index"])

        stats = compute_global_stats(tau, rewards, mask)
        return {
            "tau": tau,
            "rewards": rewards,
            "mask": mask,
            "stats": stats,
            "rho": normalizer.transform_rho(rho),
            "rho_raw": rho,
            "indices": indices,
        }

    return collate


def compute_global_stats(tau: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rows = []
    for idx in range(tau.shape[0]):
        rows.append(_sample_stats(tau[idx], rewards[idx], mask[idx]))
    return torch.stack(rows, dim=0)


def _sample_stats(tau: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dist = tau[..., 0]
    queue = tau[..., 1] if tau.shape[-1] > 1 else torch.zeros_like(dist)
    reward_values = rewards[mask]
    dist_values = dist[mask]
    queue_values = queue[mask]

    per_vehicle_valid = mask.any(dim=1)
    masked_rewards = rewards.masked_fill(~mask, -float("inf"))
    best = masked_rewards.max(dim=1).values[per_vehicle_valid]
    top2 = torch.topk(masked_rewards, k=min(2, masked_rewards.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        gap = torch.zeros_like(top2[:, 0])
    else:
        gap = top2[:, 0] - top2[:, 1]
    gap = gap[per_vehicle_valid & torch.isfinite(gap)]

    stats = []
    stats.extend(_basic_stats(reward_values))
    stats.extend(_basic_stats(dist_values))
    stats.extend(_basic_stats(queue_values))
    stats.extend(
        [
            _safe_mean(best),
            _safe_std(best),
            _safe_mean(gap),
            _safe_std(gap),
            float(mask.float().mean().item()),
            float((~mask).float().mean().item()),
            float(per_vehicle_valid.float().mean().item()),
        ]
    )
    return torch.tensor(stats, dtype=torch.float32)


def _basic_stats(values: torch.Tensor) -> list[float]:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return [0.0] * 10
    quantiles = torch.quantile(values.float(), torch.tensor([0.25, 0.5, 0.75], device=values.device))
    return [
        float(values.mean().item()),
        float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        float(values.min().item()),
        float(values.max().item()),
        float(quantiles[0].item()),
        float(quantiles[1].item()),
        float(quantiles[2].item()),
        float((values > 0).float().mean().item()),
        float((values == 0).float().mean().item()),
        float(torch.log1p(torch.tensor(float(values.numel()))).item()),
    ]


def _safe_mean(values: torch.Tensor) -> float:
    values = values[torch.isfinite(values)]
    return float(values.mean().item()) if values.numel() else 0.0


def _safe_std(values: torch.Tensor) -> float:
    values = values[torch.isfinite(values)]
    return float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0


def split_indices(
    dataset: RhoTensorDataset,
    seed: int,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    group_split: bool = False,
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    n = len(dataset)
    if n < 3:
        raise ValueError("At least 3 samples are required for train/valid/test splits.")

    if group_split and dataset.groups is not None:
        group_to_indices: dict[Any, list[int]] = {}
        for idx, group in enumerate(dataset.groups):
            group_to_indices.setdefault(group, []).append(idx)
        groups = list(group_to_indices.keys())
        rng.shuffle(groups)
        train_groups, valid_groups, test_groups = _ratio_cut(groups, train_ratio, valid_ratio)
        return {
            "train": [i for g in train_groups for i in group_to_indices[g]],
            "valid": [i for g in valid_groups for i in group_to_indices[g]],
            "test": [i for g in test_groups for i in group_to_indices[g]],
        }

    indices = list(range(n))
    rng.shuffle(indices)
    train, valid, test = _ratio_cut(indices, train_ratio, valid_ratio)
    return {"train": train, "valid": valid, "test": test}


def _ratio_cut(items: list[Any], train_ratio: float, valid_ratio: float) -> tuple[list[Any], list[Any], list[Any]]:
    n = len(items)
    train_end = max(1, min(n - 2, int(math.floor(n * train_ratio))))
    valid_end = max(train_end + 1, min(n - 1, train_end + int(math.floor(n * valid_ratio))))
    return items[:train_end], items[train_end:valid_end], items[valid_end:]


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def describe_tensor_list(values: Iterable[torch.Tensor]) -> list[tuple[int, ...]]:
    return [tuple(int(v) for v in item.shape) for item in values]
