"""Road-vehicle rho prediction model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    tau_dim: int = 2
    num_roads: int = 114
    stats_dim: int = 37
    hidden_dim: int = 128
    road_encoder_layers: int = 3
    attention_heads: int = 4
    dropout: float = 0.1
    aggregator: str = "deepsets"
    vehicle_chunk_size: int = 64
    checkpoint_road_encoder: bool = False

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**data)


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        fill_value = -torch.finfo(scores.dtype).max
        scores = scores.masked_fill(~mask, fill_value)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class RoadEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        local_dim = config.tau_dim + 2
        self.input_proj = nn.Linear(local_dim, config.hidden_dim)
        self.road_embedding = nn.Embedding(config.num_roads, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.road_encoder_layers,
            enable_nested_tensor=False,
        )
        self.pool = MaskedAttentionPool(config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.vehicle_chunk_size = int(config.vehicle_chunk_size)
        self.checkpoint_road_encoder = bool(config.checkpoint_road_encoder)

    def forward(self, tau: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, n_vehicle, n_road, _ = tau.shape
        if n_road > self.road_embedding.num_embeddings:
            raise ValueError(
                f"Input has {n_road} roads, but model was configured for "
                f"{self.road_embedding.num_embeddings} roads."
            )

        road_ids = torch.arange(n_road, device=tau.device)
        road_embedding = self.road_embedding(road_ids).view(1, 1, n_road, -1)
        chunk_size = self.vehicle_chunk_size if self.vehicle_chunk_size > 0 else n_vehicle
        pooled_chunks: list[torch.Tensor] = []

        for start in range(0, n_vehicle, chunk_size):
            end = min(start + chunk_size, n_vehicle)
            tau_chunk = tau[:, start:end]
            rewards_chunk = rewards[:, start:end]
            mask_chunk = mask[:, start:end]

            local = torch.cat(
                [tau_chunk, rewards_chunk.unsqueeze(-1), mask_chunk.float().unsqueeze(-1)],
                dim=-1,
            )
            x = self.input_proj(local)
            x = x + road_embedding
            x = x.reshape(batch * (end - start), n_road, -1)
            flat_mask = mask_chunk.reshape(batch * (end - start), n_road)

            all_invalid = ~flat_mask.any(dim=1)
            key_padding_mask = ~flat_mask
            if all_invalid.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_invalid, 0] = False

            if self.checkpoint_road_encoder and self.training:
                pooled = checkpoint(self._encode_and_pool, x, flat_mask, use_reentrant=False)
            else:
                pooled = self._encode_and_pool(x, flat_mask)
            pooled_chunks.append(self.norm(pooled).view(batch, end - start, -1))

        return torch.cat(pooled_chunks, dim=1)

    def _encode_and_pool(self, x: torch.Tensor, flat_mask: torch.Tensor) -> torch.Tensor:
        all_invalid = ~flat_mask.any(dim=1)
        key_padding_mask = ~flat_mask
        if all_invalid.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid, 0] = False
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.pool(encoded, flat_mask)


class DeepSetsAggregator(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, vehicles: torch.Tensor, vehicle_mask: torch.Tensor) -> torch.Tensor:
        x = self.phi(vehicles)
        mask = vehicle_mask.unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1)
        mean = (x * mask).sum(dim=1) / count
        x_max = x.masked_fill(~mask, -torch.finfo(x.dtype).max)
        max_values = x_max.max(dim=1).values
        max_values = torch.where(torch.isfinite(max_values), max_values, torch.zeros_like(max_values))
        var = ((x - mean.unsqueeze(1)).square() * mask).sum(dim=1) / count
        std = var.clamp_min(0).sqrt()
        return self.out(torch.cat([mean, max_values, std], dim=-1))


class MeanMaxAggregator(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, vehicles: torch.Tensor, vehicle_mask: torch.Tensor) -> torch.Tensor:
        mask = vehicle_mask.unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1)
        mean = (vehicles * mask).sum(dim=1) / count
        x_max = vehicles.masked_fill(~mask, -torch.finfo(vehicles.dtype).max)
        max_values = x_max.max(dim=1).values
        max_values = torch.where(torch.isfinite(max_values), max_values, torch.zeros_like(max_values))
        var = ((vehicles - mean.unsqueeze(1)).square() * mask).sum(dim=1) / count
        std = var.clamp_min(0).sqrt()
        return self.out(torch.cat([mean, max_values, std], dim=-1))


class AttentionAggregator(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = MaskedAttentionPool(hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, vehicles: torch.Tensor, vehicle_mask: torch.Tensor) -> torch.Tensor:
        return self.out(self.pool(self.phi(vehicles), vehicle_mask))


def make_aggregator(name: str, hidden_dim: int, dropout: float) -> nn.Module:
    if name == "deepsets":
        return DeepSetsAggregator(hidden_dim, dropout)
    if name == "meanmax":
        return MeanMaxAggregator(hidden_dim)
    if name == "attention":
        return AttentionAggregator(hidden_dim, dropout)
    raise ValueError(f"Unknown aggregator: {name}")


class RhoPredictionModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.road_encoder = RoadEncoder(config)
        self.aggregator = make_aggregator(config.aggregator, config.hidden_dim, config.dropout)
        self.stats_branch = nn.Sequential(
            nn.Linear(config.stats_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.mu = nn.Linear(128, 1)
        self.log_sigma = nn.Linear(128, 1)
        self.log_nu = nn.Linear(128, 1)

    def forward(
        self,
        tau: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        vehicle_mask = mask.any(dim=-1)
        vehicle_repr = self.road_encoder(tau, rewards, mask)
        global_repr = self.aggregator(vehicle_repr, vehicle_mask)
        stats_repr = self.stats_branch(stats)
        fused = self.fusion(torch.cat([global_repr, stats_repr], dim=-1))
        with torch.autocast(device_type=fused.device.type, enabled=False):
            fused_fp32 = torch.nan_to_num(fused.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            mu = self.mu(fused_fp32).squeeze(-1)
            log_sigma = self.log_sigma(fused_fp32).squeeze(-1)
            log_nu = self.log_nu(fused_fp32).squeeze(-1)

            mu = torch.nan_to_num(mu, nan=0.0, posinf=1e4, neginf=-1e4)
            log_sigma = torch.nan_to_num(log_sigma, nan=0.0, posinf=8.0, neginf=-12.0).clamp(-12.0, 8.0)
            log_nu = torch.nan_to_num(log_nu, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)

            sigma = F.softplus(log_sigma) + 1e-5
            nu = F.softplus(log_nu) + 2.0
        return {"mu": mu, "sigma": sigma, "nu": nu}
