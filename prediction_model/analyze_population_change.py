"""Analyze high-dimensional MASDiff population changes.

The input pickles are MASDiff ``Population`` files.  Each individual contains
high-dimensional car-road tensors:

* tau: [num_car, num_road, 2]
* rewards: [num_car, num_road]
* rho: scalar fitness measured by SUMO simulation

This script writes compact CSV summaries, reduced-dimensional embeddings, and
charts that explain how ``initial_population.pkl`` differs from
``final_population.pkl``.  The analysis deliberately treats final population as
an evolved selected population, not as row-wise descendants of the initial
population.  Row-wise deltas are still emitted for diagnostics, but the report
labels them as distribution alignment rather than lineage causality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

try:
    from .types_compat import install_pickle_compat
except ImportError:  # pragma: no cover
    from types_compat import install_pickle_compat


GROUPS = ("initial", "final")
SUMMARY_METRICS = [
    "rho",
    "mse_est_inv1p",
    "valid_ratio",
    "reward_mean",
    "reward_std",
    "reward_median",
    "best_reward_mean",
    "top2_gap_mean",
    "tau0_mean",
    "tau0_std",
    "tau1_mean",
    "tau1_std",
    "highdim_l2_norm",
]
POSITION_KEYS = ("tau0_distance", "tau1_queue", "reward", "valid_ratio", "rho_broadcast")


@dataclass(frozen=True)
class ProjectionSpec:
    n_features: int
    n_components: int
    buckets: np.ndarray
    signs: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze initial/final MASDiff populations with summary statistics and high-dimensional PCA."
    )
    parser.add_argument("--initial", default="dataset/initial_population.pkl")
    parser.add_argument("--final", default="dataset/final_population.pkl")
    parser.add_argument("--output-dir", default="dataset/population_change_analysis")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug limit; 0 means all individuals.")
    parser.add_argument("--force", action="store_true", help="Regenerate outputs even when CSV files exist.")
    parser.add_argument("--projection-dim", type=int, default=128, help="Feature hashing dimension before PCA.")
    parser.add_argument("--projection-seed", type=int, default=20260702)
    parser.add_argument("--tau0-clip", type=float, default=3000.0)
    parser.add_argument("--tau1-clip", type=float, default=40.0)
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--selection-temperatures", nargs="*", type=float, default=[1.0, 1.5, 5.0, 20.0, 100.0])
    parser.add_argument("--heatmap-value-percentiles", nargs=2, type=float, default=[2.0, 98.0])
    parser.add_argument("--heatmap-delta-percentile", type=float, default=95.0)
    parser.add_argument(
        "--docs-image-dir",
        default="",
        help="Optional Sphinx image directory. When set, generated PNG charts are copied there.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    initial_path = Path(args.initial)
    final_path = Path(args.final)

    print("Loading populations ...")
    initial_population = load_population(initial_path, args.max_samples)
    final_population = load_population(final_path, args.max_samples)

    print("Summarizing individuals ...")
    initial_rows = summarize_population(initial_population, "initial")
    final_rows = summarize_population(final_population, "final")
    write_dict_csv(output_dir / "initial_population_summary.csv", initial_rows)
    write_dict_csv(output_dir / "final_population_summary.csv", final_rows)

    print("Building high-dimensional random-projection features ...")
    projected_rows, projected_features = build_projected_feature_table(
        initial_population,
        final_population,
        projection_dim=args.projection_dim,
        seed=args.projection_seed,
        tau0_clip=args.tau0_clip,
        tau1_clip=args.tau1_clip,
        reward_scale=args.reward_scale,
    )
    coords, variance_ratio = pca_from_projected(projected_features, n_components=2)
    for row, xy in zip(projected_rows, coords, strict=True):
        row["pca1"] = float(xy[0])
        row["pca2"] = float(xy[1])
    write_dict_csv(output_dir / "highdim_embedding.csv", projected_rows)

    print("Computing distribution and selection diagnostics ...")
    distribution_rows = distribution_change_table(initial_rows, final_rows)
    write_dict_csv(output_dir / "distribution_change_summary.csv", distribution_rows)
    paired_rows = paired_row_deltas(initial_rows, final_rows)
    write_dict_csv(output_dir / "row_aligned_deltas_diagnostic.csv", paired_rows)
    pressure_rows = selection_pressure_table(
        [float(row["rho"]) for row in initial_rows],
        temperatures=args.selection_temperatures,
    )
    write_dict_csv(output_dir / "selection_pressure.csv", pressure_rows)
    lineage = lineage_summary(final_rows)

    print("Computing car-road position summaries ...")
    initial_position = compute_position_means(initial_population)
    final_position = compute_position_means(final_population)
    position_rows = position_change_table(initial_position, final_position)
    write_dict_csv(output_dir / "position_change_summary.csv", position_rows)

    print("Writing charts ...")
    chart_paths = make_charts(
        initial_rows,
        final_rows,
        projected_rows,
        distribution_rows,
        pressure_rows,
        initial_position,
        final_position,
        charts_dir,
        value_percentiles=tuple(args.heatmap_value_percentiles),
        delta_percentile=float(args.heatmap_delta_percentile),
    )

    report = build_report(
        initial_path=initial_path,
        final_path=final_path,
        output_dir=output_dir,
        initial_rows=initial_rows,
        final_rows=final_rows,
        distribution_rows=distribution_rows,
        position_rows=position_rows,
        pressure_rows=pressure_rows,
        lineage=lineage,
        variance_ratio=variance_ratio,
        runtime_seconds=time.time() - start,
        chart_paths=chart_paths,
    )
    (output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(output_dir / "analysis_report.md", report)

    if args.docs_image_dir:
        copy_charts_to_docs(chart_paths, Path(args.docs_image_dir))

    print(json.dumps(report["headline"], indent=2, ensure_ascii=False))
    print(f"saved={output_dir}")


def load_population(path: Path, max_samples: int) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    install_pickle_compat()
    start = time.time()
    with path.open("rb") as f:
        population = pickle.load(f)
    if not isinstance(population, list):
        raise TypeError(f"{path} must contain a list, got {type(population).__name__}")
    if max_samples > 0:
        population = population[: max_samples]
    print(f"Loaded {len(population)} individuals from {path} in {time.time() - start:.1f}s")
    return population


def summarize_population(population: list[Any], group: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, individual in enumerate(population):
        rows.append(summarize_individual(individual, group, idx))
    return rows


def summarize_individual(individual: Any, group: str, idx: int) -> dict[str, Any]:
    tau, rewards, valid = tensors_and_valid_mask(individual, idx)
    reward_values = rewards[valid & torch.isfinite(rewards)]
    best_rewards, top2_gaps = best_reward_stats(rewards, valid)
    tau0 = tau[..., 0][valid & torch.isfinite(tau[..., 0])]
    tau1 = tau[..., 1][valid & torch.isfinite(tau[..., 1])] if tau.shape[-1] > 1 else torch.empty(0)
    metadata = getattr(individual, "metadata", {}) or {}
    rho = float(getattr(individual, "rho", 0.0))

    row: dict[str, Any] = {
        "group": group,
        "index": idx,
        "rho": rho,
        "mse_est_inv1p": mse_from_inv1p_rho(rho),
        "n_vehicle": int(tau.shape[0]),
        "n_road": int(tau.shape[1]),
        "tau_dim": int(tau.shape[2]),
        "valid_count": int(valid.sum().item()),
        "invalid_count": int(valid.numel() - valid.sum().item()),
        "valid_ratio": float(valid.float().mean().item()) if valid.numel() else 0.0,
        "tau_nan_count": int(torch.isnan(tau).sum().item()),
        "tau_inf_count": int(torch.isinf(tau).sum().item()),
        "reward_nan_count": int(torch.isnan(rewards).sum().item()),
        "reward_inf_count": int(torch.isinf(rewards).sum().item()),
        "reward_positive_ratio": ratio(reward_values > 0),
        "reward_zero_ratio": ratio(reward_values == 0),
        "highdim_l2_norm": highdim_norm(tau, rewards, valid),
        "metadata_iteration_k": metadata_number(metadata, "iteration_k"),
        "metadata_parent_rho": metadata_number(metadata, "parent_rho"),
        "metadata_delta_parent_rho": rho - metadata_number(metadata, "parent_rho")
        if metadata.get("parent_rho") is not None
        else "",
        "metadata_source_warmstart": bool(metadata.get("source_warmstart", False)),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=json_default),
    }
    row.update(prefixed_stats("reward", reward_values))
    row.update(prefixed_stats("best_reward", best_rewards, quantiles=False))
    row.update(prefixed_stats("top2_gap", top2_gaps, quantiles=False))
    row.update(prefixed_stats("tau0", tau0))
    row.update(prefixed_stats("tau1", tau1))
    return row


def tensors_and_valid_mask(individual: Any, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tau = torch.as_tensor(getattr(individual, "tau"), dtype=torch.float32).cpu()
    rewards = torch.as_tensor(getattr(individual, "rewards"), dtype=torch.float32).cpu()
    if tau.dim() != 3:
        raise ValueError(f"Individual {idx} tau must be [car,road,dim], got {tuple(tau.shape)}")
    if rewards.shape != tau.shape[:2]:
        raise ValueError(f"Individual {idx} rewards shape {tuple(rewards.shape)} does not match tau {tuple(tau.shape)}")
    valid = torch.isfinite(rewards) & torch.isfinite(tau).all(dim=-1)
    return tau, rewards, valid


def mse_from_inv1p_rho(rho: float) -> float:
    if rho <= 0:
        return float("inf")
    return float(1.0 / rho - 1.0)


def metadata_number(metadata: dict[str, Any], key: str) -> float | str:
    if key not in metadata or metadata[key] is None:
        return ""
    try:
        return float(metadata[key])
    except Exception:
        return ""


def ratio(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def prefixed_stats(prefix: str, values: torch.Tensor, quantiles: bool = True) -> dict[str, float]:
    values = values[torch.isfinite(values)].float()
    if values.numel() == 0:
        out = {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
        if quantiles:
            out.update({f"{prefix}_q25": 0.0, f"{prefix}_median": 0.0, f"{prefix}_q75": 0.0})
        return out
    out = {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }
    if quantiles:
        qs = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75]))
        out.update(
            {
                f"{prefix}_q25": float(qs[0].item()),
                f"{prefix}_median": float(qs[1].item()),
                f"{prefix}_q75": float(qs[2].item()),
            }
        )
    return out


def best_reward_stats(rewards: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    has_valid = valid.any(dim=1)
    if not bool(has_valid.any()):
        return torch.empty(0), torch.empty(0)
    masked = rewards.masked_fill(~valid, -float("inf"))
    best = masked.max(dim=1).values[has_valid]
    k = min(2, masked.shape[1])
    top = torch.topk(masked, k=k, dim=1).values
    if k == 1:
        gap = torch.zeros_like(top[:, 0])
    else:
        gap = top[:, 0] - top[:, 1]
    gap = gap[has_valid & torch.isfinite(gap)]
    return best[torch.isfinite(best)], gap


def highdim_norm(tau: torch.Tensor, rewards: torch.Tensor, valid: torch.Tensor) -> float:
    tau0 = torch.nan_to_num(tau[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
    tau1 = torch.nan_to_num(tau[..., 1], nan=0.0, posinf=0.0, neginf=0.0) if tau.shape[-1] > 1 else torch.zeros_like(tau0)
    rew = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
    stacked = torch.stack([tau0, tau1, rew, valid.float()], dim=0)
    return float(torch.linalg.vector_norm(stacked).item())


def build_projected_feature_table(
    initial_population: list[Any],
    final_population: list[Any],
    *,
    projection_dim: int,
    seed: int,
    tau0_clip: float,
    tau1_clip: float,
    reward_scale: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    all_population = [("initial", idx, ind) for idx, ind in enumerate(initial_population)]
    all_population.extend(("final", idx, ind) for idx, ind in enumerate(final_population))
    if not all_population:
        raise ValueError("No individuals to embed.")

    first_tau, first_rewards, first_valid = tensors_and_valid_mask(all_population[0][2], 0)
    first_vector = normalized_highdim_vector(first_tau, first_rewards, first_valid, tau0_clip, tau1_clip, reward_scale)
    spec = make_projection_spec(first_vector.size, projection_dim, seed)

    rows: list[dict[str, Any]] = []
    features = np.zeros((len(all_population), projection_dim), dtype=np.float64)
    for row_idx, (group, idx, individual) in enumerate(all_population):
        tau, rewards, valid = tensors_and_valid_mask(individual, idx)
        vector = normalized_highdim_vector(tau, rewards, valid, tau0_clip, tau1_clip, reward_scale)
        if vector.size != spec.n_features:
            raise ValueError(f"Individual {group}/{idx} high-dimensional length changed from {spec.n_features} to {vector.size}")
        projected = feature_hash(vector, spec)
        features[row_idx] = projected
        rows.append(
            {
                "group": group,
                "index": idx,
                "rho": float(getattr(individual, "rho", 0.0)),
                "projected_norm": float(np.linalg.norm(projected)),
            }
        )
    return rows, zscore(features)


def normalized_highdim_vector(
    tau: torch.Tensor,
    rewards: torch.Tensor,
    valid: torch.Tensor,
    tau0_clip: float,
    tau1_clip: float,
    reward_scale: float,
) -> np.ndarray:
    tau_np = tau.detach().cpu().numpy().astype(np.float32, copy=False)
    rewards_np = rewards.detach().cpu().numpy().astype(np.float32, copy=False)
    valid_np = valid.detach().cpu().numpy().astype(np.float32, copy=False)

    tau0 = np.nan_to_num(tau_np[..., 0], nan=0.0, posinf=tau0_clip, neginf=0.0)
    tau0 = np.clip(tau0, 0.0, tau0_clip) / max(tau0_clip, 1e-6)
    if tau_np.shape[-1] > 1:
        tau1 = np.nan_to_num(tau_np[..., 1], nan=0.0, posinf=tau1_clip, neginf=0.0)
    else:
        tau1 = np.zeros_like(tau0)
    tau1 = np.clip(tau1, 0.0, tau1_clip) / max(tau1_clip, 1e-6)
    reward = np.nan_to_num(rewards_np, nan=0.0, posinf=reward_scale, neginf=0.0)
    reward = np.clip(reward, 0.0, reward_scale) / max(reward_scale, 1e-6)
    return np.concatenate(
        [
            tau0.reshape(-1),
            tau1.reshape(-1),
            reward.reshape(-1),
            valid_np.reshape(-1),
        ]
    ).astype(np.float32, copy=False)


def make_projection_spec(n_features: int, n_components: int, seed: int) -> ProjectionSpec:
    n_components = max(2, int(n_components))
    rng = np.random.default_rng(int(seed))
    buckets = rng.integers(0, n_components, size=n_features, dtype=np.int32)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n_features)
    return ProjectionSpec(n_features=n_features, n_components=n_components, buckets=buckets, signs=signs)


def feature_hash(vector: np.ndarray, spec: ProjectionSpec) -> np.ndarray:
    projected = np.bincount(spec.buckets, weights=vector * spec.signs, minlength=spec.n_components)
    return projected.astype(np.float64) / math.sqrt(max(spec.n_features / spec.n_components, 1.0))


def zscore(features: np.ndarray) -> np.ndarray:
    mean = np.nanmean(features, axis=0, keepdims=True)
    std = np.nanstd(features, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (features - mean) / std


def pca_from_projected(features: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, list[float]]:
    x = np.asarray(features, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:n_components].T
    denom = float(np.sum(singular_values**2))
    if denom <= 0.0:
        ratio = [0.0 for _ in range(n_components)]
    else:
        ratio = [float(v) for v in (singular_values[:n_components] ** 2 / denom)]
    return coords, ratio


def distribution_change_table(initial_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for metric in SUMMARY_METRICS:
        initial = vector(initial_rows, metric)
        final = vector(final_rows, metric)
        rows.append(distribution_stats(metric, initial, final))
    return rows


def distribution_stats(metric: str, initial: np.ndarray, final: np.ndarray) -> dict[str, Any]:
    delta_mean = safe_mean(final) - safe_mean(initial)
    pooled = np.concatenate([initial, final])
    pooled_std = safe_std(pooled)
    return {
        "metric": metric,
        "initial_mean": safe_mean(initial),
        "initial_std": safe_std(initial),
        "initial_median": safe_median(initial),
        "final_mean": safe_mean(final),
        "final_std": safe_std(final),
        "final_median": safe_median(final),
        "mean_delta": delta_mean,
        "median_delta": safe_median(final) - safe_median(initial),
        "standardized_mean_delta": delta_mean / pooled_std if pooled_std > 0 else 0.0,
        "ks_stat": ks_2sample_stat(initial, final),
        "overlap_coefficient": histogram_overlap(initial, final),
    }


def ks_2sample_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if a.size == 0 or b.size == 0:
        return 0.0
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def histogram_overlap(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    values = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if values.size == 0 or np.nanmax(values) == np.nanmin(values):
        return 1.0
    hist_a, edges = np.histogram(a, bins=bins, range=(float(values.min()), float(values.max())), density=True)
    hist_b, _ = np.histogram(b, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.minimum(hist_a, hist_b) * widths))


def paired_row_deltas(initial_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_by_index = {int(row["index"]): row for row in final_rows}
    rows: list[dict[str, Any]] = []
    for initial in initial_rows:
        idx = int(initial["index"])
        final = final_by_index.get(idx)
        if final is None:
            continue
        row: dict[str, Any] = {"index": idx}
        for metric in SUMMARY_METRICS:
            row[f"initial_{metric}"] = float(initial.get(metric, 0.0))
            row[f"final_{metric}"] = float(final.get(metric, 0.0))
            row[f"delta_{metric}"] = row[f"final_{metric}"] - row[f"initial_{metric}"]
        rows.append(row)
    return rows


def selection_pressure_table(rhos: list[float], temperatures: Iterable[float]) -> list[dict[str, Any]]:
    scores = np.asarray(rhos, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    if scores.size == 0:
        return rows
    for temperature in temperatures:
        logits = float(temperature) * scores
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-300))))
        effective_n = float(np.exp(entropy))
        uniform = 1.0 / len(scores)
        rows.append(
            {
                "temperature": float(temperature),
                "rho_min": float(np.min(scores)),
                "rho_max": float(np.max(scores)),
                "rho_range": float(np.max(scores) - np.min(scores)),
                "max_probability": float(np.max(probs)),
                "min_probability": float(np.min(probs)),
                "uniform_probability": uniform,
                "effective_sample_size": effective_n,
                "effective_sample_ratio": effective_n / len(scores),
                "top10_probability_mass": float(np.sum(np.sort(probs)[-min(10, len(probs)) :])),
            }
        )
    return rows


def lineage_summary(final_rows: list[dict[str, Any]]) -> dict[str, Any]:
    parent_delta = []
    iteration_values = []
    for row in final_rows:
        value = row.get("metadata_delta_parent_rho", "")
        if value != "":
            parent_delta.append(float(value))
        iteration = row.get("metadata_iteration_k", "")
        if iteration != "":
            iteration_values.append(float(iteration))
    if not parent_delta:
        return {
            "available": False,
            "message": "No parent_rho metadata found in final population; row-aligned deltas are not lineage deltas.",
        }
    deltas = np.asarray(parent_delta, dtype=np.float64)
    return {
        "available": True,
        "count": int(deltas.size),
        "delta_parent_rho_mean": safe_mean(deltas),
        "delta_parent_rho_median": safe_median(deltas),
        "delta_parent_rho_positive_ratio": float(np.mean(deltas > 0)),
        "iteration_min": float(np.min(iteration_values)) if iteration_values else "",
        "iteration_max": float(np.max(iteration_values)) if iteration_values else "",
    }


def compute_position_means(population: list[Any]) -> dict[str, np.ndarray]:
    if not population:
        raise ValueError("population is empty")
    tau0, rewards0, _ = tensors_and_valid_mask(population[0], 0)
    n_vehicle, n_road = int(tau0.shape[0]), int(tau0.shape[1])
    sums = {key: np.zeros((n_vehicle, n_road), dtype=np.float64) for key in POSITION_KEYS}
    counts = {key: np.zeros((n_vehicle, n_road), dtype=np.float64) for key in POSITION_KEYS}

    for idx, individual in enumerate(population):
        tau, rewards, valid = tensors_and_valid_mask(individual, idx)
        if tau.shape[:2] != (n_vehicle, n_road):
            raise ValueError(f"Individual {idx} has shape {tuple(tau.shape[:2])}, expected {(n_vehicle, n_road)}")
        valid_np = valid.numpy()
        add_position(sums["tau0_distance"], counts["tau0_distance"], tau[..., 0], valid)
        if tau.shape[-1] > 1:
            add_position(sums["tau1_queue"], counts["tau1_queue"], tau[..., 1], valid)
        add_position(sums["reward"], counts["reward"], rewards, valid)
        sums["valid_ratio"] += valid_np.astype(np.float64)
        counts["valid_ratio"] += 1.0
        sums["rho_broadcast"][valid_np] += float(getattr(individual, "rho", 0.0))
        counts["rho_broadcast"][valid_np] += 1.0
    return {key: divide_with_nan(sums[key], counts[key]) for key in POSITION_KEYS}


def add_position(total: np.ndarray, count: np.ndarray, values: torch.Tensor, valid: torch.Tensor) -> None:
    finite = valid & torch.isfinite(values)
    finite_np = finite.numpy()
    values_np = values.numpy()
    total[finite_np] += values_np[finite_np].astype(np.float64)
    count[finite_np] += 1.0


def divide_with_nan(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    out = np.full_like(total, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def position_change_table(initial: dict[str, np.ndarray], final: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in POSITION_KEYS:
        delta = final[key] - initial[key]
        finite = delta[np.isfinite(delta)]
        rows.append(
            {
                "metric": key,
                "initial_mean": safe_mean(initial[key][np.isfinite(initial[key])]),
                "final_mean": safe_mean(final[key][np.isfinite(final[key])]),
                "delta_mean": safe_mean(finite),
                "delta_median": safe_median(finite),
                "delta_std": safe_std(finite),
                "delta_abs_p95": float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 0.0,
                "delta_positive_ratio": float(np.mean(finite > 0)) if finite.size else 0.0,
                "delta_negative_ratio": float(np.mean(finite < 0)) if finite.size else 0.0,
            }
        )
    return rows


def make_charts(
    initial_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    projected_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    pressure_rows: list[dict[str, Any]],
    initial_position: dict[str, np.ndarray],
    final_position: dict[str, np.ndarray],
    charts_dir: Path,
    *,
    value_percentiles: tuple[float, float],
    delta_percentile: float,
) -> list[Path]:
    paths = [
        plot_distribution(initial_rows, final_rows, charts_dir, "rho", xlabel="rho"),
        plot_distribution(initial_rows, final_rows, charts_dir, "mse_est_inv1p", xlabel="MSE estimated from rho=1/(1+MSE)"),
        plot_distribution(initial_rows, final_rows, charts_dir, "reward_mean", xlabel="reward_mean"),
        plot_distribution_change_bar(distribution_rows, charts_dir),
        plot_highdim_pca(projected_rows, charts_dir),
        plot_highdim_pca_by_rho(projected_rows, charts_dir),
        plot_selection_pressure(pressure_rows, charts_dir),
    ]
    heatmap_titles = {
        "tau0_distance": "tau0 distance",
        "tau1_queue": "tau1 queue",
        "reward": "reward",
        "valid_ratio": "valid-position ratio",
        "rho_broadcast": "rho broadcast to valid positions",
    }
    for key, title in heatmap_titles.items():
        path = charts_dir / f"{key}_position_heatmap.png"
        plot_position_triptych(
            initial_position[key],
            final_position[key],
            path,
            title,
            value_percentiles=value_percentiles,
            delta_percentile=delta_percentile,
        )
        paths.append(path)
    return paths


def plot_distribution(
    initial_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    charts_dir: Path,
    metric: str,
    *,
    xlabel: str,
) -> Path:
    initial = vector(initial_rows, metric)
    final = vector(final_rows, metric)
    path = charts_dir / f"{metric}_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = safe_bins(np.concatenate([initial, final]))
    ax.hist(initial, bins=bins, alpha=0.55, label="initial")
    ax.hist(final, bins=bins, alpha=0.55, label="final")
    ax.set_title(f"{metric} distribution")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_distribution_change_bar(rows: list[dict[str, Any]], charts_dir: Path) -> Path:
    path = charts_dir / "distribution_change_bar.png"
    labels = [str(row["metric"]) for row in rows]
    values = [float(row["standardized_mean_delta"]) for row in rows]
    colors = ["#c44e52" if value >= 0 else "#4c72b0" for value in values]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Standardized mean shift: final - initial")
    ax.set_ylabel("delta mean / pooled std")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_highdim_pca(projected_rows: list[dict[str, Any]], charts_dir: Path) -> Path:
    path = charts_dir / "highdim_pca_population_change.png"
    groups = np.asarray([row["group"] for row in projected_rows])
    x = vector(projected_rows, "pca1")
    y = vector(projected_rows, "pca2")
    fig, ax = plt.subplots(figsize=(8, 6))
    for group, color in [("initial", "#4c72b0"), ("final", "#dd8452")]:
        mask = groups == group
        ax.scatter(x[mask], y[mask], s=22, alpha=0.72, label=group, color=color)
    draw_row_links(ax, projected_rows)
    ax.set_title("PCA after feature hashing of full tau/reward tensors")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_highdim_pca_by_rho(projected_rows: list[dict[str, Any]], charts_dir: Path) -> Path:
    path = charts_dir / "highdim_pca_by_rho.png"
    x = vector(projected_rows, "pca1")
    y = vector(projected_rows, "pca2")
    rho = vector(projected_rows, "rho")
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x, y, c=rho, cmap="viridis", s=24, alpha=0.78)
    ax.set_title("High-dimensional PCA colored by rho")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="rho")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def draw_row_links(ax: Any, rows: list[dict[str, Any]]) -> None:
    by_key: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault(int(row["index"]), {})[str(row["group"])] = row
    for pair in by_key.values():
        if "initial" not in pair or "final" not in pair:
            continue
        a = pair["initial"]
        b = pair["final"]
        ax.plot([a["pca1"], b["pca1"]], [a["pca2"], b["pca2"]], color="gray", alpha=0.12, linewidth=0.7)


def plot_selection_pressure(rows: list[dict[str, Any]], charts_dir: Path) -> Path:
    path = charts_dir / "selection_pressure.png"
    temps = [float(row["temperature"]) for row in rows]
    ratios = [float(row["effective_sample_ratio"]) for row in rows]
    top_mass = [float(row["top10_probability_mass"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temps, ratios, marker="o", label="effective sample ratio")
    ax.plot(temps, top_mass, marker="o", label="top-10 probability mass")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_title("Softmax selection pressure from raw rho")
    ax.set_xlabel("temperature")
    ax.set_ylabel("probability / ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_position_triptych(
    initial: np.ndarray,
    final: np.ndarray,
    path: Path,
    title: str,
    *,
    value_percentiles: tuple[float, float],
    delta_percentile: float,
) -> None:
    delta = final - initial
    vmin, vmax = shared_limits(initial, final, value_percentiles)
    dlim = symmetric_limit(delta, delta_percentile)
    value_cmap = masked_cmap("viridis")
    delta_cmap = masked_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=-dlim, vcenter=0.0, vmax=dlim)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    panels = [
        ("initial", initial, value_cmap, vmin, vmax, None),
        ("final", final, value_cmap, vmin, vmax, None),
        ("final - initial", delta, delta_cmap, None, None, norm),
    ]
    for ax, (panel_title, matrix, cmap, lo, hi, panel_norm) in zip(axes, panels, strict=True):
        image = ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            norm=panel_norm,
        )
        ax.set_title(panel_title)
        ax.set_xlabel("road_index")
        ax.set_ylabel("car_index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{title} position mean heatmap")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def shared_limits(initial: np.ndarray, final: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    values = np.concatenate([initial[np.isfinite(initial)], final[np.isfinite(final)]])
    if values.size == 0:
        return 0.0, 1.0
    lo_p, hi_p = sorted([float(percentiles[0]), float(percentiles[1])])
    lo = float(np.nanpercentile(values, lo_p))
    hi = float(np.nanpercentile(values, hi_p))
    if lo == hi:
        pad = max(abs(lo) * 1e-6, 1e-9)
        return lo - pad, hi + pad
    return lo, hi


def symmetric_limit(delta: np.ndarray, percentile: float) -> float:
    values = np.abs(delta[np.isfinite(delta)])
    if values.size == 0:
        return 1.0
    limit = float(np.nanpercentile(values, min(max(float(percentile), 0.0), 100.0)))
    if limit <= 0.0:
        limit = float(np.nanmax(values))
    return limit if limit > 0.0 else 1.0


def masked_cmap(name: str) -> Any:
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("#f2f2f2")
    return cmap


def build_report(
    *,
    initial_path: Path,
    final_path: Path,
    output_dir: Path,
    initial_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    pressure_rows: list[dict[str, Any]],
    lineage: dict[str, Any],
    variance_ratio: list[float],
    runtime_seconds: float,
    chart_paths: list[Path],
) -> dict[str, Any]:
    distribution_by_metric = {str(row["metric"]): row for row in distribution_rows}
    rho = distribution_by_metric["rho"]
    mse = distribution_by_metric["mse_est_inv1p"]
    reward = distribution_by_metric["reward_mean"]
    tau0 = distribution_by_metric["tau0_mean"]
    tau1 = distribution_by_metric["tau1_mean"]
    strongest = sorted(distribution_rows, key=lambda row: abs(float(row["standardized_mean_delta"])), reverse=True)[:6]
    headline = {
        "n_initial": len(initial_rows),
        "n_final": len(final_rows),
        "rho_mean_initial": rho["initial_mean"],
        "rho_mean_final": rho["final_mean"],
        "rho_mean_delta": rho["mean_delta"],
        "mse_est_mean_initial": mse["initial_mean"],
        "mse_est_mean_final": mse["final_mean"],
        "mse_est_mean_delta": mse["mean_delta"],
        "reward_mean_delta": reward["mean_delta"],
        "tau0_mean_delta": tau0["mean_delta"],
        "tau1_mean_delta": tau1["mean_delta"],
        "highdim_pca_variance_ratio": variance_ratio,
    }
    return {
        "inputs": {"initial": str(initial_path), "final": str(final_path), "output_dir": str(output_dir)},
        "headline": headline,
        "distribution_changes": distribution_rows,
        "strongest_standardized_shifts": strongest,
        "position_changes": position_rows,
        "selection_pressure": pressure_rows,
        "lineage": lineage,
        "methods": [
            "Loaded MASDiff population pickles with a lightweight src.pipeline.types compatibility shim.",
            "Summarized each individual into interpretable scalar features instead of flattening tensors into CSV.",
            "Embedded full high-dimensional tau/reward/valid tensors with deterministic feature hashing, then applied PCA.",
            "Compared final and initial as distributions; row-aligned deltas are diagnostic only because MASDiff keeps top-M selected individuals.",
            "Computed car-road position heatmaps by averaging each car_index/road_index cell over all individuals.",
            "Estimated softmax selection pressure from raw rho to diagnose whether elite sampling is nearly uniform.",
        ],
        "charts": [str(path) for path in chart_paths],
        "runtime_seconds": round(float(runtime_seconds), 3),
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    h = report["headline"]
    lines = [
        "# Population Change Analysis",
        "",
        "## Headline",
        "",
        f"- n_initial: {h['n_initial']}",
        f"- n_final: {h['n_final']}",
        f"- rho mean: {h['rho_mean_initial']:.8g} -> {h['rho_mean_final']:.8g}",
        f"- rho mean delta: {h['rho_mean_delta']:.8g}",
        f"- estimated MSE mean: {h['mse_est_mean_initial']:.8g} -> {h['mse_est_mean_final']:.8g}",
        f"- reward_mean delta: {h['reward_mean_delta']:.8g}",
        f"- tau0_mean delta: {h['tau0_mean_delta']:.8g}",
        f"- tau1_mean delta: {h['tau1_mean_delta']:.8g}",
        f"- high-dimensional PCA variance ratio: {h['highdim_pca_variance_ratio']}",
        "",
        "## Strongest Standardized Distribution Shifts",
        "",
    ]
    for row in report["strongest_standardized_shifts"]:
        lines.append(
            f"- {row['metric']}: mean_delta={row['mean_delta']:.8g}, "
            f"standardized={row['standardized_mean_delta']:.4g}, ks={row['ks_stat']:.4g}"
        )
    lines.extend(["", "## Methods", ""])
    for item in report["methods"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Charts", ""])
    for chart in report["charts"]:
        lines.append(f"- `{chart}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_charts_to_docs(chart_paths: list[Path], docs_image_dir: Path) -> None:
    docs_image_dir.mkdir(parents=True, exist_ok=True)
    for path in chart_paths:
        if path.suffix.lower() == ".png" and path.exists():
            shutil.copy2(path, docs_image_dir / path.name)
    print(f"Copied {len(chart_paths)} charts to {docs_image_dir}")


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def vector(rows: Iterable[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, 0.0)
        values.append(float(value) if value != "" else float("nan"))
    return np.asarray(values, dtype=np.float64)


def safe_bins(values: np.ndarray, bins: int = 30) -> int | np.ndarray:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return bins
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= max(abs(lo), abs(hi), 1.0) * 1e-10:
        center = 0.5 * (lo + hi)
        width = max(abs(center) * 1e-6, 1e-9)
        return np.linspace(center - width, center + width, min(10, bins) + 1)
    return bins


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if values.size else 0.0


def safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanstd(values)) if values.size else 0.0


def safe_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmedian(values)) if values.size else 0.0


def json_default(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return {"type": "tensor", "shape": list(value.shape)}
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return {"type": "ndarray", "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
