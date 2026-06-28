"""Analyze changes from an initial MASDiff population pickle to a final one."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

try:
    from .types_compat import install_pickle_compat
except ImportError:
    from types_compat import install_pickle_compat


SUMMARY_COLUMNS = [
    "index",
    "rho",
    "n_vehicle",
    "n_road",
    "tau_dim",
    "valid_count",
    "invalid_count",
    "valid_ratio",
    "tau_nan_count",
    "tau_inf_count",
    "reward_nan_count",
    "reward_inf_count",
    "reward_mean",
    "reward_std",
    "reward_min",
    "reward_q25",
    "reward_median",
    "reward_q75",
    "reward_max",
    "reward_positive_ratio",
    "reward_zero_ratio",
    "best_reward_mean",
    "best_reward_std",
    "best_reward_max",
    "top2_gap_mean",
    "top2_gap_std",
    "metadata_json",
]

TAU_PREFIXES = ("tau0", "tau1")
TAU_STATS = ("mean", "std", "min", "q25", "median", "q75", "max")
DELTA_COLUMNS = [
    "rho",
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
]
EMBEDDING_COLUMNS = [
    "rho",
    "valid_ratio",
    "reward_mean",
    "reward_std",
    "reward_median",
    "reward_positive_ratio",
    "best_reward_mean",
    "top2_gap_mean",
    "tau0_mean",
    "tau0_std",
    "tau1_mean",
    "tau1_std",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert initial/final population pickles to CSV and analyze paired changes."
    )
    parser.add_argument("--initial", default="dataset/initial_population.pkl")
    parser.add_argument("--final", default="dataset/final_population.pkl")
    parser.add_argument("--output-dir", default="outputs/population_change_analysis")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug limit; 0 means all individuals.")
    parser.add_argument("--force", action="store_true", help="Regenerate summary CSV files even if they exist.")
    parser.add_argument(
        "--write-long-csv",
        action="store_true",
        help="Also write vehicle-road long CSV gzip files. This can be very large.",
    )
    parser.add_argument(
        "--long-csv-max-samples",
        type=int,
        default=10,
        help="Safety cap for long CSV output; use 0 for all samples.",
    )
    parser.add_argument(
        "--embedding",
        choices=["auto", "pca", "tsne", "none"],
        default="auto",
        help="Low-dimensional visualization method. auto writes PCA and t-SNE when sklearn is available.",
    )
    parser.add_argument(
        "--no-position-heatmaps",
        action="store_true",
        help="Skip car-road position heatmaps for tau0_distance, tau1_queue, reward, and rho.",
    )
    parser.add_argument(
        "--heatmap-value-percentiles",
        nargs=2,
        type=float,
        default=[2.0, 98.0],
        metavar=("LOW", "HIGH"),
        help="Percentile limits for initial/final heatmaps. Clipping improves visual contrast.",
    )
    parser.add_argument(
        "--heatmap-delta-percentile",
        type=float,
        default=95.0,
        help="Absolute delta percentile used for symmetric final-initial color limits.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_csv = output_dir / "initial_population_summary.csv"
    final_csv = output_dir / "final_population_summary.csv"
    comparison_csv = output_dir / "population_change_summary.csv"

    start = time.time()
    if args.force or not initial_csv.exists():
        summarize_population(
            Path(args.initial),
            initial_csv,
            output_dir / "initial_population_long.csv.gz",
            args.max_samples,
            args.write_long_csv,
            args.long_csv_max_samples,
        )
    if args.force or not final_csv.exists():
        summarize_population(
            Path(args.final),
            final_csv,
            output_dir / "final_population_long.csv.gz",
            args.max_samples,
            args.write_long_csv,
            args.long_csv_max_samples,
        )

    initial_rows = read_summary_csv(initial_csv)
    final_rows = read_summary_csv(final_csv)
    comparison_rows = make_comparison_rows(initial_rows, final_rows)
    write_comparison_csv(comparison_csv, comparison_rows)

    stats = analyze_comparison(comparison_rows)
    stats["inputs"] = {
        "initial": str(Path(args.initial)),
        "final": str(Path(args.final)),
        "initial_csv": str(initial_csv),
        "final_csv": str(final_csv),
        "comparison_csv": str(comparison_csv),
        "max_samples": args.max_samples,
    }
    stats["methods"] = method_descriptions(args.embedding)
    stats["runtime_seconds"] = round(time.time() - start, 3)

    chart_paths = make_charts(initial_rows, final_rows, comparison_rows, output_dir, args.embedding, args.tsne_perplexity)
    if not args.no_position_heatmaps:
        chart_paths.extend(
            make_position_heatmaps(
                Path(args.initial),
                Path(args.final),
                output_dir,
                args.max_samples,
                tuple(args.heatmap_value_percentiles),
                args.heatmap_delta_percentile,
            )
        )
    stats["charts"] = [str(path) for path in chart_paths]

    with (output_dir / "analysis_report.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    write_markdown_report(output_dir / "analysis_report.md", stats)

    print(json.dumps(stats["headline"], indent=2, ensure_ascii=False))
    print(f"saved={output_dir}")


def summarize_population(
    pickle_path: Path,
    summary_csv: Path,
    long_csv_gz: Path,
    max_samples: int,
    write_long_csv: bool,
    long_csv_max_samples: int,
) -> None:
    if not pickle_path.exists():
        raise FileNotFoundError(pickle_path)

    install_pickle_compat()
    print(f"Loading {pickle_path} ...")
    start = time.time()
    with pickle_path.open("rb") as f:
        population = pickle.load(f)
    elapsed = time.time() - start
    print(f"Loaded {len(population)} individuals from {pickle_path.name} in {elapsed:.1f}s")

    limit = len(population) if max_samples <= 0 else min(max_samples, len(population))
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    long_writer = None
    long_file = None
    if write_long_csv:
        long_file = gzip.open(long_csv_gz, "wt", encoding="utf-8", newline="")
        long_writer = csv.DictWriter(
            long_file,
            fieldnames=["index", "vehicle", "road", "valid", "tau0", "tau1", "reward"],
        )
        long_writer.writeheader()

    columns = summary_columns()
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for idx in range(limit):
            individual = population[idx]
            row = summarize_individual(individual, idx)
            writer.writerow({column: row.get(column, "") for column in columns})
            if long_writer is not None and should_write_long(idx, long_csv_max_samples):
                write_long_rows(long_writer, individual, idx)

    if long_file is not None:
        long_file.close()
    del population
    print(f"Wrote {summary_csv}")


def summary_columns() -> list[str]:
    columns = list(SUMMARY_COLUMNS)
    for prefix in TAU_PREFIXES:
        for stat in TAU_STATS:
            columns.append(f"{prefix}_{stat}")
    return columns


def summarize_individual(individual: Any, idx: int) -> dict[str, Any]:
    tau = torch.as_tensor(individual.tau, dtype=torch.float32).cpu()
    rewards = torch.as_tensor(individual.rewards, dtype=torch.float32).cpu()
    if tau.dim() != 3:
        raise ValueError(f"Individual {idx} tau must be 3-D, got {tuple(tau.shape)}")
    if rewards.shape != tau.shape[:2]:
        raise ValueError(f"Individual {idx} rewards shape {tuple(rewards.shape)} does not match tau {tuple(tau.shape)}")

    valid = torch.isfinite(rewards) & torch.isfinite(tau).all(dim=-1)
    valid_count = int(valid.sum().item())
    total_count = int(valid.numel())
    reward_values = rewards[valid & torch.isfinite(rewards)]
    best_rewards, top2_gaps = best_reward_stats(rewards, valid)
    metadata = getattr(individual, "metadata", {}) or {}

    row: dict[str, Any] = {
        "index": idx,
        "rho": float(individual.rho),
        "n_vehicle": int(tau.shape[0]),
        "n_road": int(tau.shape[1]),
        "tau_dim": int(tau.shape[2]),
        "valid_count": valid_count,
        "invalid_count": total_count - valid_count,
        "valid_ratio": valid_count / max(total_count, 1),
        "tau_nan_count": int(torch.isnan(tau).sum().item()),
        "tau_inf_count": int(torch.isinf(tau).sum().item()),
        "reward_nan_count": int(torch.isnan(rewards).sum().item()),
        "reward_inf_count": int(torch.isinf(rewards).sum().item()),
        "reward_positive_ratio": ratio(reward_values > 0),
        "reward_zero_ratio": ratio(reward_values == 0),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=json_default),
    }
    row.update(prefixed_stats("reward", reward_values))
    row.update(prefixed_stats("best_reward", best_rewards, include_quantiles=False))
    row.update(prefixed_stats("top2_gap", top2_gaps, include_quantiles=False))

    for channel_idx, prefix in enumerate(TAU_PREFIXES):
        if channel_idx < tau.shape[-1]:
            values = tau[..., channel_idx][valid & torch.isfinite(tau[..., channel_idx])]
        else:
            values = torch.empty(0)
        row.update(prefixed_stats(prefix, values))
    return row


def prefixed_stats(prefix: str, values: torch.Tensor, include_quantiles: bool = True) -> dict[str, float]:
    values = values[torch.isfinite(values)].float()
    if values.numel() == 0:
        base = {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
        if include_quantiles:
            base.update({f"{prefix}_q25": 0.0, f"{prefix}_median": 0.0, f"{prefix}_q75": 0.0})
        return base

    result = {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }
    if include_quantiles:
        qs = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75], device=values.device))
        result.update(
            {
                f"{prefix}_q25": float(qs[0].item()),
                f"{prefix}_median": float(qs[1].item()),
                f"{prefix}_q75": float(qs[2].item()),
            }
        )
    return result


def best_reward_stats(rewards: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    per_vehicle_valid = valid.any(dim=1)
    if not per_vehicle_valid.any():
        return torch.empty(0), torch.empty(0)
    masked_rewards = rewards.masked_fill(~valid, -float("inf"))
    best = masked_rewards.max(dim=1).values[per_vehicle_valid]
    top2 = torch.topk(masked_rewards, k=min(2, masked_rewards.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        gap = torch.zeros_like(top2[:, 0])
    else:
        gap = top2[:, 0] - top2[:, 1]
    gap = gap[per_vehicle_valid & torch.isfinite(gap)]
    return best[torch.isfinite(best)], gap


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


def ratio(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def should_write_long(idx: int, long_csv_max_samples: int) -> bool:
    return long_csv_max_samples <= 0 or idx < long_csv_max_samples


def write_long_rows(writer: csv.DictWriter, individual: Any, idx: int) -> None:
    tau = torch.as_tensor(individual.tau, dtype=torch.float32).cpu()
    rewards = torch.as_tensor(individual.rewards, dtype=torch.float32).cpu()
    valid = torch.isfinite(rewards) & torch.isfinite(tau).all(dim=-1)
    tau1 = tau[..., 1] if tau.shape[-1] > 1 else torch.zeros_like(tau[..., 0])
    for vehicle in range(tau.shape[0]):
        for road in range(tau.shape[1]):
            writer.writerow(
                {
                    "index": idx,
                    "vehicle": vehicle,
                    "road": road,
                    "valid": int(bool(valid[vehicle, road])),
                    "tau0": float(tau[vehicle, road, 0]),
                    "tau1": float(tau1[vehicle, road]),
                    "reward": float(rewards[vehicle, road]),
                }
            )


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key == "metadata_json":
                    parsed[key] = value
                elif key in {"index", "n_vehicle", "n_road", "tau_dim", "valid_count", "invalid_count"}:
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = float(value) if value != "" else 0.0
            rows.append(parsed)
    return rows


def make_comparison_rows(initial_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_by_index = {int(row["index"]): row for row in final_rows}
    rows = []
    for initial in initial_rows:
        idx = int(initial["index"])
        if idx not in final_by_index:
            continue
        final = final_by_index[idx]
        row: dict[str, Any] = {"index": idx}
        for column in DELTA_COLUMNS:
            row[f"initial_{column}"] = float(initial.get(column, 0.0))
            row[f"final_{column}"] = float(final.get(column, 0.0))
            row[f"delta_{column}"] = row[f"final_{column}"] - row[f"initial_{column}"]
        rows.append(row)
    return rows


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No paired rows to analyze.")

    stats: dict[str, Any] = {"n_pairs": len(rows), "metrics": {}}
    for metric in DELTA_COLUMNS:
        initial = vector(rows, f"initial_{metric}")
        final = vector(rows, f"final_{metric}")
        delta = vector(rows, f"delta_{metric}")
        stats["metrics"][metric] = paired_metric_stats(initial, final, delta)

    rho = stats["metrics"]["rho"]
    valid_ratio = stats["metrics"]["valid_ratio"]
    reward_mean = stats["metrics"]["reward_mean"]
    stats["headline"] = {
        "n_pairs": len(rows),
        "rho_mean_initial": rho["initial_mean"],
        "rho_mean_final": rho["final_mean"],
        "rho_mean_delta": rho["delta_mean"],
        "rho_delta_effect_size_dz": rho["cohen_dz"],
        "rho_initial_final_corr": rho["initial_final_corr"],
        "valid_ratio_mean_delta": valid_ratio["delta_mean"],
        "reward_mean_delta": reward_mean["delta_mean"],
    }
    return stats


def paired_metric_stats(initial: np.ndarray, final: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    result = {
        "initial_mean": safe_mean(initial),
        "initial_std": safe_std(initial),
        "final_mean": safe_mean(final),
        "final_std": safe_std(final),
        "delta_mean": safe_mean(delta),
        "delta_std": safe_std(delta),
        "delta_median": safe_median(delta),
        "delta_min": safe_min(delta),
        "delta_max": safe_max(delta),
        "delta_positive_count": int(np.sum(delta > 0)),
        "delta_negative_count": int(np.sum(delta < 0)),
        "delta_zero_count": int(np.sum(delta == 0)),
        "initial_final_corr": safe_corr(initial, final),
        "cohen_dz": cohen_dz(delta),
    }
    result.update(optional_paired_test(delta))
    return result


def optional_paired_test(delta: np.ndarray) -> dict[str, float | str]:
    if len(delta) < 2 or safe_std(delta) == 0.0:
        return {"paired_t_stat": 0.0, "paired_t_pvalue": 1.0, "paired_test": "not_applicable"}
    t_stat = safe_mean(delta) / (safe_std(delta) / math.sqrt(len(delta)))
    try:
        from scipy import stats as scipy_stats

        test = scipy_stats.ttest_1samp(delta, popmean=0.0, nan_policy="omit")
        return {
            "paired_t_stat": float(test.statistic),
            "paired_t_pvalue": float(test.pvalue),
            "paired_test": "paired t-test via scipy.stats.ttest_1samp on final-initial deltas",
        }
    except Exception:
        return {
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float("nan"),
            "paired_test": "paired t-statistic only; scipy not available for p-value",
        }


def make_charts(
    initial_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    output_dir: Path,
    embedding: str,
    tsne_perplexity: float,
) -> list[Path]:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_distribution(initial_rows, final_rows, charts_dir, "rho"),
        plot_distribution(initial_rows, final_rows, charts_dir, "reward_mean"),
        plot_initial_final_scatter(comparison_rows, charts_dir, "rho"),
        plot_delta_histogram(comparison_rows, charts_dir, "rho"),
        plot_delta_boxplot(comparison_rows, charts_dir),
    ]
    if embedding in {"auto", "pca"}:
        paths.append(plot_pca_embedding(initial_rows, final_rows, charts_dir))
    if embedding in {"auto", "tsne"}:
        tsne_path = plot_tsne_embedding(initial_rows, final_rows, charts_dir, tsne_perplexity)
        if tsne_path is not None:
            paths.append(tsne_path)
    return paths


def make_position_heatmaps(
    initial_path: Path,
    final_path: Path,
    output_dir: Path,
    max_samples: int,
    value_percentiles: tuple[float, float],
    delta_percentile: float,
) -> list[Path]:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    print("Computing car-road position heatmaps ...")
    initial = compute_position_means(initial_path, max_samples)
    final = compute_position_means(final_path, max_samples)
    specs = [
        ("tau0_distance", "tau0_distance", "viridis"),
        ("tau1_queue", "tau1_queue", "viridis"),
        ("reward", "reward", "viridis"),
        ("rho", "rho broadcast to valid car-road positions", "viridis"),
    ]
    paths = []
    for key, title, cmap in specs:
        path = charts_dir / f"{key}_position_heatmap.png"
        plot_position_triptych(initial[key], final[key], path, title, cmap, value_percentiles, delta_percentile)
        paths.append(path)
    return paths


def compute_position_means(pickle_path: Path, max_samples: int) -> dict[str, np.ndarray]:
    install_pickle_compat()
    print(f"Loading {pickle_path} for position heatmaps ...")
    with pickle_path.open("rb") as f:
        population = pickle.load(f)
    limit = len(population) if max_samples <= 0 else min(max_samples, len(population))
    if limit == 0:
        raise ValueError(f"No individuals available in {pickle_path}")

    first_tau = torch.as_tensor(population[0].tau, dtype=torch.float32).cpu()
    if first_tau.dim() != 3:
        raise ValueError(f"First individual tau must be 3-D, got {tuple(first_tau.shape)}")
    n_vehicle, n_road, tau_dim = (int(v) for v in first_tau.shape)

    sums = {
        "tau0_distance": np.zeros((n_vehicle, n_road), dtype=np.float64),
        "tau1_queue": np.zeros((n_vehicle, n_road), dtype=np.float64),
        "reward": np.zeros((n_vehicle, n_road), dtype=np.float64),
        "rho": np.zeros((n_vehicle, n_road), dtype=np.float64),
    }
    counts = {key: np.zeros((n_vehicle, n_road), dtype=np.float64) for key in sums}

    for idx in range(limit):
        individual = population[idx]
        tau = torch.as_tensor(individual.tau, dtype=torch.float32).cpu()
        rewards = torch.as_tensor(individual.rewards, dtype=torch.float32).cpu()
        if tau.shape[:2] != (n_vehicle, n_road):
            raise ValueError(
                f"Individual {idx} shape {tuple(tau.shape[:2])} does not match first sample {(n_vehicle, n_road)}"
            )
        if rewards.shape != tau.shape[:2]:
            raise ValueError(f"Individual {idx} rewards shape {tuple(rewards.shape)} does not match tau {tuple(tau.shape)}")

        valid = torch.isfinite(rewards) & torch.isfinite(tau).all(dim=-1)
        valid_np = valid.numpy()
        add_position_values(sums["tau0_distance"], counts["tau0_distance"], tau[..., 0], valid)
        if tau_dim > 1:
            add_position_values(sums["tau1_queue"], counts["tau1_queue"], tau[..., 1], valid)
        add_position_values(sums["reward"], counts["reward"], rewards, valid)
        sums["rho"][valid_np] += float(individual.rho)
        counts["rho"][valid_np] += 1.0

    del population
    means = {}
    for key in sums:
        means[key] = divide_with_nan(sums[key], counts[key])
    return means


def add_position_values(total: np.ndarray, count: np.ndarray, values: torch.Tensor, valid: torch.Tensor) -> None:
    finite = valid & torch.isfinite(values)
    finite_np = finite.numpy()
    values_np = values.numpy()
    total[finite_np] += values_np[finite_np].astype(np.float64)
    count[finite_np] += 1.0


def divide_with_nan(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    result = np.full_like(total, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def plot_position_triptych(
    initial: np.ndarray,
    final: np.ndarray,
    path: Path,
    title: str,
    cmap: str,
    value_percentiles: tuple[float, float],
    delta_percentile: float,
) -> None:
    delta = final - initial
    value_vmin, value_vmax = shared_color_limits(initial, final, value_percentiles)
    delta_abs = symmetric_delta_limit(delta, delta_percentile)
    value_cmap = masked_cmap(cmap)
    delta_cmap = masked_cmap("coolwarm")
    delta_norm = TwoSlopeNorm(vmin=-delta_abs, vcenter=0.0, vmax=delta_abs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    panels = [
        ("initial", initial, value_cmap, value_vmin, value_vmax, None),
        ("final", final, value_cmap, value_vmin, value_vmax, None),
        ("final - initial", delta, delta_cmap, None, None, delta_norm),
    ]
    for ax, (panel_title, matrix, panel_cmap, vmin, vmax, norm) in zip(axes, panels, strict=True):
        image = ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            cmap=panel_cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_title(panel_title)
        ax.set_xlabel("road_index")
        ax.set_ylabel("car_index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{title} position mean heatmap "
        f"(value p{value_percentiles[0]:g}-p{value_percentiles[1]:g}, delta +/-p{delta_percentile:g})"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def shared_color_limits(initial: np.ndarray, final: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    values = np.concatenate([initial[np.isfinite(initial)], final[np.isfinite(final)]])
    if values.size == 0:
        return 0.0, 1.0
    lower, upper = sorted(float(value) for value in percentiles)
    lo = float(np.nanpercentile(values, lower))
    hi = float(np.nanpercentile(values, upper))
    if lo == hi:
        pad = max(abs(lo) * 1e-6, 1e-9)
        return lo - pad, hi + pad
    return lo, hi


def symmetric_delta_limit(delta: np.ndarray, percentile: float) -> float:
    values = np.abs(delta[np.isfinite(delta)])
    if values.size == 0:
        return 1.0
    clipped_percentile = min(max(float(percentile), 0.0), 100.0)
    limit = float(np.nanpercentile(values, clipped_percentile))
    if limit <= 0.0:
        limit = float(np.nanmax(values))
    if limit <= 0.0:
        return 1.0
    return limit


def masked_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("#f2f2f2")
    return cmap


def plot_distribution(initial_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]], charts_dir: Path, column: str) -> Path:
    initial = vector(initial_rows, column)
    final = vector(final_rows, column)
    path = charts_dir / f"{column}_distribution.png"
    plt.figure(figsize=(8, 5))
    bins = safe_bins(np.concatenate([initial, final]))
    plt.hist(initial, bins=bins, alpha=0.55, label="initial")
    plt.hist(final, bins=bins, alpha=0.55, label="final")
    plt.title(f"{column} distribution")
    plt.xlabel(column)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_initial_final_scatter(rows: list[dict[str, Any]], charts_dir: Path, column: str) -> Path:
    initial = vector(rows, f"initial_{column}")
    final = vector(rows, f"final_{column}")
    path = charts_dir / f"{column}_initial_vs_final.png"
    lo = float(min(np.min(initial), np.min(final)))
    hi = float(max(np.max(initial), np.max(final)))
    plt.figure(figsize=(6, 6))
    plt.scatter(initial, final, s=18, alpha=0.75)
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
    plt.title(f"{column}: initial vs final")
    plt.xlabel(f"initial {column}")
    plt.ylabel(f"final {column}")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_delta_histogram(rows: list[dict[str, Any]], charts_dir: Path, column: str) -> Path:
    delta = vector(rows, f"delta_{column}")
    path = charts_dir / f"{column}_delta_histogram.png"
    plt.figure(figsize=(8, 5))
    plt.hist(delta, bins=safe_bins(delta), alpha=0.8)
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.title(f"{column} delta: final - initial")
    plt.xlabel(f"delta {column}")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def safe_bins(values: np.ndarray, bins: int = 30) -> np.ndarray | int:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return bins
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= max(abs(lo), abs(hi), 1.0) * 1e-10:
        center = 0.5 * (lo + hi)
        width = max(abs(center) * 1e-6, 1e-9)
        return np.linspace(center - width, center + width, min(bins, 10) + 1)
    return bins


def plot_delta_boxplot(rows: list[dict[str, Any]], charts_dir: Path) -> Path:
    columns = ["rho", "valid_ratio", "reward_mean", "best_reward_mean", "tau0_mean", "tau1_mean"]
    values = [vector(rows, f"delta_{column}") for column in columns]
    path = charts_dir / "selected_delta_boxplot.png"
    plt.figure(figsize=(10, 5))
    plt.boxplot(values, tick_labels=columns, showfliers=False)
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.title("Selected metric deltas")
    plt.ylabel("final - initial")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_pca_embedding(initial_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]], charts_dir: Path) -> Path:
    features, labels, indices = embedding_matrix(initial_rows, final_rows)
    coords = pca_2d(features)
    path = charts_dir / "pca_population_change.png"
    plot_embedding_with_arrows(coords, labels, indices, path, "PCA of population summary features")
    return path


def plot_tsne_embedding(
    initial_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    charts_dir: Path,
    perplexity: float,
) -> Path | None:
    try:
        from sklearn.manifold import TSNE
    except Exception:
        print("sklearn is not available; skipping t-SNE chart.")
        return None
    features, labels, indices = embedding_matrix(initial_rows, final_rows)
    n_samples = features.shape[0]
    safe_perplexity = min(float(perplexity), max(2.0, (n_samples - 1) / 3.0))
    coords = TSNE(n_components=2, perplexity=safe_perplexity, init="pca", learning_rate="auto").fit_transform(features)
    path = charts_dir / "tsne_population_change.png"
    plot_embedding_with_arrows(coords, labels, indices, path, "t-SNE of population summary features")
    return path


def embedding_matrix(initial_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], np.ndarray]:
    paired_final = {int(row["index"]): row for row in final_rows}
    rows = []
    labels = []
    indices = []
    for initial in initial_rows:
        idx = int(initial["index"])
        if idx not in paired_final:
            continue
        rows.append([float(initial.get(column, 0.0)) for column in EMBEDDING_COLUMNS])
        labels.append("initial")
        indices.append(idx)
        final = paired_final[idx]
        rows.append([float(final.get(column, 0.0)) for column in EMBEDDING_COLUMNS])
        labels.append("final")
        indices.append(idx)
    features = np.asarray(rows, dtype=np.float64)
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (features - mean) / std, labels, np.asarray(indices, dtype=np.int64)


def pca_2d(features: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(features, dtype=torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    coords = x @ vh[:2].T
    return coords.cpu().numpy()


def plot_embedding_with_arrows(coords: np.ndarray, labels: list[str], indices: np.ndarray, path: Path, title: str) -> None:
    labels_array = np.asarray(labels)
    plt.figure(figsize=(8, 6))
    initial_mask = labels_array == "initial"
    final_mask = labels_array == "final"
    plt.scatter(coords[initial_mask, 0], coords[initial_mask, 1], s=18, alpha=0.65, label="initial")
    plt.scatter(coords[final_mask, 0], coords[final_mask, 1], s=18, alpha=0.65, label="final")
    unique_indices = np.unique(indices)
    for idx in unique_indices:
        pair = np.where(indices == idx)[0]
        if len(pair) == 2:
            a, b = pair[0], pair[1]
            plt.arrow(
                coords[a, 0],
                coords[a, 1],
                coords[b, 0] - coords[a, 0],
                coords[b, 1] - coords[a, 1],
                length_includes_head=True,
                head_width=0.02,
                alpha=0.18,
                color="gray",
            )
    plt.title(title)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_markdown_report(path: Path, stats: dict[str, Any]) -> None:
    headline = stats["headline"]
    lines = [
        "# Population Change Analysis",
        "",
        "## Headline",
        "",
        f"- Paired individuals: {headline['n_pairs']}",
        f"- rho mean: {headline['rho_mean_initial']:.8g} -> {headline['rho_mean_final']:.8g}",
        f"- rho mean delta: {headline['rho_mean_delta']:.8g}",
        f"- rho paired effect size dz: {headline['rho_delta_effect_size_dz']:.4g}",
        f"- rho initial/final correlation: {headline['rho_initial_final_corr']:.4g}",
        f"- valid ratio mean delta: {headline['valid_ratio_mean_delta']:.8g}",
        f"- reward mean delta: {headline['reward_mean_delta']:.8g}",
        "",
        "## Methods",
        "",
    ]
    for item in stats["methods"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Charts", ""])
    for chart in stats["charts"]:
        lines.append(f"- `{chart}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def method_descriptions(embedding: str) -> list[str]:
    return [
        "Loaded MASDiff population pickles with the local compatibility shim for src.pipeline.types.Individual.",
        "Converted each population to a sample-level CSV with one row per individual; tensor fields are summarized instead of fully flattened by default.",
        "Used finite masks to exclude NaN/Inf tau or reward entries from continuous statistics.",
        "Computed paired final-initial deltas by individual index for rho, valid ratio, reward summaries, best-reward summaries, top-2 reward gap, and tau channel summaries.",
        "Reported paired mean/std/median/min/max deltas, sign counts, Pearson initial-final correlation, Cohen's dz, and a paired t-test when scipy is available.",
        f"Generated distribution, paired scatter, delta histogram, boxplot, and {embedding} low-dimensional visualization charts.",
    ]


def vector(rows: Iterable[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)


def safe_mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if values.size else 0.0


def safe_std(values: np.ndarray) -> float:
    return float(np.nanstd(values)) if values.size else 0.0


def safe_median(values: np.ndarray) -> float:
    return float(np.nanmedian(values)) if values.size else 0.0


def safe_min(values: np.ndarray) -> float:
    return float(np.nanmin(values)) if values.size else 0.0


def safe_max(values: np.ndarray) -> float:
    return float(np.nanmax(values)) if values.size else 0.0


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or np.nanstd(a) == 0.0 or np.nanstd(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cohen_dz(delta: np.ndarray) -> float:
    std = safe_std(delta)
    if std == 0.0:
        return 0.0
    return safe_mean(delta) / std


if __name__ == "__main__":
    main()
