"""Screen road-level reward importance for MASDiff SUMO populations.

This script is a low-cost observational analysis.  It estimates which
``road_index`` columns in the reward matrix are associated with ``rho`` or
``delta_rho``.  It does not prove causality; true causal importance still needs
counterfactual SUMO runs where one road's reward column is perturbed, DQN is
retrained, and rho is recomputed.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .types_compat import install_pickle_compat
except ImportError:  # pragma: no cover
    from types_compat import install_pickle_compat


FEATURES = ("reward_mean", "reward_std", "reward_max", "tau0_mean", "tau1_mean", "valid_ratio")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze road-level reward importance for rho.")
    parser.add_argument("--initial", default="dataset/initial_population.pkl")
    parser.add_argument("--final", default="dataset/final_population.pkl")
    parser.add_argument("--output-dir", default="dataset/road_importance_analysis")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug limit; 0 means all individuals.")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    initial = load_population(Path(args.initial), args.max_samples)
    final = load_population(Path(args.final), args.max_samples)

    initial_features, initial_meta = road_feature_matrices(initial, "initial")
    final_features, final_meta = road_feature_matrices(final, "final")

    road_rows = analyze_roads(
        initial_features=initial_features,
        final_features=final_features,
        initial_meta=initial_meta,
        final_meta=final_meta,
        ridge_alpha=float(args.ridge_alpha),
    )
    write_csv(output_dir / "road_importance.csv", road_rows)

    mutant_rows = mutant_child_reward_analysis(final_features, final_meta, ridge_alpha=float(args.ridge_alpha))
    write_csv(output_dir / "mutant_child_reward_vs_parent_delta.csv", mutant_rows)

    row_delta_rows = row_aligned_delta_analysis(initial_features, final_features, initial_meta, final_meta)
    write_csv(output_dir / "row_aligned_road_delta_diagnostic.csv", row_delta_rows)

    chart_paths = make_charts(road_rows, mutant_rows, row_delta_rows, charts_dir, top_k=int(args.top_k))
    report = build_report(
        initial=Path(args.initial),
        final=Path(args.final),
        output_dir=output_dir,
        road_rows=road_rows,
        mutant_rows=mutant_rows,
        row_delta_rows=row_delta_rows,
        final_meta=final_meta,
        chart_paths=chart_paths,
        runtime_seconds=time.time() - start,
    )
    (output_dir / "road_importance_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown_report(output_dir / "road_importance_report.md", report)

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


def road_feature_matrices(population: list[Any], group: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if not population:
        raise ValueError("population is empty")
    tau0 = torch.as_tensor(getattr(population[0], "tau"), dtype=torch.float32).cpu()
    n_road = int(tau0.shape[1])
    values = {feature: np.zeros((len(population), n_road), dtype=np.float64) for feature in FEATURES}
    meta_rows: list[dict[str, Any]] = []

    for idx, individual in enumerate(population):
        tau = torch.as_tensor(getattr(individual, "tau"), dtype=torch.float32).cpu()
        rewards = torch.as_tensor(getattr(individual, "rewards"), dtype=torch.float32).cpu()
        if tau.dim() != 3 or rewards.shape != tau.shape[:2]:
            raise ValueError(f"Invalid shape at {group}/{idx}: tau={tuple(tau.shape)}, rewards={tuple(rewards.shape)}")
        if int(tau.shape[1]) != n_road:
            raise ValueError(f"Road count changed at {group}/{idx}: {tau.shape[1]} vs {n_road}")
        valid = torch.isfinite(rewards) & torch.isfinite(tau).all(dim=-1)
        valid_np = valid.numpy()
        rewards_np = rewards.numpy()
        tau_np = tau.numpy()

        values["reward_mean"][idx] = column_mean(rewards_np, valid_np)
        values["reward_std"][idx] = column_std(rewards_np, valid_np)
        values["reward_max"][idx] = column_max(rewards_np, valid_np)
        values["tau0_mean"][idx] = column_mean(tau_np[..., 0], valid_np)
        values["tau1_mean"][idx] = column_mean(tau_np[..., 1], valid_np) if tau_np.shape[-1] > 1 else 0.0
        values["valid_ratio"][idx] = valid_np.mean(axis=0)

        metadata = getattr(individual, "metadata", {}) or {}
        rho = float(getattr(individual, "rho", 0.0))
        parent_rho = metadata_float(metadata, "parent_rho")
        meta_rows.append(
            {
                "group": group,
                "index": idx,
                "rho": rho,
                "parent_rho": parent_rho,
                "delta_parent_rho": rho - parent_rho if parent_rho is not None else "",
                "iteration_k": metadata_float(metadata, "iteration_k"),
                "initial_index": metadata_float(metadata, "initial_index"),
                "is_mutant": parent_rho is not None,
            }
        )
    return values, meta_rows


def column_mean(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.where(valid & np.isfinite(values), values, np.nan)
    with np.errstate(all="ignore"):
        out = np.nanmean(arr, axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def column_std(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.where(valid & np.isfinite(values), values, np.nan)
    with np.errstate(all="ignore"):
        out = np.nanstd(arr, axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def column_max(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.where(valid & np.isfinite(values), values, np.nan)
    with np.errstate(all="ignore"):
        out = np.nanmax(arr, axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def metadata_float(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def analyze_roads(
    *,
    initial_features: dict[str, np.ndarray],
    final_features: dict[str, np.ndarray],
    initial_meta: list[dict[str, Any]],
    final_meta: list[dict[str, Any]],
    ridge_alpha: float,
) -> list[dict[str, Any]]:
    n_road = final_features["reward_mean"].shape[1]
    initial_rho = vector(initial_meta, "rho")
    final_rho = vector(final_meta, "rho")
    row_delta_rho = final_rho[: min(len(final_rho), len(initial_rho))] - initial_rho[: min(len(final_rho), len(initial_rho))]

    final_reward_beta = ridge_coefficients(final_features["reward_mean"], final_rho, ridge_alpha)
    initial_reward_beta = ridge_coefficients(initial_features["reward_mean"], initial_rho, ridge_alpha)

    rows: list[dict[str, Any]] = []
    for road in range(n_road):
        final_reward = final_features["reward_mean"][:, road]
        initial_reward = initial_features["reward_mean"][:, road]
        delta_reward = (
            final_features["reward_mean"][: row_delta_rho.size, road]
            - initial_features["reward_mean"][: row_delta_rho.size, road]
        )
        evidence = [
            abs(safe_spearman(final_reward, final_rho)),
            abs(safe_spearman(initial_reward, initial_rho)),
            abs(safe_spearman(delta_reward, row_delta_rho)),
        ]
        row = {
            "road_index": road,
            "initial_reward_mean": safe_mean(initial_reward),
            "final_reward_mean": safe_mean(final_reward),
            "reward_mean_shift": safe_mean(final_reward) - safe_mean(initial_reward),
            "initial_tau1_mean": safe_mean(initial_features["tau1_mean"][:, road]),
            "final_tau1_mean": safe_mean(final_features["tau1_mean"][:, road]),
            "tau1_mean_shift": safe_mean(final_features["tau1_mean"][:, road])
            - safe_mean(initial_features["tau1_mean"][:, road]),
            "initial_valid_ratio": safe_mean(initial_features["valid_ratio"][:, road]),
            "final_valid_ratio": safe_mean(final_features["valid_ratio"][:, road]),
            "final_reward_rho_pearson": safe_pearson(final_reward, final_rho),
            "final_reward_rho_spearman": safe_spearman(final_reward, final_rho),
            "initial_reward_rho_pearson": safe_pearson(initial_reward, initial_rho),
            "initial_reward_rho_spearman": safe_spearman(initial_reward, initial_rho),
            "row_aligned_delta_reward_delta_rho_spearman": safe_spearman(delta_reward, row_delta_rho),
            "row_aligned_delta_reward_delta_rho_pearson": safe_pearson(delta_reward, row_delta_rho),
            "final_reward_ridge_beta": float(final_reward_beta[road]),
            "initial_reward_ridge_beta": float(initial_reward_beta[road]),
        }
        row["screening_score"] = float(np.nanmean(evidence))
        rows.append(row)

    normalize_beta_score(rows, "final_reward_ridge_beta", "final_reward_ridge_beta_score")
    for row in rows:
        parts = [
            abs(float(row["final_reward_rho_spearman"])),
            abs(float(row["row_aligned_delta_reward_delta_rho_spearman"])),
            float(row["final_reward_ridge_beta_score"]),
        ]
        row["composite_screening_score"] = float(np.nanmean(parts))
    rows.sort(key=lambda item: float(item["composite_screening_score"]), reverse=True)
    return rows


def mutant_child_reward_analysis(
    final_features: dict[str, np.ndarray],
    final_meta: list[dict[str, Any]],
    *,
    ridge_alpha: float,
) -> list[dict[str, Any]]:
    mutant_indices = [idx for idx, row in enumerate(final_meta) if row.get("delta_parent_rho") != ""]
    if len(mutant_indices) < 3:
        return []
    y = np.asarray([float(final_meta[idx]["delta_parent_rho"]) for idx in mutant_indices], dtype=np.float64)
    x = final_features["reward_mean"][mutant_indices]
    beta = ridge_coefficients(x, y, ridge_alpha)
    rows = []
    for road in range(x.shape[1]):
        reward_col = x[:, road]
        rows.append(
            {
                "road_index": road,
                "n_mutants_with_parent_rho": len(mutant_indices),
                "child_reward_vs_delta_parent_rho_pearson": safe_pearson(reward_col, y),
                "child_reward_vs_delta_parent_rho_spearman": safe_spearman(reward_col, y),
                "child_reward_delta_parent_rho_ridge_beta": float(beta[road]),
                "child_reward_mean": safe_mean(reward_col),
                "delta_parent_rho_mean": safe_mean(y),
            }
        )
    normalize_beta_score(rows, "child_reward_delta_parent_rho_ridge_beta", "ridge_beta_score")
    for row in rows:
        row["mutant_screening_score"] = float(
            np.nanmean(
                [
                    abs(float(row["child_reward_vs_delta_parent_rho_spearman"])),
                    float(row["ridge_beta_score"]),
                ]
            )
        )
    rows.sort(key=lambda item: float(item["mutant_screening_score"]), reverse=True)
    return rows


def row_aligned_delta_analysis(
    initial_features: dict[str, np.ndarray],
    final_features: dict[str, np.ndarray],
    initial_meta: list[dict[str, Any]],
    final_meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    n = min(len(initial_meta), len(final_meta))
    if n < 3:
        return []
    y = vector(final_meta[:n], "rho") - vector(initial_meta[:n], "rho")
    rows = []
    for road in range(final_features["reward_mean"].shape[1]):
        dx = final_features["reward_mean"][:n, road] - initial_features["reward_mean"][:n, road]
        rows.append(
            {
                "road_index": road,
                "note": "diagnostic_only_not_lineage",
                "delta_reward_mean": safe_mean(dx),
                "delta_rho_mean": safe_mean(y),
                "delta_reward_delta_rho_pearson": safe_pearson(dx, y),
                "delta_reward_delta_rho_spearman": safe_spearman(dx, y),
                "abs_delta_reward_mean": safe_mean(np.abs(dx)),
            }
        )
    rows.sort(key=lambda item: abs(float(item["delta_reward_delta_rho_spearman"])), reverse=True)
    return rows


def ridge_coefficients(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xz = zscore(np.asarray(x, dtype=np.float64))
    yz = zscore(np.asarray(y, dtype=np.float64).reshape(-1, 1)).reshape(-1)
    if xz.shape[0] < 3 or np.nanstd(yz) == 0:
        return np.zeros(xz.shape[1], dtype=np.float64)
    xtx = xz.T @ xz
    reg = max(float(alpha), 0.0) * np.eye(xtx.shape[0])
    try:
        return np.linalg.solve(xtx + reg, xz.T @ yz)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + reg) @ xz.T @ yz


def normalize_beta_score(rows: list[dict[str, Any]], source_key: str, target_key: str) -> None:
    if not rows:
        return
    max_abs = max(abs(float(row[source_key])) for row in rows)
    denom = max(max_abs, 1e-12)
    for row in rows:
        row[target_key] = abs(float(row[source_key])) / denom


def make_charts(
    road_rows: list[dict[str, Any]],
    mutant_rows: list[dict[str, Any]],
    row_delta_rows: list[dict[str, Any]],
    charts_dir: Path,
    *,
    top_k: int,
) -> list[Path]:
    paths = [
        plot_top_roads(
            road_rows,
            charts_dir / "road_importance_top_roads.png",
            score_key="composite_screening_score",
            title="Top road screening scores",
            top_k=top_k,
        ),
        plot_metric_by_road(
            sorted(road_rows, key=lambda row: int(row["road_index"])),
            charts_dir / "road_reward_rho_spearman.png",
            metric_key="final_reward_rho_spearman",
            title="Final reward_mean vs rho Spearman by road",
        ),
        plot_metric_by_road(
            sorted(road_rows, key=lambda row: int(row["road_index"])),
            charts_dir / "road_reward_mean_shift.png",
            metric_key="reward_mean_shift",
            title="Road reward_mean shift: final - initial",
        ),
    ]
    if mutant_rows:
        paths.append(
            plot_top_roads(
                mutant_rows,
                charts_dir / "mutant_parent_delta_top_roads.png",
                score_key="mutant_screening_score",
                title="Mutant child reward vs parent delta_rho",
                top_k=top_k,
            )
        )
    if row_delta_rows:
        paths.append(
            plot_top_roads(
                row_delta_rows,
                charts_dir / "row_aligned_delta_top_roads.png",
                score_key="delta_reward_delta_rho_spearman",
                title="Row-aligned delta diagnostic",
                top_k=top_k,
                absolute=True,
            )
        )
    return paths


def plot_top_roads(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    score_key: str,
    title: str,
    top_k: int,
    absolute: bool = False,
) -> Path:
    selected = rows[: max(1, int(top_k))]
    labels = [str(int(row["road_index"])) for row in selected]
    values = [float(row[score_key]) for row in selected]
    if absolute:
        values = [abs(value) for value in values]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values, color="#4c72b0")
    ax.set_title(title)
    ax.set_xlabel("road_index")
    ax.set_ylabel(score_key)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_metric_by_road(rows: list[dict[str, Any]], path: Path, *, metric_key: str, title: str) -> Path:
    x = [int(row["road_index"]) for row in rows]
    y = [float(row[metric_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, y, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("road_index")
    ax.set_ylabel(metric_key)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_report(
    *,
    initial: Path,
    final: Path,
    output_dir: Path,
    road_rows: list[dict[str, Any]],
    mutant_rows: list[dict[str, Any]],
    row_delta_rows: list[dict[str, Any]],
    final_meta: list[dict[str, Any]],
    chart_paths: list[Path],
    runtime_seconds: float,
) -> dict[str, Any]:
    n_mutants = sum(1 for row in final_meta if row.get("is_mutant"))
    top_roads = road_rows[:10]
    top_mutant_roads = mutant_rows[:10]
    headline = {
        "n_roads": len(road_rows),
        "n_final_individuals": len(final_meta),
        "n_final_mutants_with_parent_rho": n_mutants,
        "top_composite_road_indices": [int(row["road_index"]) for row in top_roads],
        "top_mutant_delta_road_indices": [int(row["road_index"]) for row in top_mutant_roads],
    }
    return {
        "inputs": {"initial": str(initial), "final": str(final), "output_dir": str(output_dir)},
        "headline": headline,
        "top_roads": top_roads,
        "top_mutant_roads": top_mutant_roads,
        "top_row_aligned_delta_roads": row_delta_rows[:10],
        "methods": [
            "Computed per-road reward/tau/valid features by averaging over valid car positions.",
            "Estimated road association with rho using Pearson, Spearman, and ridge coefficients.",
            "Used final mutant parent_rho metadata when available; this tests child road reward levels against child-parent delta_rho, not reward deltas.",
            "Computed row-aligned initial/final reward deltas only as a diagnostic because final rows are selected top-M population entries, not guaranteed descendants.",
            "True causal importance requires counterfactual SUMO reruns with a single road reward column perturbed and DQN retrained.",
        ],
        "charts": [str(path) for path in chart_paths],
        "runtime_seconds": round(float(runtime_seconds), 3),
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Road Importance Analysis",
        "",
        "## Headline",
        "",
        f"- Roads: {report['headline']['n_roads']}",
        f"- Final individuals: {report['headline']['n_final_individuals']}",
        f"- Final mutants with parent_rho: {report['headline']['n_final_mutants_with_parent_rho']}",
        f"- Top composite roads: {report['headline']['top_composite_road_indices']}",
        f"- Top mutant-delta roads: {report['headline']['top_mutant_delta_road_indices']}",
        "",
        "## Methods",
        "",
    ]
    for item in report["methods"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Charts", ""])
    for chart in report["charts"]:
        lines.append(f"- `{chart}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def vector(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, "")
        values.append(float(value) if value != "" and value is not None else np.nan)
    return np.asarray(values, dtype=np.float64)


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return np.nan_to_num((arr - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size else 0.0


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[mask]
    bb = bb[mask]
    if aa.size < 3 or np.nanstd(aa) < 1e-12 or np.nanstd(bb) < 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[mask]
    bb = bb[mask]
    if aa.size < 3:
        return 0.0
    return safe_pearson(rankdata(aa), rankdata(bb))


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


if __name__ == "__main__":
    main()
