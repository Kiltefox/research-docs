"""Counterfactual road-reward intervention runner.

This script lives in ``prediction_model`` but imports MASDiff modules from a
sibling checkout.  It does not modify MASDiff files.

For a selected individual and road_index, the script perturbs one reward column,
rebuilds DQN training data, retrains policies, reruns SUMO, and recomputes rho.
That is the causal estimand we need for "if this road's R changes, how does rho
change?"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
import types
from dataclasses import dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal road reward interventions through MASDiff/SUMO.")
    parser.add_argument("--masdiff-root", default="", help="Path to sibling MASDiff checkout.")
    parser.add_argument(
        "--config",
        default="",
        help="MASDiff YAML config used to instantiate Q/environment/DQN/metric.",
    )
    parser.add_argument("--population", default="dataset/final_population.pkl")
    parser.add_argument("--output-dir", default="dataset/road_causal_intervention")
    parser.add_argument("--roads", nargs="+", type=int, default=[12, 24, 67])
    parser.add_argument("--all-roads", action="store_true", help="Override --roads and intervene on every road column.")
    parser.add_argument("--individual-indices", nargs="*", type=int, default=[])
    parser.add_argument("--top-n-by-rho", type=int, default=3, help="Used when --individual-indices is omitted.")
    parser.add_argument("--mode", choices=["add", "scale", "set"], default="add")
    parser.add_argument("--values", nargs="+", type=float, default=[-0.5, 0.5])
    parser.add_argument("--clamp-min", type=float, default=0.0)
    parser.add_argument("--clamp-max", type=float, default=5.0)
    parser.add_argument("--dqn-device", default="", help="Optional override for dqn_module.cfg.device.")
    parser.add_argument("--dqn-epochs", type=int, default=0, help="Optional override for dqn_module.cfg.epochs.")
    parser.add_argument("--recompute-baseline", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip completed road/index/mode/value rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without running DQN/SUMO interventions.")
    args = parser.parse_args()

    repo_cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_masdiff_root = repo_root.parent / "MASDiff"
    masdiff_root = resolve_path(args.masdiff_root, base=repo_root.parent) if args.masdiff_root else default_masdiff_root
    config_path = (
        resolve_path(args.config, base=repo_root)
        if args.config
        else masdiff_root / "experiments" / "sumo_ryl_transfer" / "source_train.yaml"
    )
    population_path = resolve_path(args.population, base=script_dir)
    output_dir = resolve_path(args.output_dir, base=script_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    add_masdiff_to_path(masdiff_root)
    set_seeds(args.seed)

    # MASDiff configs use many relative paths (maps, q cache, outputs), so run
    # MASDiff code from its own root while keeping all prediction_model paths
    # resolved absolutely.
    os.chdir(masdiff_root)
    try:
        cfg, q_provider, environment, dqn_module, metric = load_masdiff_components(config_path)
        apply_dqn_overrides(dqn_module, device=args.dqn_device, epochs=args.dqn_epochs)
        population = load_population(population_path)
        selected_indices = choose_individuals(population, args.individual_indices, args.top_n_by_rho)
        roads = all_road_indices(population) if args.all_roads else list(args.roads)
        validate_roads(population, roads)
        validate_experience_buffers(population, selected_indices)

        setup = {
            "masdiff_root": str(masdiff_root),
            "config": str(config_path),
            "population": str(population_path),
            "output_dir": str(output_dir),
            "roads": roads,
            "individual_indices": selected_indices,
            "mode": args.mode,
            "values": args.values,
            "recompute_baseline": bool(args.recompute_baseline),
            "dqn_device_override": args.dqn_device,
            "dqn_epochs_override": int(args.dqn_epochs),
            "dry_run": bool(args.dry_run),
            "algorithm_N": int(cfg.algorithm.N),
        }
        (output_dir / "intervention_setup.json").write_text(json.dumps(setup, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(setup, indent=2, ensure_ascii=False))

        if args.dry_run:
            print("dry_run=true; no DQN training or SUMO simulation was executed.")
            return

        q = q_provider.load_or_create_q(environment)
        results_csv = output_dir / "road_intervention_results.csv"
        completed = load_completed_keys(results_csv) if args.resume else set()

        baseline_cache: dict[int, float] = {}
        ensure_results_header(results_csv)
        for individual_index in selected_indices:
            individual = population[individual_index]
            stored_rho = float(getattr(individual, "rho", 0.0))
            baseline_rho = ""
            if args.recompute_baseline:
                baseline_rho = recompute_rho(
                    individual,
                    rewards=torch.as_tensor(individual.rewards, dtype=torch.float32).clone(),
                    dqn_module=dqn_module,
                    environment=environment,
                    metric=metric,
                    q=q,
                )
                baseline_cache[individual_index] = float(baseline_rho)

            for road in roads:
                for value in args.values:
                    key = (individual_index, road, args.mode, float(value))
                    if key in completed:
                        print(f"skip completed {key}")
                        continue
                    row = run_intervention(
                        individual=individual,
                        individual_index=individual_index,
                        road_index=road,
                        mode=args.mode,
                        value=float(value),
                        clamp_min=float(args.clamp_min),
                        clamp_max=float(args.clamp_max),
                        stored_rho=stored_rho,
                        baseline_rho=baseline_cache.get(individual_index, baseline_rho),
                        dqn_module=dqn_module,
                        environment=environment,
                        metric=metric,
                        q=q,
                    )
                    append_result(results_csv, row)
                    print(json.dumps(row, ensure_ascii=False))

        summary_rows = summarize_results(read_result_rows(results_csv))
        write_csv(output_dir / "road_intervention_summary.csv", summary_rows)
        importance_rows = aggregate_road_importance(read_result_rows(results_csv))
        write_csv(output_dir / "road_causal_importance.csv", importance_rows)
        if summary_rows:
            plot_summary(summary_rows, output_dir / "road_intervention_effects.png")
        if importance_rows:
            plot_importance_ranking(importance_rows, output_dir / "road_causal_importance_ranking.png")
        print(f"saved={output_dir}")
    finally:
        os.chdir(repo_cwd)


def add_masdiff_to_path(masdiff_root: Path) -> None:
    if not masdiff_root.exists():
        raise FileNotFoundError(f"MASDiff root not found: {masdiff_root}")
    root_str = str(masdiff_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def set_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_masdiff_components(config_path: Path) -> tuple[Any, Any, Any, Any, Any]:
    from src.config.loader import load_config
    from src.utils.import_utils import instantiate

    cfg = load_config(config_path)
    q_provider = instantiate(cfg.q_provider)
    environment = instantiate(cfg.environment)
    dqn_module = instantiate(cfg.dqn_module)
    metric = instantiate(cfg.metric)
    return cfg, q_provider, environment, dqn_module, metric


def apply_dqn_overrides(dqn_module: Any, *, device: str, epochs: int) -> None:
    cfg = getattr(dqn_module, "cfg", None)
    if cfg is None:
        return
    updates: dict[str, Any] = {}
    if device:
        updates["device"] = str(device)
    if int(epochs) > 0:
        updates["epochs"] = int(epochs)
    if not updates:
        return
    if is_dataclass(cfg):
        dqn_module.cfg = replace(cfg, **updates)
    else:
        for key, value in updates.items():
            setattr(cfg, key, value)


def load_population(path: Path) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    install_population_pickle_compat()
    start = time.time()
    with path.open("rb") as f:
        population = pickle.load(f)
    if not isinstance(population, list):
        raise TypeError(f"population pickle must contain list, got {type(population).__name__}")
    print(f"Loaded {len(population)} individuals from {path} in {time.time() - start:.1f}s")
    return population


@dataclass
class PickleIndividual:
    tau: Any = None
    rewards: Any = None
    rho: float = 0.0
    experience_buffers: list[list[Any]] = field(default_factory=list)
    policies: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.tau = state.get("tau")
        self.rewards = state.get("rewards")
        self.rho = float(state.get("rho", 0.0))
        self.experience_buffers = state.get("experience_buffers", []) or []
        self.policies = []
        metadata = state.get("metadata", {}) or {}
        self.metadata = metadata if isinstance(metadata, dict) else {}


def install_population_pickle_compat() -> None:
    """Register only src.pipeline.types to avoid importing MASDiff pipeline package."""

    PickleIndividual.__module__ = "src.pipeline.types"
    src_mod = sys.modules.get("src")
    if src_mod is None:
        src_mod = types.ModuleType("src")
        src_mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules["src"] = src_mod

    pipeline_mod = types.ModuleType("src.pipeline")
    pipeline_mod.__path__ = []  # type: ignore[attr-defined]
    types_mod = types.ModuleType("src.pipeline.types")
    types_mod.Individual = PickleIndividual
    types_mod.Population = list

    setattr(pipeline_mod, "types", types_mod)
    setattr(src_mod, "pipeline", pipeline_mod)
    sys.modules["src.pipeline"] = pipeline_mod
    sys.modules["src.pipeline.types"] = types_mod


def choose_individuals(population: list[Any], explicit_indices: list[int], top_n_by_rho: int) -> list[int]:
    if explicit_indices:
        indices = [int(idx) for idx in explicit_indices]
    else:
        ranked = sorted(range(len(population)), key=lambda idx: float(getattr(population[idx], "rho", 0.0)), reverse=True)
        indices = ranked[: max(1, int(top_n_by_rho))]
    for idx in indices:
        if idx < 0 or idx >= len(population):
            raise IndexError(f"individual index out of range: {idx}")
    return indices


def validate_roads(population: list[Any], roads: list[int]) -> None:
    if not population:
        raise ValueError("population is empty")
    rewards = torch.as_tensor(population[0].rewards)
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be [car,road], got {tuple(rewards.shape)}")
    n_road = int(rewards.shape[1])
    for road in roads:
        if road < 0 or road >= n_road:
            raise IndexError(f"road_index {road} out of range [0,{n_road})")


def all_road_indices(population: list[Any]) -> list[int]:
    if not population:
        raise ValueError("population is empty")
    rewards = torch.as_tensor(population[0].rewards)
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be [car,road], got {tuple(rewards.shape)}")
    return list(range(int(rewards.shape[1])))


def validate_experience_buffers(population: list[Any], indices: list[int]) -> None:
    missing = []
    for idx in indices:
        buffers = getattr(population[idx], "experience_buffers", None)
        if not buffers:
            missing.append(idx)
    if missing:
        raise ValueError(
            "Selected individuals have empty experience_buffers. "
            "Load with real MASDiff classes, not prediction_model.types_compat. "
            f"Missing indices: {missing}"
        )


def run_intervention(
    *,
    individual: Any,
    individual_index: int,
    road_index: int,
    mode: str,
    value: float,
    clamp_min: float,
    clamp_max: float,
    stored_rho: float,
    baseline_rho: float | str,
    dqn_module: Any,
    environment: Any,
    metric: Any,
    q: Any,
) -> dict[str, Any]:
    rewards = torch.as_tensor(individual.rewards, dtype=torch.float32).clone()
    before = rewards[:, road_index].detach().float().cpu()
    modified = apply_intervention(rewards, road_index, mode, value, clamp_min, clamp_max)
    after = modified[:, road_index].detach().float().cpu()
    start = time.time()
    rho = recompute_rho(
        individual,
        rewards=modified,
        dqn_module=dqn_module,
        environment=environment,
        metric=metric,
        q=q,
    )
    runtime = time.time() - start
    base_for_effect = float(baseline_rho) if baseline_rho != "" else stored_rho
    return {
        "individual_index": int(individual_index),
        "road_index": int(road_index),
        "mode": mode,
        "value": float(value),
        "stored_rho": float(stored_rho),
        "baseline_rho": baseline_rho,
        "intervened_rho": float(rho),
        "effect_vs_stored": float(rho) - float(stored_rho),
        "effect_vs_baseline": float(rho) - float(base_for_effect),
        "reward_before_mean": float(before.mean().item()),
        "reward_after_mean": float(after.mean().item()),
        "reward_before_std": float(before.std(unbiased=False).item()),
        "reward_after_std": float(after.std(unbiased=False).item()),
        "runtime_seconds": round(runtime, 3),
    }


def apply_intervention(
    rewards: torch.Tensor,
    road_index: int,
    mode: str,
    value: float,
    clamp_min: float,
    clamp_max: float,
) -> torch.Tensor:
    out = rewards.clone()
    if mode == "add":
        out[:, road_index] = out[:, road_index] + float(value)
    elif mode == "scale":
        out[:, road_index] = out[:, road_index] * float(value)
    elif mode == "set":
        out[:, road_index] = float(value)
    else:  # pragma: no cover
        raise ValueError(f"unknown mode={mode}")
    out[:, road_index] = torch.clamp(out[:, road_index], min=float(clamp_min), max=float(clamp_max))
    return out


def recompute_rho(
    individual: Any,
    *,
    rewards: torch.Tensor,
    dqn_module: Any,
    environment: Any,
    metric: Any,
    q: Any,
) -> float:
    training_data = dqn_module.build_training_data(individual.experience_buffers, rewards)
    policies = dqn_module.train_per_agent(training_data)
    simulation_data = environment.simulate_evaluate(policies)
    return float(metric.compute_rho(q, simulation_data))


def load_completed_keys(path: Path) -> set[tuple[int, int, str, float]]:
    if not path.exists():
        return set()
    rows = read_result_rows(path)
    return {
        (int(row["individual_index"]), int(row["road_index"]), str(row["mode"]), float(row["value"]))
        for row in rows
    }


def ensure_results_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    write_csv(path, [])


def append_result(path: Path, row: dict[str, Any]) -> None:
    fieldnames = result_fieldnames()
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["road_index"]), str(row["mode"]), float(row["value"]))
        grouped.setdefault(key, []).append(row)
    summary = []
    for (road, mode, value), items in sorted(grouped.items()):
        effects = np.asarray([float(item["effect_vs_baseline"]) for item in items], dtype=np.float64)
        summary.append(
            {
                "road_index": road,
                "mode": mode,
                "value": value,
                "n": int(effects.size),
                "effect_mean": float(np.mean(effects)) if effects.size else 0.0,
                "effect_std": float(np.std(effects)) if effects.size else 0.0,
                "effect_median": float(np.median(effects)) if effects.size else 0.0,
                "positive_ratio": float(np.mean(effects > 0)) if effects.size else 0.0,
                "negative_ratio": float(np.mean(effects < 0)) if effects.size else 0.0,
            }
        )
    return summary


def aggregate_road_importance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["road_index"]), []).append(row)
    output = []
    for road, items in grouped.items():
        effects = np.asarray([float(item["effect_vs_baseline"]) for item in items], dtype=np.float64)
        values = np.asarray([float(item["value"]) for item in items], dtype=np.float64)
        pos_mask = values > 0
        neg_mask = values < 0
        plus_mean = float(np.mean(effects[pos_mask])) if np.any(pos_mask) else ""
        minus_mean = float(np.mean(effects[neg_mask])) if np.any(neg_mask) else ""
        directional_slope = ""
        if np.any(pos_mask) and np.any(neg_mask):
            value_span = float(np.mean(values[pos_mask]) - np.mean(values[neg_mask]))
            if abs(value_span) > 1e-12:
                directional_slope = (float(plus_mean) - float(minus_mean)) / value_span
        output.append(
            {
                "road_index": road,
                "n": int(effects.size),
                "importance_abs_effect_mean": float(np.mean(np.abs(effects))) if effects.size else 0.0,
                "importance_abs_effect_max": float(np.max(np.abs(effects))) if effects.size else 0.0,
                "signed_effect_mean": float(np.mean(effects)) if effects.size else 0.0,
                "positive_ratio": float(np.mean(effects > 0)) if effects.size else 0.0,
                "negative_ratio": float(np.mean(effects < 0)) if effects.size else 0.0,
                "plus_value_effect_mean": plus_mean,
                "minus_value_effect_mean": minus_mean,
                "directional_slope": directional_slope,
            }
        )
    output.sort(key=lambda row: float(row["importance_abs_effect_mean"]), reverse=True)
    return output


def plot_summary(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [f"{row['road_index']}:{row['value']:g}" for row in rows]
    values = [float(row["effect_mean"]) for row in rows]
    colors = ["#c44e52" if value < 0 else "#55a868" for value in values]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.7), 4.8))
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Causal road reward intervention effects")
    ax.set_xlabel("road_index:intervention_value")
    ax.set_ylabel("mean rho effect vs recomputed baseline")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_importance_ranking(rows: list[dict[str, Any]], path: Path, top_k: int = 30) -> None:
    selected = rows[: min(int(top_k), len(rows))]
    labels = [str(int(row["road_index"])) for row in selected]
    values = [float(row["importance_abs_effect_mean"]) for row in selected]
    fig, ax = plt.subplots(figsize=(max(10, len(selected) * 0.45), 5.2))
    ax.bar(labels, values, color="#4c72b0")
    ax.set_title("Road causal importance ranking")
    ax.set_xlabel("road_index")
    ax.set_ylabel("mean abs rho effect vs recomputed baseline")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else result_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def result_fieldnames() -> list[str]:
    return [
        "individual_index",
        "road_index",
        "mode",
        "value",
        "stored_rho",
        "baseline_rho",
        "intervened_rho",
        "effect_vs_stored",
        "effect_vs_baseline",
        "reward_before_mean",
        "reward_after_mean",
        "reward_before_std",
        "reward_after_std",
        "runtime_seconds",
    ]


if __name__ == "__main__":
    main()
