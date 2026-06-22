"""Export the compact rho prediction dataset to CSV files."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
from typing import Any, Iterable, TextIO

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sumo_ryl_prediction_dataset.pt to CSV.")
    parser.add_argument("--input", default="dataset/sumo_ryl_prediction_dataset.pt", help="Input .pt dataset path.")
    parser.add_argument("--output-dir", default="outputs/dataset_csv", help="Directory for exported CSV files.")
    parser.add_argument(
        "--format",
        choices=["summary", "long", "both"],
        default="summary",
        help="summary writes one row per individual; long writes one row per vehicle-road pair.",
    )
    parser.add_argument(
        "--long-gzip",
        action="store_true",
        help="Write long CSV as .csv.gz. Recommended for the full dataset.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit for quick exports.")
    parser.add_argument(
        "--long-valid-only",
        action="store_true",
        help="For long export, write only valid finite vehicle-road rows.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(input_path)
    sample_count = len(dataset["rho"])
    if args.max_samples is not None:
        sample_count = min(sample_count, max(0, args.max_samples))

    if args.format in {"summary", "both"}:
        summary_path = output_dir / "sumo_ryl_prediction_summary.csv"
        export_summary(dataset, sample_count, summary_path)
        print(f"Saved summary CSV: {summary_path}")

    if args.format in {"long", "both"}:
        suffix = ".csv.gz" if args.long_gzip else ".csv"
        long_path = output_dir / f"sumo_ryl_prediction_long{suffix}"
        export_long(dataset, sample_count, long_path, valid_only=args.long_valid_only)
        print(f"Saved long CSV: {long_path}")


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}")
    for key in ("tau", "rewards", "rho"):
        if key not in payload:
            raise KeyError(f"Dataset is missing required key: {key}")
    return payload


def export_summary(dataset: dict[str, Any], sample_count: int, path: Path) -> None:
    fieldnames = [
        "index",
        "group",
        "rho",
        "n_vehicle",
        "n_road",
        "tau_dim",
        "valid_count",
        "valid_ratio",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
        "reward_q25",
        "reward_median",
        "reward_q75",
        "tau0_mean",
        "tau0_std",
        "tau0_min",
        "tau0_max",
        "tau0_q25",
        "tau0_median",
        "tau0_q75",
        "tau1_mean",
        "tau1_std",
        "tau1_min",
        "tau1_max",
        "tau1_q25",
        "tau1_median",
        "tau1_q75",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(sample_count):
            tau, rewards, rho, group = get_sample(dataset, idx)
            mask = finite_mask(tau, rewards)
            n_vehicle, n_road, tau_dim = tau.shape
            row = {
                "index": idx,
                "group": "" if group is None else group,
                "rho": float(rho),
                "n_vehicle": int(n_vehicle),
                "n_road": int(n_road),
                "tau_dim": int(tau_dim),
                "valid_count": int(mask.sum().item()),
                "valid_ratio": float(mask.float().mean().item()),
            }
            row.update(prefix_stats("reward", rewards[mask]))
            row.update(prefix_stats("tau0", tau[..., 0][mask]))
            if tau_dim > 1:
                row.update(prefix_stats("tau1", tau[..., 1][mask]))
            else:
                row.update(prefix_stats("tau1", torch.empty(0)))
            writer.writerow(row)


def export_long(dataset: dict[str, Any], sample_count: int, path: Path, *, valid_only: bool) -> None:
    with open_text(path) as f:
        writer = csv.writer(f)
        writer.writerow(["index", "group", "rho", "vehicle", "road", "tau0", "tau1", "reward", "valid"])
        for idx in range(sample_count):
            tau, rewards, rho, group = get_sample(dataset, idx)
            mask = finite_mask(tau, rewards)
            n_vehicle, n_road, tau_dim = tau.shape
            group_value = "" if group is None else group
            rho_value = float(rho)
            for vehicle in range(n_vehicle):
                for road in range(n_road):
                    valid = bool(mask[vehicle, road].item())
                    if valid_only and not valid:
                        continue
                    tau0 = safe_float(tau[vehicle, road, 0])
                    tau1 = safe_float(tau[vehicle, road, 1]) if tau_dim > 1 else ""
                    reward = safe_float(rewards[vehicle, road])
                    writer.writerow([idx, group_value, rho_value, vehicle, road, tau0, tau1, reward, int(valid)])


def get_sample(dataset: dict[str, Any], idx: int) -> tuple[torch.Tensor, torch.Tensor, float, Any]:
    tau_store = dataset["tau"]
    rewards_store = dataset["rewards"]
    groups = dataset.get("groups")
    tau = tau_store[idx] if isinstance(tau_store, list) else tau_store[idx]
    rewards = rewards_store[idx] if isinstance(rewards_store, list) else rewards_store[idx]
    rho = dataset["rho"][idx]
    group = None if groups is None else groups[idx]
    return (
        torch.as_tensor(tau, dtype=torch.float32),
        torch.as_tensor(rewards, dtype=torch.float32),
        float(torch.as_tensor(rho).item()),
        group,
    )


def finite_mask(tau: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(rewards) & torch.isfinite(tau).all(dim=-1)


def prefix_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    values = values[torch.isfinite(values)].float()
    if values.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_q75": 0.0,
        }
    quantiles = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75]))
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
        f"{prefix}_q25": float(quantiles[0].item()),
        f"{prefix}_median": float(quantiles[1].item()),
        f"{prefix}_q75": float(quantiles[2].item()),
    }


def safe_float(value: torch.Tensor) -> float | str:
    value = torch.as_tensor(value)
    return float(value.item()) if torch.isfinite(value).item() else ""


def open_text(path: Path) -> Iterable[TextIO]:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


if __name__ == "__main__":
    main()
