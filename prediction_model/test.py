"""Evaluate a trained rho prediction checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .data import Normalizer, RhoTensorDataset, SubsetDataset, make_collate_fn, save_json
    from .metrics import prediction_interval, regression_metrics
    from .model import ModelConfig, RhoPredictionModel
except ImportError:
    from data import Normalizer, RhoTensorDataset, SubsetDataset, make_collate_fn, save_json
    from metrics import prediction_interval, regression_metrics
    from model import ModelConfig, RhoPredictionModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Test rho prediction model.")
    parser.add_argument("--data", default="dataset/sumo_ryl_prediction_dataset.pt")
    parser.add_argument("--checkpoint", default="runs/rho_prediction/best.pt")
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--output-dir", default="outputs/rho_prediction")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalizer = Normalizer.from_dict(checkpoint["normalizer"])
    config = ModelConfig.from_dict(checkpoint["model_config"])
    dataset = RhoTensorDataset(args.data)
    indices = list(range(len(dataset))) if args.split == "all" else checkpoint["splits"][args.split]

    loader = DataLoader(
        SubsetDataset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(normalizer, augment=False),
    )

    device = torch.device(args.device)
    model = RhoPredictionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pred_parts: dict[str, list[torch.Tensor]] = {"mu": [], "sigma": [], "nu": []}
    target_scaled: list[torch.Tensor] = []
    target_raw: list[torch.Tensor] = []
    sample_indices: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            moved = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                pred = model(moved["tau"], moved["rewards"], moved["mask"], moved["stats"])
            for key in pred_parts:
                pred_parts[key].append(pred[key].detach().cpu())
            target_scaled.append(batch["rho"].detach().cpu())
            target_raw.append(batch["rho_raw"].detach().cpu())
            sample_indices.append(batch["indices"].detach().cpu())

    pred_cat = {key: torch.cat(parts, dim=0) for key, parts in pred_parts.items()}
    target_scaled_cat = torch.cat(target_scaled)
    target_raw_cat = torch.cat(target_raw)
    metrics = regression_metrics(pred_cat, target_scaled_cat, target_raw_cat, normalizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / f"{args.split}_metrics.json", metrics)
    _write_predictions(
        output_dir / f"{args.split}_predictions.csv",
        torch.cat(sample_indices),
        pred_cat,
        target_raw_cat,
        normalizer,
    )
    print(metrics)


def _write_predictions(
    path: Path,
    indices: torch.Tensor,
    pred: dict[str, torch.Tensor],
    target_raw: torch.Tensor,
    normalizer: Normalizer,
) -> None:
    lower, upper = prediction_interval(pred, normalizer)
    mu_raw = normalizer.inverse_rho(pred["mu"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "rho_true", "rho_pred", "rho_lower_90", "rho_upper_90", "sigma_scaled", "nu"],
        )
        writer.writeheader()
        for row in range(indices.numel()):
            writer.writerow(
                {
                    "index": int(indices[row].item()),
                    "rho_true": float(target_raw[row].item()),
                    "rho_pred": float(mu_raw[row].item()),
                    "rho_lower_90": float(lower[row].item()),
                    "rho_upper_90": float(upper[row].item()),
                    "sigma_scaled": float(pred["sigma"][row].item()),
                    "nu": float(pred["nu"][row].item()),
                }
            )


if __name__ == "__main__":
    main()
