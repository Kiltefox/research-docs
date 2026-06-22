"""Train and test a plain MLP baseline on the full flattened input."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from .data import Normalizer, RhoTensorDataset, finite_mask, save_json, split_indices
except ImportError:
    from data import Normalizer, RhoTensorDataset, finite_mask, save_json, split_indices


@dataclass
class BaselineConfig:
    input_dim: int
    hidden_dims: list[int]
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineConfig":
        return cls(
            input_dim=int(data["input_dim"]),
            hidden_dims=[int(value) for value in data["hidden_dims"]],
            dropout=float(data.get("dropout", 0.1)),
        )


class FullInputMLP(nn.Module):
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FullInputDataset(Dataset):
    def __init__(self, dataset: RhoTensorDataset, indices: list[int], normalizer: Normalizer) -> None:
        self.dataset = dataset
        self.indices = indices
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int) -> dict[str, torch.Tensor]:
        idx = self.indices[row]
        sample = self.dataset[idx]
        raw_tau = sample["tau"].float()
        raw_rewards = sample["rewards"].float()
        valid = finite_mask(raw_tau, raw_rewards)

        clean_tau = torch.nan_to_num(raw_tau, nan=0.0, posinf=0.0, neginf=0.0)
        clean_rewards = torch.nan_to_num(raw_rewards, nan=0.0, posinf=0.0, neginf=0.0)
        tau = self.normalizer.transform_tau(clean_tau)
        rewards = self.normalizer.transform_rewards(clean_rewards)

        x = torch.cat(
            [
                tau.reshape(-1),
                rewards.reshape(-1),
                valid.float().reshape(-1),
            ],
            dim=0,
        ).float()

        rho_raw = sample["rho"].float().reshape(())
        rho = self.normalizer.transform_rho(rho_raw)
        return {
            "x": x,
            "rho": rho.float().reshape(()),
            "rho_raw": rho_raw,
            "index": torch.tensor(idx, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-input MLP baseline for rho prediction.")
    parser.add_argument("--mode", choices=["train", "test", "all"], default="all")
    parser.add_argument("--data", default="dataset/sumo_ryl_prediction_dataset.pt")
    parser.add_argument("--reference-config", default="runs/rho_prediction/config.json")
    parser.add_argument("--output-dir", default="runs/full_input_baseline")
    parser.add_argument("--test-output-dir", default="outputs/full_input_baseline")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--group-split", action="store_true")
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="Use the first N training samples for train/valid/test to debug memorization.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        train(args)
    if args.mode in ("test", "all"):
        checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.output_dir) / "best.pt"
        test(args, checkpoint)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    dataset = RhoTensorDataset(args.data)
    splits, normalizer = load_reference(args.reference_config, dataset, args.seed, args.group_split)
    if args.overfit_samples > 0:
        overfit = splits["train"][: args.overfit_samples]
        if not overfit:
            raise ValueError("--overfit-samples requires a non-empty training split.")
        splits = {"train": overfit, "valid": overfit, "test": overfit}
    input_dim = full_input_dim(dataset)
    config = BaselineConfig(input_dim=input_dim, hidden_dims=args.hidden_dims, dropout=args.dropout)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "config.json",
        {
            "args": vars(args),
            "model": config.to_dict(),
            "normalizer": normalizer.to_dict(),
            "splits": splits,
            "sample_shape": list(dataset.sample_shape),
            "input_parts": ["tau", "rewards", "valid_mask"],
        },
    )

    train_loader = make_loader(dataset, splits["train"], normalizer, args, shuffle=True)
    valid_loader = make_loader(dataset, splits["valid"], normalizer, args, shuffle=False)

    device = torch.device(args.device)
    model = FullInputMLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_valid = float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    start = time.time()
    show_progress = not args.no_progress

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, args, show_progress, epoch)
        valid = evaluate(model, valid_loader, device, normalizer, args.amp, show_progress, f"epoch {epoch:03d} valid")
        row = {"epoch": epoch, "train_mse": train_loss, **{f"valid_{key}": value for key, value in valid.items()}}
        history.append(row)
        write_history(output_dir / "history.csv", history)
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} "
            f"valid_rmse={valid['rmse']:.6f} valid_mae={valid['mae']:.6f}"
        )

        if valid["rmse"] < best_valid:
            best_valid = valid["rmse"]
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, config, normalizer, splits, args, epoch, valid)
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    save_checkpoint(output_dir / "last.pt", model, config, normalizer, splits, args, epoch, valid)
    elapsed = (time.time() - start) / 60.0
    print(f"Training complete in {elapsed:.1f} min. Best valid RMSE: {best_valid:.6f}")


def test(args: argparse.Namespace, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset = RhoTensorDataset(args.data)
    normalizer = Normalizer.from_dict(checkpoint["normalizer"])
    config = BaselineConfig.from_dict(checkpoint["model_config"])
    indices = checkpoint["splits"]["test"]

    loader = make_loader(dataset, indices, normalizer, args, shuffle=False)
    device = torch.device(args.device)
    model = FullInputMLP(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    metrics, pred, target, sample_indices = predict(model, loader, device, normalizer, args.amp, not args.no_progress, "test")

    output_dir = Path(args.test_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "test_metrics.json", metrics)
    write_predictions(output_dir / "test_predictions.csv", sample_indices, target, pred)
    print(json.dumps(metrics, indent=2))
    print(f"saved={output_dir}")


def load_reference(
    reference_config: str | Path,
    dataset: RhoTensorDataset,
    seed: int,
    group_split: bool,
) -> tuple[dict[str, list[int]], Normalizer]:
    path = Path(reference_config)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        return config["splits"], Normalizer.from_dict(config["normalizer"])

    splits = split_indices(dataset, seed=seed, group_split=group_split)
    return splits, Normalizer.fit(dataset, splits["train"])


def full_input_dim(dataset: RhoTensorDataset) -> int:
    n_vehicle, n_road, tau_dim = dataset.sample_shape
    return n_vehicle * n_road * (tau_dim + 2)


def make_loader(
    dataset: RhoTensorDataset,
    indices: list[int],
    normalizer: Normalizer,
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        FullInputDataset(dataset, indices, normalizer),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: FullInputMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    show_progress: bool,
    epoch: int,
) -> float:
    model.train()
    total = 0.0
    count = 0
    iterator = progress(loader, f"epoch {epoch:03d} train", show_progress)
    for batch in iterator:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            pred = model(batch["x"])
            loss = torch.nn.functional.mse_loss(pred.float(), batch["rho"].float())
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item()) * int(batch["rho"].numel())
        count += int(batch["rho"].numel())
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(mse=f"{total / max(count, 1):.6f}")
    return total / max(count, 1)


@torch.no_grad()
def evaluate(
    model: FullInputMLP,
    loader: DataLoader,
    device: torch.device,
    normalizer: Normalizer,
    amp: bool,
    show_progress: bool,
    desc: str,
) -> dict[str, float]:
    metrics, _, _, _ = predict(model, loader, device, normalizer, amp, show_progress, desc)
    return metrics


@torch.no_grad()
def predict(
    model: FullInputMLP,
    loader: DataLoader,
    device: torch.device,
    normalizer: Normalizer,
    amp: bool,
    show_progress: bool,
    desc: str,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    pred_scaled_parts = []
    target_raw_parts = []
    index_parts = []
    iterator = progress(loader, desc, show_progress)
    for batch in iterator:
        moved = to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            pred_scaled = model(moved["x"])
        pred_scaled_parts.append(pred_scaled.detach().cpu())
        target_raw_parts.append(batch["rho_raw"].detach().cpu())
        index_parts.append(batch["index"].detach().cpu())
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(samples=sum(part.numel() for part in target_raw_parts))

    pred_scaled = torch.cat(pred_scaled_parts)
    pred_raw = normalizer.inverse_rho(pred_scaled)
    target_raw = torch.cat(target_raw_parts)
    indices = torch.cat(index_parts)
    return regression_metrics(pred_raw, target_raw), pred_raw, target_raw, indices


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.float()
    target = target.float()
    err = pred - target
    abs_err = err.abs()
    target_mean = target.mean()
    ss_res = err.square().sum()
    ss_tot = (target - target_mean).square().sum()
    r2 = 1.0 - ss_res / ss_tot if float(ss_tot) > 0 else torch.tensor(float("nan"))
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(err.square().mean().sqrt()),
        "bias": float(err.mean()),
        "r2": float(r2),
        "corr": float(corr(pred, target)),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std(unbiased=False)),
        "target_mean": float(target_mean),
        "target_std": float(target.std(unbiased=False)),
        "pred_min": float(pred.min()),
        "pred_max": float(pred.max()),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
    }


def corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = a_centered.square().sum().sqrt() * b_centered.square().sum().sqrt()
    if float(denom) <= 0:
        return torch.tensor(float("nan"))
    return (a_centered * b_centered).sum() / denom


def save_checkpoint(
    path: Path,
    model: FullInputMLP,
    config: BaselineConfig,
    normalizer: Normalizer,
    splits: dict[str, list[int]],
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": config.to_dict(),
            "normalizer": normalizer.to_dict(),
            "splits": splits,
            "args": vars(args),
            "epoch": epoch,
            "valid_metrics": metrics,
        },
        path,
    )


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, indices: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "rho_true", "rho_pred", "error", "abs_error"])
        writer.writeheader()
        for idx, y, y_hat in zip(indices.tolist(), target.tolist(), pred.tolist(), strict=True):
            error = float(y_hat - y)
            writer.writerow(
                {
                    "index": int(idx),
                    "rho_true": float(y),
                    "rho_pred": float(y_hat),
                    "error": error,
                    "abs_error": abs(error),
                }
            )


def progress(loader: DataLoader, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(loader, desc=desc, total=len(loader), dynamic_ncols=True, leave=False)
    return loader


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


if __name__ == "__main__":
    main()
