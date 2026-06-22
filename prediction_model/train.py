"""Train the rho prediction model."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from .data import (
        STATS_DIM,
        Normalizer,
        RhoTensorDataset,
        SubsetDataset,
        make_collate_fn,
        save_json,
        split_indices,
    )
    from .metrics import regression_metrics, student_t_nll
    from .model import ModelConfig, RhoPredictionModel
except ImportError:
    from data import (
        STATS_DIM,
        Normalizer,
        RhoTensorDataset,
        SubsetDataset,
        make_collate_fn,
        save_json,
        split_indices,
    )
    from metrics import regression_metrics, student_t_nll
    from model import ModelConfig, RhoPredictionModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rho prediction model.")
    parser.add_argument("--data", default="dataset/sumo_ryl_prediction_dataset.pt")
    parser.add_argument("--output-dir", default="runs/rho_prediction")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--mean-loss-weight",
        type=float,
        default=1.0,
        help="Auxiliary MSE weight on the predicted mean in normalized rho space.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--road-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--aggregator",
        choices=["deepsets", "meanmax", "attention"],
        default="deepsets",
        help="Vehicle-level aggregation module.",
    )
    parser.add_argument("--vehicle-chunk-size", type=int, default=64)
    parser.add_argument("--checkpoint-road", action="store_true", help="Recompute road encoder activations to save VRAM.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--group-split", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tau-noise-std", type=float, default=0.0)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="Use the first N training samples for train and valid to debug whether the model can memorize.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {data_path}. Run "
            "`python prepare_dataset.py` first."
        )

    torch.manual_seed(args.seed)
    dataset = RhoTensorDataset(data_path)
    splits = split_indices(dataset, seed=args.seed, group_split=args.group_split)
    if args.overfit_samples > 0:
        overfit = splits["train"][: args.overfit_samples]
        if not overfit:
            raise ValueError("--overfit-samples requires a non-empty training split.")
        splits = {"train": overfit, "valid": overfit, "test": splits["test"]}
    normalizer = Normalizer.fit(dataset, splits["train"])
    n_vehicle, n_road, tau_dim = dataset.sample_shape

    config = ModelConfig(
        tau_dim=tau_dim,
        num_roads=n_road,
        stats_dim=STATS_DIM,
        hidden_dim=args.hidden_dim,
        road_encoder_layers=args.road_layers,
        attention_heads=args.heads,
        dropout=args.dropout,
        aggregator=args.aggregator,
        vehicle_chunk_size=args.vehicle_chunk_size,
        checkpoint_road_encoder=args.checkpoint_road,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "config.json",
        {
            "args": vars(args),
            "model": config.to_dict(),
            "normalizer": normalizer.to_dict(),
            "splits": splits,
            "sample_shape": [n_vehicle, n_road, tau_dim],
        },
    )

    train_loader = _loader(dataset, splits["train"], normalizer, args, shuffle=True, augment=True)
    valid_loader = _loader(dataset, splits["valid"], normalizer, args, shuffle=False, augment=False)

    device = torch.device(args.device)
    model = RhoPredictionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_valid = float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    start = time.time()
    show_progress = not args.no_progress

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.grad_clip,
            scaler,
            args.amp,
            args.mean_loss_weight,
            epoch=epoch,
            show_progress=show_progress,
        )
        valid = evaluate(
            model,
            valid_loader,
            device,
            normalizer,
            args.amp,
            epoch=epoch,
            show_progress=show_progress,
        )
        row = {"epoch": epoch, "train_nll": train_loss, **{f"valid_{k}": v for k, v in valid.items()}}
        history.append(row)
        _write_history(output_dir / "history.csv", history)
        print(
            f"epoch={epoch:03d} train_nll={train_loss:.5f} "
            f"valid_nll={valid['nll']:.5f} valid_mae={valid['mae']:.5f}"
        )

        if valid["nll"] < best_valid:
            best_valid = valid["nll"]
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, config, normalizer, splits, args, best_epoch, best_valid)
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    save_checkpoint(output_dir / "last.pt", model, config, normalizer, splits, args, epoch, valid["nll"])
    print(f"Training complete in {(time.time() - start) / 60:.1f} min. Best valid NLL: {best_valid:.5f}")


def _loader(
    dataset: RhoTensorDataset,
    indices: list[int],
    normalizer: Normalizer,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    augment: bool,
) -> DataLoader:
    return DataLoader(
        SubsetDataset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=make_collate_fn(normalizer, augment=augment, tau_noise_std=args.tau_noise_std),
    )


def train_one_epoch(
    model: RhoPredictionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    scaler: torch.amp.GradScaler,
    amp: bool,
    mean_loss_weight: float,
    epoch: int = 0,
    show_progress: bool = True,
) -> float:
    model.train()
    total = 0.0
    count = 0
    iterator = _progress(loader, desc=f"epoch {epoch:03d} train", enabled=show_progress)
    for batch in iterator:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            pred = model(batch["tau"], batch["rewards"], batch["mask"], batch["stats"])
            nll = student_t_nll(pred, batch["rho"])
            mean_loss = torch.nn.functional.mse_loss(pred["mu"].float(), batch["rho"].float())
            loss = nll + mean_loss_weight * mean_loss
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item()) * int(batch["rho"].numel())
        count += int(batch["rho"].numel())
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{total / max(count, 1):.4f}",
                nll=f"{float(nll.item()):.4f}",
                mse=f"{float(mean_loss.item()):.4f}",
            )
    return total / max(count, 1)


@torch.no_grad()
def evaluate(
    model: RhoPredictionModel,
    loader: DataLoader,
    device: torch.device,
    normalizer: Normalizer,
    amp: bool = False,
    epoch: int = 0,
    show_progress: bool = True,
) -> dict[str, float]:
    model.eval()
    preds: dict[str, list[torch.Tensor]] = {"mu": [], "sigma": [], "nu": []}
    target_scaled: list[torch.Tensor] = []
    target_raw: list[torch.Tensor] = []
    iterator = _progress(loader, desc=f"epoch {epoch:03d} valid", enabled=show_progress)
    for batch in iterator:
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            pred = model(batch["tau"], batch["rewards"], batch["mask"], batch["stats"])
        for key in preds:
            preds[key].append(pred[key].detach().cpu())
        target_scaled.append(batch["rho"].detach().cpu())
        target_raw.append(batch["rho_raw"].detach().cpu())
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(samples=sum(part.numel() for part in target_scaled))
    pred_cat = {key: torch.cat(value, dim=0) for key, value in preds.items()}
    return regression_metrics(pred_cat, torch.cat(target_scaled), torch.cat(target_raw), normalizer)


def _progress(loader: DataLoader, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(loader, desc=desc, total=len(loader), dynamic_ncols=True, leave=False)
    return loader


def save_checkpoint(
    path: Path,
    model: RhoPredictionModel,
    config: ModelConfig,
    normalizer: Normalizer,
    splits: dict[str, list[int]],
    args: argparse.Namespace,
    epoch: int,
    valid_nll: float,
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
            "valid_nll": valid_nll,
        },
        path,
    )


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
