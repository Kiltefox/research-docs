"""Convert the raw MASDiff population pickle into a compact tensor cache."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import torch

try:
    from .types_compat import install_pickle_compat
except ImportError:
    from types_compat import install_pickle_compat


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SUMO rho prediction dataset.")
    parser.add_argument(
        "--input",
        default="./dataset/sumo_ryl_initial_population.pkl",
        help="Raw population pickle path.",
    )
    parser.add_argument(
        "--output",
        default="./dataset/sumo_ryl_prediction_dataset.pt",
        help="Compact torch dataset path.",
    )
    parser.add_argument(
        "--group-key",
        default=None,
        help="Optional metadata key used later for group split.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    install_pickle_compat()
    start = time.time()
    with input_path.open("rb") as f:
        population = pickle.load(f)
    print(f"Loaded {len(population)} individuals in {time.time() - start:.1f}s")

    tau_items: list[torch.Tensor] = []
    reward_items: list[torch.Tensor] = []
    rho_items: list[float] = []
    groups: list[Any] = []
    shapes: list[tuple[int, ...]] = []

    for idx, individual in enumerate(population):
        tau = torch.as_tensor(individual.tau, dtype=torch.float32).cpu()
        rewards = torch.as_tensor(individual.rewards, dtype=torch.float32).cpu()
        rho = float(individual.rho)
        if tau.dim() != 3:
            raise ValueError(f"Individual {idx} tau must be 3-D, got {tuple(tau.shape)}")
        if rewards.shape != tau.shape[:2]:
            raise ValueError(
                f"Individual {idx} rewards shape {tuple(rewards.shape)} does not match tau {tuple(tau.shape)}"
            )
        tau_items.append(tau)
        reward_items.append(rewards)
        rho_items.append(rho)
        shapes.append(tuple(int(v) for v in tau.shape))
        metadata = getattr(individual, "metadata", {}) or {}
        groups.append(metadata.get(args.group_key, idx) if args.group_key else idx)

    tau_payload: list[torch.Tensor] | torch.Tensor = tau_items
    rewards_payload: list[torch.Tensor] | torch.Tensor = reward_items
    if len(set(shapes)) == 1:
        tau_payload = torch.stack(tau_items, dim=0)
        rewards_payload = torch.stack(reward_items, dim=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tau": tau_payload,
            "rewards": rewards_payload,
            "rho": torch.tensor(rho_items, dtype=torch.float32),
            "groups": groups,
            "metadata": {
                "source": str(input_path),
                "num_samples": len(rho_items),
                "tau_shapes": sorted(set(shapes)),
                "group_key": args.group_key,
            },
        },
        output_path,
    )
    print(f"Saved compact dataset to {output_path}")


if __name__ == "__main__":
    main()
