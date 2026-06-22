"""Compatibility helpers for loading MASDiff population pickles."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Individual:
    """Slim replacement for ``src.pipeline.types.Individual``."""

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
        metadata = state.get("metadata", {})
        self.metadata = metadata if isinstance(metadata, dict) else {}
        self.experience_buffers = []
        self.policies = []


def install_pickle_compat() -> None:
    """Register a minimal module path required by the source pickle."""

    Individual.__module__ = "src.pipeline.types"

    src_mod = sys.modules.setdefault("src", types.ModuleType("src"))
    pipeline_mod = sys.modules.setdefault("src.pipeline", types.ModuleType("src.pipeline"))
    types_mod = types.ModuleType("src.pipeline.types")
    types_mod.Individual = Individual
    types_mod.Population = list

    setattr(src_mod, "pipeline", pipeline_mod)
    setattr(pipeline_mod, "types", types_mod)
    sys.modules["src.pipeline.types"] = types_mod
