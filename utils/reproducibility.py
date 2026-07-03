"""Utilities for reproducible DDQN tutoring experiments."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is required for the agent but kept safe here.
    torch = None


def set_global_seed(seed: int | None) -> None:
    """Set random seeds for Python, NumPy, and PyTorch when available."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except AttributeError:
            pass

        try:
            torch.use_deterministic_algorithms(True)
        except (AttributeError, RuntimeError):
            pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def save_experiment_config(config: dict[str, Any], output_path: str | Path) -> Path:
    """Save experiment configuration values to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(config), file, indent=2, sort_keys=True)

    return output_path


def create_experiment_run_dir(base_dir: str | Path = "outputs/runs") -> Path:
    """Create a timestamped experiment run directory."""
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
