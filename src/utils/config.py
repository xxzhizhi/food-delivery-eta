"""Configuration management using YAML files."""

from pathlib import Path
from typing import Any

import yaml


def load_config(filepath: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath) as f:
        return yaml.safe_load(f)
