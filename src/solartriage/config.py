"""Load the YAML configuration and resolve project paths."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["raw_dir"] = ROOT / cfg["data"]["raw_dir"]
    cfg["data"]["processed_dir"] = ROOT / cfg["data"]["processed_dir"]
    return cfg


def ensure_dirs() -> None:
    for d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
