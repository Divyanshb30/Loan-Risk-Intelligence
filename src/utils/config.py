from pathlib import Path
import yaml

# Project root = two levels up from this file (src/utils/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Loads config relative to project root — works regardless of
    where the script or notebook is called from.
    """
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    return PROJECT_ROOT
