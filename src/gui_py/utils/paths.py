"""
Path Utilities for CGROOT++ GUI Application

Centralizes path management to ensure consistency across widgets.
"""
from pathlib import Path

def get_project_root() -> Path:
    """Get the project root directory."""
    # Assumes this file is in src/gui_py/utils/
    return Path(__file__).parent.parent.parent.parent

def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "src" / "data"

def get_logs_dir() -> Path:
    """Get the logs directory, creating if necessary."""
    logs_dir = get_data_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

def get_models_dir() -> Path:
    """Get the trained models directory, creating if necessary."""
    models_dir = get_data_dir() / "trained-model"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

def get_samples_dir() -> Path:
    """Get the samples directory."""
    return get_data_dir() / "samples"

def get_datasets_dir() -> Path:
    """Get the datasets directory."""
    return get_data_dir() / "datasets"
