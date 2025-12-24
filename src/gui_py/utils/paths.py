"""
Path Utilities for CGROOT++ GUI Application

Centralizes path management to ensure consistency across widgets.
"""
import sys
from pathlib import Path

def get_project_root() -> Path:
    """Get the project root directory (Writable Location)."""
    if getattr(sys, 'frozen', False):
        # If frozen (PyInstaller), use the executable's directory
        return Path(sys.executable).parent
    # Assumes this file is in src/gui_py/utils/
    return Path(__file__).parent.parent.parent.parent

def get_bundled_resource_root() -> Path:
    """Get the root for READ-ONLY bundled resources."""
    if getattr(sys, 'frozen', False):
        # PyInstaller bundled content
        return Path(sys._MEIPASS)
    return get_project_root()

def get_data_dir() -> Path:
    """Get the USER data directory (Writable)."""
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
    """
    Get the datasets directory. 
    Checks bundled resources first if frozen, then writable location.
    """
    if getattr(sys, 'frozen', False):
        # Check bundled first (e.g., standard datasets like MNIST)
        # Using --add-data "src/data;src/data" places content in _MEIPASS/src/data
        bundled_path = get_bundled_resource_root() / "src" / "data" / "datasets"
        
        if bundled_path.exists():
            return bundled_path

    return get_data_dir() / "datasets"
