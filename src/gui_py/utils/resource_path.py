import sys
import os
from pathlib import Path

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Go up 3 levels from utils/resource_path.py to get to project root
        # src/gui_py/utils -> src/gui_py -> src -> PROJECT_ROOT
        base_path = Path(__file__).resolve().parent.parent.parent.parent
        
    return str(Path(base_path) / relative_path)
