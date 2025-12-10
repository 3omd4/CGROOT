# Add build output directories to path to find the extension
import sys
from pathlib import Path

# Try to find the build directory relative to this script
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
build_dirs = [
    project_root / "build" / "bin" / "Debug",
    project_root / "build" / "bin" / "Release",
    project_root / "build" / "bin" / "RelWithDebInfo",
    project_root / "bin" / "Debug", # Linux/Make
]

for d in build_dirs:
    if d.exists():
        sys.path.append(str(d))

try:
    import cgroot_core
    print(f"Loaded CGROOT Core: {cgroot_core.__doc__}")
except ImportError as e:
    print(f"Warning: Could not import cgroot_core: {e}")
    print("Functionality relying on the core library will not work.")

from PyQt6.QtWidgets import QApplication
from mainwindow import MainWindow
from dark_theme import apply_dark_theme

def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
