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
from PyQt6.QtGui import QIcon
import ctypes
from mainwindow import MainWindow
from dark_theme import apply_dark_theme

def main():
    app = QApplication(sys.argv)
    
    # Set AppUserModelID for proper taskbar icon handling
    myappid = 'cgrooot.neuralnetwork.trainer.1.0' 
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass # Non-Windows or other issue

    # Set Window Icon
    icon_path = project_root / "icons" / "favicon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    apply_dark_theme(app)
    
    window = MainWindow()
    window.showMaximized()
    
    # Enable handling of Ctrl+C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nGUI execution interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
