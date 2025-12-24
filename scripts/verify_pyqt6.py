import os
from pathlib import Path

# Prepend PyQt6 bundled Qt6 DLL directory to DLL search path (Windows)
def add_qt6_dll_dir():
    try:
        import PyQt6  # noqa: F401
        qt_pkg_dir = Path(__import__('PyQt6').__file__).parent
        qt_bin = qt_pkg_dir / 'Qt6' / 'bin'
        if qt_bin.exists():
            os.add_dll_directory(str(qt_bin))
            print(f"Added Qt6 DLL directory: {qt_bin}")
        else:
            print(f"Qt6 DLL directory not found at: {qt_bin}")
    except Exception as e:
        print(f"Failed to set Qt6 DLL directory: {e}")


def main():
    add_qt6_dll_dir()
    try:
        from PyQt6 import QtCore, QtWidgets
        import numpy, colorama

        # PyQt6 version strings are provided via QtCore
        print("PyQt6 version:", getattr(QtCore, "PYQT_VERSION_STR", "unknown"))
        print("Qt version:", getattr(QtCore, "QT_VERSION_STR", "unknown"))

        # Optional: pyqtgraph may not be installed
        try:
            import pyqtgraph
            print("pyqtgraph version:", getattr(pyqtgraph, "__version__", "unknown"))
        except Exception:
            print("pyqtgraph not installed")

        # Optional: QtCharts (requires PyQt6-Charts)
        try:
            from PyQt6 import QtCharts
            print("QtCharts import: OK")
        except Exception as e:
            print("QtCharts import failed:", e)

        print("numpy version:", numpy.__version__)
        print("colorama version:", colorama.__version__)

        # Minimal widget creation to confirm QtWidgets loads
        app = QtWidgets.QApplication([])
        w = QtWidgets.QWidget()
        w.setWindowTitle("PyQt6 Verification")
        print("QtWidgets loaded successfully.")
    except Exception as e:
        print("Verification failed:", e)


if __name__ == '__main__':
    main()
