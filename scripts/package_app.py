import os
import sys
import subprocess
import shutil
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GUI_SCRIPT = PROJECT_ROOT / "src" / "gui_py" / "main.py"
ICON_PATH = PROJECT_ROOT / "icons" / "favicon.ico"
BUILD_DIR = PROJECT_ROOT / "build" / "bin"

def check_pyinstaller():
    if not shutil.which("pyinstaller"):
        print("Error: PyInstaller not found.")
        print("Please install it: pip install pyinstaller")
        sys.exit(1)

def find_extension_module():
    """Find cgroot_core.pyd in build directories."""
    # Priority: Release > RelWithDebInfo > Debug
    configs = ["Release", "RelWithDebInfo", "Debug"]
    
    # Also check root bin (Ninja default)
    possible_roots = [
         BUILD_DIR, # Ninja
    ]
    for cfg in configs:
        possible_roots.append(BUILD_DIR / cfg) # VS
        
    for root in possible_roots:
        if not root.exists(): continue
        for kid in root.glob("cgroot_core*.pyd"):
            return kid
            
    return None

def ensure_src_data_structure(dist_dir: str, project_root: str):
    """
    Ensures that `src/data` exists inside the distribution directory.
    If missing, it attempts to copy it from the project root.

    Args:
        dist_dir (str): Path to the built/distribution directory
        project_root (str): Path to the project root containing `data`
    """

    dist_dir = Path(dist_dir)
    src_dir = dist_dir / "src"
    data_dir = src_dir / "data"

    source_data = Path(project_root) / "data"

    # Create src directory if missing
    src_dir.mkdir(exist_ok=True)

    # If data already exists, nothing to do
    if data_dir.exists():
        return

    # If source data exists, copy it
    if source_data.exists() and source_data.is_dir():
        shutil.copytree(source_data, data_dir)
        print(f"[INFO] Copied data folder to: {data_dir}")
    else:
        print("[WARNING] 'data' folder not found in project root.")
        print("Application may not function correctly.")

def main():
    print("=== CGROOT++ Application Packager ===")
    check_pyinstaller()
    
    # 1. Find C++ Extension
    ext_path = find_extension_module()
    if not ext_path:
        print("Error: Could not find 'cgroot_core' extension module (.pyd).")
        print("Please build the project first (Release configuration recommended).")
        sys.exit(1)
        
    print(f"Found Extension: {ext_path}")
    
    # 2. Prepare PyInstaller Command
    # We use --onedir (default) for faster startup and easier debugging
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onedir",
        "--windowed", # No console
        f"--name=CGROOT_Trainer",
        f"--icon={ICON_PATH}",
        "--clean",
        
        # Add the extension module
        # Format: source;dest_folder
        # We put it in root of bundle so direct import works
        f"--add-binary={ext_path};.", 
        
        # Add icons folder
        f"--add-data={PROJECT_ROOT / 'icons'};icons", 
        
        # Paths to search for imports (src/gui_py)
        f"--paths={PROJECT_ROOT / 'src' / 'gui_py'}",
        
        # Entry point
        str(GUI_SCRIPT)
    ]
    
    print("\nRunning PyInstaller...")
    print("Command:", " ".join(cmd))
    
    try:
        subprocess.check_call(cmd, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Packaging failed with code {e.returncode}")
        sys.exit(1)
        
    print("\n=== Packaging Complete! ===")
    dist_dir = PROJECT_ROOT / "dist" / "CGROOT_Trainer"
    print(f"Executable is located at: {dist_dir / 'CGROOT_Trainer.exe'}")

    
    
    print("\n[IMPORTANT NOTE]")
    print("The application expects 'src/data' structure relative to it.")
    print("To ensure full functionality (loading samples, saving logs/models):")
    print(f"1. Navigate to: {dist_dir}")
    print("2. Create folder 'src'")
    print("3. Copy 'data' folder from project root to 'src/data'")
    print("   (Or ensure your logic handles missing paths)")


    ensure_src_data_structure(str(dist_dir), str(PROJECT_ROOT))



if __name__ == "__main__":
    main()
