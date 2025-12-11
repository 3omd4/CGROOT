import sys
from pathlib import Path

# Add build/bin/Release to path (where pyd is)
# Adjust based on your build output assumption
project_root = Path(__file__).parent.parent
build_dirs = [
    project_root / "build" / "bin" / "Release",
    project_root / "build" / "bin",
]

for d in build_dirs:
    if d.exists():
        sys.path.append(str(d))

try:
    import cgroot_core
    print(f"Loaded cgroot_core: {cgroot_core.__doc__}")
    
    arch = cgroot_core.architecture()
    
    # Check if learningRate is exposed
    # Default value check
    print(f"Initial LR (garbage or 0): {arch.learningRate}")
    
    # Set it
    arch.learningRate = 0.05
    print(f"Set LR to 0.05")
    
    # Get it back
    if abs(arch.learningRate - 0.05) < 1e-6:
        print("SUCCESS: learningRate is exposed and writable.")
    else:
        print(f"FAILURE: learningRate mismatch. Got {arch.learningRate}")
        sys.exit(1)

except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except AttributeError as e:
    print(f"AttributeError: {e} (learningRate likely not bound)")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
