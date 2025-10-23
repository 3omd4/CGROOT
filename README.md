# CGROOT++
A C++ Deep Learning Framework

## Build and Run

### Quick Start (Recommended)

**For Windows users, simply double-click `CGROOT_Manager.bat`** to open the interactive project manager with all options!

### Manual Build Instructions

#### Windows (Visual Studio)

1.  Configure the project with CMake:
    ```cmd
    cmake -B build -G "Visual Studio 16 2019"
    ```
2.  Compile the project (Debug):
    ```cmd
    cmake --build build --config Debug
    ```
3.  Compile the project (Release):
    ```cmd
    cmake --build build --config Release
    ```
4.  Run the main executable (Debug):
    ```cmd
    .\build\bin\Debug\cgrunner.exe
    ```
5.  Run the example (Debug):
    ```cmd
    .\build\bin\Debug\simple_test.exe
    ```

#### Linux/macOS

1.  Create a build directory and navigate into it:
    ```bash
    mkdir -p build && cd build
    ```
2.  Run CMake to generate the build files:
    ```bash
    cmake ..
    ```
3.  Compile the project:
    ```bash
    make
    ```
4.  Run the executable:
    ```bash
    ./cgrunner
    ```

### Available Scripts

- **`CGROOT_Manager.bat`** - Comprehensive interactive manager with all features
- **`build_debug.bat`** - Build Debug configuration only
- **`build_release.bat`** - Build Release configuration only
- **`run_debug.bat`** - Run Debug executables only
- **`run_release.bat`** - Run Release executables only
- **`build_and_run.bat`** - Interactive build and run script

## Project Structure

- `src/` - Main source code
  - `core/` - Core tensor and parameter classes
  - `autograd/` - Automatic differentiation
  - `math/` - Mathematical operations
  - `nn/` - Neural network layers
  - `optim/` - Optimizers
- `examples/` - Example programs
- `tests/` - Unit tests
