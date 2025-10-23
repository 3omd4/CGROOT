# CGROOT++
A C++ Deep Learning Framework

## Build and Run

To build and run the C++ application, follow these steps:

### Windows (MinGW)

1.  Configure the project with CMake:
    ```powershell
    cmake -B build -G "MinGW Makefiles"
    ```
2.  Compile the project:
    ```powershell
    cmake --build build
    ```
3.  Run the main executable:
    ```powershell
    .\build\bin\cgrunner.exe
    ```
4.  Run the example:
    ```powershell
    .\build\bin\simple_test.exe
    ```

### Linux/macOS

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

## Project Structure

- `src/` - Main source code
  - `core/` - Core tensor and parameter classes
  - `autograd/` - Automatic differentiation
  - `math/` - Mathematical operations
  - `nn/` - Neural network layers
  - `optim/` - Optimizers
- `examples/` - Example programs
- `tests/` - Unit tests
