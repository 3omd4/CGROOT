# 🧠 CGROOT++

<div align="center">

![C++](https://img.shields.io/badge/C++-17-blue.svg?style=for-the-badge&logo=cplusplus)
![CMake](https://img.shields.io/badge/CMake-3.10+-green.svg?style=for-the-badge&logo=cmake)
![Visual Studio](https://img.shields.io/badge/Visual%20Studio-2019-purple.svg?style=for-the-badge&logo=visual-studio)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

**A High-Performance C++ Deep Learning Framework**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🔧 Installation](#-installation) • [💡 Examples](#-examples) • [🤝 Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🔧 Installation](#-installation)
- [📖 Documentation](#-documentation)
- [💡 Examples](#-examples)
- [🏗️ Project Structure](#️-project-structure)
- [🛠️ Available Scripts](#️-available-scripts)
- [🧪 Testing](#-testing)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 Overview

CGROOT++ is a modern, high-performance deep learning framework built from the ground up in C++. Designed for both research and production use, it provides a clean, intuitive API similar to PyTorch while leveraging the speed and efficiency of C++.

### 🎯 Key Goals
- **Performance**: Optimized C++ implementation for maximum speed
- **Simplicity**: Clean, intuitive API design
- **Flexibility**: Modular architecture for easy extension
- **Education**: Clear code structure for learning deep learning internals

---

## ✨ Features

### 🧮 Core Components
- **📊 Tensor Operations**: Multi-dimensional arrays with automatic differentiation
- **🔄 Automatic Differentiation**: Dynamic computational graph with gradient computation
- **⚡ CPU Kernels**: Optimized mathematical operations for CPU execution

### 🧠 Neural Network Layers
- **🔗 Linear Layer**: Fully connected linear transformations
- **🔄 ReLU Activation**: Rectified Linear Unit activation function
- **📈 Sigmoid Activation**: Sigmoid activation function
- **🏗️ Sequential Container**: Stack multiple layers in sequence
- **🖼️ Conv2D Layer**: 2D Convolutional layer (planned)

### 📉 Loss Functions
- **📊 MSE Loss**: Mean Squared Error for regression tasks
- **🎯 Cross Entropy Loss**: Cross Entropy for classification tasks

### 🎛️ Optimizers
- **📉 SGD**: Stochastic Gradient Descent
- **⚡ Adam**: Adaptive Moment Estimation optimizer

### 🛠️ Development Tools
- **🔧 Interactive Manager**: Windows batch script for easy project management
- **🧪 Unit Tests**: Comprehensive test suite
- **📚 Examples**: Ready-to-run example programs

---

## 🚀 Quick Start

### 🪟 Windows (Recommended)

**Simply double-click `CGROOT_Manager.bat`** to open the interactive project manager with all available options!

The manager provides:
- 🔨 **Build Options**: Debug and Release configurations
- ▶️ **Run Options**: Execute examples and tests
- 🧹 **Clean Options**: Clean build directories
- 📊 **Status Check**: View project status and file locations
- 🎯 **VS Integration**: Open Visual Studio solution

### 🐧 Linux/macOS

```bash
# Clone the repository
git clone <repository-url>
cd CGROOT

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
make

# Run examples
./bin/cgrunner
./bin/simple_test
```

---

## 🔧 Installation

### Prerequisites

- **C++ Compiler**: C++17 compatible (GCC 7+, Clang 5+, MSVC 2019+)
- **CMake**: Version 3.10 or higher
- **Visual Studio**: 2019 or later (Windows)

### Windows Installation

1. **Install Visual Studio 2019** or later with C++ development tools
2. **Install CMake** (usually included with Visual Studio)
3. **Clone the repository**:
   ```cmd
   git clone <repository-url>
   cd CGROOT
   ```
4. **Run the manager**:
   ```cmd
   CGROOT_Manager.bat
   ```

### Linux Installation

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install build-essential cmake git

# Install dependencies (CentOS/RHEL)
sudo yum groupinstall "Development Tools"
sudo yum install cmake git

# Clone and build
git clone <repository-url>
cd CGROOT
mkdir build && cd build
cmake ..
make
```

### macOS Installation

```bash
# Install dependencies with Homebrew
brew install cmake git

# Clone and build
git clone <repository-url>
cd CGROOT
mkdir build && cd build
cmake ..
make
```

---

## 📖 Documentation

### 🏗️ Architecture Overview

CGROOT++ follows a modular architecture with clear separation of concerns:

```
src/
├── core/           # Core tensor and parameter classes
├── autograd/       # Automatic differentiation system
├── math/           # Mathematical operations and kernels
├── nn/             # Neural network layers and modules
└── optim/          # Optimization algorithms
```

### 🔧 API Reference

#### Tensor Operations
```cpp
#include "core/tensor.h"

// Create tensors
auto a = Tensor<float>({2, 3});  // 2x3 tensor
auto b = Tensor<float>({3, 4});  // 3x4 tensor

// Basic operations
auto c = a + b;                  // Element-wise addition
auto d = a.matmul(b);            // Matrix multiplication
auto e = a.relu();               // ReLU activation
```

#### Neural Network Layers
```cpp
#include "nn/linear.h"
#include "nn/relu.h"
#include "nn/sequential.h"

// Create a simple neural network
auto model = Sequential<float>();
model.add(std::make_shared<Linear<float>>(784, 128));
model.add(std::make_shared<ReLU<float>>());
model.add(std::make_shared<Linear<float>>(128, 10));
```

#### Training Loop
```cpp
#include "nn/mse_loss.h"
#include "optim/sgd.h"

// Define loss and optimizer
auto criterion = MSELoss<float>();
auto optimizer = SGD<float>(model.parameters(), 0.01);

// Training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    optimizer.zero_grad();
    auto output = model.forward(input);
    auto loss = criterion.forward(output, target);
    loss.backward();
    optimizer.step();
}
```

---

## 💡 Examples

### 📁 Available Examples

| Example | Description | Status |
|---------|-------------|--------|
| `simple_test.cpp` | Basic tensor operations demo | ✅ Ready |
| `xor_solver.cpp` | XOR problem solver with MLP | 🚧 In Development |

### 🚀 Running Examples

#### Windows
```cmd
# Using the manager
CGROOT_Manager.bat
# Select option 3 or 4 to run examples

# Or manually
.\build\bin\Debug\simple_test.exe
.\build\bin\Debug\cgrunner.exe
```

#### Linux/macOS
```bash
./bin/simple_test
./bin/cgrunner
```

### 📝 Example: Simple Tensor Operations

```cpp
#include <iostream>
#include "core/tensor.h"

int main() {
    // Create tensors
    auto a = Tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = Tensor<float>({2, 3}, {2, 3, 4, 5, 6, 7});
    
    // Perform operations
    auto c = a + b;
    auto d = a * b;
    
    // Print results
    std::cout << "Tensor a:\n" << a << std::endl;
    std::cout << "Tensor b:\n" << b << std::endl;
    std::cout << "a + b:\n" << c << std::endl;
    std::cout << "a * b:\n" << d << std::endl;
    
    return 0;
}
```

---

## 🏗️ Project Structure

```
CGROOT/
├── 📁 src/                    # Source code
│   ├── 📁 core/              # Core tensor and parameter classes
│   │   ├── tensor.h/cpp      # Main tensor implementation
│   │   ├── parameter.h       # Parameter wrapper for learnable weights
│   │   └── shape.h           # Shape utilities
│   ├── 📁 autograd/          # Automatic differentiation
│   │   ├── graph.h/cpp       # Computational graph
│   │   └── op_nodes.h/cpp    # Operation nodes
│   ├── 📁 math/              # Mathematical operations
│   │   └── cpu_kernels.h/cpp # CPU-optimized kernels
│   ├── 📁 nn/                # Neural network layers
│   │   ├── module.h          # Base module class
│   │   ├── linear.h          # Linear layer
│   │   ├── relu.h            # ReLU activation
│   │   ├── sigmoid.h         # Sigmoid activation
│   │   ├── sequential.h      # Sequential container
│   │   ├── conv2d.h          # 2D Convolution
│   │   ├── mse_loss.h        # MSE loss function
│   │   └── cross_entropy_loss.h # Cross entropy loss
│   └── 📁 optim/             # Optimizers
│       ├── optimizer.h       # Base optimizer class
│       ├── sgd.h             # SGD optimizer
│       └── adam.h            # Adam optimizer
├── 📁 examples/              # Example programs
│   ├── simple_test.cpp       # Basic functionality demo
│   └── xor_solver.cpp        # XOR problem solver
├── 📁 tests/                 # Unit tests
│   ├── test_tensor.cpp       # Tensor operation tests
│   └── test_autograd.cpp     # Autograd tests
├── 📁 build/                 # Build output directory
├── 📄 CMakeLists.txt         # CMake configuration
├── 📄 CGROOT_Manager.bat     # Windows project manager
└── 📄 README.md              # This file
```

---

## 🛠️ Available Scripts

### 🪟 Windows Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| **`CGROOT_Manager.bat`** | 🎯 Interactive project manager with all features | Double-click or run from command line |
| **`build_debug.bat`** | 🔨 Build Debug configuration only | `build_debug.bat` |
| **`build_release.bat`** | 🔨 Build Release configuration only | `build_release.bat` |
| **`run_debug.bat`** | ▶️ Run Debug executables only | `run_debug.bat` |
| **`run_release.bat`** | ▶️ Run Release executables only | `run_release.bat` |
| **`build_and_run.bat`** | 🔄 Interactive build and run script | `build_and_run.bat` |

### 🐧 Linux/macOS Commands

```bash
# Build commands
make                    # Build all targets
make cgrunner          # Build main executable only
make simple_test       # Build example only

# Run commands
./bin/cgrunner         # Run main executable
./bin/simple_test      # Run example

# Clean commands
make clean             # Clean build files
rm -rf build/          # Remove entire build directory
```

---

## 🧪 Testing

### 🧪 Running Tests

#### Windows
```cmd
# Using the manager
CGROOT_Manager.bat
# Select option 8 to check project status

# Or manually run test executables
.\build\bin\Debug\simple_test.exe
```

#### Linux/macOS
```bash
# Run tests
make test
# Or run individual test executables
./bin/simple_test
```

### 📋 Test Coverage

| Component | Test File | Status | Coverage |
|-----------|-----------|--------|----------|
| **Tensor Operations** | `test_tensor.cpp` | 🚧 Planned | Basic math operations |
| **Automatic Differentiation** | `test_autograd.cpp` | 🚧 Planned | Gradient computation |
| **Neural Network Layers** | Integration tests | 🚧 Planned | Forward/backward passes |
| **Optimizers** | Integration tests | 🚧 Planned | Parameter updates |

---

## 🤝 Contributing

We welcome contributions to CGROOT++! Here's how you can help:

### 🐛 Reporting Issues
- Use the GitHub issue tracker
- Provide detailed reproduction steps
- Include system information and error messages

### 💡 Suggesting Features
- Open a GitHub issue with the "enhancement" label
- Describe the use case and expected behavior
- Consider contributing the implementation

### 🔧 Development Setup

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/CGROOT.git
   cd CGROOT
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and test thoroughly
5. **Commit your changes**:
   ```bash
   git commit -m "Add: your feature description"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

### 📋 Development Guidelines

- **Code Style**: Follow existing code conventions
- **Documentation**: Update README and code comments
- **Testing**: Add tests for new features
- **Performance**: Consider performance implications
- **Compatibility**: Ensure cross-platform compatibility

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch** for API design inspiration
- **Eigen** for mathematical operations reference
- **CMake** for cross-platform build system
- **Visual Studio** for excellent C++ development tools

---

<div align="center">

**Made with ❤️ by the CGROOT++ Team**

[⭐ Star us on GitHub](https://github.com/yourusername/CGROOT) • [🐛 Report Issues](https://github.com/yourusername/CGROOT/issues) • [💬 Discussions](https://github.com/yourusername/CGROOT/discussions)

</div>