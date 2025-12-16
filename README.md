# 🧠 CGROOT++

<div align="center">

![C++](https://img.shields.io/badge/C++-17-blue.svg?style=for-the-badge&logo=cplusplus)
![CMake](https://img.shields.io/badge/CMake-3.10+-green.svg?style=for-the-badge&logo=cmake)
![Visual Studio](https://img.shields.io/badge/Visual%20Studio-2019-purple.svg?style=for-the-badge&logo=visual-studio)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

**A High-Performance Educational C++ Deep Learning Framework**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🔧 Installation](#-installation) • [💡 Examples](#-examples) • [🤝 Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [👥 Our Team](#-our-team)
- [🗺️ Development Roadmap](#️-development-roadmap)
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

**CGroot++** is a mini educational machine learning (ML) framework designed specifically for ML developers and the open-source community. Its primary goal is to serve as an educational tool, demystifying the internal workings of ML models. The project's unique value proposition lies in its combination of being fully open-source, having a strong educational focus, and including capabilities for explaining model decisions.

The framework is built using a hybrid technical stack to balance performance and usability. The core computational engine is written in **C++** for maximum efficiency, while **Python** is leveraged for a user-friendly GUI and plotting capabilities. The project uses **CMake** for building project files and is designed for entirely **local** deployment, as it functions as a standalone framework without needing a backend, frontend, or database.

### 🎯 Key Goals

- **Education**: Clear, well-documented code structure for learning deep learning internals
- **Explainability**: Capabilities for explaining model decisions and internal workings
- **Performance**: Optimized C++ implementation for maximum speed
- **Simplicity**: Clean, intuitive API design similar to PyTorch
- **Open Source**: Fully open-source framework for the community
- **Local Deployment**: Standalone framework requiring no external dependencies

---

## ✨ Features

### 🎯 Core Features

#### 🧠 **Neural Network Engine**

- **Core Matrix Operations**: High-performance tensor operations, Multi-dimensional arrays with automatic differentiation
- **Automatic Differentiation**: Dynamic computational graph with gradient computation
- **Forward & Backward Propagation**: Complete automatic differentiation
- **Model Class**: Sequential container for stacking layers
- **CPU Kernels**: Optimized mathematical operations for CPU execution

#### 🔗 **Core Layers**

- **Dense Layer**: Fully connected linear transformations
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax
- **Sequential Container**: Stack multiple layers in sequence

#### 📉 **Loss Functions**

- **Mean Squared Error (MSE)**: For regression tasks
- **Binary Cross-Entropy**: For binary classification tasks
- **Categorical Cross-Entropy**: For multi-class classification tasks

#### 🎛️ **Optimizers**

- **Stochastic Gradient Descent (SGD)**: The fundamental baseline optimizer
- **Momentum**: Common improvement on SGD
- **Adam**: Popular and effective adaptive optimizer

#### 🔧 **Initialization & Training**

- **Weight Initialization**: Glorot (Xavier) and He initialization methods
- **User-friendly API**: Intuitive interface similar to PyTorch
- **Performance Tracking**: Loss and accuracy monitoring after each epoch
- **Data Batching**: Efficient data loading and batching mechanism

### 🚀 Secondary Features (Planned)

#### 🏗️ **Advanced Layers**

- **Convolutional Layer (Conv2D)**: 2D convolution operations
- **Pooling Layer**: Max Pooling and Average Pooling
- **Dropout Layer**: Regularization technique

#### 🛡️ **Regularization Techniques**

- **L2 Regularization**: Weight decay for preventing overfitting
- **Dropout**: Random neuron deactivation during training

#### 🎛️ **Training Control**

- **Early Stopping**: Prevent overfitting by monitoring validation loss
- **Model Saving/Loading**: Persist trained models

#### 📊 **Additional Loss Functions**

- **Mean Absolute Error (MAE)**: For robust regression tasks

### 🛠️ Development Tools

- **🔧 Interactive Manager**: Windows batch script for easy project management
- **🧪 Unit Tests**: Comprehensive test suite
- **📚 Examples**: Ready-to-run example programs
- **📖 Documentation**: Detailed API reference and tutorials

---

## 👥 Our Team

The CGROOT++ project is developed by a dedicated team of software engineering students who share a passion for machine learning and educational technology. We work collaboratively to create a comprehensive deep learning framework that serves both educational and practical purposes.

### 🎯 **Our Mission**

To build an open-source, educational machine learning framework that demystifies the internal workings of neural networks while providing high-performance capabilities for real-world applications.

### 🤝 **Collaborative Approach**

- **Unified Development**: We work together as one cohesive team
- **Shared Knowledge**: Regular code reviews and knowledge sharing sessions
- **Collective Ownership**: Every team member contributes to all aspects of the project
- **Continuous Learning**: We learn from each other and grow together as developers

### 👨‍💻 **Team Members**

- **Mohamed Emad-Eldeen**
- **George Esmat**
- **Ziad Khalid**
- **Ahmed Hasan**
- **Mohamed Amgd**
- **Antony Ghayes**

---

## 🗺️ Development Roadmap

### 🎯 **Current Focus: Core Foundation**

- **Tensor Operations**: Multi-dimensional array implementation with memory management
- **Shape Management**: Tensor shape and stride calculations
- **Parameter System**: Learnable weights with proper initialization
- **Basic Kernels**: CPU-optimized mathematical operations

### 🧠 **Next Phase: Automatic Differentiation**

- **Computational Graph**: Dynamic graph construction and management
- **Operation Nodes**: Individual operation implementations (Add, Mul, MatMul, etc.)
- **Backward Propagation**: Gradient computation and accumulation
- **Gradient Checking**: Numerical gradient verification

### 🏗️ **Future Development: Neural Networks**

- **Module System**: Base class for all neural network components
- **Linear Layer**: Fully connected layer implementation
- **Activation Functions**: ReLU, Sigmoid, Tanh implementations
- **Sequential Container**: Layer stacking and forward pass

### 🚀 **Advanced Features (Planned)**

- **Convolutional Layers**: Conv2D implementation with im2col
- **Pooling Layers**: Max Pooling and Average Pooling
- **Regularization**: Dropout and L2 Regularization
- **Training Controls**: Early stopping and model persistence

### 🔮 **Long-term Vision**

- **GPU Support**: CUDA kernels for accelerated computation
- **Python Bindings**: Seamless integration with Python ecosystem
- **Visualization Tools**: Model architecture and training visualization
- **Advanced Optimizers**: RMSprop, AdaGrad, and other optimizers
- **More Layer Types**: BatchNorm, LayerNorm, and attention mechanisms

---

## 🚀 Quick Start

### 🪟 Windows

#### Option 1: Launch GUI Directly

```cmd
python scripts/CGROOT_Manager.py --gui
```

#### Option 2: Full Build and Package

```cmd
python scripts/CGROOT_Manager.py --full
```

This will:

- Kill any zombie processes
- Clean and rebuild the project (Release)
- Install PyInstaller (if needed)
- Package the app as a standalone `.exe`
- Launch the packaged executable

#### Option 3: Interactive Manager

```cmd
python scripts/CGROOT_Manager.py
```

Provides a menu with all build and run options.

### 🐧 Linux/macOS

```bash
# Build the C++ core
mkdir build && cd build
cmake ..
make

# Launch the GUI
python3 src/gui_py/main.py
```

---

## 💻 GUI Application

CGROOT++ includes a comprehensive PyQt6-based GUI for training and testing neural networks.

### ✨ Features

- **📈 Real-time Training Visualization**: Live preview of training samples and predictions
- **🗺️ Feature Maps**: Visualize intermediate layer activations
- **📊 Metrics Tracking**: Interactive charts for loss and accuracy
- **⚙️ Configuration**: Complete control over model architecture and hyperparameters
- **💾 Model Persistence**: Save and load trained models
- **🔍 Inference**: Test models on individual images
- **📝 Comprehensive Logging**: All actions logged with timestamps

### ⌨️ Keyboard Shortcuts

| Shortcut | Action         |
| -------- | -------------- |
| `Ctrl+O` | Load Dataset   |
| `Ctrl+T` | Start Training |
| `Ctrl+S` | Stop Training  |
| `F1`     | Show Help      |

### 📋 Workflow

1. **Load Dataset** (File → Load Dataset or `Ctrl+O`)

   - Select MNIST-format image file
   - Label file auto-detected
   - Supports MNIST and Fashion-MNIST

2. **Configure Model** (Configuration Tab)

   - Set architecture (layers, neurons, kernels)
   - Choose optimizer (SGD, Momentum, Adam)
   - Adjust hyperparameters
   - Optional: Enable validation split

3. **Train** (Training Tab)

   - Click "Start Training" or press `Ctrl+T`
   - Monitor real-time preview and metrics
   - View feature maps at each epoch
   - Stop anytime with `Ctrl+S`

4. **Save Model** (Training Tab)

   - Click "Store Model"
   - Choose location (defaults to `src/data/trained-model`)
   - Saves weights and configuration

5. **Inference** (Inference Tab)
   - Load saved model
   - Select test image
   - View prediction and confidence scores

---

## 🔧 Framework & Technology Stack

The project's technical stack uses **C++** for core efficiency and **Python** for the GUI and plotting functionalities. **CMake** is utilized for building the project files, and the framework is designed for **local** deployment, as hosting, database, frontend, and backend components are not required.

### 🛠️ **Technical Stack**

- **Programming Language(s)**: C++ for efficiency, Python for GUI & plotting
- **Deployment/Hosting**: Local; hosting isn't needed for an ML framework
- **Other Tools/Libraries**: CMake for building project files
- **Target Platform**: Cross-platform (Windows, Linux, macOS)

### 🎯 **Design Philosophy**

- **Educational Focus**: Clear, well-documented code for learning ML internals
- **Performance**: Optimized C++ implementation for maximum speed
- **Simplicity**: Clean, intuitive API design similar to PyTorch
- **Modularity**: Well-structured components for easy extension
- **Local Deployment**: Standalone framework with no external dependencies

### 🌟 **Unique Value Proposition**

| **What is the project's name?**               | CGroot++                                                       |
| --------------------------------------------- | -------------------------------------------------------------- |
| **What is the core purpose of the software?** | Mini educational ML framework                                  |
| **Who is the target audience?**               | ML developers + open-source communities                        |
| **What is the unique value proposition?**     | Open source + educational purpose + explaining model decisions |

### 🎓 **Educational Benefits**

- **Transparency**: Every component is clearly documented and easy to understand
- **Learning Path**: Step-by-step implementation of ML concepts from scratch
- **Model Explainability**: Built-in capabilities for understanding model decisions
- **Hands-on Experience**: Direct interaction with low-level ML operations
- **Community Learning**: Open-source nature encourages collaborative learning

---

## 🔧 Installation

### Prerequisites

- **C++ Compiler**: C++17 compatible (GCC 7+, Clang 5+, MSVC 2019+)
- **CMake**: Version 3.10 or higher
- **Visual Studio**: 2019 or later (Windows)
- **Python**: 3.8+ (required for GUI)
- **Qt6**: Required for PyQt6 GUI application

### Python Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

Required packages:

- `PyQt6` - GUI framework
- `pyqtgraph` - Plotting and visualization
- `numpy` - Numerical operations
- `colorama` - Terminal colors

### Windows Installation

1.  **Install Visual Studio 2019** or later with C++ development tools and CMake
2.  **Install Python 3.8+** from [python.org](https://python.org)
3.  **Install Qt6** (CMake will attempt to find it)
4.  **Clone the repository**:
    ```cmd
    git clone <repository-url>
    cd CGROOT
    ```
5.  **Install Python dependencies**:
    ```cmd
    pip install -r requirements.txt
    ```
6.  **Build the project**:
    ```cmd
    python scripts/CGROOT_Manager.py --build
    ```
7.  **Launch the GUI**:
    ```cmd
    python scripts/CGROOT_Manager.py --gui
    ```

### Linux Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install build-essential cmake git python3 python3-pip
sudo apt install qt6-base-dev  # Qt6 for PyQt6

# Install Python dependencies
pip3 install -r requirements.txt

# Clone and build
git clone <repository-url>
cd CGROOT
mkdir build && cd build
cmake ..
make

# Launch GUI
python3 src/gui_py/main.py
```

### macOS Installation

```bash
# Install dependencies with Homebrew
brew install cmake git python qt@6

# Install Python dependencies
pip3 install -r requirements.txt

# Clone and build
git clone <repository-url>
cd CGROOT
mkdir build && cd build
cmake ..
make

# Launch GUI
python3 src/gui_py/main.py
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

| Example           | Description                  | Status            |
| ----------------- | ---------------------------- | ----------------- |
| `simple_test.cpp` | Basic tensor operations demo | ✅ Ready          |
| `xor_solver.cpp`  | XOR problem solver with MLP  | 🚧 In Development |

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
│   │   ├── op_nodes.h/cpp    # Operation nodes
│   │   └── grad_fn.h         # Base class for gradient functions
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
│   ├── 📁 optim/             # Optimizers
│   │   ├── optimizer.h       # Base optimizer class
│   │   ├── sgd.h             # SGD optimizer
│   │   └── adam.h            # Adam optimizer
│   └── 📁 gui_py/             # Python GUI application
│       ├── main.py           # Main GUI entry point
│       └── components/       # GUI components
├── 📁 examples/              # Example programs
│   ├── simple_test.cpp       # Basic functionality demo
│   └── xor_solver.cpp        # XOR problem solver
├── 📁 tests/                 # Unit tests
│   ├── test_tensor.cpp       # Tensor operation tests
│   └── test_autograd.cpp     # Autograd tests
├── 📁 scripts/               # Utility scripts
│   ├── CGROOT_Manager.py     # Cross-platform project manager
│   └── package_app.py        # Script for packaging GUI
├── 📁 build/                 # Build output directory
├── 📄 CMakeLists.txt         # CMake configuration
├── 📄 CGROOT_Manager.bat     # Windows batch script (deprecated by Python manager)
├── 📄 requirements.txt       # Python dependencies
└── 📄 README.md              # This file
```

---

## 🛠️ Available Scripts

### 🪟 Python Manager (Cross-platform)

| Command                                            | Description                                   |
| -------------------------------------------------- | --------------------------------------------- |
| `python scripts/CGROOT_Manager.py`                 | Interactive menu with all options             |
| `python scripts/CGROOT_Manager.py --build`         | Build Release configuration                   |
| `python scripts/CGROOT_Manager.py --clean --build` | Clean and build                               |
| `python scripts/CGROOT_Manager.py --gui`           | Launch GUI application                        |
| `python scripts/CGROOT_Manager.py --full`          | **Full cycle**: clean → build → package → run |
| `python scripts/CGROOT_Manager.py --test`          | Run test executables                          |

### 📦 Packaging

```bash
# Create standalone executable
python scripts/package_app.py

# Output will be in: dist/CGROOT_Trainer/CGROOT_Trainer.exe
```

### 🐧 Linux/macOS Commands

```bash
# Build commands
make                    # Build all targets
make cgroot_core       # Build C++ core only

# Run GUI
python3 src/gui_py/main.py

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

| Component                     | Test File           | Status     | Coverage                |
| ----------------------------- | ------------------- | ---------- | ----------------------- |
| **Tensor Operations**         | `test_tensor.cpp`   | 🚧 Planned | Basic math operations   |
| **Automatic Differentiation** | `test_autograd.cpp` | 🚧 Planned | Gradient computation    |
| **Neural Network Layers**     | Integration tests   | 🚧 Planned | Forward/backward passes |
| **Optimizers**                | Integration tests   | 🚧 Planned | Parameter updates       |

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

## 🔗 Repository & Links

- **GitHub Repository**: [https://github.com/3omd4/CGROOT](https://github.com/3omd4/CGROOT)
- **Gantt Chart**: [Project Timeline](https://www.notion.so/28fa5133a8ef8068aeb9c2e69dc66e37?pvs=21)
- **Issues & Discussions**: [GitHub Issues](https://github.com/3omd4/CGROOT/issues)

---

<div align="center">

**Made with ❤️ by the CGROOT++ Team**

[⭐ Star us on GitHub](https://github.com/3omd4/CGROOT) • [🐛 Report Issues](https://github.com/3omd4/CGROOT/issues) • [💬 Discussions](https://github.com/3omd4/CGROOT/discussions)

</div>
