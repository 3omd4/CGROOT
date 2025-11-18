# CGROOT++ Project Structure

This document describes the complete project structure after reorganization.

## Directory Structure

```
CGROOT/
├─ CMakeLists.txt                  # Build configuration for C++ core engine
├─ README.md                       # Project overview and instructions
├─ LICENSE                         # Open-source license
├─ docs/                           # Documentation
│   └─ user_guide.md               # How to use framework and GUI
│
├─ src/                            # Source code
│   ├─ core/                       # Core ML engine
│   │   ├─ layers/                 # Neural network layers
│   │   │   ├─ dense.*              # Dense (fully connected) layer
│   │   │   ├─ conv2d.*            # Convolutional layer
│   │   │   ├─ pooling.*           # Pooling layers (Max, Avg)
│   │   │   ├─ dropout.*           # Dropout layer
│   │   │   ├─ layer.*             # Base layer class
│   │   │   └─ layers.h            # Layer declarations
│   │   │
│   │   ├─ activations/            # Activation functions
│   │   │   ├─ relu.h 
│   │   │   ├─ sigmoid.h 
│   │   │   ├─ tanh.h 
│   │   │   └─ softmax.h 
│   │   │
│   │   ├─ losses/                 # Loss functions
│   │   │   ├─ mse.h 
│   │   │   ├─ binary_crossentropy.h 
│   │   │   └─ categorical_crossentropy.h 
│   │   │
│   │   ├─ optimizers/             # Optimizers
│   │   │   ├─ sgd.h 
│   │   │   ├─ momentum.h 
│   │   │   └─ adam.h 
│   │   │
│   │   ├─ utils/                  # Helper utilities
│   │   │   ├─ weight_init.h 
│   │   │   ├─ data_loader.*       # Data loading utilities
│   │   │   ├─ mnist_loader.*      # MNIST/Fashion-MNIST loader
│   │   │   └─ metrics.h 
│   │   │
│   │   └─ model.*                 # Sequential model class and training loop
│   │
│   ├─ gui/                        # GUI source code
│   │   ├─ widgets/                 # Individual GUI components
│   │   │   ├─ configurationwidget.* 
│   │   │   ├─ imageviewerwidget.* 
│   │   │   ├─ inferencewidget.* 
│   │   │   ├─ metricswidget.* 
│   │   │   └─ trainingwidget.* 
│   │   │
│   │   ├─ controllers/             # Bridge between GUI & core engine
│   │   │   └─ modelcontroller.* 
│   │   │
│   │   ├─ mainwindow.*            # Main GUI window
│   │   └─ main.cpp                # GUI entry point
│   │
│   ├─ autograd/                   # Automatic differentiation
│   │   ├─ graph.*
│   │   └─ op_nodes.*
│   │
│   ├─ math/                       # Mathematical operations
│   │   ├─ cpu_kernels.*
│   │   └─ matrix_ops.*
│   │
│   ├─ examples/                   # Small demos to test framework
│   │   ├─ xor_demo.* 
│   │   ├─ simple_classification.* 
│   │   └─ regression_demo.* 
│   │
│   └─ main.cpp                    # Core engine entry point
│
├─ tests/                          # Unit tests for core engine
│   ├─ test_layers.* 
│   ├─ test_activations.* 
│   ├─ test_optimizers.*
│   └─ test_matrices_operations.*
│
└─ build/                          # Compiled binaries (auto-generated)
```

## Key Changes from Previous Structure

1. **Consolidated Core Components**: All core ML functionality moved under `src/core/`
2. **Organized by Function**: Layers, activations, losses, optimizers in separate directories
3. **GUI Reorganization**: Widgets and controllers separated into subdirectories
4. **Examples Moved**: Examples now in `src/examples/` instead of root `examples/`
5. **Utils Consolidation**: Data loaders and utilities in `src/core/utils/`

## Build Targets

- `cgroot_lib`: Core library
- `cgrunner`: Command-line runner
- `fashion_mnist_gui`: Qt6 GUI application
- `simple_test`: Example program
- `xor_solver`: XOR demo
- `test_tensor`: Tensor tests
- `test_autograd`: Autograd tests

## Include Paths

When including files, use relative paths from `src/`:

```cpp
// Core components
#include "core/model.h"
#include "core/layers/dense.h"
#include "core/activations/relu.h"
#include "core/losses/mse.h"
#include "core/optimizers/sgd.h"

// GUI components
#include "gui/widgets/trainingwidget.h"
#include "gui/controllers/modelcontroller.h"
```

