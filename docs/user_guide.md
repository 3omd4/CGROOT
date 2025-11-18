# CGROOT++ User Guide

## Overview

CGROOT++ is a comprehensive deep learning framework written in C++ with a Qt6-based GUI for training and testing neural networks on the Fashion-MNIST dataset.

## Project Structure

```
CGROOT/
├─ CMakeLists.txt                  # Build configuration
├─ README.md                       # Project overview
├─ LICENSE                         # Open-source license
├─ docs/                           # Documentation
│   └─ user_guide.md               # This file
│
├─ src/                            # Source code
│   ├─ core/                       # Core ML engine
│   │   ├─ layers/                 # Neural network layers
│   │   ├─ activations/            # Activation functions
│   │   ├─ losses/                 # Loss functions
│   │   ├─ optimizers/             # Optimizers
│   │   ├─ utils/                  # Helper utilities
│   │   └─ model.*                 # Sequential model class
│   │
│   ├─ gui/                        # GUI source code
│   │   ├─ widgets/                # GUI components
│   │   ├─ controllers/            # Bridge between GUI & core
│   │   ├─ mainwindow.*            # Main GUI window
│   │   └─ main.cpp                # GUI entry point
│   │
│   └─ examples/                   # Example demos
│
├─ tests/                          # Unit tests
└─ build/                          # Compiled binaries
```

## Building the Project

### Prerequisites

- CMake 3.10 or higher
- C++17 compatible compiler
- Qt6 (for GUI, optional)

### Build Steps

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Using the Framework

### Core API

#### Creating a Model

```cpp
#include "core/model.h"
#include "core/layers/dense.h"
#include "core/activations/relu.h"

// Create model architecture
architecture arch;
arch.numOfLayers = 3;
// ... configure architecture

// Create model
NNModel model(arch, numClasses, imageHeight, imageWidth);
```

#### Training

```cpp
// Load dataset
auto dataset = MNISTLoader::load_training_data(images_path, labels_path);

// Train on each sample
for (const auto& image : dataset->images) {
    std::vector<std::vector<unsigned char>> imageData = convertToImageFormat(image);
    model.train(imageData, image.label);
}
```

#### Inference

```cpp
// Classify an image
int predictedClass = model.classify(imageData);
```

## Using the GUI

### Starting the GUI

```bash
./build/bin/fashion_mnist_gui
```

### Workflow

1. **Load Dataset**: File → Load Dataset
2. **Configure Model**: Configuration tab
3. **Start Training**: Training tab → Start Training
4. **Monitor Metrics**: Metrics tab (real-time charts)
5. **Run Inference**: Inference tab

## Framework Components

### Layers

- **Dense**: Fully connected layer
- **Conv2D**: 2D convolutional layer
- **Pooling**: Max/Average pooling
- **Dropout**: Regularization layer

### Activations

- **ReLU**: Rectified Linear Unit
- **Sigmoid**: Sigmoid function
- **Tanh**: Hyperbolic tangent
- **Softmax**: Softmax for classification

### Loss Functions

- **MSE**: Mean Squared Error
- **Binary Cross-Entropy**: For binary classification
- **Categorical Cross-Entropy**: For multi-class classification

### Optimizers

- **SGD**: Stochastic Gradient Descent
- **Momentum**: SGD with momentum
- **Adam**: Adaptive moment estimation

## Examples

See `src/examples/` for example programs demonstrating framework usage.

## Testing

Run unit tests:

```bash
./build/bin/test_tensor
./build/bin/test_autograd
```

## License

See LICENSE file for details.

