# Fashion-MNIST Neural Network GUI

A comprehensive Qt6-based graphical user interface for training and testing neural networks on the Fashion-MNIST dataset.

## Features

### 🎯 Main Features

- **Real-time Training Visualization**: Live charts showing loss and accuracy metrics
- **Interactive Configuration Panel**: Modify all model and training parameters
- **Image Display**: View Fashion-MNIST images with zoom and pan capabilities
- **Inference Results**: Display predictions with confidence scores and class probabilities
- **Logging System**: Comprehensive logging output for all operations
- **Model Management**: Load and save trained models

### 📊 Components

1. **MainWindow**: Central application window with tabbed interface
2. **TrainingWidget**: Training controls and parameter configuration
3. **InferenceWidget**: Run inference on test images
4. **ConfigurationWidget**: Model architecture and hyperparameter settings
5. **MetricsWidget**: Real-time charts for loss and accuracy
6. **ImageViewerWidget**: Display and interact with Fashion-MNIST images
7. **ModelController**: Background thread controller for model operations

## Building

### Prerequisites

- Qt6 (Core, Widgets, Charts)
- CMake 3.10 or higher
- C++17 compatible compiler

### Build Instructions

```bash
mkdir build
cd build
cmake ..
make
# or on Windows with Visual Studio
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

The GUI executable will be created in `build/bin/fashion_mnist_gui`

## Usage

1. **Load Dataset**: Use File → Load Dataset to load Fashion-MNIST training data
2. **Configure Model**: Go to Configuration tab to set model parameters
3. **Start Training**: Go to Training tab and click "Start Training"
4. **Monitor Progress**: Watch real-time metrics in the Metrics tab
5. **Run Inference**: Use Inference tab to test the model on images

## Architecture

The GUI uses Qt6's signal/slot mechanism for communication between components:

- **MainWindow** coordinates all widgets and handles user actions
- **ModelController** runs in a separate thread to prevent UI blocking
- **Real-time updates** are achieved through Qt signals and timers
- **Charts** update automatically when new metrics are available

## Fashion-MNIST Classes

The dataset contains 10 classes:
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Notes

- The GUI is designed to work with the existing CGroot++ neural network framework
- Model loading/saving functionality needs to be implemented based on your model serialization format
- Configuration saving/loading uses JSON format (implementation needed)

