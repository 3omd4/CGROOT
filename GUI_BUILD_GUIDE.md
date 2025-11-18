# Fashion-MNIST GUI Build Guide

## Overview

This document provides complete instructions for building and using the Qt6-based GUI for the Fashion-MNIST neural network trainer.

## Project Structure

```
src/gui/
├── main.cpp                 # Application entry point
├── mainwindow.h/cpp         # Main application window
├── trainingwidget.h/cpp     # Training controls and parameters
├── inferencewidget.h/cpp    # Inference interface
├── configurationwidget.h/cpp # Model configuration panel
├── metricswidget.h/cpp      # Real-time metrics charts
├── imageviewerwidget.h/cpp  # Image display with zoom/pan
├── modelcontroller.h/cpp    # Background model operations
└── README.md                # GUI documentation
```

## Prerequisites

### Required Software

1. **Qt6** (version 6.0 or higher)
   - Components needed: Core, Widgets, Charts
   - Download from: https://www.qt.io/download

2. **CMake** (version 3.10 or higher)
   - Download from: https://cmake.org/download/

3. **C++ Compiler**
   - Windows: Visual Studio 2019 or later (with C++17 support)
   - Linux: GCC 7+ or Clang 5+
   - macOS: Xcode 10+

### Installing Qt6

#### Windows
```powershell
# Using Qt Online Installer
# Download from https://www.qt.io/download
# Select: Qt 6.x.x → MSVC 2019 64-bit → Qt Charts
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install qt6-base-dev qt6-charts-dev cmake build-essential
```

#### macOS
```bash
brew install qt@6
brew link qt@6
```

## Building the GUI

### Step 1: Configure CMake

```bash
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/qt6
```

**Windows (Visual Studio):**
```powershell
cmake .. -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH="C:/Qt/6.x.x/msvc2019_64"
```

**Linux:**
```bash
cmake .. -DCMAKE_PREFIX_PATH=/usr/lib/qt6
```

**macOS:**
```bash
cmake .. -DCMAKE_PREFIX_PATH=$(brew --prefix qt@6)
```

### Step 2: Build

**Windows:**
```powershell
cmake --build . --config Release
```

**Linux/macOS:**
```bash
make -j4
```

### Step 3: Run

The executable will be in `build/bin/fashion_mnist_gui`

**Windows:**
```powershell
.\bin\Release\fashion_mnist_gui.exe
```

**Linux/macOS:**
```bash
./bin/fashion_mnist_gui
```

## Features

### 1. Main Window
- Tabbed interface with 4 main sections
- Menu bar with File, Training, Inference, Help menus
- Toolbar with quick access buttons
- Status bar with progress indicator
- Log output panel at bottom

### 2. Training Tab
- Start/Stop training controls
- Training parameters:
  - Number of epochs
  - Batch size
  - Learning rate
  - Optimizer selection (SGD, Adam, RMSprop)
  - Validation split percentage
  - Checkpoint saving options

### 3. Inference Tab
- Load and display Fashion-MNIST images
- Run inference on single images or batches
- Display predictions with confidence scores
- Class probability distribution
- Interactive image viewer with zoom/pan

### 4. Configuration Tab
- **Model Configuration:**
  - Number of classes (default: 10 for Fashion-MNIST)
  - Image dimensions (28x28 for Fashion-MNIST)
  - Number of layers
  - Layer types and sizes

- **Training Configuration:**
  - Optimizer settings
  - Learning rate, weight decay, momentum
  - Epochs and batch size
  - Validation settings

- **Network Architecture:**
  - Layer configuration
  - Activation functions
  - Convolutional layer parameters

### 5. Metrics Tab
- **Real-time Charts:**
  - Training loss over epochs
  - Training accuracy over epochs
  - Auto-scaling axes
  - Smooth line plots

- **Statistics Display:**
  - Current epoch, loss, accuracy
  - Best loss and accuracy achieved
  - Real-time updates during training

## Usage Workflow

### Training a Model

1. **Load Dataset:**
   - File → Load Dataset
   - Select Fashion-MNIST images file (train-images-idx3-ubyte)
   - Select Fashion-MNIST labels file (train-labels-idx1-ubyte)

2. **Configure Model:**
   - Go to Configuration tab
   - Set model parameters (classes, image size, layers)
   - Set training parameters (epochs, batch size, learning rate)

3. **Start Training:**
   - Go to Training tab
   - Review parameters
   - Click "Start Training"
   - Monitor progress in Metrics tab

4. **Monitor Training:**
   - Watch real-time loss and accuracy charts
   - Check log output for detailed information
   - View progress bar in status bar

5. **Save Model:**
   - File → Save Model
   - Choose location and filename

### Running Inference

1. **Load Model:**
   - File → Load Model (if not already trained)

2. **Run Inference:**
   - Go to Inference tab
   - Set number of samples
   - Click "Run Inference"
   - View predictions with confidence scores

3. **View Results:**
   - See predicted class and confidence
   - View probability distribution for all classes
   - Images are displayed with predictions

## Fashion-MNIST Classes

The dataset contains 10 clothing categories:

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

## Technical Details

### Architecture

- **Qt6 Widgets**: Traditional widget-based UI (not QML)
- **Qt Charts**: Real-time plotting and visualization
- **Multi-threading**: Model operations run in background thread
- **Signal/Slot**: Qt's event-driven communication
- **MOC**: Qt's Meta-Object Compiler for signals/slots

### Threading Model

- **Main Thread**: UI rendering and user interaction
- **Worker Thread**: Model training and inference operations
- **Communication**: Qt signals/slots for thread-safe updates

### Real-time Updates

- Charts update every 100ms during training
- Metrics are emitted from worker thread
- UI updates are queued to main thread automatically
- Smooth animations with Qt Charts

## Troubleshooting

### Qt6 Not Found

**Error:** `Could not find a package configuration file provided by "Qt6"`

**Solution:**
```bash
# Set CMAKE_PREFIX_PATH to Qt6 installation
cmake .. -DCMAKE_PREFIX_PATH=/path/to/qt6
```

### Missing Qt Charts

**Error:** `Could not find Qt6Charts`

**Solution:**
- Install Qt Charts module
- Ensure Qt6 Charts is selected during Qt installation

### Compilation Errors

**Error:** `'QMainWindow' file not found`

**Solution:**
- Verify Qt6 is properly installed
- Check CMAKE_PREFIX_PATH points to Qt6
- Ensure Qt6 Widgets component is installed

### Runtime Errors

**Error:** Application crashes on startup

**Solution:**
- Check that all Qt6 DLLs are in PATH (Windows)
- Verify Qt6 libraries are linked correctly
- Check log output for specific error messages

## Customization

### Changing Theme

Edit `src/gui/main.cpp` to modify the color palette:

```cpp
darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
// Modify colors as needed
```

### Adding New Metrics

1. Add new series to `MetricsWidget`
2. Update `ModelController` to emit new metrics
3. Connect signals in `MainWindow`

### Extending Configuration

1. Add new widgets to `ConfigurationWidget`
2. Update `ModelParameters` or `TrainingParameters` structs
3. Connect to `ModelController::updateParameters()`

## Integration with Existing Code

The GUI integrates with your existing neural network code:

- Uses `NNModel` from `src/model/model.h`
- Uses `MNISTLoader` from `src/data/mnist_loader.h`
- Compatible with existing layer implementations
- Works with your training/inference functions

## Next Steps

1. **Implement Model Serialization:**
   - Add save/load functionality in `ModelController`
   - Use binary or JSON format

2. **Add Configuration Persistence:**
   - Implement JSON save/load in `ConfigurationWidget`
   - Store user preferences

3. **Enhance Visualization:**
   - Add confusion matrix
   - Add layer visualization
   - Add weight histograms

4. **Performance Optimization:**
   - Add GPU support indicators
   - Optimize chart updates
   - Add batch processing visualization

## Support

For issues or questions:
- Check the log output in the GUI
- Review Qt6 documentation: https://doc.qt.io/qt-6/
- Check CMake configuration output

## License

This GUI is part of the CGroot++ project.

