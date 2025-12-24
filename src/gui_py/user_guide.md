# CGROOT++ Native GUI User Guide

Welcome to the **CGROOT++ Neural Network Trainer**. This application allows you to train, monitor, and test C++ neural network models using a modern Python interface.

## 1. Getting Started

### Launching the Application
Run the following command from your terminal:
```bash
python src/gui_py/main.py
```

## 2. Workflow Overview

The typical workflow consists of three steps:
1.  **Load Data**: Import your training dataset.
2.  **Train**: Configure hyperparameters and train the model.
3.  **Inference**: Test the trained model on new images.

---

## 3. Step-by-Step Instructions

### Step 1: Loading the Dataset
Before training, you must load the MNIST dataset.
1.  Go to the Menu Bar: **File** -> **Load Dataset...**
2.  **Select Images File**: Browse to your data folder (e.g., `src/data`) and select the images file (e.g., `t10k-images.idx3-ubyte`).
3.  **Select Labels File**: Next, select the corresponding labels file (e.g., `t10k-labels.idx1-ubyte`).
   
> **Note**: If successful, you will see a "Loaded X images" message in the Log Output at the bottom of the window.

### Step 2: Training the Model
Switch to the **Training** tab.
1.  **Configure Parameters**:
    *   **Epochs**: Number of times to iterate over the entire dataset (e.g., 10).
    *   **Batch Size**: Number of samples per update (e.g., 32).
    *   **Learning Rate**: Step size for the optimizer (e.g., 0.01).
    *   **Optimizer**: Select SGD, Adam, etc.
2.  **Start Training**: Click the green **Start Training** button.
3.  **Monitor Progress**:
    *   The **Log Consle** will show per-epoch progress.
    *   The **Metrics** tab (and status bar) will update with the current Loss and Accuracy.

### Step 3: Running Inference
Once training is complete, you can test the model's performance on individual images.
1.  Switch to the **Inference** tab.
2.  **Load Image**: Click **Load Image** to select a specific digit image (PNG/JPG) or use a sample from the test set if implemented.
3.  **Run Inference**: Click **Run Inference**.
    *   **Prediction**: The predicted digit will appear in large text.
    *   **Confidence**: The model's confidence percentage.
    *   **Probabilities**: A ranked list showing the probability for all classes (0-9).

### Step 4: Analyzing Metrics
Switch to the **Metrics** tab to visualize performance.
*   **Charts**: View real-time graphs for **Training Loss** and **Accuracy** over epochs.
*   **Stats**: Check "Best Loss" and "Best Accuracy" achieved during the session.

---

## 4. Configuration
Use the **Configuration** tab to adjust model architecture before training (if supported by the core):
*   **Model**: Set image dimensions (28x28 for MNIST) and number of layers.
*   **Training**: Detailed hyperparameter tuning.

## 5. Troubleshooting
*   **"Core library not loaded"**: Ensure you have built the C++ core using `python scripts/CGROOT_Manager.py --build`.
*   **Dataset Errors**: Ensure you select the *uncompressed* `.ubyte` files, not `.zip` files.
