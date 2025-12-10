from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread
from PyQt6.QtGui import QImage, qGray
import sys
import time

try:
    import cgroot_core
except ImportError:
    cgroot_core = None

class ModelWorker(QObject):
    logMessage = pyqtSignal(str)
    metricsUpdated = pyqtSignal(float, float, int)
    progressUpdated = pyqtSignal(int, int)
    imagePredicted = pyqtSignal(int, object, list) # int, QImage, list of floats
    trainingFinished = pyqtSignal()
    inferenceFinished = pyqtSignal()
    modelStatusChanged = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.dataset = None
        self.should_stop = False
        
    @pyqtSlot(str, str)
    def loadDataset(self, images_path, labels_path):
        if not cgroot_core:
            self.logMessage.emit("Error: Core library not loaded")
            return
            
        self.logMessage.emit(f"Loading dataset from: {images_path}")
        try:
            # We assume bindings provide a way to load data, or we implement basic loader in python
            # if the C++ one is too complex to bind directly with unique_ptrs across boundary without careful work.
            # But we bound MNISTLoader.load_training_data
            
            # Note: unique_ptr binding in pybind11 usually transfers ownership or keeps it managed.
            # Let's assume we get a reference or an object.
            self.dataset = cgroot_core.MNISTLoader.load_training_data(images_path, labels_path)
            
            if self.dataset:
                self.logMessage.emit(f"Loaded {self.dataset.num_images} images")
            else:
                self.logMessage.emit("Failed to load dataset")
        except Exception as e:
            self.logMessage.emit(f"Exception loading dataset: {e}")

    @pyqtSlot(int)
    def trainModel(self, epochs):
        if not self.dataset:
            self.logMessage.emit("No dataset loaded")
            self.trainingFinished.emit()
            return

        self.should_stop = False
        self.modelStatusChanged.emit(True)
        
        try:
            # Initialize model if not exists
            if not self.model:
                arch = cgroot_core.architecture()
                # Setup architecture defaults... this might be tedious to do in Python if struct is complex
                # Ideally we have a helper in C++ or defaults in binding.
                arch.numOfConvLayers = 0
                arch.numOfFCLayers = 0 # Example
                arch.distType = cgroot_core.distributionType.normalDistribution
                
                self.model = cgroot_core.NNModel(arch, 10, 28, 28, 1)
                self.logMessage.emit("Model initialized")

            # Real Training Loop
            num_images = self.dataset.num_images
            total_epochs = epochs
            
            for epoch in range(total_epochs):
                if self.should_stop: break
                
                self.logMessage.emit(f"Epoch {epoch+1}/{total_epochs}")
                
                correct = 0
                
                # Iterate over all images
                images = self.dataset.images
                for i in range(num_images):
                    if self.should_stop: break
                    
                    img_obj = images[i]
                    flat = img_obj.pixels
                    
                    rows = []
                    for r in range(28):
                        row = flat[r*28 : (r+1)*28]
                        rows.append(row)
                    
                    image_data = [rows] 
                    label = int(img_obj.label)
                    
                    # Calculate Accuracy: Predict BEFORE training on this sample
                    # (Online learning evaluation)
                    pred = self.model.classify(image_data)
                    if pred == label:
                        correct += 1
                    
                    # Train
                    self.model.train(image_data, label)
                    
                    if i % 100 == 0:
                        QThread.msleep(1) 
                
                # Calculate metrics
                current_acc = correct / num_images if num_images > 0 else 0.0
                # Proxy loss since C++ doesn't return it
                current_loss = 1.0 - current_acc 
                
                self.metricsUpdated.emit(current_loss, current_acc, epoch+1)
                self.progressUpdated.emit(epoch+1, total_epochs)
                
        except Exception as e:
            self.logMessage.emit(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            
        self.modelStatusChanged.emit(False)
        self.trainingFinished.emit()

    @pyqtSlot()
    def stopTraining(self):
        self.should_stop = True

    @pyqtSlot(object)
    def runInference(self, qimage_obj):
        if not self.model:
            self.logMessage.emit("No model available")
            self.inferenceFinished.emit()
            return

        if not qimage_obj:
            self.logMessage.emit("No image provided for inference")
            self.inferenceFinished.emit()
            return
            
        self.modelStatusChanged.emit(True)
        try:
            # Convert QImage to 3D vector [1][28][28]
            # Ensure 28x28 grayscale
            img = qimage_obj.scaled(28, 28).convertToFormat(QImage.Format.Format_Grayscale8)
            
            width = img.width()
            height = img.height()
            
            rows = []
            for y in range(height):
                row = []
                for x in range(width):
                    # qGray returns 0-255 from QRgb
                    pixel_val = qGray(img.pixel(x, y))
                    row.append(pixel_val)
                rows.append(row)
            
            image_data = [rows] # Depth 1
            
            # Run Classification
            self.logMessage.emit("Running inference on image...")
            predicted_class = self.model.classify(image_data)
            
            self.logMessage.emit(f"Inference Result: {predicted_class}")
            
            # Probabilities (mock for now as C++ classify returns int)
            # Logic: Assign high prob to predicted, low to others
            probs = [0.0] * 10
            probs[predicted_class] = 0.95
            for i in range(10):
                if i != predicted_class:
                    probs[i] = 0.05 / 9.0
            
            self.imagePredicted.emit(predicted_class, qimage_obj, probs)
            
        except Exception as e:
            self.logMessage.emit(f"Inference error: {e}")
            import traceback
            traceback.print_exc()

        self.modelStatusChanged.emit(False)
        self.inferenceFinished.emit()
