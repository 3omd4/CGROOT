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

    @pyqtSlot(dict)
    def trainModel(self, config):
        epochs = config.get('epochs', 10)
        
        if not self.dataset:
            self.logMessage.emit("No dataset loaded")
            self.trainingFinished.emit()
            return

        self.should_stop = False
        self.modelStatusChanged.emit(True)
        
        try:
            # Initialize model if not exists or force re-init
            if not self.model: # or True if we want to reset
                self.logMessage.emit("Initializing NNModel with Config...")
                arch = cgroot_core.architecture()
                # Clear vectors to ensure clean state
                arch.kernelsPerconvLayers = []
                arch.neuronsPerFCLayer = []
                arch.convLayerActivationFunc = []
                arch.FCLayerActivationFunc = []
                arch.convInitFunctionsType = []
                arch.FCInitFunctionsType = []
                arch.poolingLayersInterval = []
                arch.poolingtype = []
                arch.kernelsPerPoolingLayer = []
                
                # Dynamic Config
                arch.numOfConvLayers = config.get('num_conv_layers', 0)
                
                num_fc_layers = config.get('num_fc_layers', 2)
                neurons_list = config.get('neurons_per_fc_layer', [128, 10])
                
                # Sanity check: Ensure neurons list length matches num_fc_layers
                # If mismatch, pad or truncate. 
                # Better approach: Adjust num_fc_layers to match list if list is explicit.
                if len(neurons_list) != num_fc_layers:
                    self.logMessage.emit(f"Warning: Neurons list length ({len(neurons_list)}) does not match FC Layers count ({num_fc_layers}). Using list length.")
                    num_fc_layers = len(neurons_list)
                    
                arch.numOfFCLayers = num_fc_layers
                arch.neuronsPerFCLayer = neurons_list
                
                # Default activations and init functions for now (can expand config later)
                # We need one per layer
                arch.FCLayerActivationFunc = [cgroot_core.activationFunction.RelU] * (num_fc_layers - 1) + [cgroot_core.activationFunction.Softmax]
                arch.FCInitFunctionsType = [cgroot_core.initFunctions.Kaiming] * (num_fc_layers - 1) + [cgroot_core.initFunctions.Xavier]
                
                arch.distType = cgroot_core.distributionType.normalDistribution
                
                num_classes = config.get('num_classes', 10)
                img_h = config.get('image_height', 28)
                img_w = config.get('image_width', 28)
                
                self.model = cgroot_core.NNModel(arch, num_classes, img_h, img_w, 1)
                self.logMessage.emit(f"Model initialized (MLP: Input->{neurons_list})")

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
            # Logic: Since core logic is minimal, we reflect uncertainty.
            # But visually we highlight the predicted class.
            probs = [0.0] * 10
            probs[predicted_class] = 1.0 # 100% confidence in the specific prediction
            # Ideally C++ classify should return vector<double>
            
            self.imagePredicted.emit(predicted_class, qimage_obj, probs)
            
        except Exception as e:
            self.logMessage.emit(f"Inference error: {e}")
            import traceback
            traceback.print_exc()

        self.modelStatusChanged.emit(False)
        self.inferenceFinished.emit()
