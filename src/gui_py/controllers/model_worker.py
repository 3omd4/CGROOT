from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, QMetaObject, Qt
from PyQt6.QtGui import QImage, qGray, qRgb
import sys
import time
import random
import math

try:
    import cgroot_core
except ImportError:
    cgroot_core = None

class ModelWorker(QObject):
    logMessage = pyqtSignal(str)
    metricsUpdated = pyqtSignal(float, float, int)  # loss, accuracy, epoch
    progressUpdated = pyqtSignal(int, int)
    imagePredicted = pyqtSignal(int, object, list) # int, QImage, list of floats
    trainingFinished = pyqtSignal()
    inferenceFinished = pyqtSignal()
    modelStatusChanged = pyqtSignal(bool)
    
    # Internal signals for thread-safe callback emissions
    _internal_log = pyqtSignal(str)
    _internal_progress = pyqtSignal(int, int, float, float)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.dataset = None
        self.should_stop = False
        
        # Thread safety: Track if training is active to prevent post-finish callbacks
        self._training_active = False
        
        # Callback lifetime management: Store references to prevent GC during C++ execution
        self._progress_callback_ref = None
        self._log_callback_ref = None
        
        # Connect internal signals (these are thread-safe automatically)
        self._internal_log.connect(self._emit_log_from_thread)
        self._internal_progress.connect(self._emit_progress_from_thread)
        
    @pyqtSlot(str, str)
    def loadDataset(self, images_path, labels_path):
        if not cgroot_core:
            self.logMessage.emit("Error: Core library not loaded")
            return
            
        self.logMessage.emit(f"Loading dataset from: {images_path}")
        try:
            self.dataset = cgroot_core.MNISTLoader.load_training_data(images_path, labels_path)
            
            if self.dataset:
                self.logMessage.emit(f"Loaded {self.dataset.num_images} images")
            else:
                self.logMessage.emit("Failed to load dataset")
        except Exception as e:
            self.logMessage.emit(f"Exception loading dataset: {e}")
            import traceback
            traceback.print_exc()

    def _convert_mnist_to_image_format(self, flat_pixels, height=28, width=28):
        """Convert flat MNIST pixel array to [depth][height][width] format"""
        image_data = [[]]  # Single depth channel
        for y in range(height):
            row = []
            for x in range(width):
                idx = y * width + x
                if idx < len(flat_pixels):
                    row.append(int(flat_pixels[idx]))
                else:
                    row.append(0)
            image_data[0].append(row)
        return image_data

    def _calculate_loss_from_probs(self, probs, true_label):
        """Calculate cross-entropy loss from probabilities"""
        if not probs or len(probs) == 0:
            return 1.0
        
        # Cross-entropy loss: -log(p_true)
        true_prob = probs[true_label] if true_label < len(probs) else 0.0
        # Add small epsilon to avoid log(0)
        true_prob = max(true_prob, 1e-10)
        loss = -math.log(true_prob)
        return loss

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
            # Initialize model if not exists
            if not self.model:
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
                num_classes = config.get('num_classes', 10)
                
                # IMPORTANT: If the last FC layer size equals num_classes, it's likely meant
                # to be the output layer, which is created separately by NNModel.
                # Remove it to avoid double-counting and bad allocation.
                if neurons_list and neurons_list[-1] == num_classes:
                    self.logMessage.emit(f"Detected output layer in FC layers ({neurons_list[-1]}), removing it as output layer is created separately")
                    neurons_list = neurons_list[:-1]
                    num_fc_layers = len(neurons_list)
                    self.logMessage.emit(f"Adjusted to {num_fc_layers} hidden FC layers: {neurons_list}")
                
                # Ensure we have at least one hidden layer
                if not neurons_list or len(neurons_list) == 0:
                    self.logMessage.emit(f"WARNING: No hidden layers specified! Adding default hidden layer with 128 neurons")
                    neurons_list = [128]
                    num_fc_layers = 1
                
                # Sanity check: Ensure neurons list length matches num_fc_layers
                if len(neurons_list) != num_fc_layers:
                    self.logMessage.emit(f"Warning: Neurons list length ({len(neurons_list)}) does not match FC Layers count ({num_fc_layers}). Using list length.")
                    num_fc_layers = len(neurons_list)
                    
                arch.numOfFCLayers = num_fc_layers
                arch.neuronsPerFCLayer = neurons_list
                
                # self.logMessage.emit(f"DEBUG: After setting arch.neuronsPerFCLayer:")
                # self.logMessage.emit(f"  Type: {type(neurons_list)}")
                # self.logMessage.emit(f"  Value: {neurons_list}")
                # self.logMessage.emit(f"  Length: {len(neurons_list)}")
                # self.logMessage.emit(f"  arch.numOfFCLayers: {arch.numOfFCLayers}")
                
                # Default activations and init functions
                # All FC layers use ReLU except output uses Softmax (handled by output layer)
                arch.FCLayerActivationFunc = [cgroot_core.activationFunction.RelU] * num_fc_layers
                arch.FCInitFunctionsType = [cgroot_core.initFunctions.Xavier] * num_fc_layers
                
                arch.distType = cgroot_core.distributionType.normalDistribution
                
                img_h = config.get('image_height', 28)
                img_w = config.get('image_width', 28)
                
                # Optimizer Config
                opt_type_str = config.get('optimizer', 'Adam')
                lr = config.get('learning_rate', 0.001)
                wd = config.get('weight_decay', 0.0001)
                mom = config.get('momentum', 0.9)
                
                # Setup OptimizerConfig
                arch.optConfig.learningRate = lr
                arch.optConfig.weightDecay = wd
                arch.optConfig.momentum = mom
                arch.optConfig.beta1 = 0.9
                arch.optConfig.beta2 = 0.999
                arch.optConfig.epsilon = 1e-8
                
                if opt_type_str == "Adam":
                    arch.optConfig.type = cgroot_core.OptimizerType.Adam
                elif opt_type_str == "RMSprop" or opt_type_str == "RMSProp":
                    arch.optConfig.type = cgroot_core.OptimizerType.RMSprop
                else:
                    arch.optConfig.type = cgroot_core.OptimizerType.SGD

                # Set legacy learningRate for compatibility
                arch.learningRate = lr
                
                # Set batch size
                batch_size = config.get('batch_size', 32)
                arch.batch_size = batch_size

                # Debug logging to understand the architecture before model creation
                # self.logMessage.emit(f"================ Architecture Debug Info ================")
                # self.logMessage.emit(f"Number of Conv Layers: {arch.numOfConvLayers}")
                # self.logMessage.emit(f"Number of FC Layers: {arch.numOfFCLayers}")
                # self.logMessage.emit(f"Neurons per FC Layer: {arch.neuronsPerFCLayer}")
                # self.logMessage.emit(f"FC Activation Functions: {arch.FCLayerActivationFunc}")
                # self.logMessage.emit(f"FC Init Functions: {arch.FCInitFunctionsType}")
                # self.logMessage.emit(f"Image dimensions: {img_h}x{img_w}x1")
                # self.logMessage.emit(f"Number of classes: {num_classes}")
                # self.logMessage.emit(f"Calculated input size (flattened): {img_h * img_w * 1}")

                # self.logMessage.emit(f"Convolutional layers kernels: {arch.kernelsPerconvLayers}")
                # self.logMessage.emit(f"Fully connected layers neurons: {arch.neuronsPerFCLayer}")
                # self.logMessage.emit(f"Convolutional layers activation functions: {arch.convLayerActivationFunc}")
                # self.logMessage.emit(f"Fully connected layers activation functions: {arch.FCLayerActivationFunc}")
                # self.logMessage.emit(f"Convolutional layers init functions: {arch.convInitFunctionsType}")
                # self.logMessage.emit(f"Fully connected layers init functions: {arch.FCInitFunctionsType}")
                # self.logMessage.emit(f"Pooling layers interval: {arch.poolingLayersInterval}")
                # self.logMessage.emit(f"Pooling layers type: {arch.poolingtype}")
                # self.logMessage.emit(f"Pooling layers kernels: {arch.kernelsPerPoolingLayer}")


                # self.logMessage.emit(f"Distribution type: {arch.distType}")

                # self.logMessage.emit(f"Learning rate: {arch.learningRate}")
                # self.logMessage.emit(f"Batch size: {arch.batch_size}")
                # self.logMessage.emit(f"Optimizer type: {arch.optConfig.type}")
                # self.logMessage.emit(f"Optimizer learning rate: {arch.optConfig.learningRate}")
                # self.logMessage.emit(f"Optimizer weight decay: {arch.optConfig.weightDecay}")
                # self.logMessage.emit(f"Optimizer momentum: {arch.optConfig.momentum}")
                # self.logMessage.emit(f"Optimizer beta1: {arch.optConfig.beta1}")
                # self.logMessage.emit(f"Optimizer beta2: {arch.optConfig.beta2}")
                # self.logMessage.emit(f"Optimizer epsilon: {arch.optConfig.epsilon}")
                # self.logMessage.emit(f"Batch size: {batch_size}")



                # self.logMessage.emit(f"=============================================================")

                # Validate configuration before creating model
                validation_errors = []
                
                if num_fc_layers != len(neurons_list):
                    validation_errors.append(
                        f"FC layer count mismatch: num_fc_layers={num_fc_layers} but "
                        f"neurons_per_fc_layer has {len(neurons_list)} values")
                
                if num_fc_layers != len(arch.FCLayerActivationFunc):
                    validation_errors.append(
                        f"FC activation function count mismatch: expected {num_fc_layers}, "
                        f"got {len(arch.FCLayerActivationFunc)}")
                
                if num_fc_layers != len(arch.FCInitFunctionsType):
                    validation_errors.append(
                        f"FC init function count mismatch: expected {num_fc_layers}, "
                        f"got {len(arch.FCInitFunctionsType)}")
                
                if any(n <= 0 for n in neurons_list):
                    validation_errors.append(
                        f"All neuron counts must be positive. Got: {neurons_list}")
                
                if img_h <= 0 or img_w <= 0:
                    validation_errors.append(
                        f"Image dimensions must be positive. Got: {img_h}x{img_w}")
                
                if num_classes <= 0:
                    validation_errors.append(
                        f"Number of classes must be positive. Got: {num_classes}")
                
                if validation_errors:
                    error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in validation_errors)
                    self.logMessage.emit(f"ERROR: {error_msg}")
                    raise ValueError(error_msg)

                # self.logMessage.emit(f"Architecture configured")
                # self.logMessage.emit(f"=== ABOUT TO CREATE MODEL ===")
                # self.logMessage.emit(f"Creating NNModel with:")
                # self.logMessage.emit(f"  num_classes={num_classes}")
                # self.logMessage.emit(f"  img_h={img_h}, img_w={img_w}, depth=1")
                # self.logMessage.emit(f"  neurons_list={neurons_list}")
                # self.logMessage.emit(f"  Expected output layer input size: {neurons_list[-1] if neurons_list else 'N/A'}")
                # self.logMessage.emit(f"  Expected output layer output size: {num_classes}")
                
                self.model = cgroot_core.NNModel(arch, num_classes, img_h, img_w, 1)
                self.logMessage.emit(f"Model initialized successfully")

            # Create training configuration for C++
            train_config = cgroot_core.TrainingConfig()
            train_config.epochs = epochs
            train_config.batch_size = config.get('batch_size', 32)
            train_config.validation_split = config.get('validation_split', 0.2)
            train_config.use_validation = config.get('use_validation', True)
            train_config.shuffle = True
            train_config.random_seed = 42
            
            # Define thread-safe callbacks for C++ to call
            # These use QMetaObject.invokeMethod to safely cross thread boundaries
            def progress_callback(epoch, total_epochs, loss, accuracy):
                """
                Thread-safe progress callback.
                Emits internal signal which is automatically thread-safe in Qt.
                """
                self._internal_progress.emit(epoch, total_epochs, loss, accuracy)
            
            def log_callback(message):
                """
                Thread-safe log callback.
                Emits internal signal which is automatically thread-safe in Qt.
                """
                self._internal_log.emit(message)


            # Store callback references to prevent garbage collection during C++ execution
            self._progress_callback_ref = progress_callback
            self._log_callback_ref = log_callback
            self._training_active = True
            
            self.logMessage.emit(f"=== Training Session Start ===")
            self.logMessage.emit(f"Worker thread ID: {int(QThread.currentThreadId())}")
            
            # Call the C++ train_epochs method (this is where all the magic happens!)
            self.logMessage.emit(f"Starting C++ training loop for {epochs} epochs...")
            # history = self.model.train_epochs(
            #     self.dataset, 
            #     train_config,
            #     progress_callback,
            #     log_callback
            # )
            
            self.logMessage.emit(f"Training completed! Total epochs: {len(history)}")
            
        except Exception as e:
            self.logMessage.emit(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # CRITICAL: Clear training state and callbacks even if exception occurs
            # This MUST happen before emitting trainingFinished signal
            self._training_active = False
            self._progress_callback_ref = None
            self._log_callback_ref = None
            self.logMessage.emit(f"=== Training Session End ===")
            
            # Reset model status and notify completion
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
            img = qimage_obj.scaled(28, 28).convertToFormat(QImage.Format.Format_Grayscale8)
            
            width = img.width()
            height = img.height()
            
            rows = []
            for y in range(height):
                row = []
                for x in range(width):
                    pixel_val = qGray(img.pixel(x, y))
                    row.append(pixel_val)
                rows.append(row)
            
            image_data = [rows]  # Depth 1
            
            # Run Classification
            self.logMessage.emit("Running inference on image...")
            predicted_class = self.model.classify(image_data)
            
            self.logMessage.emit(f"Inference Result: {predicted_class}")
            
            # Get probabilities from model
            probs = self.model.getProbabilities()
            if not probs or len(probs) == 0:
                probs = [0.0] * 10
                probs[predicted_class] = 1.0
            
            confidence = max(probs) if probs else 0.0
            self.logMessage.emit(f"Inference Confidence: {confidence:.4f}")
            
            self.imagePredicted.emit(predicted_class, qimage_obj, probs)
            
        except Exception as e:
            self.logMessage.emit(f"Inference error: {e}")
            import traceback
            traceback.print_exc()

        self.modelStatusChanged.emit(False)
        self.inferenceFinished.emit()
    
    # ========================================================================
    # Thread-Safe Signal Emission Helpers
    # ========================================================================
    
    @pyqtSlot(str)
    def _emit_log_from_thread(self, message):
        """
        Thread-safe log emission helper.
        Called via QMetaObject.invokeMethod from C++ callback thread.
        Only emits if training is still active to prevent post-finish emissions.
        """
        if self._training_active:
            self.logMessage.emit(message)
    
    @pyqtSlot(int, int, float, float)
    def _emit_progress_from_thread(self, epoch, total_epochs, loss, accuracy):
        """
        Thread-safe progress/metrics emission helper.
        Called via QMetaObject.invokeMethod from C++ callback thread.
        Only emits if training is still active to prevent post-finish emissions.
        """
        if self._training_active:
            self.metricsUpdated.emit(loss, accuracy, epoch)
            self.progressUpdated.emit(epoch, total_epochs)
    
    # ========================================================================
    # Lifecycle Management
    # ========================================================================
    
    def cleanup(self):
        """
        Explicit cleanup method for safe resource deallocation.
        Call this before destroying the worker object.
        
        Destruction order:
        1. Stop training flag
        2. Clear callback references
        3. Destroy model
        4. Clear dataset
        """
        self.logMessage.emit("=== ModelWorker Cleanup Start ===")
        
        # 1. Stop any ongoing training
        self.should_stop = True
        self._training_active = False
        
        #2. Clear callback references to allow Python GC
        self._progress_callback_ref = None
        self._log_callback_ref = None
        
        # 3. Destroy C++ model object
        if self.model:
            self.logMessage.emit("Destroying C++ model...")
            try:
                del self.model
                self.model = None
            except Exception as e:
                self.logMessage.emit(f"Error destroying model: {e}")
        
        # 4. Clear dataset
        if self.dataset:
            try:
                del self.dataset
                self.dataset = None
            except Exception as e:
                self.logMessage.emit(f"Error clearing dataset: {e}")
        
        self.logMessage.emit("=== ModelWorker Cleanup Complete ===")
    
    def __del__(self):
        """
        Destructor safety net - ensures cleanup on object destruction.
        Not guaranteed to be called, so explicit cleanup() is preferred.
        """
        try:
            if hasattr(self, 'model') and self.model:
                self.cleanup()
        except:
            pass  # Avoid exceptions in destructor
