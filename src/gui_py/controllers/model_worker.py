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

# ========================== Training Thread ==========================
class TrainingThread(QThread):
    progress = pyqtSignal(int, int, float, float)  # epoch, total, loss, acc
    log = pyqtSignal(str)
    finished = pyqtSignal(list)  # history

    def __init__(self, model, dataset, config):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.config = config
        self._stop_requested = False

    def run(self):
        # Thread-safe callbacks for C++ training
        def progress_callback(epoch, total, loss, acc):
            if not self._stop_requested:
                self.progress.emit(epoch, total, loss, acc)

        def log_callback(msg):
            if not self._stop_requested:
                self.log.emit(msg)

        def stop_check():
            return self._stop_requested

        try:
            history = self.model.train_epochs(
                self.dataset,
                self.config,
                progress_callback,
                log_callback,
                stop_check
            )
            if not self._stop_requested:
                self.finished.emit(history)
        except Exception as e:
            self.log.emit(f"Training error: {e}")

    def stop(self):
        self._stop_requested = True

# ========================== Loader Thread ==========================
class LoaderThread(QThread):
    loaded = pyqtSignal(object)  # dataset object
    log = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, images_path, labels_path):
        super().__init__()
        self.images_path = images_path
        self.labels_path = labels_path

    def run(self):
        try:
            self.log.emit(f"Loading dataset from: {self.images_path}")
            if not cgroot_core:
                self.error.emit("Core library not loaded")
                return

            dataset = cgroot_core.MNISTLoader.load_training_data(self.images_path, self.labels_path)
            self.loaded.emit(dataset)
        except Exception as e:
            self.error.emit(f"Exception loading dataset: {e}")

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
        self._training_active = False
        self._train_thread = None
        self._loader_thread = None
        
        # Connect internal signals (these are thread-safe automatically)
        self._internal_log.connect(self._emit_log_from_thread)
        self._internal_progress.connect(self._emit_progress_from_thread)
        
    
    @pyqtSlot(str, str)
    def loadDataset(self, images_path, labels_path):
        """Load dataset asynchronously in a background thread."""
        if not cgroot_core:
            self.logMessage.emit("Error: Core library not loaded")
            return
            
        self.logMessage.emit(f"Starting async dataset load...")
        
        # Create and start loader thread
        self._loader_thread = LoaderThread(images_path, labels_path)
        self._loader_thread.log.connect(self.logMessage.emit)
        self._loader_thread.error.connect(self._on_loader_error)
        self._loader_thread.loaded.connect(self._on_dataset_loaded)
        self._loader_thread.start()
    
    @pyqtSlot(object)
    def _on_dataset_loaded(self, dataset):
        """Handle successful dataset loading."""
        self.dataset = dataset
        if self.dataset:
            self.logMessage.emit(f"Dataset loaded successfully: {self.dataset.num_images} images")
        else:
            self.logMessage.emit("Failed to load dataset (returned None)")
        
        # Clean up thread
        self._loader_thread = None

    @pyqtSlot(str)
    def _on_loader_error(self, error_msg):
        """Handle dataset loading error."""
        self.logMessage.emit(error_msg)
        self._loader_thread = None



    @pyqtSlot(dict)
    def trainModel(self, config):
        """Train the model asynchronously using a separate thread."""
        if not self.dataset:
            self.logMessage.emit("No dataset loaded")
            self.trainingFinished.emit()
            return

        if not self.model:
            self._initialize_model(config)

        # Create training configuration
        train_config = self._create_training_config(config)

        # Start training in separate thread
        self._train_thread = TrainingThread(self.model, self.dataset, train_config)
        self._train_thread.progress.connect(self._internal_progress.emit)
        self._train_thread.log.connect(self._internal_log.emit)
        self._train_thread.finished.connect(self._on_training_finished)
        self._training_active = True
        self.should_stop = False
        self.modelStatusChanged.emit(True)
        self.logMessage.emit("=== Training Session Start ===")
        self._train_thread.start()
    
    def _on_training_finished(self, history):
        """Handle training completion."""
        self._training_active = False
        self.modelStatusChanged.emit(False)
        self.logMessage.emit("=== Training Session End ===")
        self.logMessage.emit(f"Training completed! Total epochs: {len(history)}")
        self.trainingFinished.emit()
        self._train_thread = None
    
    
    def _initialize_model(self, config):
        """Initialize the neural network model with the given configuration using C++ factory."""
        self.logMessage.emit("Initializing NNModel with Config (via C++ Factory)...")
        try:
            self.model = cgroot_core.create_model(config)
            self.logMessage.emit("Model initialized successfully")
        except Exception as e:
            self.logMessage.emit(f"Error initializing model: {e}")
            raise e

    
    def _create_training_config(self, config):
        """Create a TrainingConfig object from the provided configuration dict."""
        train_config = cgroot_core.TrainingConfig()
        train_config.epochs = config.get('epochs', 10)
        train_config.batch_size = config.get('batch_size', 32)
        train_config.validation_split = config.get('validation_split', 0.2)
        train_config.use_validation = config.get('use_validation', True)
        train_config.shuffle = True
        train_config.random_seed = 42
        return train_config

    @pyqtSlot()
    def stopTraining(self):
        """Stop the training thread."""
        self.should_stop = True
        if hasattr(self, "_train_thread") and self._train_thread:
            self._train_thread.stop()
            self._train_thread.wait(5000)  # Wait up to 5 seconds for thread to finish

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
            # Convert QImage to 8-bit grayscale
            img = qimage_obj.scaled(28, 28).convertToFormat(QImage.Format.Format_Grayscale8)
            
            width = img.width()
            height = img.height()
            stride = img.bytesPerLine()
            
            # fast access to bytes
            bits = img.bits()
            bits.setsize(img.sizeInBytes())
            
            # Run Classification using C++ helper
            self.logMessage.emit("Running inference on image (C++ optimized)...")
            predicted_class = cgroot_core.classify_pixels(self.model, bits, width, height, stride)
            
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
