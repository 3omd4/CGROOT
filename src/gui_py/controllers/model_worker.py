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
    # epoch, total, loss, acc, feature_maps(object), pred, qimage, probs, current_idx, layer_type, true_label
    progress = pyqtSignal(int, int, float, float, object, int, object, list, int, int, int)
    log = pyqtSignal(str)

    def __init__(self, model, dataset, config):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.config = config
        self._stop_requested = False
        self.layer_index = 1 # Default to layer 1
        self._visualizations_enabled = True
        
    def set_visualizations_enabled(self, enabled):
        self._visualizations_enabled = enabled
        
    def set_layer_index(self, idx):
        self.layer_index = idx

    def run(self):
        # Thread-safe callbacks for C++ training
        def progress_callback(epoch, total, loss, acc, current_idx=-1):
            if not self._stop_requested:
                # Safe to access model here as C++ execution is paused in callback
                maps = None
                layer_type = -1
                
                # Fetch feature maps ONLY at epoch end (or if we decide to do it more often)
                if self._visualizations_enabled:
                    # Fetch feature maps ONLY at epoch end (or if we decide to do it more often)
                    # But careful: per-sample updates shouldn't fetch heavy maps.
                    try:
                        # Access dynamic layer index
                        target_layer = self.layer_index
                        if target_layer < 0: target_layer = 0
                        
                        # Get maps. If invalid layer, this returns []
                        maps = self.model.getLayerFeatureMaps(target_layer)
                        layer_type = self.model.getLayerType(target_layer)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        # No fallback! Let it be empty list or None so UI knows it failed/is invalid.
                        maps = [] # Explicitly empty so UI clears it
                        layer_type = -1
                    
                    # Also, pick a random image from dataset to show as "preview"
                    # This ensures the user sees 'Activity'
                    preview_img_data = None
                    preview_pred = -1
                    preview_probs = []
                    preview_label = -1
                    
                    if self.dataset and hasattr(self.dataset, 'num_images') and self.dataset.num_images > 0:
                        try:
                            # Use current_idx if valid (>=0), otherwise random if we want (or skip)
                            # The C++ code emits -1 at epoch end.
                            idx = -1
                            if current_idx >= 0:
                                idx = current_idx
                                # User requested NO INFERENCE during training.
                                preview_pred = -1
                                preview_probs = []
                                    
                                # Prepare QImage for display
                                # Access properties directly as defined in bindings
                                width = self.dataset.image_width
                                height = self.dataset.image_height
                                
                                # Access pixels from MNISTImage object
                                # self.dataset.images is a list of MNISTImage objects
                                # images[idx].pixels is the vector of uint8
                                img_data_vec = self.dataset.images[idx].pixels
                                preview_label = self.dataset.images[idx].label
                                
                                if img_data_vec:
                                    # Create QImage from raw data
                                    # QImage constructor from data requires bytes, so we might need to convert
                                    # vector<uint8_t> (which pybind gives as list of int) to bytes.
                                    # Or loop setPixel if that's safer/easier given the format.
                                    # Flattened list of ints.
                                    
                                    # Infer depth
                                    pixel_count = len(img_data_vec)
                                    area = width * height
                                    depth = pixel_count // area
                                    
                                    if depth == 1:
                                        q_img = QImage(width, height, QImage.Format.Format_Grayscale8)
                                        for y in range(height):
                                            for x in range(width):
                                                idx_pixel = y * width + x
                                                val = int(img_data_vec[idx_pixel])
                                                q_img.setPixel(x, y, qRgb(val, val, val))
                                    elif depth == 3:
                                        q_img = QImage(width, height, QImage.Format.Format_RGB888)
                                        # Data is CHW (Planar)
                                        r_start = 0
                                        g_start = area
                                        b_start = 2 * area
                                        
                                        for y in range(height):
                                            for x in range(width):
                                                offset = y * width + x
                                                r = int(img_data_vec[r_start + offset])
                                                g = int(img_data_vec[g_start + offset])
                                                b = int(img_data_vec[b_start + offset])
                                                q_img.setPixel(x, y, qRgb(r, g, b))
                                    else:
                                        # Fallback or error
                                        q_img = None
                                    
                                    preview_img_data = q_img
                            else:
                                # If current_idx is -1 (e.g., epoch end), we might still want to show a random image
                                # or just skip updating the image. For now, we'll keep it as None.
                                pass
                        except Exception as e:
                            print(f"Error getting preview image or performing inference: {e}")
                else:
                    # Visualizations DISABLED
                    maps = []
                    layer_type = -1
                    preview_img_data = None
                    preview_pred = -1
                    preview_probs = []
                    preview_label = -1
                
                # Emit
                # We need to distinguish what we are updating in the UI.
                # The _internal_progress signal handles all.
                # If 'maps' is empty list, UI won't update maps (if we handle it right in Controller/MainWindow/Widget)
                # If 'preview_img_data' is None, UI won't update image.
                
                # Emit includes current_idx and layer_type
                
                self.progress.emit(epoch, total, loss, acc, maps, preview_pred, preview_img_data, preview_probs, current_idx, layer_type, preview_label)

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
            # if not self._stop_requested:
            #     self.finished.emit() # Removed to prevent double emission (QThread emits finished automatically)
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
    featureMapsReady = pyqtSignal(list, int, bool) # maps, layer_type, is_epoch_end
    trainingPreviewReady = pyqtSignal(int, object, list, int) # int, QImage, list of floats, int (true_label) (Training Only)
    imagePredicted = pyqtSignal(int, object, list) # int, QImage, list of floats (Inference Only)
    trainingFinished = pyqtSignal()
    inferenceFinished = pyqtSignal()
    modelStatusChanged = pyqtSignal(bool)
    configurationLoaded = pyqtSignal(dict) # New signal for config
    metricsCleared = pyqtSignal() # Signal to clear metrics graph
    metricsSetEpoch = pyqtSignal(int) # Signal to set epoch for metrics graph
    datasetInfoLoaded = pyqtSignal(int, int, int, int) # num_images, width, height, depth
    
    # Internal signals for thread-safe callback emissions
    _internal_log = pyqtSignal(str)
    _internal_progress = pyqtSignal(int, int, float, float, object, int, object, list, int, int, int)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.dataset = None
        self.should_stop = False
        self._training_active = False
        self._visualizations_enabled = True # Default
        self._train_thread = None
        self._loader_thread = None
        
        # Connect internal signals (these are thread-safe automatically)
        self._internal_log.connect(self._emit_log_from_thread)
        self._internal_progress.connect(self._emit_progress_from_thread)
        
        # Log Buffer
        self.log_buffer = [] 
        
        # Connect logMessage to internal buffer capture so all logs are saved
        self.logMessage.connect(self._capture_log)
        
    def _capture_log(self, message):
        """Capture all log messages to the buffer."""
        if hasattr(self, 'log_buffer'):
            self.log_buffer.append(message) 
        
    
    @pyqtSlot(str, str)
    def loadDataset(self, images_path, labels_path):
        """Load dataset asynchronously in a background thread."""
        if not cgroot_core:
            self.logMessage.emit("Error: Core library not loaded")
            return
            
        import os
        abs_img_path = os.path.abspath(images_path)
        abs_lbl_path = os.path.abspath(labels_path)
        self.logMessage.emit(f"Loading dataset from:\n  Images: {abs_img_path}\n  Labels: {abs_lbl_path}")
        
        # Create and start loader thread
        self._loader_thread = LoaderThread(images_path, labels_path)
        self._loader_thread.log.connect(self.logMessage.emit)
        self._loader_thread.error.connect(self._on_loader_error)
        self._loader_thread.loaded.connect(self._on_dataset_loaded)
        self._loader_thread.finished.connect(self._on_loader_finished)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)
        self._loader_thread.start()
    
    @pyqtSlot()
    def _on_loader_finished(self):
        """Cleanup loader thread reference safely."""
        self._loader_thread = None

    @pyqtSlot(object)
    def _on_dataset_loaded(self, dataset):
        """Handle successful dataset loading."""
        self.dataset = dataset
        if self.dataset:
            # Use getattr with default 1 for depth since binding might not be updated during runtime immediately without recompile
            d = getattr(self.dataset, 'depth', 1) 
            self.logMessage.emit(f"Dataset loaded successfully: {self.dataset.num_images} images, {self.dataset.image_width}x{self.dataset.image_height}x{d}")
            self.datasetInfoLoaded.emit(self.dataset.num_images, self.dataset.image_width, self.dataset.image_height, d)
        else:
            self.logMessage.emit("Failed to load dataset (returned None)")
        
        # Thread cleanup handled by finished signal


    @pyqtSlot(str)
    def _on_loader_error(self, error_msg):
        """Handle dataset loading error."""
        self.logMessage.emit(error_msg)
        # Thread cleanup handled by finished signal




    @pyqtSlot(bool)
    def setVisualizationsEnabled(self, enabled):
        """Enable or disable heavy visualization data generation during training."""
        self._visualizations_enabled = enabled
        if hasattr(self, "_train_thread") and self._train_thread:
            self._train_thread.set_visualizations_enabled(enabled)

    @pyqtSlot(int)
    def setTargetLayer(self, layer_idx):
        """Update the layer index to visualize."""
        if hasattr(self, "_train_thread") and self._train_thread:
            self._train_thread.set_layer_index(layer_idx)
        # Store for future threads
        self._last_layer_idx = layer_idx 

    @pyqtSlot(dict)
    def trainModel(self, config):
        """Train the model asynchronously using a separate thread."""
        if not self.dataset:
            self.logMessage.emit("No dataset loaded")
            self.trainingFinished.emit()
            return

        if not self.model:
            # Debug: print config types
            # msg = "Config Types: " + ", ".join([f"{k}: {type(v).__name__}" for k, v in config.items()])
            # self.logMessage.emit(msg)
            self._initialize_model(config)
            
        # Log Full Configuration
        # import json
        # self.logMessage.emit("=== Training Configuration ===")
        # self.logMessage.emit(json.dumps(config, indent=4))
        # self.logMessage.emit("================================")

        # Store config for later save
        self._last_config = config
        
        # Reset Log Buffer
        self.log_buffer = []
        
        # Start Time
        import time
        self._start_time = time.time()
        
        # Create training configuration
        train_config = self._create_training_config(config)

        # Start training in separate thread
        self._train_thread = TrainingThread(self.model, self.dataset, train_config)
        self._train_thread.progress.connect(self._internal_progress.emit)
        self._train_thread.log.connect(self._internal_log.emit)
        self._train_thread.finished.connect(self._on_training_finished)
        self._train_thread.finished.connect(self._train_thread.deleteLater) # Ensures thread object is safely deleted
        self._training_active = True
        self.should_stop = False
        self.modelStatusChanged.emit(True)
        self.logMessage.emit("=== Training Session Start ===")
        # Pass last layer index if set
        if hasattr(self, "_last_layer_idx"):
            self._train_thread.set_layer_index(self._last_layer_idx)
        
        self._train_thread.set_visualizations_enabled(self._visualizations_enabled)
            
        self._train_thread.start()
    
    
    def _on_training_finished(self):
        """Handle training completion."""
        self._training_active = False
        
        # Calculate duration
        if hasattr(self, '_start_time'):
            import time
            duration = time.time() - self._start_time
            # Format duration
            hours, rem = divmod(duration, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
            self.logMessage.emit(f"Total Training Time: {time_str}")
            
        self.modelStatusChanged.emit(False)
        self.logMessage.emit("=== Training Session End ===")
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

    @pyqtSlot(str, dict)
    def storeModel(self, folder_path, config=None):
        """Store the current model and its configuration to the specified folder."""
        if not self.model:
            self.logMessage.emit("Error: No model to store.")
            return

        self.logMessage.emit(f"Storing model to: {folder_path} ...")
        self.modelStatusChanged.emit(True)
        try:
            import json
            from pathlib import Path
            from datetime import datetime
            
            folder = Path(folder_path)
            folder.mkdir(parents=True, exist_ok=True)
            
            # Store weights using C++ model.store()
            # The C++ function creates a timestamped file: model_param_YYYYMMDD_HHMMSS.bin
            success = self.model.store(str(folder))
            
            # Also save configuration with the same timestamp
            # Use passed config if available, else fallback to last known
            config_to_save = config if config else (getattr(self, '_last_config', None))
            
            if success and config_to_save:
                # Generate timestamp in the same format as C++ (YYYYMMDD_HHMMSS)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                config_file = folder / f"model_config_{timestamp}.json"
                
                # Cleanup: Remove transient keys like _full_logs before saving JSON
                clean_config = config_to_save.copy()
                if '_full_logs' in clean_config:
                    del clean_config['_full_logs']
                
                with open(config_file, 'w') as f:
                    json.dump(clean_config, f, indent=4)
                self.logMessage.emit(f"Configuration saved to: {config_file}")
            
            if success:
                self.logMessage.emit("Model stored successfully.")
                
                # Save Log Buffer to .txt
                log_file = folder / f"model_log_{timestamp}.txt"
                try:
                    with open(log_file, 'w') as f:
                        # Prefer full logs passed from UI
                        if config and '_full_logs' in config:
                            f.write(config['_full_logs'])
                        else:
                            f.write("\n".join(self.log_buffer))
                    self.logMessage.emit(f"Log saved to: {log_file}")
                except Exception as e:
                    self.logMessage.emit(f"Error saving log file: {e}")

            else:
                self.logMessage.emit("Failed to store model.")
                
        except Exception as e:
            self.logMessage.emit(f"Error storing model: {e}")
            import traceback
            traceback.print_exc()
        self.modelStatusChanged.emit(False)

    @pyqtSlot(str)
    def loadModel(self, file_path):
        """Load model parameters from the specified file.
        
        This function will:
        1. Look for model_config.json in the same directory
        2. Initialize the model architecture using the config
        3. Load the weights from the specified file
        """
        from pathlib import Path
        import json
        
        self.logMessage.emit(f"Loading model from: {file_path} ...")
        self.modelStatusChanged.emit(True)
        
        try:
            # Find the config file in the same directory
            model_file = Path(file_path)
            
            if model_file.is_dir():
                # If file_path is a directory, look for .bin file
                bin_files = list(model_file.glob("model_param*.bin"))
                if not bin_files:
                    self.logMessage.emit("Error: No model_param*.bin file found in directory")
                    self.modelStatusChanged.emit(False)
                    return
                model_file = bin_files[0]  # Use the first one
                self.logMessage.emit(f"Found model file: {model_file}")
            
            # Extract timestamp from the model file name (e.g., model_param_20231216_143025.bin)
            import re
            timestamp_match = re.search(r'model_param_(\d{8}_\d{6})\.bin', model_file.name)
            
            # Try to find config file with matching timestamp first
            config_file = None
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                timestamped_config = model_file.parent / f"model_config_{timestamp}.json"
                if timestamped_config.exists():
                    config_file = timestamped_config
                    self.logMessage.emit(f"Found timestamped config: {config_file}")
            
            # Fallback to generic model_config.json if timestamped version not found
            if not config_file:
                generic_config = model_file.parent / "model_config.json"
                if generic_config.exists():
                    config_file = generic_config
                    self.logMessage.emit(f"Using generic config file: {config_file}")
            
            # Load configuration
            if not config_file:
                self.logMessage.emit(f"Warning: No config file found for {model_file.name}")
                self.logMessage.emit("Attempting to load weights into existing model...")
                
                if not self.model:
                    self.logMessage.emit("Error: Model not initialized and no config file found.")
                    self.logMessage.emit("Please either:")
                    self.logMessage.emit("  1. Ensure model_config_TIMESTAMP.json exists in the same directory, OR")
                    self.logMessage.emit("  2. Initialize the model architecture first (e.g., start training)")
                    self.modelStatusChanged.emit(False)
                    return
            else:
                # Load and apply configuration
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                
                self.logMessage.emit(f"Loaded configuration from: {config_file}")
                self._last_config = config
                self.configurationLoaded.emit(config) # Emit signal for UI update
                
                # Initialize model with this configuration
                if self.model:
                    self.logMessage.emit("Re-initializing model with loaded configuration...")
                    del self.model
                    self.model = None
                
                self._initialize_model(config)
            
            # Now load the weights
            success = self.model.load(str(model_file))
            
            if success:
                self.logMessage.emit("Model loaded successfully.")
                
                # Restore Metrics Graph from History
                if hasattr(self.model, "getTrainingHistory"):
                    history = self.model.getTrainingHistory()
                    if history:
                        self.metricsCleared.emit() # Clear existing graph data before restoring
                        self.metricsSetEpoch.emit(len(history) + 1) # Set total epochs for metrics graph
                        self.logMessage.emit(f"Restoring {len(history)} epochs of history...")
                        
                        # # Debug: Log first and last history item
                        # if len(history) > 0:

                        #     first = history[0]
                        #     last = history[-1]
                        #     self.logMessage.emit(f"  First: Epoch {first.epoch}, Loss {first.train_loss:.4f}, Acc {first.train_accuracy:.2f}")
                        #     self.logMessage.emit(f"  Last:  Epoch {last.epoch}, Loss {last.train_loss:.4f}, Acc {last.train_accuracy:.2f}")

                        for m in history:
                            # m is TrainingMetrics object (bound from C++)
                            # emit metricsUpdated signal
                            # We emit one by one to populate graph
                            # Note: graph might need 'epoch' index 1-based or 0-based.
                            # MetricsUpdated(loss, acc, epoch)
                            self.metricsUpdated.emit(m.train_loss, m.train_accuracy, m.epoch)
                    else:
                         self.logMessage.emit("Model loaded but no training history found (legacy format or empty).")

            else:
                self.logMessage.emit("Failed to load model weights.")
                
        except Exception as e:
            self.logMessage.emit(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            
        self.modelStatusChanged.emit(False)

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
    
    @pyqtSlot(int, int, float, float, object, int, object, list, int, int, int)
    def _emit_progress_from_thread(self, epoch, total_epochs, loss, accuracy, feature_maps, pred_class, q_image, probs, current_idx, layer_type, true_label):
        """
        Thread-safe progress/metrics emission helper.
        Called via QMetaObject.invokeMethod from C++ callback thread.
        Only emits if training is still active to prevent post-finish emissions.
        """
        if self._training_active:

            is_epoch_end = (current_idx == -1)

            if is_epoch_end:
                self.metricsUpdated.emit(loss, accuracy, epoch)
                self.progressUpdated.emit(epoch, total_epochs)
                
            # Emit feature maps if they were updated (checked by not None)
            # Empty list [] IS valid (it means clear/unknown)
            if feature_maps is not None:
                 self.featureMapsReady.emit(feature_maps, layer_type, is_epoch_end)
            
            # Emit image if present (per sample)
            if q_image:
                self.trainingPreviewReady.emit(pred_class, q_image, probs, true_label)
    
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
        except Exception:
            pass  # Avoid exceptions in destructor
