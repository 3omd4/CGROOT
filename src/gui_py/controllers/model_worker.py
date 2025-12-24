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

# Imported from workers package
from workers.training_thread import TrainingThread
from workers.loader_thread import LoaderThread
from workers.testing_thread import TestingThread
from utils.visualization_manager import VisualizationManager

class ModelWorker(QObject):
    logMessage = pyqtSignal(str)
    metricsUpdated = pyqtSignal(float, float, float, float, int)  # t_loss, t_acc, v_loss, v_acc, epoch
    progressUpdated = pyqtSignal(int, int) # value, maximum
    featureMapsReady = pyqtSignal(list, int, bool) # maps, layer_type, is_epoch_end
    trainingPreviewReady = pyqtSignal(int, object, list, int) # int, QImage, list of floats, int (true_label) (Training Only)
    imagePredicted = pyqtSignal(int, object, list) # int, QImage, list of floats (Inference Only)
    trainingFinished = pyqtSignal()
    inferenceFinished = pyqtSignal()
    modelStatusChanged = pyqtSignal(bool) # isTraining
    configurationLoaded = pyqtSignal(dict) # New signal for config
    metricsCleared = pyqtSignal() # Signal to clear metrics graph
    metricsSetEpoch = pyqtSignal(int) # Signal to set epoch for metrics graph
    datasetInfoLoaded = pyqtSignal(int, int, int, int) # num_images, width, height, depth
    modelInfoLoaded = pyqtSignal(int, int, int) # w, h, d
    evaluationFinished = pyqtSignal(float, float, list) # loss, acc, confusion_matrix
    
    # Internal signals for thread-safe callback emissions
    _internal_log = pyqtSignal(str)
    _internal_progress = pyqtSignal(int, int, float, float, float, float, object, int, object, list, int, int, int)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.dataset = None
        self.should_stop = False
        self._training_active = False
        self._visualizations_enabled = True # Default
        self._train_thread = None
        self._loader_thread = None
        
        self.last_viz_update = 0
        self.viz_interval = 0.1 # Max 10 updates per second

        
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
            d = getattr(self.dataset, 'depth', 1) 
            self.logMessage.emit(f"Dataset loaded successfully: {self.dataset.num_images} images, {self.dataset.image_width}x{self.dataset.image_height}x{d}")
            self.datasetInfoLoaded.emit(self.dataset.num_images, self.dataset.image_width, self.dataset.image_height, d)
        else:
            self.logMessage.emit("Failed to load dataset (returned None)")

    @pyqtSlot(str)
    def _on_loader_error(self, error_msg):
        """Handle dataset loading error."""
        self.logMessage.emit(error_msg)

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
        
        # Log layer-specific configuration for verification
        self.logMessage.emit("=== Layer Configuration Details ===")
        
        # Conv layers
        if 'num_conv_layers' in config and config['num_conv_layers'] > 0:
            self.logMessage.emit(f"Conv Layers: {config['num_conv_layers']}")
            if 'kernels_per_layer' in config:
                self.logMessage.emit(f"  Kernels: {config['kernels_per_layer']}")
            if 'kernel_dims' in config:
                self.logMessage.emit(f"  Kernel Dims: {config['kernel_dims']}")
            if 'conv_activations' in config:
                self.logMessage.emit(f"  Activations: {config['conv_activations']}")
            if 'conv_init_types' in config:
                self.logMessage.emit(f"  Init Types: {config['conv_init_types']}")
            if 'conv_paddings' in config:
                self.logMessage.emit(f"  Paddings: {config['conv_paddings']}")
            if 'conv_strides' in config:
                self.logMessage.emit(f"  Strides: {config['conv_strides']}")
        
        # Pooling layers
        if 'pooling_intervals' in config and config['pooling_intervals']:
            self.logMessage.emit(f"Pooling: {config['pooling_type']} at intervals {config['pooling_intervals']}")
            if 'pooling_strides' in config:
                self.logMessage.emit(f"  Strides: {config['pooling_strides']}")
        
        # FC layers
        if 'num_fc_layers' in config:
            self.logMessage.emit(f"FC Layers: {config['num_fc_layers']}")
            if 'neurons_per_fc_layer' in config:
                self.logMessage.emit(f"  Neurons: {config['neurons_per_fc_layer']}")
            if 'fc_activations' in config:
                self.logMessage.emit(f"  Activations: {config['fc_activations']}")
            if 'fc_init_types' in config:
                self.logMessage.emit(f"  Init Types: {config['fc_init_types']}")
        
        self.logMessage.emit("===================================")
        
        try:
            if cgroot_core is None:
                raise ImportError("C++ Core module (cgroot_core) not loaded. Please rebuild the project.")
            self.model = cgroot_core.create_model(config)
            self.logMessage.emit("Model initialized successfully")
            return True
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
        """Stop the training thread and save the current model state."""
        if not self._training_active:
            self.logMessage.emit("Training is not currently active.")
            return
            
        self.logMessage.emit("Stopping training and saving model...")
        self.should_stop = True
        self._training_active = False # Ensure flag is off
        # Stop the training thread gracefully
        self.modelStatusChanged.emit(False)
        if hasattr(self, "_train_thread") and self._train_thread:
            self._train_thread.stop()
            self.logMessage.emit("Training thread stopped")
            # self._train_thread.wait(1000)  # Wait up to 1 second for thread to finish
        
        # Auto-save the model after stopping
        if self.model:
            try:
                from pathlib import Path
                # Get default model directory
                script_dir = Path(__file__).parent
                project_root = script_dir.parent.parent
                auto_save_dir = project_root / "src" / "data" / "trained-model" / "auto_save"
                auto_save_dir.mkdir(parents=True, exist_ok=True)
                
                # Get configuration for saving
                config_to_save = getattr(self, '_last_config', None)
                
                # Call storeModel to save weights and configuration
                self.logMessage.emit(f"Auto-saving model to: {auto_save_dir}")
                self.storeModel(str(auto_save_dir), config_to_save)
                self.logMessage.emit("Model saved successfully after training stop.")
            except Exception as e:
                self.logMessage.emit(f"Warning: Failed to auto-save model: {e}")
        else:
            self.logMessage.emit("No model to save.")

    @pyqtSlot(dict)
    def resetModel(self, config):
        """Reset and re-initialize the model with current config (Random Weights)."""
        if self._training_active:
            self.logMessage.emit("Cannot reset model while training is active. Please stop training first.")
            return

        self.logMessage.emit("Resetting model with random weights...")
        self.logMessage.emit("** WARNING: Training history cleared **")
        
        # Re-initialize model
        if self._initialize_model(config):
             # Clear history signals if any UI components listen to them
             # Currently we just re-emit initialized status
             self.modelStatusChanged.emit(False) # Training not active
             w = self.model.getInputWidth()
             h = self.model.getInputHeight()
             d = self.model.getInputDepth()
             self.modelInfoLoaded.emit(w, h, d) # Inform UI of new (clean) state
             self.logMessage.emit("Model reset successfully.")
        else:
             self.logMessage.emit("Failed to reset model.")

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
                
                # Emit model dimensions specifically to assist GUI in guessing dataset type
                try:
                    w = self.model.getInputWidth()
                    h = self.model.getInputHeight()
                    d = self.model.getInputDepth()
                    self.modelInfoLoaded.emit(w, h, d)
                except Exception as e:
                    self.logMessage.emit(f"Warning: Could not get model info: {e}")

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
                            self.metricsUpdated.emit(m.train_loss, m.train_accuracy, m.val_loss, m.val_accuracy, m.epoch)
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
            # Get expected dimensions from model
            w = self.model.getInputWidth()
            h = self.model.getInputHeight()
            d = self.model.getInputDepth()
            
            # Format image based on depth
            fmt = QImage.Format.Format_Grayscale8 if d == 1 else QImage.Format.Format_RGB888
            
            img = qimage_obj.scaled(w, h).convertToFormat(fmt)

            
            width = img.width()
            height = img.height()
            stride = img.bytesPerLine()
            
            # fast access to bytes
            bits = img.bits()
            bits.setsize(img.sizeInBytes())

            # Run Classification using C++ helper
            self.logMessage.emit(f"Running inference (Input: {width}x{height}x{d})...")

            # Note: C++ classify_pixels logic must match the QImage format (RGB888 vs Grayscale8)
            predicted_class = cgroot_core.classify_pixels(self.model, bits, width, height, stride)
            
            self.logMessage.emit(f"Inference Result: {predicted_class}")
            
            # Get probabilities from model
            probs = self.model.getProbabilities()
            if not probs or len(probs) == 0:
                probs = [0.0] * 10
                probs[predicted_class] = 1.0
            
            confidence = max(probs) if probs else 0.0
            self.logMessage.emit(f"Inference Confidence: {confidence:.4f}")
            
            # Emit signal with results
            self.imagePredicted.emit(predicted_class, qimage_obj, probs)
            
        except Exception as e:
            self.logMessage.emit(f"Inference error: {e}")
            import traceback
            traceback.print_exc()

        self.modelStatusChanged.emit(False)
        self.inferenceFinished.emit()
    
    # Thread-Safe Signal Emission Helpers
    @pyqtSlot(str)
    def _emit_log_from_thread(self, message):
        """
        Thread-safe log emission helper.
        Called via QMetaObject.invokeMethod from C++ callback thread.
        Only emits if training is still active to prevent post-finish emissions.
        """
        if self._training_active:
            self.logMessage.emit(message)
    
    @pyqtSlot(int, int, float, float, float, float, object, int, object, list, int, int, int)
    def _emit_progress_from_thread(self, epoch, total_epochs, t_loss, t_acc, v_loss, v_acc, feature_maps, pred_class, q_image, probs, current_idx, layer_type, true_label):
        """
        Thread-safe progress/metrics emission helper.
        Called via QMetaObject.invokeMethod from C++ callback thread.
        Only emits if training is still active to prevent post-finish emissions.
        """
        if self._training_active:

            is_epoch_end = (current_idx == -1)
            current_time = time.time()
            
            # Helper to check if we should emit heavy data
            should_emit_viz = (current_time - self.last_viz_update > self.viz_interval) or is_epoch_end

            if is_epoch_end:
                self.metricsUpdated.emit(t_loss, t_acc, v_loss, v_acc, epoch)
                self.progressUpdated.emit(epoch, total_epochs)
                
            # Emit feature maps if they were updated (checked by not None)
            # Empty list [] IS valid (it means clear/unknown)
            if feature_maps is not None and should_emit_viz:
                 self.featureMapsReady.emit(feature_maps, layer_type, is_epoch_end)
            
            # Emit image if present (per sample)
            if q_image and should_emit_viz:
                self.trainingPreviewReady.emit(pred_class, q_image, probs, true_label)
            
            if should_emit_viz:
                self.last_viz_update = current_time
    
    # Lifecycle Management

    def cleanup(self):
        """
        Explicit cleanup method for safe resource deallocation.
        Call this before destroying the worker object.
        """
        self.logMessage.emit("=== ModelWorker Cleanup Start ===")
        
        # 1. Stop any ongoing training
        self.should_stop = True
        self._training_active = False
        
        #2. Clear callback references to allow Python GC
        self._progress_callback_ref = None
        self._log_callback_ref = None
        
        # 2b. Stop thread if running
        if hasattr(self, "_train_thread") and self._train_thread:
            self.stopTraining() # sets flag and waits

        # 2c. Stop loader thread
        if hasattr(self, "_loader_thread") and self._loader_thread:
            if self._loader_thread.isRunning():
                self.logMessage.emit("Waiting for loader thread to finish...")
                self._loader_thread.quit()
                self._loader_thread.wait(2000) # Wait max 2s

        # 2d. Stop testing thread
        if hasattr(self, "_test_thread") and self._test_thread:
            if self._test_thread.isRunning():
                self.logMessage.emit("Stopping testing thread...")
                self._test_thread.stop()
                self._test_thread.quit()
                self._test_thread.wait(2000) # Wait max 2s

        # 3. Destroy C++ model object
        if self.model:
            self.logMessage.emit("Destroying C++ model...")
            try:
                # Extra check: wait again if needed? stopTraining already waits.
                if hasattr(self, "_train_thread") and self._train_thread:
                    if self._train_thread.isRunning():
                        self.logMessage.emit("Warning: Training thread still running during model destruction!")
                        # self._train_thread.wait() # Force wait

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
    
    @pyqtSlot(str, str)
    def runTesting(self, images_path, labels_path):
        """Run evaluation on a test dataset asynchronously."""
        if not self.model:
            self.logMessage.emit("Error: No model loaded to test.")
            return

        self.logMessage.emit(f"Starting Test on: {images_path}")
        self.modelStatusChanged.emit(True)

        # Create and start testing thread
        self._test_thread = TestingThread(self.model, images_path, labels_path)
        
        # Connect signals using internal slots to ensure thread safety
        self._test_thread.log.connect(self._internal_log.emit)
        self._test_thread.progress.connect(self.progressUpdated)
        self._test_thread.results.connect(self._on_testing_finished_results)
        self._test_thread.error.connect(self._on_testing_error)
        self._test_thread.finished.connect(self._on_testing_thread_finished)
        self._test_thread.finished.connect(self._test_thread.deleteLater)
        
        self.should_stop = False 
        self._test_thread.start()

    @pyqtSlot(float, float, list)
    def _on_testing_finished_results(self, loss, accuracy, matrix):
        """Handle successful testing results."""
        self.logMessage.emit("=== Evaluation Results ===")
        self.logMessage.emit(f"Test Accuracy: {accuracy * 100:.2f}%")
        self.logMessage.emit(f"Test Loss: {loss:.4f}")
        self.logMessage.emit("==========================")
        self.evaluationFinished.emit(loss, accuracy, matrix)

    @pyqtSlot(str)
    def _on_testing_error(self, error_msg):
        self.logMessage.emit(error_msg)

    @pyqtSlot()
    def _on_testing_thread_finished(self):
        """Cleanup testing thread."""
        self._test_thread = None
        self.modelStatusChanged.emit(False)
