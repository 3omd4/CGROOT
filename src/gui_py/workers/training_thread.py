from PyQt6.QtCore import QThread, pyqtSignal
try:
    import cgroot_core
except ImportError:
    cgroot_core = None
    
# Import VisualizationManager
from utils.visualization_manager import VisualizationManager

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
                    try:
                        # Access dynamic layer index
                        target_layer = self.layer_index
                        if target_layer < 0: target_layer = 0
                        
                        # Get maps. If invalid layer, this returns []
                        maps = self.model.getLayerFeatureMaps(target_layer)
                        layer_type = self.model.getLayerType(target_layer)
                        
                        # Process maps using Manager (optional step for normalization/coloring if moved there)
                        maps = VisualizationManager.process_feature_maps(maps, layer_type)
                        
                    except Exception as e:
                        # No fallback! Let it be empty list or None so UI knows it failed/is invalid.
                        maps = [] # Explicitly empty so UI clears it
                        layer_type = -1
                    
                    # Preview Image Generation
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
                                # User requested NO INFERENCE during training to save time/complexity
                                preview_pred = -1
                                preview_probs = []
                                    
                                # Prepare QImage for display
                                width = self.dataset.image_width
                                height = self.dataset.image_height
                                depth = getattr(self.dataset, 'depth', 1)
                                
                                # Access raw pixels from MNISTImage object
                                # images[idx].pixels is the vector of uint8
                                img_data_vec = self.dataset.images[idx].pixels
                                preview_label = self.dataset.images[idx].label
                                
                                # Use VisualizationManager to create QImage
                                preview_img_data = VisualizationManager.create_preview_image(
                                    img_data_vec, width, height, depth
                                )

                            else:
                                # If current_idx is -1 (e.g., epoch end), skip updating image
                                pass
                        except Exception as e:
                            print(f"Error getting preview image or calling VizManager: {e}")
                else:
                    # Visualizations DISABLED
                    maps = []
                    layer_type = -1
                    preview_img_data = None
                    preview_pred = -1
                    preview_probs = []
                    preview_label = -1
                
                # Emit
                self.progress.emit(epoch, total, loss, acc, maps, preview_pred, preview_img_data, preview_probs, current_idx, layer_type, preview_label)

        def log_callback(msg):
            if not self._stop_requested:
                self.log.emit(msg)

        def stop_check():
            return self._stop_requested

        try:
            # Check if model has train_epochs
            if hasattr(self.model, 'train_epochs'):
                history = self.model.train_epochs(
                    self.dataset,
                    self.config,
                    progress_callback,
                    log_callback,
                    stop_check
                )
            else:
                self.log.emit("Error: Model definition missing train_epochs method")
                
        except Exception as e:
            self.log.emit(f"Training error: {e}")

    def stop(self):
        self._stop_requested = True
