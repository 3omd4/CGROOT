from PyQt6.QtCore import QThread, pyqtSignal
try:
    import cgroot_core
except ImportError:
    cgroot_core = None

class LoaderThread(QThread):
    loaded = pyqtSignal(object)  # dataset object
    log = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, images_path, labels_path):
        super().__init__()
        self.setObjectName("LoaderThread")
        self.images_path = images_path
        self.labels_path = labels_path

    def run(self):
        try:
            if not cgroot_core:
                self.error.emit("Core library not loaded")
                return

            # Load dataset in this thread
            import os
            from utils.custom_loader import CustomDatasetLoader
            
            dataset = None # Initialize dataset
            if os.path.isdir(self.images_path):
                 # Custom Folder Loading
                 # We assume 28x28 grayscale for now, or we can make it smart later.
                 # Let's default to 28x28x1 to match MNIST since that's what the UI defaults to.
                 # Ideally we pass this config.
                 self.log.emit(f"Detected directory. Using Custom Loader on: {self.images_path}")
                 dataset = CustomDatasetLoader.load_from_folder(self.images_path, target_width=28, target_height=28, grayscale=True)
            else:
                 # Standard MNIST File Loading
                 dataset = cgroot_core.MNISTLoader.load_training_data(self.images_path, self.labels_path)

            if dataset:
                self.loaded.emit(dataset)
            else:
                self.error.emit("Failed to load dataset (returned None)")
                
        except Exception as e:
            self.error.emit(f"Error loading dataset: {e}")
