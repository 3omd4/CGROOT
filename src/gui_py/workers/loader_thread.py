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
