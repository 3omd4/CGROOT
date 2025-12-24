from PyQt6.QtCore import QThread, pyqtSignal
try:
    import cgroot_core
except ImportError:
    cgroot_core = None

class TestingThread(QThread):
    progress = pyqtSignal(int, int) # current, total
    log = pyqtSignal(str)
    # loss, accuracy, confusion_matrix
    results = pyqtSignal(float, float, list) 
    error = pyqtSignal(str)

    def __init__(self, model, images_path, labels_path):
        super().__init__()
        self.setObjectName("TestingThread")
        self.model = model
        self.images_path = images_path
        self.labels_path = labels_path
        self._stop_requested = False

    def run(self):
        try:
            if not cgroot_core:
                self.error.emit("Core library not loaded")
                return

            # Load dataset in this thread (or passed? No, better load here or pass loaded dataset)
            # ModelWorker loaded it in runTesting previously.
            # Loading is fast, but better to do it here to avoid blocking UI if large.
            # actually `cgroot_core.MNISTLoader.load_test_data` might take a second.
            
            test_dataset = cgroot_core.MNISTLoader.load_test_data(self.images_path, self.labels_path)
            if not test_dataset:
                self.error.emit("Failed to load test dataset")
                return
            
            self.log.emit(f"Loaded {test_dataset.num_images} test images. Starting Evaluation...")

            # Callback for C++ evaluate
            def progress_cb(epoch, total, loss, acc, idx):
                if not self._stop_requested:
                    self.progress.emit(idx, test_dataset.num_images)

            # Check function for C++ to stop
            # Note: Model::evaluate might not take a stop function in current bindings?
            # Creating a dummy one just in case, but if C++ doesn't support it, we can't stop mid-loop easily
            # unless we modify C++.
            # However, looking at TrainingThread:
            # history = self.model.train_epochs(..., stop_check)
            # Let's check if evaluate supports it.
            # ModelWorker previous code: `self.model.evaluate(test_dataset, progress_cb)`
            # It seems it only takes progress_cb.
            # If C++ evaluate doesn't support stop, we can't truly stop it mid-execution.
            # BUT, we can at least unblock the UI so the user can interact.
            # And we can ignore the results.
            
            # Use a wrapper if possible? 
            # If C++ evaluate is monolithic, we can't stop it.
            # But the requirement is "Fix Testing Thread Blocking". 
            # Moving to thread fixes the Blocking. Stopping is secondary but desired.
            
            loss, accuracy, matrix = self.model.evaluate(test_dataset, progress_cb)
            
            if not self._stop_requested:
                self.results.emit(loss, accuracy, matrix)
            else:
                self.log.emit("Testing stopped by user.")

        except Exception as e:
            self.error.emit(f"Evaluation Failed: {e}")
            
    def stop(self):
        self._stop_requested = True
