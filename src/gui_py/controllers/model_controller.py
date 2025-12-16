from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from .model_worker import ModelWorker

class ModelController(QObject):
    # Signals to GUI
    logMessage = pyqtSignal(str)
    metricsUpdated = pyqtSignal(float, float, int) # loss, accuracy, epoch
    progressUpdated = pyqtSignal(int, int) # value, maximum
    imagePredicted = pyqtSignal(int, object, list) # predictedClass, QImage, probabilities
    trainingFinished = pyqtSignal()
    modelStatusChanged = pyqtSignal(bool) # isTraining
    featureMapsReady = pyqtSignal(list, int, bool) # maps, layer_type, is_epoch_end
    trainingPreviewReady = pyqtSignal(int, object, list, int) # NEW
    
    # Signals to Worker
    requestLoadDataset = pyqtSignal(str, str)
    requestTrain = pyqtSignal(dict) # CHANGED: Accepts config dict now
    requestStop = pyqtSignal()
    requestInference = pyqtSignal(object)
    setTargetLayer = pyqtSignal(int)
    requestStoreModel = pyqtSignal(str) # folderPath
    requestLoadModel = pyqtSignal(str) # filePath
    setVisualizationsEnabled = pyqtSignal(bool) # NEW: Toggle visualizations

    def __init__(self):
        super().__init__()
        
        self.thread = QThread()
        self.worker = ModelWorker()
        self.worker.moveToThread(self.thread)
        
        # Connect Worker -> Controller
        self.worker.logMessage.connect(self.logMessage)
        self.worker.metricsUpdated.connect(self.metricsUpdated)
        self.worker.progressUpdated.connect(self.progressUpdated)
        self.worker.imagePredicted.connect(self.imagePredicted)
        self.worker.trainingPreviewReady.connect(self.trainingPreviewReady) # NEW
        self.worker.trainingFinished.connect(self.trainingFinished)
        self.worker.modelStatusChanged.connect(self.modelStatusChanged)
        self.worker.featureMapsReady.connect(self.featureMapsReady)
        
        # Connect Controller -> Worker
        self.requestLoadDataset.connect(self.worker.loadDataset)
        self.requestTrain.connect(self.worker.trainModel)
        self.requestStop.connect(self.worker.stopTraining)
        self.requestInference.connect(self.worker.runInference)
        self.setTargetLayer.connect(self.worker.setTargetLayer)
        self.requestStoreModel.connect(self.worker.storeModel)
        self.requestLoadModel.connect(self.worker.loadModel)
        self.setVisualizationsEnabled.connect(self.worker.setVisualizationsEnabled)
        
        self.thread.start()
        
    def cleanup(self):
        self.requestStop.emit()
        self.thread.quit()
        self.thread.wait()