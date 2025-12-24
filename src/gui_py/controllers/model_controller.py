from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from .model_worker import ModelWorker

class ModelController(QObject):
    # Signals to GUI
    logMessage = pyqtSignal(str)
    metricsUpdated = pyqtSignal(float, float, float, float, int) # train_loss, train_acc, val_loss, val_acc, epoch
    progressUpdated = pyqtSignal(int, int) # value, maximum
    imagePredicted = pyqtSignal(int, object, list) # predictedClass, QImage, probabilities
    trainingFinished = pyqtSignal()
    modelStatusChanged = pyqtSignal(bool) # isTraining
    featureMapsReady = pyqtSignal(list, int, bool) # maps, layer_type, is_epoch_end
    trainingPreviewReady = pyqtSignal(int, object, list, int) # int: epoch, object: image, list: probabilities, int: predicted_class
    configurationLoaded = pyqtSignal(dict) # dict: configuration
    metricsCleared = pyqtSignal() # Signal to clear metrics graph
    metricsSetEpoch = pyqtSignal(int) # Signal to set epoch for metrics graph
    datasetInfoLoaded = pyqtSignal(int, int, int, int) # num_images, width, height, depth
    modelInfoLoaded = pyqtSignal(int, int, int) # width, height, depth (From loaded model)
    evaluationFinished = pyqtSignal(float, float, list) # loss, acc, confusion_matrix

    
    # Signals to Worker
    requestLoadDataset = pyqtSignal(str, str)
    requestTrain = pyqtSignal(dict) # Accepts configuration dict
    requestStop = pyqtSignal()
    requestInference = pyqtSignal(object)
    setTargetLayer = pyqtSignal(int)
    requestStoreModel = pyqtSignal(str, dict) # folderPath, configuration
    requestLoadModel = pyqtSignal(str) # filePath
    requestResetModel = pyqtSignal(dict) # configuration
    requestTest = pyqtSignal(str, str) # images_path, labels_path
    setVisualizationsEnabled = pyqtSignal(bool) # Toggle visualizations

    def __init__(self):
        super().__init__()
        
        self.thread = QThread()
        self.thread.setObjectName("ModelController_WorkerThread")
        self.worker = ModelWorker()
        self.worker.moveToThread(self.thread)
        
        # Connect Worker -> Controller
        self.worker.logMessage.connect(self.logMessage)
        self.worker.metricsUpdated.connect(self.metricsUpdated)
        self.worker.progressUpdated.connect(self.progressUpdated)
        self.worker.imagePredicted.connect(self.imagePredicted)
        self.worker.trainingPreviewReady.connect(self.trainingPreviewReady)
        self.worker.trainingFinished.connect(self.trainingFinished)
        self.worker.modelStatusChanged.connect(self.modelStatusChanged)
        self.worker.featureMapsReady.connect(self.featureMapsReady)
        self.worker.configurationLoaded.connect(self.configurationLoaded)
        self.worker.metricsCleared.connect(self.metricsCleared)
        self.worker.metricsSetEpoch.connect(self.metricsSetEpoch)
        self.worker.datasetInfoLoaded.connect(self.datasetInfoLoaded)
        self.worker.modelInfoLoaded.connect(self.modelInfoLoaded)
        self.worker.evaluationFinished.connect(self.evaluationFinished)
        
        # Connect Controller -> Worker
        self.requestLoadDataset.connect(self.worker.loadDataset)
        self.requestTrain.connect(self.worker.trainModel)
        self.requestStop.connect(self.worker.stopTraining)
        self.requestInference.connect(self.worker.runInference)
        self.requestStop.connect(self.worker.stopTraining)
        self.requestInference.connect(self.worker.runInference)
        self.setTargetLayer.connect(self.worker.setTargetLayer)
        self.requestStoreModel.connect(self.worker.storeModel)
        self.requestLoadModel.connect(self.worker.loadModel)
        self.requestTest.connect(self.worker.runTesting)
        self.setVisualizationsEnabled.connect(self.worker.setVisualizationsEnabled)
        self.requestResetModel.connect(self.worker.resetModel)
        
        self.thread.start()
        
    def cleanup(self):
        self.requestStop.emit()
        self.thread.quit()
        self.thread.wait()