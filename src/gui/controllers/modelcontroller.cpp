#include "modelcontroller.h"
#include "modelworker.h"

ModelController::ModelController(QObject *parent)
    : QObject(parent), m_worker(new ModelWorker()),
      m_workerThread(new QThread(this)), m_isTraining(false) {
  // Move worker to thread
  m_worker->moveToThread(m_workerThread);

  // Connect Worker -> Controller (Forward to GUI)
  connect(m_worker, &ModelWorker::logMessage, this,
          &ModelController::logMessage);
  connect(m_worker, &ModelWorker::metricsUpdated, this,
          &ModelController::metricsUpdated);
  connect(m_worker, &ModelWorker::progressUpdated, this,
          &ModelController::progressUpdated);
  connect(m_worker, &ModelWorker::imagePredicted, this,
          &ModelController::imagePredicted);
  connect(m_worker, &ModelWorker::trainingFinished, this,
          &ModelController::trainingFinished);
  connect(m_worker, &ModelWorker::modelStatusChanged, this,
          &ModelController::modelStatusChanged);
  connect(m_worker, &ModelWorker::inferenceFinished, this, [this]() {
    emit logMessage("Inference finished.");
    emit modelStatusChanged(false);
  });

  // Connect Controller -> Worker
  connect(this, &ModelController::requestLoadDataset, m_worker,
          &ModelWorker::loadDataset);
  connect(this, &ModelController::requestTrain, m_worker,
          &ModelWorker::trainModel);
  connect(this, &ModelController::requestStop, m_worker,
          &ModelWorker::stopTraining);
  connect(this, &ModelController::requestInference, m_worker,
          &ModelWorker::runInference);

  m_workerThread->start();
}

ModelController::~ModelController() {
  emit requestStop();
  m_workerThread->quit();
  m_workerThread->wait();
  delete m_worker; // Safe to delete after thread has fully stopped
}

void ModelController::loadDataset(const std::string &imagesPath,
                                  const std::string &labelsPath) {
  emit requestLoadDataset(QString::fromStdString(imagesPath),
                          QString::fromStdString(labelsPath));
}

void ModelController::loadModel(const std::string &modelPath) {
  emit logMessage("Load Model implementation pending in Worker");
}

void ModelController::saveModel(const std::string &modelPath) {
  emit logMessage("Save Model implementation pending in Worker");
}

void ModelController::startTraining() { emit requestTrain(10); }

void ModelController::stopTraining() { emit requestStop(); }

void ModelController::startInference() { emit requestInference(); }

void ModelController::updateParameters() {
  // TODO
}
