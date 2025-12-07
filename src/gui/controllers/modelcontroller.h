#ifndef MODELCONTROLLER_H
#define MODELCONTROLLER_H

#include <QImage>
#include <QObject>
#include <QThread>
#include <QVector>
#include <string>

class ModelWorker;

/**
 * @brief The ModelController class acts as an interface between the GUI and the
 * ModelWorker. It manages the worker thread and forwards signals/slots. This
 * class should exist in the Main GUI Thread.
 */
class ModelController : public QObject {
  Q_OBJECT

public:
  explicit ModelController(QObject *parent = nullptr);
  ~ModelController();

  // Commands forwarded to worker
  void loadDataset(const std::string &imagesPath,
                   const std::string &labelsPath);
  void loadModel(const std::string &modelPath);
  void saveModel(const std::string &modelPath);
  void startTraining();
  void stopTraining();
  void startInference();
  void updateParameters(); // Placeholder

signals:
  // Signals to Worker
  void requestLoadDataset(const QString &imagesPath, const QString &labelsPath);
  void requestTrain(int epochs);
  void requestStop();
  void requestInference();

  // Signals to GUI
  void metricsUpdated(double loss, double accuracy, int epoch);
  void progressUpdated(int value, int maximum);
  void logMessage(const QString &message);
  void imagePredicted(int predictedClass, const QImage &image,
                      const QVector<double> &probabilities);
  void trainingFinished();
  void modelStatusChanged(bool isTraining);

private:
  ModelWorker *m_worker;
  QThread *m_workerThread;

  // State to act as cache if needed
  bool m_isTraining;
};

#endif // MODELCONTROLLER_H
