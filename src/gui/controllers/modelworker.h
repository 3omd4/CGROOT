#ifndef MODELWORKER_H
#define MODELWORKER_H

#include <QImage>
#include <QMutex>
#include <QObject>
#include <QVector>
#include <atomic>
#include <memory>
#include <vector>

class NNModel;

namespace cgroot {
namespace data {
namespace MNISTLoader {
struct MNISTDataset;
}
} // namespace data
} // namespace cgroot

/**
 * @brief The ModelWorker class handles long-running AI model operations on a
 * background thread.
 */
class ModelWorker : public QObject {
  Q_OBJECT

public:
  explicit ModelWorker(QObject *parent = nullptr);
  ~ModelWorker();

public slots:
  void loadDataset(const QString &imagesPath, const QString &labelsPath);
  void initializeModel();
  void trainModel(int epochs);
  void stopTraining();
  void runInference();

signals:
  void logMessage(const QString &message);
  void metricsUpdated(double loss, double accuracy, int epoch);
  void progressUpdated(int value, int maximum);
  void imagePredicted(int predictedClass, const QImage &image,
                      const QVector<double> &probabilities);
  void trainingFinished();
  void inferenceFinished();
  void modelStatusChanged(bool isActive);

private:
  QImage convertMNISTImageToQImage(const std::vector<uint8_t> &pixels,
                                   int width, int height);

  std::unique_ptr<NNModel> m_model;
  std::unique_ptr<cgroot::data::MNISTLoader::MNISTDataset> m_trainingDataset;
  std::unique_ptr<cgroot::data::MNISTLoader::MNISTDataset> m_testDataset;

  std::atomic<bool> m_shouldStop;
};

#endif // MODELWORKER_H