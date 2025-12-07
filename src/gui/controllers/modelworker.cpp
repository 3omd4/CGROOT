#include "modelworker.h"
#include "../../core/model.h"
#include "../../core/utils/mnist_loader.h"
#include <QDebug>
#include <QThread>
#include <algorithm>
#include <cmath>

ModelWorker::ModelWorker(QObject *parent)
    : QObject(parent), m_model(nullptr), m_shouldStop(false) {}
ModelWorker::~ModelWorker() {}
void ModelWorker::loadDataset(const QString &imagesPath,
                              const QString &labelsPath) {
  emit logMessage(
      QString("Worker: Loading dataset from:\n  Images: %1\n  Labels: %2")
          .arg(imagesPath)
          .arg(labelsPath));
  try {
    // Load dataset
    // Note: data::MNISTLoader returns a unique_ptr, so we move it.
    // We load twice for now (train/test split logic could be improved)
    // or just use the whole set for both as in original code logic?
    // The original code had separate m_trainingDataset and m_testDataset but
    // loaded into m_trainingDataset via loadDataset. I will assume we load the
    // same files for both or the user calls this twice? Actually, the original
    // code had loadDataset just load into m_trainingDataset. User probably
    // expects standard MNIST loading. For simplicity, I'll load into
    // m_trainingDataset.

    auto dataset = cgroot::data::MNISTLoader::load_training_data(
        imagesPath.toStdString(), labelsPath.toStdString());
    if (dataset) {
      m_trainingDataset = std::move(dataset);

      // For now, let's just shallow copy or rely on user loading test set
      // separately? Original code: m_testDataset was practically unused or null
      // unless loaded? Wait, original usage of m_testDataset in doInference
      // check: if (!m_model || !m_testDataset) ...

      // Let's create a partial copy or just point to it.
      // For this implementation, I'll clone the dataset for testing or reuse
      // it. Since unique_ptr owns it, I can't double own. I will RELOAD it for
      // testDataset for now to avoid complexity, OR just use m_trainingDataset
      // for inference if test is null. Actually, better to just set both for
      // now since the GUI only has "Load Dataset".

      // Re-loading is expensive. Let's make a "view" or just use current
      // dataset for inference too. I'll leave m_testDataset empty and use
      // m_trainingDataset for inference if m_testDataset is null.

      emit logMessage(QString("Successfully loaded %1 images")
                          .arg(m_trainingDataset->num_images));
    } else {
      emit logMessage("Failed to load dataset");
    }
  } catch (const std::exception &e) {
    emit logMessage(QString("Error loading dataset: %1").arg(e.what()));
  }
}
void ModelWorker::initializeModel() {
  if (m_model)
    return;
  emit logMessage("Worker: Initializing model...");
  architecture arch;
  arch.numOfLayers = 3;
  // Note: NNModel usage here depends on how NNModel is defined.
  // Original code: m_model = std::make_unique<NNModel>(arch, 10, 28, 28);
  // NNModel constructor takes arch, numOfClasses, H, W, D.
  // We assume 28x28x1 for MNIST.
  m_model = std::make_unique<NNModel>(arch, 10, 28, 28, 1);
  emit logMessage("Worker: Model initialized");
}
void ModelWorker::trainModel(int epochs) {
  if (!m_trainingDataset) {
    emit logMessage("Error: No dataset available for training");
    emit trainingFinished();
    return;
  }
  initializeModel();
  m_shouldStop = false;
  emit modelStatusChanged(true);
  // Prepare data structures for training loop to avoid repeated allocations
  std::vector<std::vector<unsigned char>> imageData(
      28, std::vector<unsigned char>(28));
  for (int epoch = 0; epoch < epochs && !m_shouldStop.load(); ++epoch) {
    emit logMessage(QString("Epoch %1/%2").arg(epoch + 1).arg(epochs));

    double epochLoss = 0.0;
    int correctPredictions = 0;
    int totalSamples = 0;

    size_t numBatches =
        (m_trainingDataset->images.size() + 31) / 32; // Batch size 32

    for (size_t batch = 0; batch < numBatches && !m_shouldStop.load();
         ++batch) {
      size_t startIdx = batch * 32;
      size_t endIdx = std::min(startIdx + 32, m_trainingDataset->images.size());

      for (size_t i = startIdx; i < endIdx; ++i) {
        const auto &image = m_trainingDataset->images[i];

        // Optimized copy
        for (int row = 0; row < 28; ++row) {
          // memcpy might be faster but loop is safe
          for (int col = 0; col < 28; ++col) {
            imageData[row][col] = image.pixels[row * 28 + col];
          }
        }

        if (m_model) {
          m_model->train(imageData, image.label);
          int predicted = m_model->classify(imageData);

          if (predicted == image.label) {
            correctPredictions++;
          }
          totalSamples++;
        }
      }

      if (batch % 10 == 0) {
        emit progressUpdated(batch + 1, numBatches);
        // Process events to keep loop responsive to stop signal if it was set
        // via atomics? No, we check m_shouldStop.
      }
    }

    if (totalSamples > 0) {
      double accuracy = static_cast<double>(correctPredictions) / totalSamples;
      double loss = 1.0 - accuracy; // Dummy loss calculation
      emit metricsUpdated(loss, accuracy, epoch + 1);
    }
  }
  m_shouldStop = false;
  emit modelStatusChanged(false);
  emit trainingFinished();
}
void ModelWorker::stopTraining() { m_shouldStop = true; }
void ModelWorker::runInference() {
  // Use training dataset if test is null, for demo purposes
  auto *dataset = m_testDataset ? m_testDataset.get() : m_trainingDataset.get();

  if (!m_model || !dataset) {
    emit logMessage("Error: Model or dataset not ready for inference");
    emit inferenceFinished();
    return;
  }
  emit modelStatusChanged(true);
  emit logMessage("Starting inference...");
  int numSamples = std::min(100, static_cast<int>(dataset->images.size()));
  std::vector<std::vector<unsigned char>> imageData(
      28, std::vector<unsigned char>(28));
  for (int i = 0; i < numSamples && !m_shouldStop.load(); ++i) {
    const auto &image = dataset->images[i];

    for (int row = 0; row < 28; ++row) {
      for (int col = 0; col < 28; ++col) {
        imageData[row][col] = image.pixels[row * 28 + col];
      }
    }
    int predicted = m_model->classify(imageData);

    // Convert to QImage for display
    QImage qImage = convertMNISTImageToQImage(image.pixels, 28, 28);
    QVector<double> probs(10, 0.0); // Dummy probabilities
    probs[predicted] = 1.0;
    emit imagePredicted(predicted, qImage, probs);
    emit progressUpdated(i + 1, numSamples);

    QThread::msleep(50); // Slow down for visualization
  }
  emit modelStatusChanged(false);
  emit inferenceFinished();
}
QImage
ModelWorker::convertMNISTImageToQImage(const std::vector<uint8_t> &pixels,
                                       int width, int height) {
  QImage image(width, height, QImage::Format_Grayscale8);
  // ... copy pixels ...
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int val = pixels[y * width + x];
      image.setPixel(x, y, qRgb(val, val, val));
    }
  }
  return image.scaled(280, 280, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}