#include "modelcontroller.h"
#include "../../core/model.h"
#include "../../core/utils/mnist_loader.h"
#include <QDebug>
#include <QImage>
#include <QThread>
#include <QMutexLocker>
#include <cmath>
#include <algorithm>

ModelController::ModelController(QObject *parent)
    : QObject(parent)
    , m_model(nullptr)
    , m_isTraining(false)
    , m_shouldStop(false)
    , m_currentEpoch(0)
    , m_totalEpochs(10)
    , m_currentLoss(0.0)
    , m_currentAccuracy(0.0)
{
}

ModelController::~ModelController()
{
    stopTraining();
}

void ModelController::loadDataset(const std::string& imagesPath, const std::string& labelsPath)
{
    QMutexLocker locker(&m_mutex);
    
    emit logMessage(QString("Loading dataset from:\n  Images: %1\n  Labels: %2")
                    .arg(QString::fromStdString(imagesPath))
                    .arg(QString::fromStdString(labelsPath)));
    
    m_datasetImagesPath = QString::fromStdString(imagesPath);
    m_datasetLabelsPath = QString::fromStdString(labelsPath);
    
    try {
        auto dataset = cgroot::data::MNISTLoader::load_training_data(imagesPath, labelsPath);
        if (dataset) {
            m_trainingDataset = std::move(dataset);
            emit logMessage(QString("Successfully loaded %1 training images")
                          .arg(m_trainingDataset->num_images));
        } else {
            emit logMessage("Failed to load dataset");
        }
    } catch (const std::exception& e) {
        emit logMessage(QString("Error loading dataset: %1").arg(e.what()));
    }
}

void ModelController::loadModel(const std::string& modelPath)
{
    QMutexLocker locker(&m_mutex);
    
    m_modelPath = QString::fromStdString(modelPath);
    emit logMessage(QString("Loading model from: %1").arg(m_modelPath));
    
    // TODO: Implement model loading
    emit logMessage("Model loading not yet implemented");
}

void ModelController::saveModel(const std::string& modelPath)
{
    QMutexLocker locker(&m_mutex);
    
    m_modelPath = QString::fromStdString(modelPath);
    emit logMessage(QString("Saving model to: %1").arg(m_modelPath));
    
    // TODO: Implement model saving
    emit logMessage("Model saving not yet implemented");
}

void ModelController::startTraining()
{
    if (m_isTraining.load()) {
        emit logMessage("Training already in progress");
        return;
    }
    
    if (!m_trainingDataset) {
        emit logMessage("No dataset loaded. Please load a dataset first.");
        return;
    }
    
    m_shouldStop = false;
    m_isTraining = true;
    emit modelStatusChanged(true);
    
    QMetaObject::invokeMethod(this, "doTraining", Qt::QueuedConnection);
}

void ModelController::stopTraining()
{
    m_shouldStop = true;
    if (m_isTraining.load()) {
        emit logMessage("Stopping training...");
    }
}

void ModelController::startInference()
{
    if (!m_model) {
        emit logMessage("No model loaded. Please load or train a model first.");
        return;
    }
    
    QMetaObject::invokeMethod(this, "doInference", Qt::QueuedConnection);
}

void ModelController::updateParameters()
{
    emit logMessage("Parameters updated");
}

void ModelController::updateMetrics()
{
    if (m_isTraining.load()) {
        emit metricsUpdated(m_currentLoss, m_currentAccuracy, m_currentEpoch);
    }
}

void ModelController::doTraining()
{
    emit logMessage("Starting training...");
    
    if (!m_trainingDataset) {
        emit logMessage("Error: No training dataset available");
        m_isTraining = false;
        emit modelStatusChanged(false);
        emit trainingFinished();
        return;
    }
    
    initializeModel();
    
    if (!m_model) {
        emit logMessage("Error: Failed to initialize model");
        m_isTraining = false;
        emit modelStatusChanged(false);
        emit trainingFinished();
        return;
    }
    
    m_totalEpochs = 10;
    m_currentEpoch = 0;
    
    for (int epoch = 0; epoch < m_totalEpochs && !m_shouldStop.load(); ++epoch) {
        m_currentEpoch = epoch;
        emit logMessage(QString("Epoch %1/%2").arg(epoch + 1).arg(m_totalEpochs));
        
        double epochLoss = 0.0;
        int correctPredictions = 0;
        int totalSamples = 0;
        
        size_t numBatches = (m_trainingDataset->images.size() + 31) / 32;
        for (size_t batch = 0; batch < numBatches && !m_shouldStop.load(); ++batch) {
            size_t startIdx = batch * 32;
            size_t endIdx = qMin(startIdx + 32, m_trainingDataset->images.size());
            
            for (size_t i = startIdx; i < endIdx; ++i) {
                const auto& image = m_trainingDataset->images[i];
                
                std::vector<std::vector<unsigned char>> imageData(28);
                for (int row = 0; row < 28; ++row) {
                    imageData[row].resize(28);
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
            
            emit progressUpdated(batch + 1, numBatches);
            QThread::msleep(10);
        }
        
        if (m_shouldStop.load()) {
            break;
        }
        
        m_currentLoss = 1.0 - (static_cast<double>(correctPredictions) / totalSamples);
        m_currentAccuracy = static_cast<double>(correctPredictions) / totalSamples;
        
        emit metricsUpdated(m_currentLoss, m_currentAccuracy, epoch + 1);
        emit logMessage(QString("Epoch %1 completed - Loss: %2, Accuracy: %3%")
                       .arg(epoch + 1)
                       .arg(m_currentLoss, 0, 'f', 4)
                       .arg(m_currentAccuracy * 100, 0, 'f', 2));
    }
    
    m_isTraining = false;
    emit modelStatusChanged(false);
    emit trainingFinished();
    emit logMessage("Training completed");
}

void ModelController::doInference()
{
    emit logMessage("Starting inference...");
    
    if (!m_model || !m_testDataset) {
        emit logMessage("Error: Model or test dataset not available");
        return;
    }
    
    int numSamples = qMin(100, static_cast<int>(m_testDataset->images.size()));
    
    for (int i = 0; i < numSamples; ++i) {
        const auto& image = m_testDataset->images[i];
        
        std::vector<std::vector<unsigned char>> imageData(28);
        for (int row = 0; row < 28; ++row) {
            imageData[row].resize(28);
            for (int col = 0; col < 28; ++col) {
                imageData[row][col] = image.pixels[row * 28 + col];
            }
        }
        
        if (m_model) {
            int predicted = m_model->classify(imageData);
            QImage qImage = convertMNISTImageToQImage(image.pixels, 28, 28);
            
            QVector<double> probabilities(10, 0.0);
            probabilities[predicted] = 1.0;
            
            emit imagePredicted(predicted, qImage, probabilities);
        }
        
        emit progressUpdated(i + 1, numSamples);
        QThread::msleep(50);
    }
    
    emit logMessage("Inference completed");
}

void ModelController::initializeModel()
{
    if (m_model) {
        return;
    }
    
    emit logMessage("Initializing model...");
    
    architecture arch;
    arch.numOfLayers = 3;
    
    m_model = std::make_unique<NNModel>(arch, 10, 28, 28);
    
    emit logMessage("Model initialized");
}

QImage ModelController::convertMNISTImageToQImage(const std::vector<uint8_t>& pixels, int width, int height)
{
    QImage image(width, height, QImage::Format_Grayscale8);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (idx < static_cast<int>(pixels.size())) {
                image.setPixel(x, y, qRgb(pixels[idx], pixels[idx], pixels[idx]));
            }
        }
    }
    
    return image.scaled(280, 280, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

