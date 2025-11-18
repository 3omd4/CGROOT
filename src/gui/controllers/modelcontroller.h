#ifndef MODELCONTROLLER_H
#define MODELCONTROLLER_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QImage>
#include <QVector>
#include <string>
#include <memory>
#include <atomic>

class NNModel;
#include "../../core/utils/mnist_loader.h"

class ModelController : public QObject
{
    Q_OBJECT

public:
    explicit ModelController(QObject *parent = nullptr);
    ~ModelController();

    void loadDataset(const std::string& imagesPath, const std::string& labelsPath);
    void loadModel(const std::string& modelPath);
    void saveModel(const std::string& modelPath);
    void startTraining();
    void stopTraining();
    void startInference();
    void updateParameters();
    void updateMetrics();

signals:
    void metricsUpdated(double loss, double accuracy, int epoch);
    void progressUpdated(int value, int maximum);
    void logMessage(const QString& message);
    void imagePredicted(int predictedClass, const QImage& image, const QVector<double>& probabilities);
    void trainingFinished();
    void modelStatusChanged(bool isTraining);

private slots:
    void doTraining();
    void doInference();

private:
    void initializeModel();
    QImage convertMNISTImageToQImage(const std::vector<uint8_t>& pixels, int width, int height);
    
    std::unique_ptr<NNModel> m_model;
    std::unique_ptr<cgroot::data::MNISTLoader::MNISTDataset> m_trainingDataset;
    std::unique_ptr<cgroot::data::MNISTLoader::MNISTDataset> m_testDataset;
    
    QMutex m_mutex;
    std::atomic<bool> m_isTraining;
    std::atomic<bool> m_shouldStop;
    
    int m_currentEpoch;
    int m_totalEpochs;
    double m_currentLoss;
    double m_currentAccuracy;
    
    QString m_datasetImagesPath;
    QString m_datasetLabelsPath;
    QString m_modelPath;
};

#endif // MODELCONTROLLER_H

