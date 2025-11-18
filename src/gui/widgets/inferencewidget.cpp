#include "inferencewidget.h"
#include "imageviewerwidget.h"
#include <QFileDialog>
#include <QListWidgetItem>
#include <QProgressBar>
#include <QGroupBox>
#include <QFormLayout>

const QStringList InferenceWidget::m_classNames = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

InferenceWidget::InferenceWidget(QWidget *parent)
    : QWidget(parent)
    , m_imageViewer(nullptr)
    , m_predictedClassLabel(nullptr)
    , m_confidenceLabel(nullptr)
    , m_probabilityList(nullptr)
    , m_runInferenceButton(nullptr)
    , m_loadImageButton(nullptr)
    , m_numSamplesSpinBox(nullptr)
    , m_inferenceProgress(nullptr)
{
    setupUI();
}

void InferenceWidget::setupUI()
{
    QHBoxLayout* mainLayout = new QHBoxLayout(this);
    
    QGroupBox* imageGroup = new QGroupBox("Input Image", this);
    QVBoxLayout* imageLayout = new QVBoxLayout(imageGroup);
    
    m_imageViewer = new ImageViewerWidget(this);
    m_imageViewer->setMinimumSize(300, 300);
    imageLayout->addWidget(m_imageViewer);
    
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    m_loadImageButton = new QPushButton("Load Image", this);
    connect(m_loadImageButton, &QPushButton::clicked, this, &InferenceWidget::onLoadImage);
    buttonLayout->addWidget(m_loadImageButton);
    buttonLayout->addStretch();
    imageLayout->addLayout(buttonLayout);
    
    mainLayout->addWidget(imageGroup, 1);
    
    QGroupBox* resultsGroup = new QGroupBox("Inference Results", this);
    QVBoxLayout* resultsLayout = new QVBoxLayout(resultsGroup);
    
    m_predictedClassLabel = new QLabel("Predicted: -", this);
    m_predictedClassLabel->setStyleSheet("QLabel { font-size: 18pt; font-weight: bold; padding: 10px; }");
    resultsLayout->addWidget(m_predictedClassLabel);
    
    m_confidenceLabel = new QLabel("Confidence: -", this);
    m_confidenceLabel->setStyleSheet("QLabel { font-size: 14pt; padding: 5px; }");
    resultsLayout->addWidget(m_confidenceLabel);
    
    resultsLayout->addWidget(new QLabel("Class Probabilities:", this));
    m_probabilityList = new QListWidget(this);
    m_probabilityList->setMaximumHeight(300);
    resultsLayout->addWidget(m_probabilityList);
    
    QGroupBox* inferenceGroup = new QGroupBox("Inference Settings", this);
    QFormLayout* inferenceLayout = new QFormLayout(inferenceGroup);
    
    m_numSamplesSpinBox = new QSpinBox(this);
    m_numSamplesSpinBox->setMinimum(1);
    m_numSamplesSpinBox->setMaximum(10000);
    m_numSamplesSpinBox->setValue(100);
    inferenceLayout->addRow("Number of Samples:", m_numSamplesSpinBox);
    
    m_runInferenceButton = new QPushButton("Run Inference", this);
    m_runInferenceButton->setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }");
    connect(m_runInferenceButton, &QPushButton::clicked, this, &InferenceWidget::onRunInference);
    inferenceLayout->addRow(m_runInferenceButton);
    
    m_inferenceProgress = new QProgressBar(this);
    m_inferenceProgress->setVisible(false);
    inferenceLayout->addRow("Progress:", m_inferenceProgress);
    
    resultsLayout->addWidget(inferenceGroup);
    resultsLayout->addStretch();
    
    mainLayout->addWidget(resultsGroup, 1);
}

void InferenceWidget::displayPrediction(int predictedClass, const QImage& image, const QVector<double>& probabilities)
{
    if (m_imageViewer) {
        m_imageViewer->displayImage(image);
    }
    
    if (predictedClass >= 0 && predictedClass < m_classNames.size()) {
        QString className = m_classNames[predictedClass];
        m_predictedClassLabel->setText(QString("Predicted: %1").arg(className));
        
        if (probabilities.size() > predictedClass) {
            double confidence = probabilities[predictedClass];
            m_confidenceLabel->setText(QString("Confidence: %1%").arg(confidence * 100, 0, 'f', 2));
        }
    }
    
    updateClassProbabilities(probabilities);
}

void InferenceWidget::updateClassProbabilities(const QVector<double>& probabilities)
{
    m_probabilityList->clear();
    
    QVector<QPair<int, double>> sortedProbs;
    for (int i = 0; i < probabilities.size() && i < m_classNames.size(); ++i) {
        sortedProbs.append(qMakePair(i, probabilities[i]));
    }
    
    std::sort(sortedProbs.begin(), sortedProbs.end(), 
        [](const QPair<int, double>& a, const QPair<int, double>& b) {
            return a.second > b.second;
        });
    
    for (const auto& pair : sortedProbs) {
        int classIdx = pair.first;
        double prob = pair.second;
        QString itemText = QString("%1: %2%")
            .arg(m_classNames[classIdx])
            .arg(prob * 100, 0, 'f', 2);
        
        QListWidgetItem* item = new QListWidgetItem(itemText, m_probabilityList);
        item->setData(Qt::UserRole, classIdx);
        m_probabilityList->addItem(item);
    }
}

void InferenceWidget::onRunInference()
{
    emit onInferenceStarted();
    m_runInferenceButton->setEnabled(false);
    m_inferenceProgress->setVisible(true);
    m_inferenceProgress->setRange(0, 0);
}

void InferenceWidget::onLoadImage()
{
    QString filePath = QFileDialog::getOpenFileName(this, 
        "Load Fashion-MNIST Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)");
    if (filePath.isEmpty()) return;
    
    QImage image(filePath);
    if (!image.isNull() && m_imageViewer) {
        m_imageViewer->displayImage(image);
    }
}

void InferenceWidget::onImageSelected(int index)
{
    // Handle image selection from dataset
}

void InferenceWidget::onInferenceStarted()
{
    m_runInferenceButton->setEnabled(false);
    m_inferenceProgress->setVisible(true);
}

void InferenceWidget::onInferenceFinished()
{
    m_runInferenceButton->setEnabled(true);
    m_inferenceProgress->setVisible(false);
}

