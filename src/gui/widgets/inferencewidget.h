#ifndef INFERENCEWIDGET_H
#define INFERENCEWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QSpinBox>
#include <QLabel>
#include <QGroupBox>
#include <QProgressBar>
#include <QListWidget>
#include <QImage>
#include <QPixmap>

class ImageViewerWidget;

class InferenceWidget : public QWidget
{
    Q_OBJECT

public:
    explicit InferenceWidget(QWidget *parent = nullptr);

public slots:
    void displayPrediction(int predictedClass, const QImage& image, const QVector<double>& probabilities);
    void onInferenceStarted();
    void onInferenceFinished();

private slots:
    void onRunInference();
    void onLoadImage();
    void onImageSelected(int index);

private:
    void setupUI();
    void updateClassProbabilities(const QVector<double>& probabilities);
    
    ImageViewerWidget* m_imageViewer;
    QLabel* m_predictedClassLabel;
    QLabel* m_confidenceLabel;
    QListWidget* m_probabilityList;
    QPushButton* m_runInferenceButton;
    QPushButton* m_loadImageButton;
    QSpinBox* m_numSamplesSpinBox;
    QProgressBar* m_inferenceProgress;
    
    static const QStringList m_classNames;
};

#endif // INFERENCEWIDGET_H

