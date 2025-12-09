#ifndef CONFIGURATIONWIDGET_H
#define CONFIGURATIONWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QTabWidget>
#include <QScrollArea>
#include <QVector>

class ConfigurationWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ConfigurationWidget(QWidget *parent = nullptr);
    
    struct ModelParameters {
        unsigned int numClasses;
        unsigned int imageWidth;
        unsigned int imageHeight;
        unsigned int numLayers;
        QStringList layerTypes;
        QStringList activationFunctions;
        QVector<unsigned int> neuronsPerLayer;
        QVector<unsigned int> convKernelSizes;
        QVector<unsigned int> numKernels;
    };
    
    struct TrainingParameters {
        int epochs;
        int batchSize;
        double learningRate;
        QString optimizer;
        double weightDecay;
        double momentum;
        bool useValidation;
        double validationSplit;
    };
    
    ModelParameters getModelParameters() const;
    TrainingParameters getTrainingParameters() const;
    
    void setModelParameters(const ModelParameters& params);
    void setTrainingParameters(const TrainingParameters& params);

signals:
    void parametersChanged();

private slots:
    void onParameterChanged();
    void onResetDefaults();
    void onLoadConfig();
    void onSaveConfig();

private:
    void setupUI();
    void setupModelTab();
    void setupTrainingTab();
    void setupNetworkTab();
    
    QTabWidget* m_tabWidget;
    
    QSpinBox* m_numClassesSpinBox;
    QSpinBox* m_imageWidthSpinBox;
    QSpinBox* m_imageHeightSpinBox;
    QSpinBox* m_numLayersSpinBox;
    
    QComboBox* m_optimizerComboBox;
    QDoubleSpinBox* m_learningRateSpinBox;
    QDoubleSpinBox* m_weightDecaySpinBox;
    QDoubleSpinBox* m_momentumSpinBox;
    QSpinBox* m_epochsSpinBox;
    QSpinBox* m_batchSizeSpinBox;
    QCheckBox* m_useValidationCheckBox;
    QDoubleSpinBox* m_validationSplitSpinBox;
    
    QPushButton* m_resetButton;
    QPushButton* m_loadConfigButton;
    QPushButton* m_saveConfigButton;
};

#endif // CONFIGURATIONWIDGET_H

