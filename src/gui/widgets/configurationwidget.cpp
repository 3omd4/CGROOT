#include "configurationwidget.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QScrollArea>

ConfigurationWidget::ConfigurationWidget(QWidget *parent)
    : QWidget(parent)
    , m_tabWidget(nullptr)
{
    setupUI();
}

void ConfigurationWidget::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    m_tabWidget = new QTabWidget(this);
    
    setupModelTab();
    setupTrainingTab();
    setupNetworkTab();
    
    mainLayout->addWidget(m_tabWidget);
    
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    m_resetButton = new QPushButton("Reset to Defaults", this);
    connect(m_resetButton, &QPushButton::clicked, this, &ConfigurationWidget::onResetDefaults);
    
    m_loadConfigButton = new QPushButton("Load Config", this);
    connect(m_loadConfigButton, &QPushButton::clicked, this, &ConfigurationWidget::onLoadConfig);
    
    m_saveConfigButton = new QPushButton("Save Config", this);
    connect(m_saveConfigButton, &QPushButton::clicked, this, &ConfigurationWidget::onSaveConfig);
    
    buttonLayout->addWidget(m_resetButton);
    buttonLayout->addWidget(m_loadConfigButton);
    buttonLayout->addWidget(m_saveConfigButton);
    buttonLayout->addStretch();
    
    mainLayout->addLayout(buttonLayout);
}

void ConfigurationWidget::setupModelTab()
{
    QScrollArea* scrollArea = new QScrollArea();
    QWidget* modelWidget = new QWidget();
    QFormLayout* modelLayout = new QFormLayout(modelWidget);
    
    m_numClassesSpinBox = new QSpinBox(this);
    m_numClassesSpinBox->setMinimum(2);
    m_numClassesSpinBox->setMaximum(1000);
    m_numClassesSpinBox->setValue(10);
    connect(m_numClassesSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &ConfigurationWidget::onParameterChanged);
    modelLayout->addRow("Number of Classes:", m_numClassesSpinBox);
    
    m_imageWidthSpinBox = new QSpinBox(this);
    m_imageWidthSpinBox->setMinimum(1);
    m_imageWidthSpinBox->setMaximum(1000);
    m_imageWidthSpinBox->setValue(28);
    connect(m_imageWidthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &ConfigurationWidget::onParameterChanged);
    modelLayout->addRow("Image Width:", m_imageWidthSpinBox);
    
    m_imageHeightSpinBox = new QSpinBox(this);
    m_imageHeightSpinBox->setMinimum(1);
    m_imageHeightSpinBox->setMaximum(1000);
    m_imageHeightSpinBox->setValue(28);
    connect(m_imageHeightSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &ConfigurationWidget::onParameterChanged);
    modelLayout->addRow("Image Height:", m_imageHeightSpinBox);
    
    m_numLayersSpinBox = new QSpinBox(this);
    m_numLayersSpinBox->setMinimum(1);
    m_numLayersSpinBox->setMaximum(100);
    m_numLayersSpinBox->setValue(3);
    connect(m_numLayersSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &ConfigurationWidget::onParameterChanged);
    modelLayout->addRow("Number of Layers:", m_numLayersSpinBox);
    
    scrollArea->setWidget(modelWidget);
    scrollArea->setWidgetResizable(true);
    m_tabWidget->addTab(scrollArea, "Model");
}

void ConfigurationWidget::setupTrainingTab()
{
    QScrollArea* scrollArea = new QScrollArea();
    QWidget* trainingWidget = new QWidget();
    QFormLayout* trainingLayout = new QFormLayout(trainingWidget);
    
    m_optimizerComboBox = new QComboBox(this);
    m_optimizerComboBox->addItems({"SGD", "Adam", "RMSprop"});
    connect(m_optimizerComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &ConfigurationWidget::onParameterChanged);
    trainingLayout->addRow("Optimizer:", m_optimizerComboBox);
    
    m_learningRateSpinBox = new QDoubleSpinBox(this);
    m_learningRateSpinBox->setMinimum(0.00001);
    m_learningRateSpinBox->setMaximum(10.0);
    m_learningRateSpinBox->setValue(0.001);
    m_learningRateSpinBox->setDecimals(5);
    connect(m_learningRateSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            this, &ConfigurationWidget::onParameterChanged);
    trainingLayout->addRow("Learning Rate:", m_learningRateSpinBox);
    
    m_weightDecaySpinBox = new QDoubleSpinBox(this);
    m_weightDecaySpinBox->setMinimum(0.0);
    m_weightDecaySpinBox->setMaximum(1.0);
    m_weightDecaySpinBox->setValue(0.0001);
    m_weightDecaySpinBox->setDecimals(5);
    connect(m_weightDecaySpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            this, &ConfigurationWidget::onParameterChanged);
    trainingLayout->addRow("Weight Decay:", m_weightDecaySpinBox);
    
    m_momentumSpinBox = new QDoubleSpinBox(this);
    m_momentumSpinBox->setMinimum(0.0);
    m_momentumSpinBox->setMaximum(1.0);
    m_momentumSpinBox->setValue(0.9);
    m_momentumSpinBox->setDecimals(3);
    connect(m_momentumSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            this, &ConfigurationWidget::onParameterChanged);
    trainingLayout->addRow("Momentum:", m_momentumSpinBox);
    
    m_epochsSpinBox = new QSpinBox(this);
    m_epochsSpinBox->setMinimum(1);
    m_epochsSpinBox->setMaximum(10000);
    m_epochsSpinBox->setValue(10);
    connect(m_epochsSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &ConfigurationWidget::onParameterChanged);
    trainingLayout->addRow("Epochs:", m_epochsSpinBox);
    
    m_batchSizeSpinBox = new QSpinBox(this);
    m_batchSizeSpinBox->setMinimum(1);
    m_batchSizeSpinBox->setMaximum(10000);
    m_batchSizeSpinBox->setValue(32);
    connect(m_batchSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &ConfigurationWidget::onParameterChanged);
    trainingLayout->addRow("Batch Size:", m_batchSizeSpinBox);
    
    m_useValidationCheckBox = new QCheckBox(this);
    m_useValidationCheckBox->setChecked(true);
    connect(m_useValidationCheckBox, &QCheckBox::toggled, 
            this, &ConfigurationWidget::onParameterChanged);
    trainingLayout->addRow("Use Validation Set:", m_useValidationCheckBox);
    
    m_validationSplitSpinBox = new QDoubleSpinBox(this);
    m_validationSplitSpinBox->setMinimum(0.0);
    m_validationSplitSpinBox->setMaximum(0.5);
    m_validationSplitSpinBox->setValue(0.2);
    m_validationSplitSpinBox->setDecimals(2);
    connect(m_validationSplitSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            this, &ConfigurationWidget::onParameterChanged);
    trainingLayout->addRow("Validation Split:", m_validationSplitSpinBox);
    
    scrollArea->setWidget(trainingWidget);
    scrollArea->setWidgetResizable(true);
    m_tabWidget->addTab(scrollArea, "Training");
}

void ConfigurationWidget::setupNetworkTab()
{
    QWidget* networkWidget = new QWidget();
    QVBoxLayout* networkLayout = new QVBoxLayout(networkWidget);
    
    QLabel* infoLabel = new QLabel(
        "Network architecture configuration will be available here.\n"
        "You can configure layer types, activation functions, and layer sizes.",
        this);
    infoLabel->setWordWrap(true);
    networkLayout->addWidget(infoLabel);
    
    m_tabWidget->addTab(networkWidget, "Network Architecture");
}

ConfigurationWidget::ModelParameters ConfigurationWidget::getModelParameters() const
{
    ModelParameters params;
    params.numClasses = m_numClassesSpinBox->value();
    params.imageWidth = m_imageWidthSpinBox->value();
    params.imageHeight = m_imageHeightSpinBox->value();
    params.numLayers = m_numLayersSpinBox->value();
    return params;
}

ConfigurationWidget::TrainingParameters ConfigurationWidget::getTrainingParameters() const
{
    TrainingParameters params;
    params.epochs = m_epochsSpinBox->value();
    params.batchSize = m_batchSizeSpinBox->value();
    params.learningRate = m_learningRateSpinBox->value();
    params.optimizer = m_optimizerComboBox->currentText();
    params.weightDecay = m_weightDecaySpinBox->value();
    params.momentum = m_momentumSpinBox->value();
    params.useValidation = m_useValidationCheckBox->isChecked();
    params.validationSplit = m_validationSplitSpinBox->value();
    return params;
}

void ConfigurationWidget::setModelParameters(const ModelParameters& params)
{
    m_numClassesSpinBox->setValue(params.numClasses);
    m_imageWidthSpinBox->setValue(params.imageWidth);
    m_imageHeightSpinBox->setValue(params.imageHeight);
    m_numLayersSpinBox->setValue(params.numLayers);
}

void ConfigurationWidget::setTrainingParameters(const TrainingParameters& params)
{
    m_epochsSpinBox->setValue(params.epochs);
    m_batchSizeSpinBox->setValue(params.batchSize);
    m_learningRateSpinBox->setValue(params.learningRate);
    int optimizerIndex = m_optimizerComboBox->findText(params.optimizer);
    if (optimizerIndex >= 0) {
        m_optimizerComboBox->setCurrentIndex(optimizerIndex);
    }
    m_weightDecaySpinBox->setValue(params.weightDecay);
    m_momentumSpinBox->setValue(params.momentum);
    m_useValidationCheckBox->setChecked(params.useValidation);
    m_validationSplitSpinBox->setValue(params.validationSplit);
}

void ConfigurationWidget::onParameterChanged()
{
    emit parametersChanged();
}

void ConfigurationWidget::onResetDefaults()
{
    m_numClassesSpinBox->setValue(10);
    m_imageWidthSpinBox->setValue(28);
    m_imageHeightSpinBox->setValue(28);
    m_numLayersSpinBox->setValue(3);
    m_optimizerComboBox->setCurrentIndex(0);
    m_learningRateSpinBox->setValue(0.001);
    m_weightDecaySpinBox->setValue(0.0001);
    m_momentumSpinBox->setValue(0.9);
    m_epochsSpinBox->setValue(10);
    m_batchSizeSpinBox->setValue(32);
    m_useValidationCheckBox->setChecked(true);
    m_validationSplitSpinBox->setValue(0.2);
    
    emit parametersChanged();
}

void ConfigurationWidget::onLoadConfig()
{
    QString filePath = QFileDialog::getOpenFileName(this, 
        "Load Configuration", "", "Config Files (*.json *.cfg);;All Files (*.*)");
    if (filePath.isEmpty()) return;
    
    QMessageBox::information(this, "Load Config", 
        "Configuration loading will be implemented with JSON parsing.");
}

void ConfigurationWidget::onSaveConfig()
{
    QString filePath = QFileDialog::getSaveFileName(this, 
        "Save Configuration", "", "Config Files (*.json);;All Files (*.*)");
    if (filePath.isEmpty()) return;
    
    QMessageBox::information(this, "Save Config", 
        "Configuration saving will be implemented with JSON serialization.");
}

