#include "trainingwidget.h"
#include <QFileDialog>
#include <QMessageBox>

TrainingWidget::TrainingWidget(QWidget *parent)
    : QWidget(parent)
    , m_startButton(nullptr)
    , m_stopButton(nullptr)
    , m_epochsSpinBox(nullptr)
    , m_batchSizeSpinBox(nullptr)
    , m_learningRateSpinBox(nullptr)
    , m_optimizerComboBox(nullptr)
    , m_useValidationCheckBox(nullptr)
    , m_validationSplitSpinBox(nullptr)
    , m_saveCheckpointsCheckBox(nullptr)
    , m_checkpointPathEdit(nullptr)
    , m_statusLabel(nullptr)
{
    setupUI();
}

void TrainingWidget::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    QGroupBox* paramsGroup = new QGroupBox("Training Parameters", this);
    QGridLayout* paramsLayout = new QGridLayout(paramsGroup);
    
    paramsLayout->addWidget(new QLabel("Number of Epochs:", this), 0, 0);
    m_epochsSpinBox = new QSpinBox(this);
    m_epochsSpinBox->setMinimum(1);
    m_epochsSpinBox->setMaximum(10000);
    m_epochsSpinBox->setValue(10);
    paramsLayout->addWidget(m_epochsSpinBox, 0, 1);
    
    paramsLayout->addWidget(new QLabel("Batch Size:", this), 1, 0);
    m_batchSizeSpinBox = new QSpinBox(this);
    m_batchSizeSpinBox->setMinimum(1);
    m_batchSizeSpinBox->setMaximum(10000);
    m_batchSizeSpinBox->setValue(32);
    paramsLayout->addWidget(m_batchSizeSpinBox, 1, 1);
    
    paramsLayout->addWidget(new QLabel("Learning Rate:", this), 2, 0);
    m_learningRateSpinBox = new QDoubleSpinBox(this);
    m_learningRateSpinBox->setMinimum(0.00001);
    m_learningRateSpinBox->setMaximum(10.0);
    m_learningRateSpinBox->setValue(0.001);
    m_learningRateSpinBox->setDecimals(5);
    m_learningRateSpinBox->setSingleStep(0.0001);
    paramsLayout->addWidget(m_learningRateSpinBox, 2, 1);
    
    paramsLayout->addWidget(new QLabel("Optimizer:", this), 3, 0);
    m_optimizerComboBox = new QComboBox(this);
    m_optimizerComboBox->addItems({"SGD", "Adam", "RMSprop"});
    paramsLayout->addWidget(m_optimizerComboBox, 3, 1);
    
    paramsLayout->addWidget(new QLabel("Validation Split:", this), 4, 0);
    m_validationSplitSpinBox = new QSpinBox(this);
    m_validationSplitSpinBox->setMinimum(0);
    m_validationSplitSpinBox->setMaximum(50);
    m_validationSplitSpinBox->setValue(20);
    m_validationSplitSpinBox->setSuffix("%");
    paramsLayout->addWidget(m_validationSplitSpinBox, 4, 1);
    
    m_useValidationCheckBox = new QCheckBox("Use Validation Set", this);
    m_useValidationCheckBox->setChecked(true);
    paramsLayout->addWidget(m_useValidationCheckBox, 5, 0, 1, 2);
    
    m_saveCheckpointsCheckBox = new QCheckBox("Save Checkpoints", this);
    paramsLayout->addWidget(m_saveCheckpointsCheckBox, 6, 0);
    
    paramsLayout->addWidget(new QLabel("Checkpoint Path:", this), 7, 0);
    m_checkpointPathEdit = new QLineEdit(this);
    m_checkpointPathEdit->setPlaceholderText("Select checkpoint directory...");
    paramsLayout->addWidget(m_checkpointPathEdit, 7, 1);
    
    QPushButton* browseButton = new QPushButton("Browse...", this);
    connect(browseButton, &QPushButton::clicked, this, [this]() {
        QString dir = QFileDialog::getExistingDirectory(this, "Select Checkpoint Directory");
        if (!dir.isEmpty()) {
            m_checkpointPathEdit->setText(dir);
        }
    });
    paramsLayout->addWidget(browseButton, 7, 2);
    
    mainLayout->addWidget(paramsGroup);
    
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    m_startButton = new QPushButton("Start Training", this);
    m_startButton->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }");
    connect(m_startButton, &QPushButton::clicked, this, &TrainingWidget::onStartClicked);
    
    m_stopButton = new QPushButton("Stop Training", this);
    m_stopButton->setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }");
    m_stopButton->setEnabled(false);
    connect(m_stopButton, &QPushButton::clicked, this, &TrainingWidget::onStopClicked);
    
    buttonLayout->addWidget(m_startButton);
    buttonLayout->addWidget(m_stopButton);
    buttonLayout->addStretch();
    
    mainLayout->addLayout(buttonLayout);
    
    m_statusLabel = new QLabel("Ready to start training", this);
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; padding: 5px; }");
    mainLayout->addWidget(m_statusLabel);
    
    mainLayout->addStretch();
}

void TrainingWidget::onStartClicked()
{
    emit startTrainingRequested();
    onTrainingStarted();
}

void TrainingWidget::onStopClicked()
{
    emit stopTrainingRequested();
    onTrainingStopped();
}

void TrainingWidget::onTrainingStarted()
{
    m_startButton->setEnabled(false);
    m_stopButton->setEnabled(true);
    m_statusLabel->setText("Training in progress...");
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: green; padding: 5px; }");
}

void TrainingWidget::onTrainingStopped()
{
    m_startButton->setEnabled(true);
    m_stopButton->setEnabled(false);
    m_statusLabel->setText("Training stopped");
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: red; padding: 5px; }");
}

