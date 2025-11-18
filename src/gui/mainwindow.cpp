#include "mainwindow.h"
#include "widgets/trainingwidget.h"
#include "widgets/inferencewidget.h"
#include "widgets/configurationwidget.h"
#include "widgets/metricswidget.h"
#include "widgets/imageviewerwidget.h"
#include "controllers/modelcontroller.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QSplitter>
#include <QGroupBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_tabWidget(nullptr)
    , m_trainingWidget(nullptr)
    , m_inferenceWidget(nullptr)
    , m_configWidget(nullptr)
    , m_metricsWidget(nullptr)
    , m_logOutput(nullptr)
    , m_progressBar(nullptr)
    , m_statusLabel(nullptr)
    , m_workerThread(nullptr)
    , m_updateTimer(nullptr)
{
    setupUI();
    setupMenuBar();
    setupToolBar();
    setupStatusBar();
    createConnections();
    
    m_modelController = std::make_unique<ModelController>();
    m_workerThread = new QThread(this);
    m_modelController->moveToThread(m_workerThread);
    m_workerThread->start();
    
    m_updateTimer = new QTimer(this);
    connect(m_updateTimer, &QTimer::timeout, this, [this]() {
        if (m_modelController) {
            m_modelController->updateMetrics();
        }
    });
    m_updateTimer->start(100);
    
    setWindowTitle("Fashion-MNIST Neural Network Trainer");
    resize(1400, 900);
}

MainWindow::~MainWindow()
{
    if (m_workerThread) {
        m_workerThread->quit();
        m_workerThread->wait();
    }
}

void MainWindow::setupUI()
{
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);
    
    m_tabWidget = new QTabWidget(this);
    m_tabWidget->setTabPosition(QTabWidget::North);
    m_tabWidget->setMovable(true);
    
    m_trainingWidget = new TrainingWidget(this);
    m_inferenceWidget = new InferenceWidget(this);
    m_configWidget = new ConfigurationWidget(this);
    m_metricsWidget = new MetricsWidget(this);
    
    m_tabWidget->addTab(m_trainingWidget, "Training");
    m_tabWidget->addTab(m_inferenceWidget, "Inference");
    m_tabWidget->addTab(m_configWidget, "Configuration");
    m_tabWidget->addTab(m_metricsWidget, "Metrics");
    
    mainLayout->addWidget(m_tabWidget);
    
    QGroupBox* logGroup = new QGroupBox("Log Output", this);
    QVBoxLayout* logLayout = new QVBoxLayout(logGroup);
    m_logOutput = new QTextEdit(this);
    m_logOutput->setReadOnly(true);
    m_logOutput->setMaximumHeight(150);
    m_logOutput->setFont(QFont("Consolas", 9));
    logLayout->addWidget(m_logOutput);
    mainLayout->addWidget(logGroup);
}

void MainWindow::setupMenuBar()
{
    QMenu* fileMenu = menuBar()->addMenu("&File");
    
    m_loadDatasetAction = fileMenu->addAction("&Load Dataset...", this, &MainWindow::onLoadDataset);
    m_loadDatasetAction->setShortcut(QKeySequence::Open);
    
    fileMenu->addSeparator();
    
    m_loadModelAction = fileMenu->addAction("Load &Model...", this, &MainWindow::onLoadModel);
    m_saveModelAction = fileMenu->addAction("&Save Model...", this, &MainWindow::onSaveModel);
    
    fileMenu->addSeparator();
    fileMenu->addAction("E&xit", this, &QWidget::close, QKeySequence::Quit);
    
    QMenu* trainingMenu = menuBar()->addMenu("&Training");
    m_startTrainingAction = trainingMenu->addAction("&Start Training", this, &MainWindow::onStartTraining);
    m_stopTrainingAction = trainingMenu->addAction("S&top Training", this, &MainWindow::onStopTraining);
    m_stopTrainingAction->setEnabled(false);
    
    QMenu* inferenceMenu = menuBar()->addMenu("&Inference");
    m_startInferenceAction = inferenceMenu->addAction("Run &Inference", this, &MainWindow::onStartInference);
    
    QMenu* helpMenu = menuBar()->addMenu("&Help");
    helpMenu->addAction("&About", this, [this]() {
        QMessageBox::about(this, "About", 
            "Fashion-MNIST Neural Network Trainer\n\n"
            "A comprehensive GUI for training and testing neural networks on Fashion-MNIST dataset.");
    });
}

void MainWindow::setupToolBar()
{
    QToolBar* toolBar = addToolBar("Main Toolbar");
    toolBar->setMovable(false);
    
    toolBar->addAction(m_loadDatasetAction);
    toolBar->addSeparator();
    toolBar->addAction(m_loadModelAction);
    toolBar->addAction(m_saveModelAction);
    toolBar->addSeparator();
    toolBar->addAction(m_startTrainingAction);
    toolBar->addAction(m_stopTrainingAction);
    toolBar->addSeparator();
    toolBar->addAction(m_startInferenceAction);
}

void MainWindow::setupStatusBar()
{
    m_statusLabel = new QLabel("Ready", this);
    statusBar()->addWidget(m_statusLabel);
    
    m_progressBar = new QProgressBar(this);
    m_progressBar->setMaximumWidth(200);
    m_progressBar->setTextVisible(true);
    statusBar()->addPermanentWidget(m_progressBar);
}

void MainWindow::createConnections()
{
    if (m_modelController) {
        connect(m_modelController.get(), &ModelController::metricsUpdated,
                this, &MainWindow::onUpdateMetrics);
        connect(m_modelController.get(), &ModelController::progressUpdated,
                this, &MainWindow::onUpdateProgress);
        connect(m_modelController.get(), &ModelController::logMessage,
                this, &MainWindow::onLogMessage);
        connect(m_modelController.get(), &ModelController::imagePredicted,
                this, &MainWindow::onImagePrediction);
        connect(m_modelController.get(), &ModelController::trainingFinished,
                this, &MainWindow::onTrainingFinished);
        connect(m_modelController.get(), &ModelController::modelStatusChanged,
                this, &MainWindow::onModelStatusChanged);
    }
    
    if (m_trainingWidget) {
        connect(m_trainingWidget, &TrainingWidget::startTrainingRequested,
                this, &MainWindow::onStartTraining);
        connect(m_trainingWidget, &TrainingWidget::stopTrainingRequested,
                this, &MainWindow::onStopTraining);
    }
    
    if (m_configWidget) {
        connect(m_configWidget, &ConfigurationWidget::parametersChanged,
                m_modelController.get(), &ModelController::updateParameters);
    }
}

void MainWindow::onLoadDataset()
{
    QString imagesPath = QFileDialog::getOpenFileName(this, 
        "Select Fashion-MNIST Images File", "", "MNIST Files (*.idx3-ubyte);;All Files (*.*)");
    if (imagesPath.isEmpty()) return;
    
    QString labelsPath = QFileDialog::getOpenFileName(this, 
        "Select Fashion-MNIST Labels File", "", "MNIST Files (*.idx1-ubyte);;All Files (*.*)");
    if (labelsPath.isEmpty()) return;
    
    if (m_modelController) {
        m_modelController->loadDataset(imagesPath.toStdString(), labelsPath.toStdString());
    }
}

void MainWindow::onLoadModel()
{
    QString filePath = QFileDialog::getOpenFileName(this, 
        "Load Model", "", "Model Files (*.model);;All Files (*.*)");
    if (filePath.isEmpty()) return;
    
    if (m_modelController) {
        m_modelController->loadModel(filePath.toStdString());
    }
}

void MainWindow::onSaveModel()
{
    QString filePath = QFileDialog::getSaveFileName(this, 
        "Save Model", "", "Model Files (*.model);;All Files (*.*)");
    if (filePath.isEmpty()) return;
    
    if (m_modelController) {
        m_modelController->saveModel(filePath.toStdString());
    }
}

void MainWindow::onStartTraining()
{
    if (m_modelController) {
        m_modelController->startTraining();
    }
    m_startTrainingAction->setEnabled(false);
    m_stopTrainingAction->setEnabled(true);
}

void MainWindow::onStopTraining()
{
    if (m_modelController) {
        m_modelController->stopTraining();
    }
    m_startTrainingAction->setEnabled(true);
    m_stopTrainingAction->setEnabled(false);
}

void MainWindow::onStartInference()
{
    if (m_modelController) {
        m_modelController->startInference();
    }
}

void MainWindow::onUpdateMetrics(double loss, double accuracy, int epoch)
{
    if (m_metricsWidget) {
        m_metricsWidget->updateMetrics(loss, accuracy, epoch);
    }
    
    QString status = QString("Epoch: %1 | Loss: %2 | Accuracy: %3%")
        .arg(epoch).arg(loss, 0, 'f', 4).arg(accuracy * 100, 0, 'f', 2);
    m_statusLabel->setText(status);
}

void MainWindow::onUpdateProgress(int value, int maximum)
{
    m_progressBar->setMaximum(maximum);
    m_progressBar->setValue(value);
    m_progressBar->setFormat(QString("%p% (%1/%2)").arg(value).arg(maximum));
}

void MainWindow::onLogMessage(const QString& message)
{
    m_logOutput->append(message);
    QTextCursor cursor = m_logOutput->textCursor();
    cursor.movePosition(QTextCursor::End);
    m_logOutput->setTextCursor(cursor);
}

void MainWindow::onImagePrediction(int predictedClass, const QImage& image, const QVector<double>& probabilities)
{
    if (m_inferenceWidget) {
        m_inferenceWidget->displayPrediction(predictedClass, image, probabilities);
    }
}

void MainWindow::onTrainingFinished()
{
    m_startTrainingAction->setEnabled(true);
    m_stopTrainingAction->setEnabled(false);
    m_statusLabel->setText("Training completed");
    QMessageBox::information(this, "Training Complete", "Model training has finished successfully!");
}

void MainWindow::onModelStatusChanged(bool isTraining)
{
    m_startTrainingAction->setEnabled(!isTraining);
    m_stopTrainingAction->setEnabled(isTraining);
}

