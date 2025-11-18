#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QProgressBar>
#include <QStatusBar>
#include <QMenuBar>
#include <QToolBar>
#include <QAction>
#include <QTimer>
#include <QThread>
#include <memory>

class TrainingWidget;
class InferenceWidget;
class ConfigurationWidget;
class MetricsWidget;
class ImageViewerWidget;
class ModelController;

QT_BEGIN_NAMESPACE
class QChartView;
class QChart;
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onLoadDataset();
    void onLoadModel();
    void onSaveModel();
    void onStartTraining();
    void onStopTraining();
    void onStartInference();
    void onUpdateMetrics(double loss, double accuracy, int epoch);
    void onUpdateProgress(int value, int maximum);
    void onLogMessage(const QString& message);
    void onImagePrediction(int predictedClass, const QImage& image, const QVector<double>& probabilities);
    void onTrainingFinished();
    void onModelStatusChanged(bool isTraining);

private:
    void setupUI();
    void setupMenuBar();
    void setupToolBar();
    void setupStatusBar();
    void createConnections();
    
    QTabWidget* m_tabWidget;
    
    TrainingWidget* m_trainingWidget;
    InferenceWidget* m_inferenceWidget;
    ConfigurationWidget* m_configWidget;
    MetricsWidget* m_metricsWidget;
    
    QTextEdit* m_logOutput;
    QProgressBar* m_progressBar;
    QLabel* m_statusLabel;
    
    QAction* m_loadDatasetAction;
    QAction* m_loadModelAction;
    QAction* m_saveModelAction;
    QAction* m_startTrainingAction;
    QAction* m_stopTrainingAction;
    QAction* m_startInferenceAction;
    
    std::unique_ptr<ModelController> m_modelController;
    QThread* m_workerThread;
    
    QTimer* m_updateTimer;
};

#endif // MAINWINDOW_H

