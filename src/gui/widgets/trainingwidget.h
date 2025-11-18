#ifndef TRAININGWIDGET_H
#define TRAININGWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QComboBox>

class TrainingWidget : public QWidget
{
    Q_OBJECT

public:
    explicit TrainingWidget(QWidget *parent = nullptr);

signals:
    void startTrainingRequested();
    void stopTrainingRequested();

public slots:
    void onTrainingStarted();
    void onTrainingStopped();

private slots:
    void onStartClicked();
    void onStopClicked();

private:
    void setupUI();
    
    QPushButton* m_startButton;
    QPushButton* m_stopButton;
    
    QSpinBox* m_epochsSpinBox;
    QSpinBox* m_batchSizeSpinBox;
    QDoubleSpinBox* m_learningRateSpinBox;
    QComboBox* m_optimizerComboBox;
    QCheckBox* m_useValidationCheckBox;
    QSpinBox* m_validationSplitSpinBox;
    QCheckBox* m_saveCheckpointsCheckBox;
    QLineEdit* m_checkpointPathEdit;
    
    QLabel* m_statusLabel;
};

#endif // TRAININGWIDGET_H

