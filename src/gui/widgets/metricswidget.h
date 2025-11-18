#ifndef METRICSWIDGET_H
#define METRICSWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QGroupBox>
#include <QTimer>
#include <memory>

QT_BEGIN_NAMESPACE
class QChart;
class QChartView;
class QLineSeries;
class QValueAxis;
class QDateTimeAxis;
QT_END_NAMESPACE

class MetricsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MetricsWidget(QWidget *parent = nullptr);
    ~MetricsWidget();

public slots:
    void updateMetrics(double loss, double accuracy, int epoch);
    void clearMetrics();

private:
    void setupUI();
    void setupCharts();
    
    QChart* m_lossChart;
    QChart* m_accuracyChart;
    QChartView* m_lossChartView;
    QChartView* m_accuracyChartView;
    
    QLineSeries* m_lossSeries;
    QLineSeries* m_accuracySeries;
    
    QValueAxis* m_lossAxisX;
    QValueAxis* m_lossAxisY;
    QValueAxis* m_accuracyAxisX;
    QValueAxis* m_accuracyAxisY;
    
    QLabel* m_currentLossLabel;
    QLabel* m_currentAccuracyLabel;
    QLabel* m_currentEpochLabel;
    QLabel* m_bestLossLabel;
    QLabel* m_bestAccuracyLabel;
    
    double m_bestLoss;
    double m_bestAccuracy;
    int m_maxDataPoints;
    
    QTimer* m_chartUpdateTimer;
};

#endif // METRICSWIDGET_H

