#include "metricswidget.h"
#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QDateTimeAxis>
#include <QValueAxis>
#include <QSplitter>
#include <QGroupBox>
#include <QFormLayout>
#include <QPainter>
#include <limits>

MetricsWidget::MetricsWidget(QWidget *parent)
    : QWidget(parent)
    , m_lossChart(nullptr)
    , m_accuracyChart(nullptr)
    , m_lossChartView(nullptr)
    , m_accuracyChartView(nullptr)
    , m_lossSeries(nullptr)
    , m_accuracySeries(nullptr)
    , m_bestLoss(std::numeric_limits<double>::max())
    , m_bestAccuracy(0.0)
    , m_maxDataPoints(1000)
{
    setupUI();
    setupCharts();
}

MetricsWidget::~MetricsWidget()
{
}

void MetricsWidget::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    QSplitter* splitter = new QSplitter(Qt::Vertical, this);
    
    QGroupBox* statsGroup = new QGroupBox("Current Statistics", this);
    QGridLayout* statsLayout = new QGridLayout(statsGroup);
    
    m_currentEpochLabel = new QLabel("Epoch: 0", this);
    m_currentLossLabel = new QLabel("Loss: 0.0000", this);
    m_currentAccuracyLabel = new QLabel("Accuracy: 0.00%", this);
    m_bestLossLabel = new QLabel("Best Loss: N/A", this);
    m_bestAccuracyLabel = new QLabel("Best Accuracy: N/A", this);
    
    statsLayout->addWidget(new QLabel("Current Epoch:", this), 0, 0);
    statsLayout->addWidget(m_currentEpochLabel, 0, 1);
    statsLayout->addWidget(new QLabel("Current Loss:", this), 1, 0);
    statsLayout->addWidget(m_currentLossLabel, 1, 1);
    statsLayout->addWidget(new QLabel("Current Accuracy:", this), 2, 0);
    statsLayout->addWidget(m_currentAccuracyLabel, 2, 1);
    statsLayout->addWidget(new QLabel("Best Loss:", this), 3, 0);
    statsLayout->addWidget(m_bestLossLabel, 3, 1);
    statsLayout->addWidget(new QLabel("Best Accuracy:", this), 4, 0);
    statsLayout->addWidget(m_bestAccuracyLabel, 4, 1);
    
    splitter->addWidget(statsGroup);
    
    QSplitter* chartsSplitter = new QSplitter(Qt::Horizontal, this);
    
    QGroupBox* lossGroup = new QGroupBox("Training Loss", this);
    QVBoxLayout* lossLayout = new QVBoxLayout(lossGroup);
    m_lossChartView = new QChartView(this);
    m_lossChartView->setRenderHint(QPainter::Antialiasing);
    lossLayout->addWidget(m_lossChartView);
    chartsSplitter->addWidget(lossGroup);
    
    QGroupBox* accuracyGroup = new QGroupBox("Training Accuracy", this);
    QVBoxLayout* accuracyLayout = new QVBoxLayout(accuracyGroup);
    m_accuracyChartView = new QChartView(this);
    m_accuracyChartView->setRenderHint(QPainter::Antialiasing);
    accuracyLayout->addWidget(m_accuracyChartView);
    chartsSplitter->addWidget(accuracyGroup);
    
    splitter->addWidget(chartsSplitter);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);
    
    mainLayout->addWidget(splitter);
}

void MetricsWidget::setupCharts()
{
    m_lossSeries = new QLineSeries();
    m_lossSeries->setName("Loss");
    m_lossSeries->setColor(QColor(255, 0, 0));
    
    m_accuracySeries = new QLineSeries();
    m_accuracySeries->setName("Accuracy");
    m_accuracySeries->setColor(QColor(0, 150, 0));
    
    m_lossChart = new QChart();
    m_lossChart->addSeries(m_lossSeries);
    m_lossChart->setTitle("Training Loss Over Time");
    m_lossChart->setAnimationOptions(QChart::NoAnimation);
    m_lossChart->legend()->setVisible(true);
    m_lossChart->legend()->setAlignment(Qt::AlignBottom);
    
    m_lossAxisX = new QValueAxis();
    m_lossAxisX->setTitleText("Epoch");
    m_lossAxisX->setLabelFormat("%d");
    m_lossAxisY = new QValueAxis();
    m_lossAxisY->setTitleText("Loss");
    m_lossAxisY->setLabelFormat("%.4f");
    
    m_lossChart->addAxis(m_lossAxisX, Qt::AlignBottom);
    m_lossChart->addAxis(m_lossAxisY, Qt::AlignLeft);
    m_lossSeries->attachAxis(m_lossAxisX);
    m_lossSeries->attachAxis(m_lossAxisY);
    
    m_accuracyChart = new QChart();
    m_accuracyChart->addSeries(m_accuracySeries);
    m_accuracyChart->setTitle("Training Accuracy Over Time");
    m_accuracyChart->setAnimationOptions(QChart::NoAnimation);
    m_accuracyChart->legend()->setVisible(true);
    m_accuracyChart->legend()->setAlignment(Qt::AlignBottom);
    
    m_accuracyAxisX = new QValueAxis();
    m_accuracyAxisX->setTitleText("Epoch");
    m_accuracyAxisX->setLabelFormat("%d");
    m_accuracyAxisY = new QValueAxis();
    m_accuracyAxisY->setTitleText("Accuracy");
    m_accuracyAxisY->setLabelFormat("%.2f");
    m_accuracyAxisY->setRange(0.0, 1.0);
    
    m_accuracyChart->addAxis(m_accuracyAxisX, Qt::AlignBottom);
    m_accuracyChart->addAxis(m_accuracyAxisY, Qt::AlignLeft);
    m_accuracySeries->attachAxis(m_accuracyAxisX);
    m_accuracySeries->attachAxis(m_accuracyAxisY);
    
    m_lossChartView->setChart(m_lossChart);
    m_accuracyChartView->setChart(m_accuracyChart);
}

void MetricsWidget::updateMetrics(double loss, double accuracy, int epoch)
{
    m_lossSeries->append(epoch, loss);
    m_accuracySeries->append(epoch, accuracy);
    
    if (m_lossSeries->count() > m_maxDataPoints) {
        m_lossSeries->removePoints(0, m_lossSeries->count() - m_maxDataPoints);
    }
    if (m_accuracySeries->count() > m_maxDataPoints) {
        m_accuracySeries->removePoints(0, m_accuracySeries->count() - m_maxDataPoints);
    }
    
    if (loss < m_bestLoss) {
        m_bestLoss = loss;
    }
    if (accuracy > m_bestAccuracy) {
        m_bestAccuracy = accuracy;
    }
    
    m_currentEpochLabel->setText(QString("Epoch: %1").arg(epoch));
    m_currentLossLabel->setText(QString("Loss: %1").arg(loss, 0, 'f', 4));
    m_currentAccuracyLabel->setText(QString("Accuracy: %1%").arg(accuracy * 100, 0, 'f', 2));
    m_bestLossLabel->setText(QString("Best Loss: %1").arg(m_bestLoss, 0, 'f', 4));
    m_bestAccuracyLabel->setText(QString("Best Accuracy: %1%").arg(m_bestAccuracy * 100, 0, 'f', 2));
    
    if (m_lossSeries->count() > 0) {
        double minLoss = std::numeric_limits<double>::max();
        double maxLoss = std::numeric_limits<double>::lowest();
        for (const QPointF& point : m_lossSeries->points()) {
            minLoss = qMin(minLoss, point.y());
            maxLoss = qMax(maxLoss, point.y());
        }
        m_lossAxisY->setRange(minLoss * 0.9, maxLoss * 1.1);
        m_lossAxisX->setRange(0, epoch);
    }
    
    if (m_accuracySeries->count() > 0) {
        m_accuracyAxisX->setRange(0, epoch);
    }
}

void MetricsWidget::clearMetrics()
{
    m_lossSeries->clear();
    m_accuracySeries->clear();
    m_bestLoss = std::numeric_limits<double>::max();
    m_bestAccuracy = 0.0;
    
    m_currentEpochLabel->setText("Epoch: 0");
    m_currentLossLabel->setText("Loss: 0.0000");
    m_currentAccuracyLabel->setText("Accuracy: 0.00%");
    m_bestLossLabel->setText("Best Loss: N/A");
    m_bestAccuracyLabel->setText("Best Accuracy: N/A");
}

