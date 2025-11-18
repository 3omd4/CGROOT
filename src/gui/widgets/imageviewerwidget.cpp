#include "imageviewerwidget.h"
#include <QWheelEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QScrollBar>

ImageViewerWidget::ImageViewerWidget(QWidget *parent)
    : QWidget(parent)
    , m_zoomFactor(1.0)
    , m_isPanning(false)
{
    setMinimumSize(200, 200);
    setMouseTracking(true);
    setAttribute(Qt::WA_OpaquePaintEvent);
}

void ImageViewerWidget::displayImage(const QImage& image)
{
    m_pixmap = QPixmap::fromImage(image);
    updateDisplay();
    update();
}

void ImageViewerWidget::displayImage(const QPixmap& pixmap)
{
    m_pixmap = pixmap;
    updateDisplay();
    update();
}

void ImageViewerWidget::clearImage()
{
    m_pixmap = QPixmap();
    m_zoomFactor = 1.0;
    m_panOffset = QPoint(0, 0);
    update();
}

void ImageViewerWidget::setZoomFactor(double factor)
{
    m_zoomFactor = qMax(0.1, qMin(10.0, factor));
    updateDisplay();
    update();
}

void ImageViewerWidget::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor(240, 240, 240));
    
    if (m_pixmap.isNull()) {
        painter.setPen(Qt::gray);
        painter.drawText(rect(), Qt::AlignCenter, "No image loaded");
        return;
    }
    
    QRect imageRect = getImageRect();
    painter.drawPixmap(imageRect, m_pixmap);
    
    painter.setPen(Qt::black);
    painter.drawRect(imageRect);
}

QRect ImageViewerWidget::getImageRect() const
{
    if (m_pixmap.isNull()) {
        return QRect();
    }
    
    QSize scaledSize = m_pixmap.size() * m_zoomFactor;
    QRect widgetRect = rect();
    
    QPoint topLeft = widgetRect.center() - QPoint(scaledSize.width() / 2, scaledSize.height() / 2);
    topLeft += m_panOffset;
    
    return QRect(topLeft, scaledSize);
}

void ImageViewerWidget::wheelEvent(QWheelEvent* event)
{
    double delta = event->angleDelta().y() / 120.0;
    double zoomStep = 0.1;
    double newZoom = m_zoomFactor + (delta * zoomStep);
    setZoomFactor(newZoom);
}

void ImageViewerWidget::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_isPanning = true;
        m_panStart = event->pos();
    }
}

void ImageViewerWidget::mouseMoveEvent(QMouseEvent* event)
{
    if (m_isPanning) {
        QPoint delta = event->pos() - m_panStart;
        m_panOffset += delta;
        m_panStart = event->pos();
        update();
    }
}

void ImageViewerWidget::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_isPanning = false;
    }
}

void ImageViewerWidget::updateDisplay()
{
    // Could add additional display updates here
}

