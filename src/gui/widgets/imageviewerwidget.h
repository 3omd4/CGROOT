#ifndef IMAGEVIEWERWIDGET_H
#define IMAGEVIEWERWIDGET_H

#include <QWidget>
#include <QPaintEvent>
#include <QPainter>
#include <QImage>
#include <QPixmap>
#include <QMouseEvent>
#include <QWheelEvent>

class ImageViewerWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ImageViewerWidget(QWidget *parent = nullptr);
    
    void displayImage(const QImage& image);
    void displayImage(const QPixmap& pixmap);
    void clearImage();
    
    void setZoomFactor(double factor);
    double zoomFactor() const { return m_zoomFactor; }

protected:
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QPixmap m_pixmap;
    double m_zoomFactor;
    QPoint m_panStart;
    QPoint m_panOffset;
    bool m_isPanning;
    
    void updateDisplay();
    QRect getImageRect() const;
};

#endif // IMAGEVIEWERWIDGET_H

