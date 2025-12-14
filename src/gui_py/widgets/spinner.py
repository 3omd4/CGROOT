from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QPainter, QColor, QPen

class SpinnerWidget(QWidget):
    def __init__(self, parent=None, size=20):
        super().__init__(parent)
        self.setFixedSize(size, size)
        
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.is_spinning = False
        self.color = QColor("#3498db") # Blue-ish
        
    def rotate(self):
        self.angle = (self.angle + 30) % 360
        self.update()
        
    def start(self):
        self.is_spinning = True
        self.timer.start(50)
        self.show()
        
    def stop(self):
        self.is_spinning = False
        self.timer.stop()
        self.hide()
        
    def paintEvent(self, event):
        if not self.is_spinning:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        pen = QPen(self.color)
        pen.setWidth(3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        
        # Draw rotating arc
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.drawArc(rect, -self.angle * 16, 270 * 16) # negative for clockwise
        
