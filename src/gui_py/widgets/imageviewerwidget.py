from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

class ImageViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = QLabel("No Image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setStyleSheet("border: 2px dashed #666;")
        
        layout.addWidget(self.image_label)
        
    def displayImage(self, image: QImage):
        if image.isNull():
            self.image_label.setText("No Image")
            self.image_label.setPixmap(QPixmap())
        else:
            # Scale pixmap to fit label while keeping aspect ratio
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
            
    def resizeEvent(self, event):
        # Re-scale image on resize if pixmap exists
        if self.image_label.pixmap() and not self.image_label.pixmap().isNull():
            # We need to store original pixmap to avoid quality loss on multiple resizes
            # For simplicity, we assume caller or some cache might refetch, 
            # or we just rely on the label's existing pixmap which might degrade.
            # Ideally store m_originalImage.
            pass
        super().resizeEvent(event)
