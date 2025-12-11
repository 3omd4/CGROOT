from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QProgressBar, QLabel, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal

class TrainingWidget(QWidget):
    startTrainingRequested = pyqtSignal()
    stopTrainingRequested = pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Params Group removed (moved to Configuration Tab)


        # Preview Group
        preview_group = QGroupBox("Training Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.image_label = QLabel("Waiting for training data...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(100, 100)
        self.image_label.setStyleSheet("QLabel { background-color: #222; border: 1px solid #444; }")
        preview_layout.addWidget(self.image_label)
        main_layout.addWidget(preview_group)
        
        # Button Layout
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()
        
        main_layout.addLayout(btn_layout)
        
        # Status Label
        self.status_label = QLabel("Ready to start training")
        self.status_label.setStyleSheet("QLabel { font-weight: bold; padding: 5px; }")
        main_layout.addWidget(self.status_label)
        
        main_layout.addStretch()
        
    def connect_signals(self):
        # Allow external control/updates via controller signals if needed
        pass
        
    def on_start_clicked(self):
        # We emit signal, main window orchestrates call to controller
        self.startTrainingRequested.emit()
        self.set_training_state(True)
        
    def on_stop_clicked(self):
        self.stopTrainingRequested.emit()
        self.set_training_state(False)
        
    def set_training_state(self, is_running):
        self.start_btn.setEnabled(not is_running)
        self.stop_btn.setEnabled(is_running)
        if is_running:
            self.status_label.setText("Training in progress...")
            self.status_label.setStyleSheet("QLabel { font-weight: bold; color: green; padding: 5px; }")
        else:
            self.status_label.setText("Training stopped")
            self.status_label.setStyleSheet("QLabel { font-weight: bold; color: red; padding: 5px; }")
            
    # Methods for controller to update UI status
    def training_finished(self):
        self.set_training_state(False)
        self.status_label.setText("Training Completed")

    def display_image(self, predicted_class, q_img, probs):
        if q_img:
            # Scale up for visibility
            pixmap = q_img.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
            from PyQt6.QtGui import QPixmap
            self.image_label.setPixmap(QPixmap.fromImage(pixmap))
            self.image_label.setText("") # Clear text
        
        if probs:
            # Optional: Show confidence of current sample (passed as probs)
            pass
