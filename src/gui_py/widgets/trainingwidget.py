from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QProgressBar, QLabel, QGroupBox, QGridLayout, 
                             QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, 
                             QLineEdit, QFileDialog)
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
        
        # Params Group
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout(params_group)
        
        # Row 0
        params_layout.addWidget(QLabel("Number of Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(10)
        params_layout.addWidget(self.epochs_spin, 0, 1)
        
        # Row 1
        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10000)
        self.batch_size_spin.setValue(32)
        params_layout.addWidget(self.batch_size_spin, 1, 1)
        
        # Row 2
        params_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 10.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        params_layout.addWidget(self.lr_spin, 2, 1)
        
        # Row 3
        params_layout.addWidget(QLabel("Optimizer:"), 3, 0)
        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["SGD", "Adam", "RMSprop"])
        params_layout.addWidget(self.opt_combo, 3, 1)
        
        # Row 4
        params_layout.addWidget(QLabel("Validation Split:"), 4, 0)
        self.val_split_spin = QSpinBox()
        self.val_split_spin.setRange(0, 50)
        self.val_split_spin.setValue(20)
        self.val_split_spin.setSuffix("%")
        params_layout.addWidget(self.val_split_spin, 4, 1)
        
        # Row 5
        self.use_val_check = QCheckBox("Use Validation Set")
        self.use_val_check.setChecked(True)
        params_layout.addWidget(self.use_val_check, 5, 0, 1, 2)
        
        # Row 6
        self.save_chk_check = QCheckBox("Save Checkpoints")
        params_layout.addWidget(self.save_chk_check, 6, 0)
        
        # Row 7 Checkpoint Path
        params_layout.addWidget(QLabel("Checkpoint Path:"), 7, 0)
        self.ckpt_path_edit = QLineEdit()
        self.ckpt_path_edit.setPlaceholderText("Select checkpoint directory...")
        params_layout.addWidget(self.ckpt_path_edit, 7, 1)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_btn.clicked.connect(self.on_browse_ckpt)
        params_layout.addWidget(self.browse_btn, 7, 2)
        
        main_layout.addWidget(params_group)
        
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
        
    def on_browse_ckpt(self):
        d = QFileDialog.getExistingDirectory(self, "Select Checkpoint Directory")
        if d:
            self.ckpt_path_edit.setText(d)
            
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
