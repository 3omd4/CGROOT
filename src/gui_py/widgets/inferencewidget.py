from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QGroupBox, QListWidget, QListWidgetItem,
                             QFormLayout, QSpinBox, QProgressBar, QFileDialog)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage
from .imageviewerwidget import ImageViewerWidget

class InferenceWidget(QWidget):
    onInferenceStarted = pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.dataset_type = "generic"
        self.current_image = None
        self.init_ui()
        
    def set_dataset_type(self, dataset_type):
        self.dataset_type = dataset_type
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Left Side: Image
        image_group = QGroupBox("Input Image")
        image_layout = QVBoxLayout(image_group)
        
        self.image_viewer = ImageViewerWidget()
        self.image_viewer.setMinimumSize(300, 300)
        image_layout.addWidget(self.image_viewer)
        
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_btn.clicked.connect(self.on_load_image)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addStretch()
        image_layout.addLayout(btn_layout)
        
        main_layout.addWidget(image_group, 1)
        
        # Right Side: Results
        results_group = QGroupBox("Inference Results")
        results_layout = QVBoxLayout(results_group)
        
        self.pred_label = QLabel("Predicted: -")
        self.pred_label.setStyleSheet("QLabel { font-size: 18pt; font-weight: bold; padding: 10px; }")
        results_layout.addWidget(self.pred_label)
        
        self.conf_label = QLabel("Confidence: -")
        self.conf_label.setStyleSheet("QLabel { font-size: 14pt; padding: 5px; }")
        results_layout.addWidget(self.conf_label)
        
        results_layout.addWidget(QLabel("Class Probabilities:"))
        self.prob_list = QListWidget()
        self.prob_list.setMaximumHeight(300)
        results_layout.addWidget(self.prob_list)
        
        # Inference Settings
        inf_group = QGroupBox("Inference Settings")
        inf_layout = QFormLayout(inf_group)
        
        self.num_samples = QSpinBox()
        self.num_samples.setRange(1, 10000)
        self.num_samples.setValue(100)
        inf_layout.addRow("Number of Samples:", self.num_samples)
        
        self.run_btn = QPushButton("Run Inference")
        self.run_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.clicked.connect(self.on_run_inference)
        inf_layout.addRow(self.run_btn)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        inf_layout.addRow("Progress:", self.progress)
        
        results_layout.addWidget(inf_group)
        results_layout.addStretch()
        
        main_layout.addWidget(results_group, 1)

    def on_load_image(self):
        # Start at src/data/samples if exists
        import os
        from pathlib import Path
        
        # We assume we are in src/gui_py/widgets/
        # Project root is ../../../
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent
        samples_dir = project_root / "src" / "data" / "samples"
        
        start_dir = str(samples_dir) if samples_dir.exists() else ""
        
        path, _ = QFileDialog.getOpenFileName(self, "Load Fashion-MNIST/MNIST Image", start_dir, 
                                            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)")
        if path:
            image = QImage(path)
            if not image.isNull():
                self.current_image = image
                self.image_viewer.displayImage(image)

    def on_run_inference(self):
        if not self.current_image:
           return
           
        self.onInferenceStarted.emit()
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0) # Indeterminate
        
        # Request controller to run inference with image
        self.controller.requestInference.emit(self.current_image)

    def displayPrediction(self, predicted_class, image, probabilities):
        if not image.isNull():
            self.image_viewer.displayImage(image)
            
        from dataset_utils import get_class_name
        
        # We default to generic or update if we know dataset
        name = get_class_name(self.dataset_type, predicted_class)
        self.pred_label.setText(f"Predicted: {name}")
            
        if probabilities and predicted_class < len(probabilities):
            conf = probabilities[predicted_class]
            self.conf_label.setText(f"Confidence: {conf*100:.2f}%")
                
        self.update_probabilities(probabilities)
        self.inference_finished() # Hide progress

    def update_probabilities(self, probabilities):
        self.prob_list.clear()
        if not probabilities: return
        
        from dataset_utils import get_class_name
        
        # Pair index with prob
        indexed_probs = [(i, p) for i, p in enumerate(probabilities)]
        # Sort desc by prob
        indexed_probs.sort(key=lambda x: x[1], reverse=True)
        
        for idx, prob in indexed_probs:
            name = get_class_name(self.dataset_type, idx) # Improve detection later
            text = f"{name}: {prob*100:.2f}%"
            item = QListWidgetItem(text)
            self.prob_list.addItem(item)

    def inference_finished(self):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
