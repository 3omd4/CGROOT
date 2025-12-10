from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                             QTabWidget, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QCheckBox, QPushButton, QScrollArea, QLabel, QMessageBox, QGroupBox, QFileDialog, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal

class ConfigurationWidget(QWidget):
    parametersChanged = pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        self.tab_widget = QTabWidget()
        
        self.setup_model_tab()
        self.setup_training_tab()
        self.setup_network_tab()
        
        main_layout.addWidget(self.tab_widget)
        
        # Bottom Buttons
        button_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset to Defaults")
        self.load_btn = QPushButton("Load Config")
        self.save_btn = QPushButton("Save Config")
        
        self.reset_btn.clicked.connect(self.on_reset_defaults)
        self.load_btn.clicked.connect(self.on_load_config)
        self.save_btn.clicked.connect(self.on_save_config)
        
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)

    def setup_model_tab(self):
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)
        
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 1000)
        self.num_classes.setValue(10)
        
        self.image_width = QSpinBox()
        self.image_width.setRange(1, 1000)
        self.image_width.setValue(28)
        
        self.image_height = QSpinBox()
        self.image_height.setRange(1, 1000)
        self.image_height.setValue(28)
        
        self.num_layers = QSpinBox()
        self.num_layers.setRange(1, 100)
        self.num_layers.setValue(3)
        
        layout.addRow("Number of Classes:", self.num_classes)
        layout.addRow("Image Width:", self.image_width)
        layout.addRow("Image Height:", self.image_height)
        layout.addRow("Number of Layers:", self.num_layers)
        
        # Connect signals
        for w in [self.num_classes, self.image_width, self.image_height, self.num_layers]:
            w.valueChanged.connect(self.on_parameter_changed)
            
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        self.tab_widget.addTab(scroll, "Model")

    def setup_training_tab(self):
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)
        
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["SGD", "Adam", "RMSprop"])
        
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.00001, 10.0)
        self.learning_rate.setValue(0.001)
        self.learning_rate.setDecimals(5)
        
        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setRange(0.0, 1.0)
        self.weight_decay.setValue(0.0001)
        self.weight_decay.setDecimals(5)
        
        self.momentum = QDoubleSpinBox()
        self.momentum.setRange(0.0, 1.0)
        self.momentum.setValue(0.9)
        self.momentum.setDecimals(3)
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(10)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 10000)
        self.batch_size.setValue(32)
        
        self.use_validation = QCheckBox()
        self.use_validation.setChecked(True)
        
        self.validation_split = QDoubleSpinBox()
        self.validation_split.setRange(0.0, 0.5)
        self.validation_split.setValue(0.2)
        
        layout.addRow("Optimizer:", self.optimizer_combo)
        layout.addRow("Learning Rate:", self.learning_rate)
        layout.addRow("Weight Decay:", self.weight_decay)
        layout.addRow("Momentum:", self.momentum)
        layout.addRow("Epochs:", self.epochs)
        layout.addRow("Batch Size:", self.batch_size)
        layout.addRow("Use Validation Set:", self.use_validation)
        layout.addRow("Validation Split:", self.validation_split)
        
        # Connect signals
        self.optimizer_combo.currentIndexChanged.connect(self.on_parameter_changed)
        self.use_validation.toggled.connect(self.on_parameter_changed)
        for w in [self.learning_rate, self.weight_decay, self.momentum, 
                  self.epochs, self.batch_size, self.validation_split]:
            w.valueChanged.connect(self.on_parameter_changed)

        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        self.tab_widget.addTab(scroll, "Training")

    def setup_network_tab(self):
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Conv Layers Config (Currently limited/disabled as per user request/core limit)
        self.num_conv_layers = QSpinBox()
        self.num_conv_layers.setRange(0, 5)
        self.num_conv_layers.setValue(0) # Default to 0
        layout.addRow("Number of Conv Layers:", self.num_conv_layers)
        
        # FC Layers Config
        self.num_fc_layers = QSpinBox()
        self.num_fc_layers.setRange(1, 10)
        self.num_fc_layers.setValue(2)
        layout.addRow("Number of FC Layers:", self.num_fc_layers)
        
        self.neurons_fc_input = QLineEdit("128, 10")
        self.neurons_fc_input.setPlaceholderText("comma separated, e.g. 128, 64, 10")
        layout.addRow("Neurons per FC Layer:", self.neurons_fc_input)
        
        info_label = QLabel("Note: Ensure the last FC layer size matches the Number of Classes.")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addRow(info_label)
        
        # Connect signals
        self.num_conv_layers.valueChanged.connect(self.on_parameter_changed)
        self.num_fc_layers.valueChanged.connect(self.on_parameter_changed)
        self.neurons_fc_input.textChanged.connect(self.on_parameter_changed)
        
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        self.tab_widget.addTab(scroll, "Network Architecture")

    def get_architecture_parameters(self):
        # Parse neurons list
        try:
            neurons_str = self.neurons_fc_input.text()
            neurons = [int(x.strip()) for x in neurons_str.split(',') if x.strip()]
        except ValueError:
            neurons = [128, 10]
            
        return {
            'num_classes': self.num_classes.value(),
            'image_width': self.image_width.value(),
            'image_height': self.image_height.value(),
            'num_conv_layers': self.num_conv_layers.value(),
            'num_fc_layers': self.num_fc_layers.value(),
            'neurons_per_fc_layer': neurons
        }

    def on_parameter_changed(self):
        self.parametersChanged.emit()

    def on_reset_defaults(self):
        self.num_classes.setValue(10)
        self.image_width.setValue(28)
        self.image_height.setValue(28)
        self.num_layers.setValue(3)
        
        self.optimizer_combo.setCurrentIndex(0) # SGD
        self.learning_rate.setValue(0.001)
        self.weight_decay.setValue(0.0001)
        self.momentum.setValue(0.9)
        self.epochs.setValue(10)
        self.batch_size.setValue(32)
        self.use_validation.setChecked(True)
        self.validation_split.setValue(0.2)
        
        self.on_parameter_changed()

    def on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "Config Files (*.json *.cfg);;All Files (*.*)")
        if path:
            QMessageBox.information(self, "Load Config", "Configuration loading will be implemented with JSON parsing.")

    def on_save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "Config Files (*.json);;All Files (*.*)")
        if path:
            QMessageBox.information(self, "Save Config", "Configuration saving will be implemented with JSON serialization.")

    # Getters for controller use
    def get_training_parameters(self):
        return {
            'epochs': self.epochs.value(),
            'batch_size': self.batch_size.value(),
            'learning_rate': self.learning_rate.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'validation_split': self.validation_split.value()
        }
