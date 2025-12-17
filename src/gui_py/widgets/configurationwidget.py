from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                             QTabWidget, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QCheckBox, QPushButton, QScrollArea, QLabel, QMessageBox, QGroupBox, QFileDialog, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal
import logging
import os
import json
import datetime

class ConfigurationWidget(QWidget):
    parametersChanged = pyqtSignal()
    vizSettingsChanged = pyqtSignal(dict) # settings_dict

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
        self.setup_gui_tab()
        
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
        
        # Ensure consistent defaults on startup
        self.on_reset_defaults()

    def setup_model_tab(self):
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)
        
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 1000)
        self.num_classes.setValue(10)
        self.num_classes.setToolTip(
            "Number of output categories your model will predict.\n"
            "Examples: 10 (MNIST digits 0-9), 2 (binary classification), 1000 (ImageNet).\n"
            "Must match the size of your final output layer."
        )
        
        self.image_width = QSpinBox()
        self.image_width.setRange(1, 1000)
        self.image_width.setValue(28)
        self.image_width.setToolTip(
            "Width of input images in pixels.\n"
            "Common sizes: 28 (MNIST), 32 (CIFAR), 224 (ImageNet), 299 (Inception).\n"
            "All training images will be resized to this dimension."
        )
        
        self.image_height = QSpinBox()
        self.image_height.setRange(1, 1000)
        self.image_height.setValue(28)
        self.image_height.setToolTip(
            "Height of input images in pixels.\n"
            "Common sizes: 28 (MNIST), 32 (CIFAR), 224 (ImageNet), 299 (Inception).\n"
            "All training images will be resized to this dimension."
        )

        self.image_depth = QSpinBox()
        self.image_depth.setRange(1, 100)
        self.image_depth.setValue(1)
        self.image_depth.setToolTip(
            "Number of color channels.\n"
            "1 = Grayscale (MNIST)\n"
            "3 = RGB (CIFAR-10, ImageNet)"
        )
        
        layout.addRow("Number of Classes:", self.num_classes)
        layout.addRow("Image Width:", self.image_width)
        layout.addRow("Image Height:", self.image_height)
        layout.addRow("Image Depth:", self.image_depth)
        
        # Connect signals
        for w in [self.num_classes, self.image_width, self.image_height, self.image_depth]:
            w.valueChanged.connect(self.on_parameter_changed)
            
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        self.tab_widget.addTab(scroll, "Model")

    def setup_training_tab(self):
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)
        
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["SGD", "SGD with Momentum", "Adam", "RMSprop"])
        self.optimizer_combo.setCurrentIndex(2) # Default to Adam
        self.optimizer_combo.setToolTip(
            "Optimization algorithm for training:\n"
            "• SGD: Simple, stable, requires careful tuning\n"
            "• SGD with Momentum: Faster than SGD, reduces oscillations\n"
            "• Adam: Best default choice, adaptive learning rates (recommended)\n"
            "• RMSprop: Good for RNNs and non-stationary problems"
        )
        
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(5)
        self.learning_rate.setRange(0.0, 10.0)
        self.learning_rate.setValue(0.001)
        self.learning_rate.setToolTip(
            "Controls how much weights change per update (step size).\n\n"
            "Typical values:\n"
            "• SGD/Momentum: 0.01 - 0.1\n"
            "• Adam/RMSprop: 0.0001 - 0.001\n\n"
            "Too high → Training diverges or oscillates wildly\n"
            "Too low → Training is extremely slow\n"
            "Start with defaults and adjust if needed."
        )
        
        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setDecimals(5)
        self.weight_decay.setRange(0.0, 1.0)
        self.weight_decay.setValue(0.01)
        self.weight_decay.setToolTip(
            "L2 regularization - penalizes large weights to prevent overfitting.\n\n"
            "Typical values: 0.0001 - 0.01\n"
            "Set to 0 to disable regularization.\n\n"
            "Use when: Validation loss increases while training loss decreases\n"
            "(indicates overfitting)."
        )
        
        self.momentum = QDoubleSpinBox()
        self.momentum.setDecimals(3)
        self.momentum.setRange(0.0, 1.0)
        self.momentum.setValue(0.9)
        self.momentum.setToolTip(
            "Momentum coefficient for SGD with Momentum optimizer.\n\n"
            "Accelerates convergence in relevant directions and dampens oscillations.\n"
            "Think of it as a 'ball rolling downhill' - builds up speed.\n\n"
            "Standard value: 0.9\n"
            "Higher (0.95-0.99): More smoothing, may overshoot\n"
            "Lower (0.5-0.8): Less smoothing, more responsive"
        )
        
        self.beta1 = QDoubleSpinBox()
        self.beta1.setDecimals(3)
        self.beta1.setRange(0.0, 1.0)
        self.beta1.setValue(0.9)
        self.beta1.setToolTip(
            "Beta1 - Exponential decay rate for first moment (mean) in Adam.\n\n"
            "Controls the momentum-like behavior in Adam.\n"
            "Standard value: 0.9 (rarely needs changing)\n\n"
            "Higher values → More weight to past gradients\n"
            "Lower values → More responsive to recent gradients"
        )
        
        self.beta2 = QDoubleSpinBox()
        self.beta2.setDecimals(4)  # Set decimals FIRST
        self.beta2.setRange(0.0, 1.0)
        self.beta2.setSingleStep(0.001)
        self.beta2.setValue(0.999)
        self.beta2.setToolTip(
            "Beta2 - Exponential decay rate for second moment (variance) in Adam.\n\n"
            "Controls the adaptive learning rate behavior.\n"
            "Standard value: 0.999 (rarely needs changing)\n\n"
            "For sparse gradients, try 0.99 or lower.\n"
            "Keep close to 1.0 for most problems."
        )
        
        self.beta = QDoubleSpinBox()
        self.beta.setDecimals(3)  # Set decimals FIRST
        self.beta.setRange(0.0, 1.0)
        self.beta.setSingleStep(0.01)
        self.beta.setValue(0.9)
        self.beta.setToolTip(
            "Beta - Decay rate for moving average in RMSprop.\n\n"
            "Controls how much history is used for adaptive learning rates.\n"
            "Standard value: 0.9\n\n"
            "Higher (0.95-0.99): More smoothing, slower adaptation\n"
            "Lower (0.8-0.85): Less smoothing, faster adaptation"
        )
        
        self.epsilon = QDoubleSpinBox()
        self.epsilon.setDecimals(10)  # Set decimals FIRST for scientific notation
        self.epsilon.setRange(0.0, 1e-5)
        self.epsilon.setSingleStep(1e-9)
        self.epsilon.setValue(1e-8)
        self.epsilon.setToolTip(
            "Epsilon - Small constant for numerical stability.\n\n"
            "Prevents division by zero in Adam and RMSprop.\n"
            "Standard value: 1e-8 (almost never needs changing)\n\n"
            "Only adjust if you encounter numerical instability.\n"
            "Larger values (1e-7) → More stable but less precise\n"
            "Smaller values (1e-9) → More precise but may be unstable"
        )
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(5)
        self.epochs.setToolTip(
            "Number of complete passes through the entire training dataset.\n\n"
            "Each epoch processes all training samples once.\n"
            "Typical values: 10-100 depending on dataset size\n\n"
            "More epochs → Better learning (up to a point)\n"
            "Too many → Overfitting (model memorizes training data)\n\n"
            "Monitor validation loss to find optimal number."
        )
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 10000)
        self.batch_size.setValue(64)
        self.batch_size.setToolTip(
            "Number of samples processed before updating weights.\n\n"
            "Common values: 32, 64, 128, 256\n\n"
            "Larger batches (128-256):\n"
            "  + More stable gradients\n"
            "  + Better GPU utilization\n"
            "  - More memory required\n\n"
            "Smaller batches (16-32):\n"
            "  + Less memory\n"
            "  + May generalize better\n"
            "  - Noisier gradients"
        )
        
        self.use_validation = QCheckBox()
        self.use_validation.setChecked(False)
        self.use_validation.setToolTip(
            "Enable validation set to monitor overfitting.\n\n"
            "When enabled, splits training data into:\n"
            "  • Training set (for learning)\n"
            "  • Validation set (for monitoring)\n\n"
            "Validation loss increasing while training loss decreases\n"
            "indicates overfitting - stop training or add regularization."
        )
        
        self.validation_split = QDoubleSpinBox()
        self.validation_split.setRange(0.0, 0.5)
        self.validation_split.setValue(0.0)
        self.validation_split.setToolTip(
            "Fraction of training data reserved for validation.\n\n"
            "Common values: 0.1 (10%), 0.2 (20%), 0.3 (30%)\n\n"
            "Example: With 1000 samples and 0.2 split:\n"
            "  • 800 samples for training\n"
            "  • 200 samples for validation\n\n"
            "Larger splits → More reliable validation, less training data\n"
            "Smaller splits → More training data, less reliable validation"
        )
        
        layout.addRow("Optimizer:", self.optimizer_combo)
        layout.addRow("Learning Rate:", self.learning_rate)
        layout.addRow("Weight Decay:", self.weight_decay)
        
        # Optimizer-specific parameters (visibility controlled dynamically)
        self.momentum_row = layout.rowCount()
        layout.addRow("Momentum:", self.momentum)
        self.beta1_row = layout.rowCount()
        layout.addRow("Beta1:", self.beta1)
        self.beta2_row = layout.rowCount()
        layout.addRow("Beta2:", self.beta2)
        self.beta_row = layout.rowCount()
        layout.addRow("Beta (RMSprop):", self.beta)
        self.epsilon_row = layout.rowCount()
        layout.addRow("Epsilon:", self.epsilon)
        
        layout.addRow("Epochs:", self.epochs)
        layout.addRow("Batch Size:", self.batch_size)
        layout.addRow("Use Validation Set:", self.use_validation)
        layout.addRow("Validation Split:", self.validation_split)
        
        # Connect signals
        self.optimizer_combo.currentTextChanged.connect(self.on_optimizer_changed)
        self.optimizer_combo.currentIndexChanged.connect(self.on_parameter_changed)
        self.use_validation.toggled.connect(self.on_parameter_changed)
        for w in [self.learning_rate, self.weight_decay, self.momentum, 
                  self.beta1, self.beta2, self.beta, self.epsilon,
                  self.epochs, self.batch_size, self.validation_split]:
            w.valueChanged.connect(self.on_parameter_changed)

        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        self.tab_widget.addTab(scroll, "Training")
        
        # Store reference to layout for dynamic parameter visibility
        self.training_layout = layout
        
        # Initialize parameter visibility (must be after tab is added)
        # Initialize parameter visibility (must be after tab is added)
        self.on_optimizer_changed(self.optimizer_combo.currentText())
        


    def setup_network_tab(self):
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Conv Layers Config
        self.num_conv_layers = QSpinBox()
        self.num_conv_layers.setRange(0, 10)
        self.num_conv_layers.setValue(2) # Default to 2
        self.num_conv_layers.setToolTip("Number of Convolutional layers.")
        layout.addRow("Number of Conv Layers:", self.num_conv_layers)

        self.kernels_per_layer = QLineEdit("32, 64")
        self.kernels_per_layer.setPlaceholderText("e.g. 32, 64")
        self.kernels_per_layer.setToolTip("Comma separated number of kernels for each Conv layer.")
        layout.addRow("Kernels per Layer:", self.kernels_per_layer)

        self.kernel_dims = QLineEdit("3x3, 3x3")
        self.kernel_dims.setPlaceholderText("e.g. 3x3, 3x3")
        self.kernel_dims.setToolTip("Comma separated dimensions (HxW) for each Conv layer. e.g. '5x5', or '5x5, 3x3'.")
        layout.addRow("Kernel Dimensions:", self.kernel_dims)

        # Pooling Config
        self.pooling_type = QComboBox()
        self.pooling_type.addItems(["Max", "Average"])
        self.pooling_type.setCurrentIndex(0) # Default to Max
        layout.addRow("Pooling Type:", self.pooling_type)

        self.pooling_intervals = QLineEdit("1, 1")
        self.pooling_intervals.setPlaceholderText("e.g. 1, 1")
        self.pooling_intervals.setToolTip("Number of Conv layers before each Pooling layer. e.g. '1' means every conv.")
        layout.addRow("Pooling Intervals:", self.pooling_intervals)
        
        # FC Layers Config
        self.num_fc_layers = QSpinBox()
        self.num_fc_layers.setRange(1, 10)
        self.num_fc_layers.setValue(2)
        layout.addRow("Number of FC Layers:", self.num_fc_layers)
        
        self.neurons_fc_input = QLineEdit("256, 10")
        self.neurons_fc_input.setPlaceholderText("comma separated, e.g. 256, 128, 10")
        layout.addRow("Neurons per FC Layer:", self.neurons_fc_input)
        
        info_label = QLabel("Note: Ensure the last FC layer size matches the Number of Classes.")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addRow(info_label)
        
        # Connect signals
        self.num_conv_layers.valueChanged.connect(self.on_parameter_changed)
        self.num_fc_layers.valueChanged.connect(self.on_parameter_changed)
        self.neurons_fc_input.textChanged.connect(self.on_parameter_changed)
        self.kernels_per_layer.textChanged.connect(self.on_parameter_changed)
        self.kernel_dims.textChanged.connect(self.on_parameter_changed)
        self.pooling_type.currentIndexChanged.connect(self.on_parameter_changed)
        self.pooling_intervals.textChanged.connect(self.on_parameter_changed)
        
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setWidgetResizable(True)
        self.tab_widget.addTab(scroll, "Network Architecture")

    def setup_gui_tab(self):
        panel = QWidget()
        layout = QFormLayout(panel)
        
        # Training Preview
        self.show_preview_cb = QCheckBox("Show Training Preview")
        self.show_preview_cb.setChecked(True)
        self.show_preview_cb.setToolTip("Toggle displaying proper training samples during training (affects performance)")
        
        # Feature Maps
        self.fm_freq_combo = QComboBox()
        self.fm_freq_combo.addItems(["Every Epoch", "Every Sample", "Never"])
        self.fm_freq_combo.setCurrentIndex(1) # Default: Every Sample
        self.fm_freq_combo.setToolTip("Control how often feature maps are updated")
        
        # New Settings
        self.auto_scroll_cb = QCheckBox("Auto-Scroll Logs")
        self.auto_scroll_cb.setChecked(True)
        self.auto_scroll_cb.setToolTip("Automatically scroll to the bottom when new logs arrive")
        
        self.chart_anim_cb = QCheckBox("Chart Animations")
        self.chart_anim_cb.setChecked(False) # Default off for performance
        self.chart_anim_cb.setToolTip("Enable animated transitions for charts (can be CPU intensive)")
        
        layout.addRow("Training Preview:", self.show_preview_cb)
        layout.addRow("Feature Maps Update:", self.fm_freq_combo)
        layout.addRow("Auto-Scroll Logs:", self.auto_scroll_cb)
        layout.addRow("Chart Animations:", self.chart_anim_cb)
        
        self.show_preview_cb.toggled.connect(self.on_viz_setting_changed)
        self.fm_freq_combo.currentTextChanged.connect(self.on_viz_setting_changed)
        self.auto_scroll_cb.toggled.connect(self.on_viz_setting_changed)
        self.chart_anim_cb.toggled.connect(self.on_viz_setting_changed)
        
        self.tab_widget.addTab(panel, "GUI Settings")

    def get_architecture_parameters(self):
        # Parse neurons list
        try:
            neurons_str = self.neurons_fc_input.text()
            neurons = [int(x.strip()) for x in neurons_str.split(',') if x.strip()]
        except ValueError:
            neurons = [128, 10]
            logging.warning("Invalid neurons per FC layer. Using default: 128, 10")
        
        # Parse CNN params
        try:
            kernels_str = self.kernels_per_layer.text()
            kernels_list = [int(x.strip()) for x in kernels_str.split(',') if x.strip()]
        except ValueError:
            kernels_list = []

        # Parse Kernel Dims (e.g. "5x5, 3x3")
        kernel_dims_list = []
        dims_str = self.kernel_dims.text()
        parts = [p.strip() for p in dims_str.split(',') if p.strip()]
        for p in parts:
            if 'x' in p:
                d = p.split('x')
                if len(d) == 2:
                    try:
                        kernel_dims_list.append((int(d[0]), int(d[1])))
                    except  Exception:
                        pass
            
        # Parse Pooling Intervals
        try:
            intervals_str = self.pooling_intervals.text()
            intervals_list = [int(x.strip()) for x in intervals_str.split(',') if x.strip()]
        except ValueError:
            intervals_list = []
            
        return {
            'num_classes': self.num_classes.value(),
            'image_width': self.image_width.value(),
            'image_height': self.image_height.value(),
            'image_depth': self.image_depth.value(),
            'num_conv_layers': self.num_conv_layers.value(),
            'num_fc_layers': self.num_fc_layers.value(),
            'neurons_per_fc_layer': neurons,
            'kernels_per_layer': kernels_list,
            'kernel_dims': kernel_dims_list,
            'pooling_type': self.pooling_type.currentText(),
            'pooling_intervals': intervals_list
        }

    def on_parameter_changed(self):
        self.parametersChanged.emit()
    
    def on_optimizer_changed(self, optimizer_name):
        """Show/hide optimizer-specific parameters based on selection"""
        # Use stored layout reference
        layout = self.training_layout
        
        # Hide all optimizer-specific parameters first
        layout.itemAt(self.momentum_row, QFormLayout.ItemRole.LabelRole).widget().hide()
        layout.itemAt(self.momentum_row, QFormLayout.ItemRole.FieldRole).widget().hide()
        layout.itemAt(self.beta1_row, QFormLayout.ItemRole.LabelRole).widget().hide()
        layout.itemAt(self.beta1_row, QFormLayout.ItemRole.FieldRole).widget().hide()
        layout.itemAt(self.beta2_row, QFormLayout.ItemRole.LabelRole).widget().hide()
        layout.itemAt(self.beta2_row, QFormLayout.ItemRole.FieldRole).widget().hide()
        layout.itemAt(self.beta_row, QFormLayout.ItemRole.LabelRole).widget().hide()
        layout.itemAt(self.beta_row, QFormLayout.ItemRole.FieldRole).widget().hide()
        layout.itemAt(self.epsilon_row, QFormLayout.ItemRole.LabelRole).widget().hide()
        layout.itemAt(self.epsilon_row, QFormLayout.ItemRole.FieldRole).widget().hide()
        
        # Show relevant parameters based on optimizer
        if optimizer_name == "SGD with Momentum":
            layout.itemAt(self.momentum_row, QFormLayout.ItemRole.LabelRole).widget().show()
            layout.itemAt(self.momentum_row, QFormLayout.ItemRole.FieldRole).widget().show()
        elif optimizer_name == "Adam":
            layout.itemAt(self.beta1_row, QFormLayout.ItemRole.LabelRole).widget().show()
            layout.itemAt(self.beta1_row, QFormLayout.ItemRole.FieldRole).widget().show()
            layout.itemAt(self.beta2_row, QFormLayout.ItemRole.LabelRole).widget().show()
            layout.itemAt(self.beta2_row, QFormLayout.ItemRole.FieldRole).widget().show()
            layout.itemAt(self.epsilon_row, QFormLayout.ItemRole.LabelRole).widget().show()
            layout.itemAt(self.epsilon_row, QFormLayout.ItemRole.FieldRole).widget().show()
        elif optimizer_name == "RMSprop":
            layout.itemAt(self.beta_row, QFormLayout.ItemRole.LabelRole).widget().show()
            layout.itemAt(self.beta_row, QFormLayout.ItemRole.FieldRole).widget().show()
            layout.itemAt(self.epsilon_row, QFormLayout.ItemRole.LabelRole).widget().show()
            layout.itemAt(self.epsilon_row, QFormLayout.ItemRole.FieldRole).widget().show()
        # SGD shows no additional parameters (only learning_rate and weight_decay)

    def on_viz_setting_changed(self):
        settings = self.get_gui_settings()
        self.vizSettingsChanged.emit(settings)
        
    def get_gui_settings(self):
        return {
            'show_preview': self.show_preview_cb.isChecked(),
            'map_frequency': self.fm_freq_combo.currentText(),
            'auto_scroll': self.auto_scroll_cb.isChecked(),
            'chart_animations': self.chart_anim_cb.isChecked()
        }

    def on_reset_defaults(self):
        self.num_classes.setValue(10)
        self.image_width.setValue(28)
        self.image_height.setValue(28)
        self.image_depth.setValue(1)
        
        self.optimizer_combo.setCurrentIndex(2) # Adam
        self.learning_rate.setValue(0.001)
        self.weight_decay.setValue(1e-4) # Slight increase for regularization
        self.momentum.setValue(0.9)
        self.beta1.setValue(0.9)
        self.beta2.setValue(0.999)
        self.beta.setValue(0.9)
        self.epsilon.setValue(1e-8)
        self.epochs.setValue(10)
        self.batch_size.setValue(64)
        self.use_validation.setChecked(False)
        self.validation_split.setValue(0.0)
        
        # Architecture Defaults (Improved for general usage including CIFAR)
        self.num_conv_layers.setValue(2)
        self.kernels_per_layer.setText("32, 64")
        self.kernel_dims.setText("3x3, 3x3")
        self.pooling_type.setCurrentIndex(0) # Max
        self.pooling_intervals.setText("1, 1") # Pool after each
        
        self.num_fc_layers.setValue(2)
        self.neurons_fc_input.setText("256, 10") # Increased capacity


        
        self.on_parameter_changed()

    def on_load_config(self):
        # Ensure config directory exists
        config_dir = os.path.join(os.getcwd(), "src", "data", "configurations")
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", config_dir, "Config Files (*.json);;All Files (*.*)")
        if path:
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                
                self.load_parameters(config)
                QMessageBox.information(self, "Config Loaded", f"Configuration loaded from:\n{path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load config: {e}")

    def load_parameters(self, config):
        """Populate UI with values from config dictionary."""
        try:
            # Block signals to prevent multiple updates
            self.blockSignals(True)
            
            if 'num_classes' in config: self.num_classes.setValue(config['num_classes'])
            if 'image_width' in config: self.image_width.setValue(config['image_width'])
            if 'image_height' in config: self.image_height.setValue(config['image_height'])
            if 'image_depth' in config: self.image_depth.setValue(config['image_depth'])
            
            if 'num_conv_layers' in config: self.num_conv_layers.setValue(config['num_conv_layers'])
            if 'kernels_per_layer' in config: 
                val = config['kernels_per_layer']
                if isinstance(val, list): val = ", ".join(map(str, val))
                self.kernels_per_layer.setText(str(val))
            if 'kernel_dims' in config:
                # Convert list of lists/tuples back to string "HxW, HxW"
                val = config['kernel_dims']
                if isinstance(val, list):
                    strs = []
                    for item in val:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            strs.append(f"{item[0]}x{item[1]}")
                    self.kernel_dims.setText(", ".join(strs))
                else:
                    self.kernel_dims.setText(str(val))
                    
            if 'pooling_type' in config: self.pooling_type.setCurrentText(config['pooling_type'])
            if 'pooling_intervals' in config:
                val = config['pooling_intervals']
                if isinstance(val, list): val = ", ".join(map(str, val))
                self.pooling_intervals.setText(str(val))

            if 'num_fc_layers' in config: self.num_fc_layers.setValue(config['num_fc_layers'])
            if 'neurons_per_fc_layer' in config:
                val = config['neurons_per_fc_layer']
                if isinstance(val, list): val = ", ".join(map(str, val))
                self.neurons_fc_input.setText(str(val))

            if 'optimizer' in config: self.optimizer_combo.setCurrentText(config['optimizer'])
            if 'learning_rate' in config: self.learning_rate.setValue(config['learning_rate'])
            if 'weight_decay' in config: self.weight_decay.setValue(config['weight_decay'])
            if 'momentum' in config: self.momentum.setValue(config['momentum'])
            if 'beta1' in config: self.beta1.setValue(config['beta1'])
            if 'beta2' in config: self.beta2.setValue(config['beta2'])
            if 'beta' in config: self.beta.setValue(config['beta'])
            if 'epsilon' in config: self.epsilon.setValue(config['epsilon'])
            if 'epochs' in config: self.epochs.setValue(config['epochs'])
            if 'batch_size' in config: self.batch_size.setValue(config['batch_size'])
            if 'validation_split' in config: self.validation_split.setValue(config['validation_split'])
            if 'use_validation' in config: self.use_validation.setChecked(config['use_validation'])
            
            # GUI Settings
            if 'show_preview' in config: self.show_preview_cb.setChecked(config['show_preview'])
            if 'map_frequency' in config: self.fm_freq_combo.setCurrentText(config['map_frequency'])
            if 'auto_scroll' in config: self.auto_scroll_cb.setChecked(config['auto_scroll'])
            if 'chart_animations' in config: self.chart_anim_cb.setChecked(config['chart_animations'])
            
            self.blockSignals(False)
            self.on_parameter_changed()
            self.on_viz_setting_changed()
            
        except Exception as e:
            self.blockSignals(False)
            logging.error(f"Error applying config: {e}")
            raise e

    def on_save_config(self):
        # Ensure config directory exists
        config_dir = os.path.join(os.getcwd(), "src", "data", "configurations")
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"config_{timestamp}.json"
        full_path = os.path.join(config_dir, filename)
        
        try:
            # Get current parameters (parsed)
            params = self.get_training_parameters()
            # Include GUI settings
            params.update(self.get_gui_settings())
            
            with open(full_path, 'w') as f:
                json.dump(params, f, indent=4)
                
            QMessageBox.information(self, "Config Saved", f"Configuration saved to:\n{full_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save config: {e}")

    # Getters for controller use
    def get_training_parameters(self):
        # Get base training params
        params = {
            'epochs': self.epochs.value(),
            'batch_size': self.batch_size.value(),
            'learning_rate': self.learning_rate.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'weight_decay': self.weight_decay.value(),
            'momentum': self.momentum.value(),
            'beta1': self.beta1.value(),
            'beta2': self.beta2.value(),
            'beta': self.beta.value(),
            'epsilon': self.epsilon.value(),
            'validation_split': self.validation_split.value(),
            'use_validation': self.use_validation.isChecked(),
        }
        # Merge with architecture params (which are correctly parsed)
        params.update(self.get_architecture_parameters())
        return params

    def set_image_dimensions(self, width, height, depth):
        """Programmatically set image dimensions."""
        self.blockSignals(True)
        self.image_width.setValue(width)
        self.image_height.setValue(height)
        self.image_depth.setValue(depth)
        self.blockSignals(False)
        self.on_parameter_changed()
