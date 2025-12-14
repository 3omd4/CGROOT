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

    def setup_model_tab(self):
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)
        
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 1000)
        self.num_classes.setValue(10)
        self.num_classes.setToolTip("The number of distinct classes/categories in your dataset (e.g. 10 for MNIST).")
        
        self.image_width = QSpinBox()
        self.image_width.setRange(1, 1000)
        self.image_width.setValue(28)
        self.image_width.setToolTip("Width of the input images in pixels.")
        
        self.image_height = QSpinBox()
        self.image_height.setRange(1, 1000)
        self.image_height.setValue(28)
        self.image_height.setToolTip("Height of the input images in pixels.")
        
        self.num_layers = QSpinBox()
        self.num_layers.setRange(1, 100)
        self.num_layers.setValue(2)
        self.num_layers.setToolTip("Total number of layers in the custom model.")
        
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
        self.optimizer_combo.setCurrentIndex(1) # Default to Adam
        self.optimizer_combo.setToolTip("The optimization algorithm. 'Adam' is generally a good default choice.")
        
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0, 10.0)
        self.learning_rate.setValue(0.01)
        self.learning_rate.setDecimals(5)
        self.learning_rate.setToolTip("Step size for the optimizer. Too high = divergent, too low = slow.")
        
        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setRange(0.0, 1.0)
        self.weight_decay.setValue(0.01)
        self.weight_decay.setDecimals(5)
        self.weight_decay.setToolTip("L2 Regularization term to prevent overfitting.")
        
        self.momentum = QDoubleSpinBox()
        self.momentum.setRange(0.0, 1.0)
        self.momentum.setValue(0.9)
        self.momentum.setDecimals(3)
        self.momentum.setToolTip("Accelerates SGD in the relevant direction and dampens oscillations.")
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(10)
        self.epochs.setToolTip("Number of full passes through the training dataset. More epochs = better accuracy but longer training.")
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 10000)
        self.batch_size.setValue(64)
        self.batch_size.setToolTip("Number of training examples used in one iteration. Larger batches = more stable training.")
        
        self.use_validation = QCheckBox()
        self.use_validation.setChecked(False)
        self.use_validation.setToolTip("If checked, a portion of training data is set aside to validate model performance.")
        
        self.validation_split = QDoubleSpinBox()
        self.validation_split.setRange(0.0, 0.5)
        self.validation_split.setValue(0.0)
        self.validation_split.setToolTip("Percentage of data to use for validation (0.2 = 20%).")
        
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
        
        # Conv Layers Config
        self.num_conv_layers = QSpinBox()
        self.num_conv_layers.setRange(0, 10)
        self.num_conv_layers.setValue(2) # Default to 2
        self.num_conv_layers.setToolTip("Number of Convolutional layers.")
        layout.addRow("Number of Conv Layers:", self.num_conv_layers)

        self.kernels_per_layer = QLineEdit("3, 3")
        self.kernels_per_layer.setPlaceholderText("e.g. 3, 3")
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

        self.pooling_intervals = QLineEdit("2, 2")
        self.pooling_intervals.setPlaceholderText("e.g. 2, 2")
        self.pooling_intervals.setToolTip("Number of Conv layers before each Pooling layer. e.g. '2' means every 2 convs.")
        layout.addRow("Pooling Intervals:", self.pooling_intervals)
        
        # FC Layers Config
        self.num_fc_layers = QSpinBox()
        self.num_fc_layers.setRange(1, 10)
        self.num_fc_layers.setValue(2)
        layout.addRow("Number of FC Layers:", self.num_fc_layers)
        
        self.neurons_fc_input = QLineEdit("64, 10")
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
                    except: pass
            
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
        self.num_layers.setValue(3)
        
        self.optimizer_combo.setCurrentIndex(1) # Adam
        self.learning_rate.setValue(0.01)  # Good default for Adam
        self.weight_decay.setValue(0.01)  # L2 regularization
        self.momentum.setValue(0.9)
        self.epochs.setValue(10)  # More epochs for better convergence
        self.batch_size.setValue(64)  # Larger batch for stability
        self.use_validation.setChecked(False)
        self.validation_split.setValue(0.0)
        
        # Better default architecture for MNIST
        self.num_fc_layers.setValue(2)
        self.neurons_fc_input.setText("64, 10")  # Larger first layer

        self.num_conv_layers.setValue(2)
        self.kernels_per_layer.setText("3, 3")
        self.kernel_dims.setText("3x3, 3x3")
        self.pooling_type.setCurrentIndex(0) # Max
        self.pooling_intervals.setText("2, 2")


        
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
                
                # Apply configuration to UI elements
                # Block signals to prevent multiple updates
                self.blockSignals(True)
                
                if 'num_classes' in config: self.num_classes.setValue(config['num_classes'])
                if 'image_width' in config: self.image_width.setValue(config['image_width'])
                if 'image_height' in config: self.image_height.setValue(config['image_height'])
                
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
                if 'epochs' in config: self.epochs.setValue(config['epochs'])
                if 'batch_size' in config: self.batch_size.setValue(config['batch_size'])
                if 'validation_split' in config: self.validation_split.setValue(config['validation_split'])
                if 'use_validation' in config: self.use_validation.setChecked(config['use_validation'])
                
                self.blockSignals(False)
                self.on_parameter_changed()
                QMessageBox.information(self, "Config Loaded", f"Configuration loaded from:\n{path}")
                
            except Exception as e:
                self.blockSignals(False)
                QMessageBox.critical(self, "Load Error", f"Failed to load config: {e}")

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
            'validation_split': self.validation_split.value(),
            'use_validation': self.use_validation.isChecked(),
        }
        # Merge with architecture params (which are correctly parsed)
        params.update(self.get_architecture_parameters())
        return params
