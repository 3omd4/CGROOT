from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QProgressBar, QLabel, QGroupBox, QGridLayout, QScrollArea, QSpinBox, QSizePolicy, QFileDialog, QCheckBox, QMessageBox)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, qRgb
from PyQt6.QtCore import Qt, pyqtSignal
try:
    from src.gui_py.utils.paths import get_datasets_dir
except ImportError:
    # Fallback if module structure differs
    def get_datasets_dir():
        from pathlib import Path
        # Fix: Datasets are in src/data/datasets
        return Path("src/data/datasets")

class TrainingWidget(QWidget):
    startTrainingRequested = pyqtSignal()
    stopTrainingRequested = pyqtSignal()
    loadModelRequested = pyqtSignal(str) # Path to model file
    storeModelRequested = pyqtSignal(str) # Path to folder
    resetModelRequested = pyqtSignal() # New Signal
    vizToggled = pyqtSignal(bool) # Signal to sync with config widget

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        
        # Default Visualization Settings
        self.viz_show_preview = True
        self.viz_map_frequency = "Every Epoch"
        
        # State tracking
        self.is_model_loaded = False
        
        self.init_ui()
        self.connect_signals()


        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Params Group removed (moved to Configuration Tab)

        # Visualization Toggle (Global)
        self.viz_checkbox = QCheckBox("Enable Real-time Visualizations (Uncheck for faster training)")
        self.viz_checkbox.setChecked(True)
        self.viz_checkbox.toggled.connect(self.on_viz_toggled)
        self.viz_checkbox.setToolTip("Disabling visualizations significantly improves training speed by reducing data transfer overhead.")
        self.viz_checkbox.setEnabled(True)  # Now enabled for user control
        main_layout.addWidget(self.viz_checkbox)

        # Preview Group
        self.preview_group = QGroupBox() # Title moved inside
        preview_layout = QVBoxLayout(self.preview_group)

        preview_header_layout = QHBoxLayout()
        
        # Header for Preview
        preview_header = QLabel("Training Preview")
        preview_header.setStyleSheet("font-weight: bold;")
        preview_header_layout.addWidget(preview_header)

        preview_header_layout.addStretch()
        
        preview_layout.addLayout(preview_header_layout)
        
        # Image Preview
        self.image_scroll_area = QScrollArea() 
        self.image_scroll_area.setWidgetResizable(True)
        self.image_scroll_area.setMinimumHeight(300) 

        self.image_label = QLabel("Waiting for training data...") 
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False) 
        self.image_label.setStyleSheet("QLabel { background-color: #222; }")
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # Avoid overlay
        
        self.image_scroll_area.setWidget(self.image_label)
        preview_layout.addWidget(self.image_scroll_area)
        
        # Info Label (e.g. "True: 5 | Pred: 5")
        self.preview_info_label = QLabel()
        self.preview_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_info_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 5px;")
        preview_layout.addWidget(self.preview_info_label)

        # Feature Maps Preview
        self.fm_group = QGroupBox() # Title moved inside
        fm_layout = QVBoxLayout(self.fm_group) 
        
        # Header Layout (Title + Controls)
        header_layout = QHBoxLayout()
        
        self.fm_title_label = QLabel("Feature Maps (Layer 1)")
        self.fm_title_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.fm_title_label)
        
        header_layout.addStretch()
        
        # Layer Selector
        layer_label = QLabel("Select Layer:")
        self.layer_spin = QSpinBox()
        self.layer_spin.setRange(0, 100) 
        self.layer_spin.setValue(1)
        self.layer_spin.valueChanged.connect(self.on_layer_changed)
        
        header_layout.addWidget(layer_label)
        header_layout.addWidget(self.layer_spin)
        
        fm_layout.addLayout(header_layout)
        
        self.scroll_area = QScrollArea() 
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(300) 
        
        self.fm_label = QLabel("Waiting for feature maps...") 
        self.fm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fm_label.setScaledContents(False) 
        self.fm_label.setStyleSheet("QLabel { background-color: #222; }")
        self.fm_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # Avoid overlay
        
        self.scroll_area.setWidget(self.fm_label)
        fm_layout.addWidget(self.scroll_area)
        
        # Top Row (Preview + Feature Maps)
        top_row = QHBoxLayout()
        top_row.addWidget(self.preview_group, 1) # 50%
        top_row.addWidget(self.fm_group, 1) # 50%
        
        main_layout.addLayout(top_row)
        
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
        
        # Save/Load Model Buttons
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        self.load_model_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.store_model_btn = QPushButton("Store Model")
        self.store_model_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 10px; }")
        self.store_model_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self.reset_btn = QPushButton("Reset Model")
        self.reset_btn.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 10px; }")
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.setToolTip("Reset model to initial random weights using current configuration")
        
        self.test_btn = QPushButton("Run Test")
        self.test_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 10px; }")
        self.test_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self.load_model_btn.clicked.connect(self.on_load_model_clicked)
        self.store_model_btn.clicked.connect(self.on_store_model_clicked)
        self.test_btn.clicked.connect(self.on_test_clicked)
        self.reset_btn.clicked.connect(self.on_reset_clicked)
        
        # Status Label
        self.status_label = QLabel("Ready to start training")
        self.status_label.setStyleSheet("QLabel { font-weight: bold; padding: 5px; }")
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.load_model_btn)
        btn_layout.addWidget(self.store_model_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.test_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.status_label)
        
        btn_layout.setContentsMargins(0, 5, 0, 5)
        
        main_layout.addLayout(btn_layout)
        
        main_layout.addStretch()
        
    def connect_signals(self):
        # Allow external control/updates via controller signals if needed
        pass
        # Connect internal signals to controller slots
        self.loadModelRequested.connect(self.controller.requestLoadModel)
        # self.storeModelRequested.connect(self.controller.requestStoreModel) # Removed: Managed by MainWindow to inject config
        
        # Listen for model changes to update internal state
        self.controller.modelInfoLoaded.connect(self.on_model_loaded)
        self.controller.modelStatusChanged.connect(self.on_model_status_changed) # True = Training Started = Model exists

    def on_start_clicked(self):
        # We emit signal, main window orchestrates call to controller
        self.startTrainingRequested.emit()
        self.set_training_state(True)
        
    def on_stop_clicked(self):
        # Confirm stop with user
        reply = QMessageBox.question(
            self, 
            "Stop Training", 
            "Do you want to stop training?\n\nThe current model state will be automatically saved to:\nsrc/data/trained-model/auto_save/",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.stopTrainingRequested.emit()
            self.set_training_state(False)
            self.status_label.setText("Stopping training and saving model...")
        
    def on_reset_clicked(self):
        reply = QMessageBox.question(
            self, 
            "Confirm Model Reset", 
            "Are you sure you want to reset the model?\n\nThis will re-initialize the neural network with random weights using the current configuration.\n\nAll training progress will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.resetModelRequested.emit()

    def _get_model_dir(self):
        from pathlib import Path
        # src/gui_py/widgets
        script_dir = Path(__file__).parent
        # Go up 3 levels to project root
        project_root = script_dir.parent.parent.parent
        model_dir = project_root / "src" / "data" / "trained-model"
        
        # Ensure it exists
        if not model_dir.exists():
             try:
                 model_dir.mkdir(parents=True, exist_ok=True)
             except Exception:
                 pass
                 
        return str(model_dir) if model_dir.exists() else ""

    def on_load_model_clicked(self):
        start_dir = self._get_model_dir()
        path, _ = QFileDialog.getOpenFileName(self, "Load Model Weights", start_dir, "Model Files (*.bin *.dat);;All Files (*.*)")
        if path:
            self.loadModelRequested.emit(path)
        
    def on_store_model_clicked(self):
        start_dir = self._get_model_dir()
        folder = QFileDialog.getExistingDirectory(self, "Select Directory to Store Model", start_dir)
        if folder:
            self.storeModelRequested.emit(folder)
            
    def on_test_clicked(self):
        if not self.is_model_loaded:
            QMessageBox.warning(self, "No Model Loaded", "Please load a model or start training before running a test.")
            return

        import os


        # Navigate to datasets dir if possible
        # Default path
        datasets_dir = get_datasets_dir()
        start_dir = str(datasets_dir) if datasets_dir.exists() else self._get_model_dir()
            
        images_path, _ = QFileDialog.getOpenFileName(self, "Select Test Images (IDX)", start_dir, "IDX Files (*.idx3-ubyte *.idx4-ubyte);;All Files (*.*)")
        
        if not images_path:
            return

        

        # Attempt to find matching labels file automatically
        # Common convention: test-images... -> test-labels...
        # or t10k-images... -> t10k-labels...

        # Attempt to auto-discover labels file
        dir_name = os.path.dirname(images_path)
        base_name = os.path.basename(images_path)

        # Common naming convention: swap 'images' with 'labels'
        # e.g. train-images.idx3-ubyte -> train-labels.idx1-ubyte
        #      train-images-idx3-ubyte -> train-labels-idx1-ubyte
        if 'images' in base_name:
            labels_name = base_name.replace('images', 'labels')
            # Fix: Also replace idx3 with idx1 to handle standard MNIST extensions
            labels_name = labels_name.replace('idx3', 'idx1') 
            labels_name = labels_name.replace('idx4', 'idx1') 

        else:
            # Fallback check for common patterns if 'images' string not explicit or different case
            labels_name = base_name.replace('idx3', 'idx1') 
            labels_name = labels_name.replace('idx4', 'idx1') 

        guess_labels_path = os.path.join(dir_name, labels_name)
        
        labels_path = ""

        if os.path.exists(guess_labels_path):
            labels_path = guess_labels_path
        else:
            # Fallback to manual selection if auto-discovery fails
            labels_path, _ = QFileDialog.getOpenFileName(
                self, "Select MNIST Labels File", dir_name,
                "MNIST Labels (*.idx1-ubyte);;All Files (*.*)"
            )
            
        if not labels_path:
            return

        if images_path and labels_path:
             self.status_label.setText("Testing training starting...")
             self.status_label.setStyleSheet("QLabel { font-weight: bold; color: blue; padding: 5px; }")
             self.controller.requestTest.emit(images_path, labels_path)
        else:
            # If user cancelled labels selection
            pass
        
    def on_layer_changed(self, val):
        self.fm_title_label.setText(f"Feature Maps (Layer {val})")
        self.controller.setTargetLayer.emit(val)

    def set_training_state(self, is_running):
        self.start_btn.setEnabled(not is_running)
        self.stop_btn.setEnabled(is_running)
        self.test_btn.setEnabled(not is_running)
        if is_running:
            self.status_label.setText("Training in progress...")
            self.status_label.setStyleSheet("QLabel { font-weight: bold; color: green; padding: 5px; }")
        else:
            self.status_label.setText("Training stopped")
            self.status_label.setStyleSheet("QLabel { font-weight: bold; color: red; padding: 5px; }")
            
    def on_model_loaded(self, w, h, d):
        self.is_model_loaded = True
        
    def on_model_status_changed(self, is_training):
        if is_training:
            self.is_model_loaded = True
            
    # Methods for controller to update UI status
    def training_finished(self):
        self.set_training_state(False)
        self.status_label.setText("Training Completed")

    def on_viz_toggled(self, checked):
        self.controller.setVisualizationsEnabled.emit(checked)
        self.vizToggled.emit(checked)  # Notify config widget
        self.preview_group.setEnabled(checked)
        self.fm_group.setEnabled(checked)
        self.preview_group.setVisible(checked)
        self.fm_group.setVisible(checked)
        if not checked:
             self.status_label.setText("Visualizations Disabled - Training runs at maximum speed")
        else:
             self.status_label.setText("Visualizations Enabled - Training runs at normal speed")

    # Dynamic limits for layer spinner
    def update_layer_limits(self, max_layers):
        self.layer_spin.setMaximum(max_layers)    

    # Visualization Settings Slot
    def set_visualization_settings(self, settings):
        if 'viz_enabled' in settings:
             # Block signals to avoid circular updates
             self.viz_checkbox.blockSignals(True)
             self.viz_checkbox.setChecked(settings.get('viz_enabled'))
             self.viz_checkbox.blockSignals(False)
             
        self.viz_show_preview = settings.get('show_preview', True)
        self.viz_map_frequency = settings.get('map_frequency', "Every Epoch")
        
        # Toggle Visibility
        is_globally_enabled = self.viz_checkbox.isChecked()
        if self.preview_group:
            # Only show if globally enabled AND preview enabled
            self.preview_group.setVisible(is_globally_enabled and self.viz_show_preview)
            self.preview_group.setEnabled(is_globally_enabled)
            
        if self.fm_group:
             self.fm_group.setVisible(is_globally_enabled)
             self.fm_group.setEnabled(is_globally_enabled)
            
        # Reset text if re-enabled and empty
        if self.viz_show_preview:
             if not self.image_label.pixmap():
                 self.image_label.setText("Waiting for training data...")
                 self.image_label.setStyleSheet("QLabel { background-color: #222; color: #white; }")

    def display_image(self, predicted_class, q_img, probs, true_label=-1):
        if not self.viz_show_preview:
            return

        if q_img:
            # Scale to label size while keeping aspect ratio
            w = self.image_label.width()
            h = self.image_label.height()
            if w <= 0: w = 150
            if h <= 0: h = 150
            
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
            # Decoupled: Removed import dataset_utils
            # Use layer_type or generic label for now.
            # Ideally Controller should pass the decoded class name if it knows the dataset
            
            # true_name = get_class_name("generic", true_label)
            # If we want detailed names, we need to know if it's Fashion MNIST. 
            # Controller or Worker knows. But for now let's just show "Label: X"
            
            text = f"True: {true_label}"
            if predicted_class != -1:
                 text += f" | Pred: {predicted_class}"
                 
            # Note: We need a place to show this text. The image_label displays pixmap OR text.
            # We should probably set it as a tooltip or overlay. 
            # Or assume we added a separate label.
            # Let's try setting setToolTip for now to be safe without changing layout structure drastically in this step.
            self.image_label.setToolTip(text)
            
            # Also, if we want to show it visibly, we might need a separate label.
            # I will modify init_ui in next step to add 'self.preview_info_label'
            if hasattr(self, 'preview_info_label'):
                self.preview_info_label.setText(text)
            else:
                 pass # Fallback
            
        
        if probs:
            # Optional: Show confidence of current sample (passed as probs)
            pass

    def display_feature_maps(self, maps_3d, layer_type=-1, is_epoch_end=False):
        """
        Display existing feature maps in a grid.
        maps_3d: List of List of List of floats [depth][height][width]
        """
        setting = self.viz_map_frequency
        if setting == "Never":
            return
        if setting == "Every Epoch" and not is_epoch_end:
            return
        # If "Every Sample", we proceed regardless of is_epoch_end
        LAYER_NAMES = {
            0: "Input",
            1: "Flatten",
            2: "Convolution",
            3: "Pooling",
            4: "Fully Connected",
            5: "Output"
        }
        
        # Update title regardless of content
        type_str = LAYER_NAMES.get(layer_type, "Unknown")
        current_layer_idx = self.layer_spin.value()
        self.fm_title_label.setText(f"Feature Maps ({current_layer_idx}: {type_str})")

        if not maps_3d:
             # Clear the label if no maps
            self.fm_label.setPixmap(QPixmap())
            self.fm_label.setText("No feature maps available for this layer")
            return

        try:
            # Ensure 3D structure: [depth][h][w]
            maps = maps_3d
            if not maps:
                return
                
            depth = len(maps)
            height = len(maps[0])
            width = len(maps[0][0])
            
            # Create a grid image
            # Say we want approx square grid
            import math
            cols = int(math.ceil(math.sqrt(depth)))
            rows = int(math.ceil(depth / cols))
            
            cell_w = width * 4 # Scale up 4x
            cell_h = height * 4
            
            grid_w = cols * cell_w + (cols + 1) * 5 # + padding
            grid_h = rows * cell_h + (rows + 1) * 5
            
            final_image = QImage(grid_w, grid_h, QImage.Format.Format_RGB32)
            final_image.fill(QColor("black"))
            
            painter = QPainter(final_image)
            
            for d in range(depth):
                r = d // cols
                c = d % cols
                
            # Global Min/Max for the ENTIRE layer (all maps)
            # This ensures relative activation strength is visible
            all_values = []
            for d in range(depth):
                all_values.extend([val for row in maps[d] for val in row])
            
            if not all_values:
                 return

            min_v = min(all_values)
            max_v = max(all_values)
            range_v = max_v - min_v if max_v != min_v else 1.0
            
            for d in range(depth):
                r = d // cols
                c = d % cols
                
                # Normalize map data using GLOBAL min/max
                fmap = maps[d]
                
                # Create QImage for this map
                # We need 8-bit grayscale
                # Manually construct bytes? Or iterate.
                # Since these are small (e.g. 24x24), iterating is fast enough.
                
                map_img = QImage(width, height, QImage.Format.Format_Grayscale8)
                
                for y in range(height):
                    for x in range(width):
                        val = fmap[y][x]
                        norm = int(255 * (val - min_v) / range_v)
                        norm = max(0, min(255, norm))
                        map_img.setPixel(x, y, qRgb(norm, norm, norm))
                        
                # Scale up
                scaled_map = map_img.scaled(cell_w, cell_h, Qt.AspectRatioMode.KeepAspectRatio)
                
                x_pos = 5 + c * (cell_w + 5)
                y_pos = 5 + r * (cell_h + 5)
                
                painter.drawImage(x_pos, y_pos, scaled_map)
                
            painter.end()
            
            self.fm_label.setPixmap(QPixmap.fromImage(final_image))
            
        except Exception as e:
            print(f"Error displaying feature maps: {e}")
            self.fm_label.setText(f"Error: {e}")
