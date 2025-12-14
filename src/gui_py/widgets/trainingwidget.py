from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QProgressBar, QLabel, QGroupBox, QGridLayout, QScrollArea, QSpinBox, QSizePolicy)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, qRgb
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
        # Image Preview
        self.image_label = QLabel("Waiting for training data...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumWidth(100)
        self.image_label.setMinimumHeight(75)
        self.image_label.setMaximumHeight(100)
        self.image_label.setStyleSheet("QLabel { background-color: #222; border: 1px solid #444; }")
        self.image_label.setScaledContents(False) # Prevent distortion
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Allow resizing
        preview_layout.addWidget(self.image_label)
        
        # Feature Maps Preview
        self.fm_group = QGroupBox("Feature Maps (Layer 1)")
        fm_layout = QVBoxLayout(self.fm_group) 
        
        # Layer Selector
        layer_select_layout = QHBoxLayout()
        layer_label = QLabel("Select Layer:")
        self.layer_spin = QSpinBox()
        self.layer_spin.setRange(0, 100) # Arbitrary max, model dependent
        self.layer_spin.setValue(1)
        self.layer_spin.valueChanged.connect(self.on_layer_changed)
        layer_select_layout.addWidget(layer_label)
        layer_select_layout.addWidget(self.layer_spin)
        layer_select_layout.addStretch()
        fm_layout.addLayout(layer_select_layout)
        
        self.scroll_area = QScrollArea() 
        self.scroll_area.setWidgetResizable(True)
        # Fix: Remove fixed height or make it small
        self.scroll_area.setMinimumHeight(300) 
        
        self.fm_label = QLabel() 
        self.fm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fm_label.setScaledContents(False) # Allow scaling
        # Fix: Remove large minimum width forcing.
        self.fm_label.setMinimumWidth(400)
        self.fm_label.setMaximumHeight(300) 
        self.fm_label.setStyleSheet("QLabel { background-color: #111; }")
        self.fm_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Allow resizing
        
        
        self.scroll_area.setWidget(self.fm_label)
        fm_layout.addWidget(self.scroll_area)
        
        main_layout.addWidget(preview_group)
        main_layout.addWidget(self.fm_group)
        
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
        
    def on_layer_changed(self, val):
        self.fm_group.setTitle(f"Feature Maps (Layer {val})")
        self.controller.setTargetLayer.emit(val)

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
            # Scale to label size while keeping aspect ratio
            w = self.image_label.width()
            h = self.image_label.height()
            if w <= 0: w = 150
            if h <= 0: h = 150
            
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setText("") # Clear text
        
        if probs:
            # Optional: Show confidence of current sample (passed as probs)
            pass

    def display_feature_maps(self, maps_3d, layer_type=-1):
        """
        Display existing feature maps in a grid.
        maps_3d: List of List of List of floats [depth][height][width]
        """
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
        self.fm_group.setTitle(f"Feature Maps ({current_layer_idx}: {type_str})")

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
                
                # Normalize map data 0-255
                # Find max/min for this map
                fmap = maps[d]
                # Flatten to find min/max
                flat = [val for row in fmap for val in row]
                min_v = min(flat)
                max_v = max(flat)
                range_v = max_v - min_v if max_v != min_v else 1.0
                
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
