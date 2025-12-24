from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
                             QStatusBar, QGroupBox, QTextEdit, QMenu, QToolBar, 
                             QLabel, QProgressBar, QMessageBox, QApplication, QSplitter, QPushButton)
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QFont
from dataset_utils import get_class_name
from widgets.confusion_matrix import ConfusionMatrixDialog

from widgets.configurationwidget import ConfigurationWidget
from widgets.trainingwidget import TrainingWidget
from widgets.inferencewidget import InferenceWidget
from widgets.metricswidget import MetricsWidget
from widgets.spinner import SpinnerWidget
from controllers.model_controller import ModelController
from utils.resource_path import resource_path
from PyQt6.QtGui import QIcon

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("CGROOT++ Neural Network Trainer")
        self.setWindowIcon(QIcon(resource_path("icons/favicon.ico")))
        self.resize(1200, 800)
        
        self.controller = ModelController()
        
        self.setup_ui()
        self.setup_menubar()
        # self.setup_toolbar()
        self.setup_statusbar()
        self.create_connections()
        
        self.log_message(f"Application Initialized. Logs saving to: src/data/logs/")
        self.log_message(f"Starting in Full Screen Mode.")
        
        # Default State
        self.dataset_type = "MNIST"
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Splitter to allow resizing
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Tabs
        self.tabs = QTabWidget()
        
        self.training_tab = TrainingWidget(self.controller)
        self.inference_tab = InferenceWidget(self.controller)
        self.metrics_tab = MetricsWidget(self.controller)
        self.config_tab = ConfigurationWidget(self.controller)
        
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.inference_tab, "Inference")
        self.tabs.addTab(self.metrics_tab, "Metrics")
        self.tabs.addTab(self.config_tab, "Configuration")
        
        splitter.addWidget(self.tabs)
        
        # Log Output
        log_group = QGroupBox("Log Output")
        log_group.setMinimumSize(200, 150)
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        # Remove fixed height to allow resizing
        # self.log_output.setMaximumHeight(150) 
        self.log_output.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.log_output)
        
        # Clear Logs Button
        # self.clear_logs_btn = QPushButton("Clear Logs")
        # self.clear_logs_btn.clicked.connect(self.log_output.clear)
        # log_layout.addWidget(self.clear_logs_btn)
        
        splitter.addWidget(log_group)
        
        # Set initial sizes (e.g., 4:1 ratio)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)

    def setup_menubar(self):
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("&File")
        
        load_dataset_act = QAction(QIcon(resource_path("icons/open.png")), "&Load Dataset...", self)
        load_dataset_act.setShortcut("Ctrl+O")
        load_dataset_act.triggered.connect(self.on_load_dataset)
        file_menu.addAction(load_dataset_act)
        
        file_menu.addSeparator()
        
        exit_act = QAction("E&xit", self)
        exit_act.setShortcut(QKeySequence.StandardKey.Quit)
        exit_act.triggered.connect(QApplication.instance().quit)
        file_menu.addAction(exit_act)
        
        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self.show_about)
        help_menu.addAction(about_act)
        
        how_to_act = QAction("&How to Use", self)
        how_to_act.setShortcut("F1")
        how_to_act.triggered.connect(self.show_how_to_use)
        help_menu.addAction(how_to_act)

    def on_load_dataset(self):
        from PyQt6.QtWidgets import QFileDialog
        from utils.paths import get_datasets_dir
        import os
        
        # Default path
        datasets_dir = get_datasets_dir()
        start_dir = str(datasets_dir) if datasets_dir.exists() else "."
            
        images_path, _ = QFileDialog.getOpenFileName(
            self, "Select Training Images", start_dir,
            "IDX Files (*.idx3-ubyte *.idx4-ubyte);;All Files (*.*)"
        )
        
        if not images_path:
            return

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
        
        if os.path.exists(guess_labels_path):
            labels_path = guess_labels_path
            self.log_message(f"Auto-detected labels file: {labels_name}")
        else:
            # Fallback to manual selection if auto-discovery fails
            labels_path, _ = QFileDialog.getOpenFileName(
                self, "Select MNIST Labels File", dir_name,
                "MNIST Labels (*.idx1-ubyte);;All Files (*.*)"
            )
            
        if not labels_path:
            return

        # Determine Dataset Type
        dataset_type = "MNIST"
        lower_path = images_path.lower()
        if "fashion" in lower_path:
            dataset_type = "Fashion-MNIST"
        elif "emnist" in lower_path:
            dataset_type = "EMNIST"
        elif "kmnist" in lower_path:
            dataset_type = "KMNIST"
        elif "mnist" in lower_path:
            dataset_type = "MNIST"
        elif "cifar" in lower_path:
            dataset_type = "CIFAR-10"
            
        self.dataset_type = dataset_type # Store for saving with model
            
        # Log details
        self.log_message(f"Selected Dataset Type: {dataset_type}")
        self.log_message(f"Images File: {os.path.basename(images_path)}")
        self.log_message(f"Labels File: {os.path.basename(labels_path)}")
        
        # Update Inference Tab
        self.inference_tab.set_dataset_type(dataset_type)
            
        self.controller.requestLoadDataset.emit(images_path, labels_path)

    def setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Quick actions with shortcuts
        
        start_action = toolbar.addAction(QIcon(resource_path("icons/play.png")), "&Start Training (Ctrl+T)", self.training_tab.on_start_clicked)
        start_action.setShortcut("Ctrl+T")
        
        stop_action = toolbar.addAction(QIcon(resource_path("icons/stop.png")), "&Stop Training (Ctrl+S)", self.training_tab.on_stop_clicked)
        stop_action.setShortcut("Ctrl+S")

    def setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Spinner in status bar
        self.spinner = SpinnerWidget(self)
        self.status_bar.addPermanentWidget(self.spinner)
        self.spinner.hide()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setTextVisible(True)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_connections(self):
        # Controller -> MainWindow
        self.controller.logMessage.connect(self.log_message)
        self.controller.progressUpdated.connect(self.update_progress)
        self.controller.metricsUpdated.connect(self.update_metrics)
        self.controller.trainingFinished.connect(self.training_finished)
        self.controller.imagePredicted.connect(self.inference_tab.displayPrediction)
        self.controller.trainingPreviewReady.connect(self.training_tab.display_image) # UPDATED
        
        self.controller.featureMapsReady.connect(self.training_tab.display_feature_maps)
        self.controller.configurationLoaded.connect(self.config_tab.load_parameters) # Update UI with config
        self.controller.configurationLoaded.connect(self.on_configuration_loaded) # Propagate to TrainingWidget via Main
        self.controller.metricsCleared.connect(self.metrics_tab.clear)
        self.controller.metricsSetEpoch.connect(self.metrics_tab.set_total_epochs)
        self.controller.datasetInfoLoaded.connect(self.on_dataset_info_loaded)
        self.controller.modelInfoLoaded.connect(self.on_model_info_loaded)
        self.controller.evaluationFinished.connect(self.on_evaluation_finished)
        
        self.config_tab.vizSettingsChanged.connect(self.on_gui_settings_changed)
        
        # Bidirectional sync: Training widget viz checkbox → Config widget
        self.training_tab.vizToggled.connect(self.config_tab.set_viz_enabled)
        
        # Initialize GUI Settings
        self.on_gui_settings_changed(self.config_tab.get_gui_settings())
        
        # Loading State Connections
        # Note: ModelWorker emits modelStatusChanged(True) for training/inference
        # But we want specifically to handle Dataset Loading which is separate.
        # ModelWorker uses log messages for dataset loading start/end, but that's hard to parse.
        # Ideally ModelWorker should have a specific signal for "loading".
        # I will add signal connections assuming I add them to ModelController/Worker later.
        
        self.controller.modelStatusChanged.connect(self.on_model_status_changed)
        
        # Widget -> Controller/MainWindow
        self.training_tab.startTrainingRequested.connect(self.start_training)
        self.training_tab.stopTrainingRequested.connect(self.stop_training)
        self.training_tab.storeModelRequested.connect(self.on_store_model_requested) # New handler
        self.training_tab.resetModelRequested.connect(self.on_reset_model_requested)

    def on_store_model_requested(self, folder_path):
        # Gather full configuration (Training + Architecture + GUI)
        train_params = self.config_tab.get_training_parameters()
        arch_params = self.config_tab.get_architecture_parameters()
        gui_params = self.config_tab.get_gui_settings()
        
        gui_params = self.config_tab.get_gui_settings()
        
        
        full_config = {**train_params, **arch_params, **gui_params}
        full_config['dataset_type'] = self.dataset_type # Save dataset type with model
        
        # Add full logs from UI to configuration payload (transient, not for config file)
        if hasattr(self, 'log_output'):
            full_config['_full_logs'] = self.log_output.toPlainText()
        
        self.controller.requestStoreModel.emit(folder_path, full_config)

    def on_reset_model_requested(self):
        self.log_message("Resetting Model with current initialization parameters...")
        # Gather full configuration
        train_params = self.config_tab.get_training_parameters()
        arch_params = self.config_tab.get_architecture_parameters()
        gui_params = self.config_tab.get_gui_settings()
        
        full_config = {**train_params, **arch_params, **gui_params}
        full_config['dataset_type'] = self.dataset_type 
        
        self.controller.requestResetModel.emit(full_config)
        
    def start_training(self):
        # Get params from configuration
        config = self.config_tab.get_training_parameters() # Epochs, LR, etc.
        arch_config = self.config_tab.get_architecture_parameters() # Layers, etc.
        
        # Merge dictionaries
        full_config = {**config, **arch_config}
        full_config['dataset_type'] = self.dataset_type # Include in training config
        
        self.metrics_tab.clear()
        self.metrics_tab.set_total_epochs(full_config['epochs'])
        self.controller.requestTrain.emit(full_config)
        
    def stop_training(self):
        self.controller.requestStop.emit()
        self.spinner.stop()
        
    def log_message(self, msg):
        print(msg) # Ensure console output
        self.log_output.append(msg)
        if hasattr(self, 'auto_scroll_logs') and self.auto_scroll_logs:
            scrollbar = self.log_output.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        
    def update_progress(self, val, max_val):
        self.progress_bar.setMaximum(max_val)
        self.progress_bar.setValue(val)
        
    def update_metrics(self, loss, acc, epoch):
        self.metrics_tab.updateMetrics(loss, acc, epoch)
        self.status_label.setText(f"Epoch: {epoch} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
        
    def training_finished(self):
        self.training_tab.training_finished()
        self.spinner.stop()
        self.status_label.setText("Training Completed")
        QMessageBox.information(self, "Training Complete", "Model training has finished successfully!")
    
    def on_gui_settings_changed(self, settings):
        """Handle GUI settings changes from ConfigurationWidget."""
        # Training Preview & Feature Maps
        self.training_tab.set_visualization_settings(settings)
        
        # Auto-Scroll Logs
        self.auto_scroll_logs = settings.get('auto_scroll', True)
        
        # Chart Animations
        animations_enabled = settings.get('chart_animations', False)
        self.metrics_tab.set_animations(animations_enabled)
        
    def show_about(self):
        about_text = """
        <h2 style='color: #2196F3;'>CGROOT++ Neural Network Trainer</h2>
        <p>A powerful, Python-based GUI for the CGROOT++ Neural Network framework.</p>
        <p>This application provides an intuitive interface for training, monitoring, and testing neural networks with real-time visualization.</p>
        
        <h3>👨‍💻 <b>Team Members</b></h3>
        <ul>
            <li><b>Mohamed Emad-Eldeen</b></li>
            <li><b>George Esmat</b></li>
            <li><b>Ziad Khalid</b></li>
            <li><b>Ahmed Hasan</b></li>
            <li><b>Mohamed Amgd</b></li>
            <li><b>Antony Ghayes</b></li>
        </ul>
        
        <p><i>Replicated from C++ Reference GUI. Built with PyQt6.</i></p>
        """
        QMessageBox.about(self, "About CGROOT++", about_text)

    def show_how_to_use(self):
        content = """
        <h2 style='color: #2196F3;'>How to Use CGROOT++</h2>
        
        <h3>1. Setup & Data Loading</h3>
        <ul>
            <li>Go to <b>File > Load Dataset</b>.</li>
            <li>Select your MNIST-format <b>images</b> file (e.g., <i>train-images.idx3-ubyte</i>).</li>
            <li>The corresponding <b>labels</b> file will be auto-detected if located in the same directory.</li>
        </ul>
        
        <h3>2. Training</h3>
        <ul>
            <li>Navigate to the <b>Training</b> tab.</li>
            <li>Click <b>Start Training</b> to begin the learning process.</li>
            <li><b>Store Model</b>: Save your current model weights to a folder.</li>
            <li><b>Load Model</b>: Load previously saved weights to resume training or for inference.</li>
            <li>Visualize real-time progress in the preview window.</li>
        </ul>
        
        <h3>3. Configuration</h3>
        <ul>
            <li>Go to the <b>Configuration</b> tab to adjust hyperparameters:</li>
            <li><b>Epochs</b>, <b>Learning Rate</b>, <b>Batch Size</b>.</li>
            <li>Select standard (SGD) or advanced (Adam) optimizers.</li>
        </ul>
        
        <h3>4. Inference</h3>
        <ul>
            <li>Switch to the <b>Inference</b> tab to test the model.</li>
            <li>Click <b>Load Image</b> (default samples in <i>src/data/samples</i>).</li>
            <li>Click <b>Run Inference</b> to see the model's prediction and confidence scores.</li>
        </ul>
        
        <h3>5. Metrics</h3>
        <ul>
            <li>Monitor <b>Loss</b> and <b>Accuracy</b> charts in the <b>Metrics</b> tab.</li>
            <li>Toggle animations in Configuration settings for smoother visualization.</li>
        </ul>
        
        <p><i>Tip: Use the Log Output panel at the bottom to check status messages and debug info.</i></p>
        """
        
        # Use a creating a larger dialog if needed, but QMessageBox.about handles rich text well enough for this length.
        # However, for "Full" walkthrough, scrolling is key. QMessageBox.about usually scrolls.
        QMessageBox.about(self, "CGROOT++ User Guide", content)
    
    def closeEvent(self, event):
        """
        Override closeEvent to ensure graceful shutdown and log saving.
        
        Shutdown sequence:
        1. Save logs to src/data/logs/
        2. Stop training
        3. Wait for worker thread with progress dialog
        4. Cleanup worker resources
        5. Accept close event
        """
        self.log_message("=== Application Shutdown Initiated ===")
        
        # 0. Save Logs
        try:
            import os
            from datetime import datetime
            
            from pathlib import Path
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            log_dir = project_root / "src" / "data" / "logs"
            
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"session_{timestamp}.txt"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(self.log_output.toPlainText())
            
            print(f"Logs saved to: {log_file}")
            
        except Exception as e:
            print(f"Failed to save logs: {e}")

        # 1. Request training stop
        self.controller.requestStop.emit()
        
        # 2. Wait for worker thread to finish with progress indication
        if hasattr(self.controller, 'workerThread') and self.controller.workerThread:
            if self.controller.workerThread.isRunning():
                # Create progress dialog to prevent appearing frozen
                from PyQt6.QtWidgets import QProgressDialog
                progress = QProgressDialog("Stopping training thread...", "Force Close", 0, 50, self)
                progress.setWindowTitle("Closing Application")
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(0)
                
                self.log_message("Waiting for worker thread to stop...")
                self.controller.workerThread.quit()
                
                # Wait up to 10 seconds with progress updates
                wait_time = 0
                max_wait = 10000  # 10 seconds
                interval = 200  # 200ms intervals
                
                while wait_time < max_wait and self.controller.workerThread.isRunning():
                    QApplication.processEvents()  # Keep GUI responsive
                    self.controller.workerThread.wait(interval)
                    wait_time += interval
                    progress.setValue(int((wait_time / max_wait) * 50))
                    
                    # Check if user clicked "Force Close"
                    if progress.wasCanceled():
                        self.log_message("Force close requested by user")
                        break
                
                if self.controller.workerThread.isRunning():
                    self.log_message("Worker thread did not stop gracefully, terminating...")
                    self.controller.workerThread.terminate()
                    self.controller.workerThread.wait(1000)  # Wait 1 more second
                else:
                    self.log_message("Worker thread stopped successfully")
                    
                progress.setValue(50)
                progress.close()
        
        # 3. Cleanup worker resources
        if hasattr(self.controller, 'worker') and self.controller.worker:
            self.log_message("Cleaning up worker resources...")
            try:
                self.controller.worker.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {e}")
        
        self.log_message("=== Application Shutdown Complete ===")
        
        # 4. Accept the close event
        event.accept()
        
        # Force process exit to ensure clean shutdown
        print("Forcing process exit...")
        QApplication.processEvents()
        import sys
        sys.exit(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
            
    def on_model_status_changed(self, is_active):
        if is_active:
             self.spinner.start()
             self.status_label.setText("Processing...")
        else:
             self.spinner.stop()
             self.status_label.setText("Ready")

    def on_dataset_load_start(self):
        self.spinner.start()
        self.status_label.setText("Loading Dataset...")
        
    def on_dataset_load_finish(self):
         self.spinner.stop()
         self.status_label.setText("Dataset Loaded")

    def on_dataset_info_loaded(self, num, w, h, d):
        """Update Configuration with dataset properties."""
        self.log_message(f"Updating Configuration: Input {w} x {h} x {d} (width, height, depth)")
        self.config_tab.set_image_dimensions(w, h, d) 

    def on_model_info_loaded(self, w, h, d):
        """Handle model info loaded -> Guess dataset type if no dataset loaded."""
        self.log_message(f"Model Properties Detected: {w}x{h}x{d}")
        
        # Heuristic to guess dataset type for labels
        dataset_type = "MNIST" # Default
        
        # CIFAR-10 is 32x32. 
        # If d=3 (RGB) OR d=1 (Grayscale CIFAR), we assume CIFAR-10
        if w == 32 and h == 32:
            self.dataset_type = "CIFAR-10"
        elif w == 28 and h == 28:
            self.dataset_type = "MNIST" # Could be Fashion too, but MNIST is safer default
        else:
            # Keep existing or default if dimensions unknown
            pass
        
        self.log_message(f"Auto-setting Dataset Type (Labels) to: {self.dataset_type}")
        self.inference_tab.set_dataset_type(self.dataset_type)
        
        # Also update config dimensions to match model
        self.config_tab.set_image_dimensions(w, h, d) 

    def on_configuration_loaded(self, config):
        """Handle side effects of config loading that affect other tabs."""
        
        # Restore dataset type if present
        if 'dataset_type' in config:
            dt = config['dataset_type']
            self.dataset_type = dt
            self.inference_tab.set_dataset_type(dt)
            self.log_message(f"Restored Dataset Type from Model: {dt}")
        
        # Update Training Tab Layer Limits
        if 'num_conv_layers' in config:
             # Total layers in viewer = Input + Flatten + ConvLayers + Pooling + FC + Output
             # This logic depends on architecture. 
             # Safe upper bound? 100 is fine, but cleaner to set exact?
             # Let's just assume 100 max is set by default, but we can lower or specific.
             pass
        
        # Actually, simpler: just call on_start or something. 
        # But wait, the user asked: "Update the layer_spin.setMaximum() dynamically... based on the configuration."
        
        # We need to calculate total layers.
        # Conv layers depend on pool intervals.
        num_conv = config.get('num_conv_layers', 0)
        num_fc = config.get('num_fc_layers', 0)
        
        # Approximate: Input(1) + Conv(N) + Pool(N or less) + Flatten(1) + FC(N) + Output(1)
        # It's hard to get exact count without instantiating model.
        # But we can set a safe upper bound or try to calc.
        # Let's just pass 50 for now, or if we want exact:
        # Core model knows. Getting it from C++ model would be ideal but async.
        
        # For now, let's just make sure it's not 100 if we only have 5 layers.
        total_est = 1 + num_conv * 2 + 1 + num_fc # Generous estimate
        self.training_tab.update_layer_limits(total_est + 5) 

    def on_evaluation_finished(self, loss, accuracy, matrix):
        """Handle evaluation completion: Show Confusion Matrix."""
        self.log_message("Measurement Complete. Showing Confusion Matrix...")
        
        # Instantiate and show dialog
        # We store it in self to prevent garbage collection
        self.cm_dialog = ConfusionMatrixDialog(self, matrix, self.dataset_type)
        self.cm_dialog.show()
