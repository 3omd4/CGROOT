from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
                             QStatusBar, QGroupBox, QTextEdit, QMenu, QToolBar, 
                             QLabel, QProgressBar, QMessageBox, QApplication, QSplitter)
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QAction, QKeySequence, QFont

from widgets.configurationwidget import ConfigurationWidget
from widgets.trainingwidget import TrainingWidget
from widgets.inferencewidget import InferenceWidget
from widgets.metricswidget import MetricsWidget
from controllers.model_controller import ModelController

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("CGROOT++ Neural Network Trainer")
        self.resize(1200, 800)
        
        self.controller = ModelController()
        
        self.setup_ui()
        self.setup_menubar()
        self.setup_toolbar()
        self.setup_statusbar()
        self.create_connections()
        
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
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        # Remove fixed height to allow resizing
        # self.log_output.setMaximumHeight(150) 
        self.log_output.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.log_output)
        
        splitter.addWidget(log_group)
        
        # Set initial sizes (e.g., 4:1 ratio)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)

    def setup_menubar(self):
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("&File")
        
        load_dataset_act = QAction("&Load Dataset...", self)
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

    def on_load_dataset(self):
        from PyQt6.QtWidgets import QFileDialog
        import os
        
        # Default path
        start_dir = "src/data/datasets"
        if not os.path.exists(start_dir):
            start_dir = "."
            
        images_path, _ = QFileDialog.getOpenFileName(
            self, "Select MNIST Images File", start_dir,
            "MNIST Images (*.idx3-ubyte);;All Files (*.*)"
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
        else:
            # Fallback check for common patterns if 'images' string not explicit or different case
            labels_name = base_name.replace('idx3', 'idx1') 

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
            
        # Log details
        self.log_message(f"Selected Dataset Type: {dataset_type}")
        self.log_message(f"Images File: {os.path.basename(images_path)}")
        self.log_message(f"Labels File: {os.path.basename(labels_path)}")
            
        self.controller.requestLoadDataset.emit(images_path, labels_path)

    def setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Quick actions could go here
        toolbar.addAction("Start Training", self.training_tab.on_start_clicked)
        toolbar.addAction("Stop Training", self.training_tab.on_stop_clicked)

    def setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
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
        self.controller.imagePredicted.connect(self.training_tab.display_image)
        
        # Widget -> Controller/MainWindow
        self.training_tab.startTrainingRequested.connect(self.start_training)
        self.training_tab.stopTrainingRequested.connect(self.stop_training)
        
    def start_training(self):
        # Get params from configuration
        config = self.config_tab.get_training_parameters() # Epochs, LR, etc.
        arch_config = self.config_tab.get_architecture_parameters() # Layers, etc.
        
        # Merge dictionaries
        full_config = {**config, **arch_config}
        
        self.metrics_tab.clear()
        self.metrics_tab.set_total_epochs(full_config['epochs'])
        self.controller.requestTrain.emit(full_config)
        
    def stop_training(self):
        self.controller.requestStop.emit()
        
    def log_message(self, msg):
        self.log_output.append(msg)
        
    def update_progress(self, val, max_val):
        self.progress_bar.setMaximum(max_val)
        self.progress_bar.setValue(val)
        
    def update_metrics(self, loss, acc, epoch):
        self.metrics_tab.updateMetrics(loss, acc, epoch)
        self.status_label.setText(f"Epoch: {epoch} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
        
    def training_finished(self):
        self.training_tab.training_finished()
        self.status_label.setText("Training Completed")
        QMessageBox.information(self, "Training Complete", "Model training has finished successfully!")
        
    def show_about(self):
        QMessageBox.about(self, "About", 
                          "CGROOT++ Neural Network Trainer\n\n"
                          "Python Implementation (PyQt6)\n"
                          "Replicated from C++ Reference GUI")
    
    def closeEvent(self, event):
        """
        Override closeEvent to ensure graceful shutdown.
        
        Shutdown sequence:
        1. Stop training
        2. Wait for worker thread
        3. Cleanup worker resources
        4. Accept close event
        """
        self.log_message("=== Application Shutdown Initiated ===")
        
        # 1. Request training stop
        self.controller.requestStop.emit()
        
        # 2. Wait for worker thread to finish
        if hasattr(self.controller, 'workerThread') and self.controller.workerThread:
            if self.controller.workerThread.isRunning():
                self.log_message("Waiting for worker thread to stop...")
                self.controller.workerThread.quit()
                
                # Wait up to 5 seconds for graceful stop
                if not self.controller.workerThread.wait(5000):
                    self.log_message("Worker thread did not stop gracefully, terminating...")
                    self.controller.workerThread.terminate()
                    self.controller.workerThread.wait()
                else:
                    self.log_message("Worker thread stopped successfully")
        
        # 3. Cleanup worker resources
        if hasattr(self.controller, 'worker') and self.controller.worker:
            self.log_message("Cleaning up worker resources...")
            self.controller.worker.cleanup()
        
        self.log_message("=== Application Shutdown Complete ===")
        
        # 4. Accept the close event
        event.accept()
