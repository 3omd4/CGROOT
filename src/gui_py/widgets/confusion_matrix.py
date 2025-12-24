from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QLabel, QFrame, QSplitter, QTextEdit, QGroupBox, QPushButton, QFileDialog,
                             QTabWidget, QWidget, QScrollArea, QListWidget, QListWidgetItem, QAbstractItemView)
from PyQt6.QtGui import QColor, QBrush, QFont, QIcon, QPixmap, QAction
from PyQt6.QtCore import Qt, QSize
from dataset_utils import get_class_name

class ConfusionMatrixDialog(QDialog):
    def __init__(self, parent, matrix, dataset_type="MNIST", train_acc=None):
        super().__init__(parent)
        self.setWindowTitle("Model Evaluation Report")
        self.resize(1400, 750) # Increased size
        self.matrix = matrix
        self.dataset_type = dataset_type
        self.train_acc = train_acc
        
        # Calculate Metrics
        self.calculate_metrics()
        
        self.init_ui()
        
    def calculate_metrics(self):
        self.num_classes = len(self.matrix)
        self.total_samples = sum(sum(row) for row in self.matrix)
        
        # Per-class metrics
        self.precision = []
        self.recall = []
        self.f1_score = []
        
        # Top Confusions (Count, True, Predicted)
        self.confusions = [] 
        
        correct_predictions = 0
        
        for c in range(self.num_classes):
            # True Positives (Diagonal)
            tp = self.matrix[c][c]
            correct_predictions += tp
            
            # False Positives (Column sum - TP)
            col_sum = sum(self.matrix[r][c] for r in range(self.num_classes))
            fp = col_sum - tp
            
            # False Negatives (Row sum - TP)
            row_sum = sum(self.matrix[c])
            fn = row_sum - tp
            
            # Precision = TP / (TP + FP)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            self.precision.append(prec)
            
            # Recall = TP / (TP + FN)
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            self.recall.append(rec)
            
            # F1 = 2 * (P * R) / (P + R)
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            self.f1_score.append(f1)
            
            # Collect confusions for this true class
            for pred_c in range(self.num_classes):
                if c != pred_c:
                    count = self.matrix[c][pred_c]
                    if count > 0:
                        self.confusions.append((count, c, pred_c))
                        
        # Global Stats
        self.accuracy = correct_predictions / self.total_samples if self.total_samples > 0 else 0.0
        self.macro_f1 = sum(self.f1_score) / len(self.f1_score) if self.f1_score else 0.0
        self.error_rate = 1.0 - self.accuracy
        
        # Sort confusions by count (descending)
        self.confusions.sort(key=lambda x: x[0], reverse=True)

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # --- Header ---
        header_layout = QHBoxLayout()
        title = QLabel(f"Evaluation Report: {self.dataset_type}")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        header_layout.addWidget(title)
        
        btn_export = QPushButton("Export Report Image")
        btn_export.clicked.connect(self.save_report)
        header_layout.addStretch()
        header_layout.addWidget(btn_export)
        main_layout.addLayout(header_layout)

        # --- Tab Widget ---
        self.tabs = QTabWidget()
        
        # Tab 1: Analysis Report
        self.tab_report = QWidget()
        self.init_report_tab()
        self.tabs.addTab(self.tab_report, "Analysis Report")
        
        # Tab 2: Help & Guide
        self.tab_help = QWidget()
        self.init_help_tab()
        self.tabs.addTab(self.tab_help, "Guide: How to Read This?")
        
        main_layout.addWidget(self.tabs)

    def init_report_tab(self):
        layout = QHBoxLayout(self.tab_report)
        
        # 1. Left Panel: Confusion Matrix
        left_panel = QGroupBox("Confusion Matrix (Heatmap)")
        left_layout = QVBoxLayout(left_panel)
        
        # Legend Label
        legend_lbl = QLabel("Rows: True (Actual)   |   Columns: Predicted")
        legend_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        legend_lbl.setStyleSheet("font-weight: bold; font-size: 12px; color: #555; margin-bottom: 5px;")
        left_layout.addWidget(legend_lbl)
        
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setRowCount(self.num_classes)
        self.table.setColumnCount(self.num_classes)
        # Update Labels to include Index: "0: Airplane", "1: Automobile"
        labels = [f"{i}: {get_class_name(self.dataset_type, i)}" for i in range(self.num_classes)]
        
        self.table.setHorizontalHeaderLabels(labels)
        self.table.setVerticalHeaderLabels(labels)
        
        # Populate
        max_val = 0
        for r in range(self.num_classes):
            for c in range(self.num_classes):
                 if self.matrix[r][c] > max_val: max_val = self.matrix[r][c]

        for r in range(self.num_classes):
            for c in range(self.num_classes):
                val = self.matrix[r][c]
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                

                # Heatmap logic
                if r == c:
                    # Diagonal (Correct): White (low) -> Dark Green (high)
                    # Interpolate from White (255,255,255) to DarkGreen (0,100,0)
                    ratio = val / max_val if max_val > 0 else 0
                    
                    r_val = int(255 * (1 - ratio) + 0 * ratio)
                    g_val = int(255 * (1 - ratio) + 128 * ratio) # 128 is nice dark green
                    b_val = int(255 * (1 - ratio) + 0 * ratio)
                    
                    bg_color = QColor(r_val, g_val, b_val)
                    
                    # Text color: White if background is dark, else Black
                    fg_color = QColor(255, 255, 255) if ratio > 0.5 else QColor(0,0,0)

                else:
                    # Off-Diagonal (Error): White (0) -> Red (high)
                    # Normalize by row sum (error relative to class count) not global max
                    # But simple max-based scaling is often robust enough.
                    # Current: err_intensity based on max_val
                    err_ratio = val / (max_val * 0.2 + 1) # Scale aggressively to show errors
                    if err_ratio > 1.0: err_ratio = 1.0
                    
                    # White -> Red (255, 0, 0)
                    g_b_val = int(255 * (1 - err_ratio))
                    
                    if val > 0:
                         bg_color = QColor(255, g_b_val, g_b_val)
                         fg_color = QColor(0,0,0)
                    else:
                         bg_color = QColor(255, 255, 255)
                         fg_color = QColor(200, 200, 200)

                item.setBackground(QBrush(bg_color))
                item.setForeground(QBrush(fg_color))
                self.table.setItem(r, c, item)
        
        self.table.setToolTip("Rows = True Labels\nColumns = Predicted Labels\n\nDarker Green = High Accuracy\nDarker Red = High Confusion")
        
        header_h = self.table.horizontalHeader()
        header_h.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        left_layout.addWidget(self.table)
        
        # 2. Right Panel: Analysis
        right_panel = QGroupBox("Performance Analysis")
        right_layout = QVBoxLayout(right_panel)
        
        # A. Executive Summary (Global Stats)
        stats_frame = QFrame()
        stats_frame.setFrameShape(QFrame.Shape.StyledPanel)
        stats_layout = QHBoxLayout(stats_frame)
        
        def create_stat(label, value, color="#2E7D32"):
            v_layout = QVBoxLayout()
            l = QLabel(label)
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v = QLabel(value)
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            v.setStyleSheet(f"color: {color};")
            v_layout.addWidget(v)
            v_layout.addWidget(l)
            return v_layout

        stats_layout.addLayout(create_stat("Test Accuracy", f"{self.accuracy*100:.1f}%"))
        stats_layout.addLayout(create_stat("Macro F1-Score", f"{self.macro_f1:.2f}", "#1976D2"))
        stats_layout.addLayout(create_stat("Error Rate", f"{self.error_rate*100:.1f}%", "#C62828"))
        
        # Training Comparison
        if self.train_acc is not None:
             gap = self.train_acc - self.accuracy
             status_color = "#2E7D32" # Blue
             status_text = "Good Fit"
             
             if gap > 0.10: # 10% gap
                 status_color = "#D32F2F" # Red
                 status_text = "Possible Overfitting (>10%)"
             elif gap < -0.05:
                status_color = "#2E7D32" # Green
                status_text = "Underfitting"
             
             train_v_layout = QVBoxLayout()
             lbl = QLabel("Training Accuracy")
             lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
             val = QLabel(f"{self.train_acc*100:.1f}%")
             val.setAlignment(Qt.AlignmentFlag.AlignCenter)
             val.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
             val.setStyleSheet(f"color: #9C27B0;")
             
             warn = QLabel(status_text)
             warn.setStyleSheet(f"color: {status_color}; font-weight: bold;")
             warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
             
             train_v_layout.addWidget(val)
             train_v_layout.addWidget(lbl)
             train_v_layout.addWidget(warn)
             stats_layout.addLayout(train_v_layout)
        
        right_layout.addWidget(stats_frame)

        # B. Class Metrics Table
        metrics_lbl = QLabel("Class Performance:")
        metrics_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        right_layout.addWidget(metrics_lbl)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics_table.setRowCount(self.num_classes)
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels(["Class", "Precision", "Recall", "F1-Score"])
        self.metrics_table.verticalHeader().setVisible(False)
        
        # Metrics Tooltips
        self.metrics_table.horizontalHeaderItem(1).setToolTip("Precision: Of all predicted as X, how many were actually X? (Avoids False Positives)")
        self.metrics_table.horizontalHeaderItem(2).setToolTip("Recall: Of all actual X, how many did we find? (Avoids False Negatives)")
        self.metrics_table.horizontalHeaderItem(3).setToolTip("F1-Score: Harmonic mean of Precision and Recall. Best single metric per class.")

        for i in range(self.num_classes):
            name = labels[i]
            p = self.precision[i]
            r = self.recall[i]
            f = self.f1_score[i]
            
            self.metrics_table.setItem(i, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{p:.2f}"))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(f"{r:.2f}"))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(f"{f:.2f}"))
            
            if f < 0.5:
                for col in range(4):
                    self.metrics_table.item(i, col).setForeground(QBrush(QColor(200, 0, 0)))

        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.metrics_table, stretch=1)
        
        # C. Top Confusions List
        conf_lbl = QLabel("Most Frequent Errors:")
        conf_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        right_layout.addWidget(conf_lbl)
        
        self.confusions_list = QListWidget()
        self.confusions_list.setToolTip("Click an item to highlight the cell in the matrix.")
        self.confusions_list.itemClicked.connect(self.on_confusion_clicked)
        
        if not self.confusions:
             self.confusions_list.addItem("No errors! Perfect score.")
        else:
            for count, true_idx, pred_idx in self.confusions:
                true_name = labels[true_idx]
                pred_name = labels[pred_idx]
                
                text = f"{count}x: {true_name} → {pred_name}"
                item = QListWidgetItem(text)
                # Store coordinates to jump to
                item.setData(Qt.ItemDataRole.UserRole, (true_idx, pred_idx))
                self.confusions_list.addItem(item)
                
        self.confusions_list.setFixedHeight(150)
        right_layout.addWidget(self.confusions_list)
        
        layout.addWidget(left_panel, 3)
        layout.addWidget(right_panel, 2)

    def on_confusion_clicked(self, item):
        coords = item.data(Qt.ItemDataRole.UserRole)
        if coords:
            r, c = coords
            self.table.setCurrentCell(r, c)
            self.table.setFocus()

    def init_help_tab(self):
        layout = QVBoxLayout(self.tab_help)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>Understanding the Model Evaluation Report</h2>
        <p>This report helps you analyze how well your neural network is performing on the test dataset.</p>
        
        <h3>1. The Confusion Matrix (Heatmap)</h3>
        <ul>
            <li><b>Rows</b> represent the <b>True (Actual)</b> labels.</li>
            <li><b>Columns</b> represent the <b>Predicted</b> labels.</li>
            <li><b>Diagonal Cells (Green):</b> Correct predictions. The darker the green, the more correct predictions.</li>
            <li><b>Off-Diagonal Cells (Red):</b> Errors. The darker the red, the more frequent the error.</li>
        </ul>
        <p><i>Example: A red cell at Row '5', Column '3' means the model saw a '5' but predicted a '3'.</i></p>

        <h3>2. Performance Metrics</h3>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr>
                <td bgcolor="#e0e0e0"><b style="color: black;">Metric</b></td>
                <td bgcolor="#e0e0e0"><b style="color: black;">What it Questions</b></td>
                <td bgcolor="#e0e0e0"><b style="color: black;">Interpretation</b></td>
            </tr>
            <tr>
                <td><b>Precision</b></td>
                <td><i>"When the model predicts 'Cat', is it really a 'Cat'?"</i></td>
                <td>High precision means few "False Alarms". Low precision means the model "hallucinates" this class often.</td>
            </tr>
            <tr>
                <td><b>Recall</b></td>
                <td><i>"Did the model find all the 'Cats' in the image set?"</i></td>
                <td>High recall means the model rarely misses this object. Low recall means the model is "blind" to this class often.</td>
            </tr>
            <tr>
                <td><b>F1-Score</b></td>
                <td><i>"Is the model balanced?"</i></td>
                <td>
                    <b>High (> 0.9):</b> Excellent. Model is robust and balanced.<br>
                    <b>Medium (0.7-0.9):</b> Good. Normal for complex tasks.<br>
                    <b>Low (< 0.5):</b> Poor. The model is confused (random guessing is 0.1 for 10 classes).
                </td>
            </tr>
        </table>
        
        <h3>3. Improving F1-Score (Configuration Guide)</h3>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr>
                <td bgcolor="#e0e0e0"><b style="color: black;">Parameter</b></td>
                <td bgcolor="#e0e0e0"><b style="color: black;">Impact on F1-Score</b></td>
            </tr>
            <tr>
                <td><b>Epochs</b></td>
                <td><b>INCREASE IT.</b> Training for too few epochs is the #1 cause of low scores. The model needs time to learn.</td>
            </tr>
            <tr>
                <td><b>Conv Layers</b></td>
                <td><b>INCREASE IT.</b> If F1 is low, the model might be "Underfitting". Adding more Convolution layers (e.g., 2 -> 3 or 4) helps it see shapes better.</td>
            </tr>
            <tr>
                <td><b>Learning Rate</b></td>
                <td><b>DECREASE IT.</b> If the score fluctuates wildy or is stuck at 0.1, your Learning Rate might be too high (e.g. 0.1). Try 0.01 or 0.001.</td>
            </tr>
            <tr>
                <td><b>Batch Size</b></td>
                <td><b>LITTLE IMPACT.</b> Increasing batch size speeds up training but rarely improves accuracy significantly. Don't rely on it for F1-Score.</td>
            </tr>
            <tr>
                <td><b>FC Layers</b></td>
                <td><b>CAUTION.</b> Adding too many Fully Connected layers can cause Overfitting (High Training Score, Low Test F1). Usually 1-2 is enough.</td>
            </tr>
        </table>
        
        <h3>4. Common Issues Diagnosis</h3>
        <ul>
            <li><b>Low Accuracy?</b> Try training for more epochs, adding more layers, or decreasing the learning rate.</li>
            <li><b>Overfitting?</b> If Training Accuracy is high (e.g., 99%) but Test Accuracy is low (e.g., 85%), your model is just memorizing the training data. Try adding <b>Weight Decay</b> or simplifying the model.</li>
            <li><b>Specific Confusions?</b> If 'Auto' is often confused with 'Truck', check if your images are blurry or if the classes look too similar.</li>
            <li><b>Dataset Imbalance:</b> If one class has High F1 and another has Low F1, check if the dataset has enough images for the low class.</li>
            <li><b>Confusion Pairs:</b> If '7' is often confused with '1', check your data. Handwritting can be ambiguous!</li>
        </ul>
        """)
        
        layout.addWidget(help_text)
        
    def save_report(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Report", f"Confusion_Matrix_{self.dataset_type}.png", "PNG Images (*.png)")
        if filename:
            pixmap = QPixmap(self.size())
            self.render(pixmap)
            pixmap.save(filename)


