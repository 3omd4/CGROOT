from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QLabel, QFrame)
from PyQt6.QtGui import QColor, QBrush, QFont
from PyQt6.QtCore import Qt
from src.gui_py.dataset_utils import get_class_name

class ConfusionMatrixDialog(QDialog):
    def __init__(self, parent, matrix, dataset_type="MNIST"):
        super().__init__(parent)
        self.setWindowTitle("Confusion Matrix")
        self.resize(800, 600)
        self.matrix = matrix
        self.dataset_type = dataset_type
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel(f"Confusion Matrix ({self.dataset_type})")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Instructions
        subtitle = QLabel("Rows: True Labels | Columns: Predicted Labels")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        # Table
        self.table = QTableWidget()
        num_classes = len(self.matrix)
        self.table.setRowCount(num_classes)
        self.table.setColumnCount(num_classes)
        
        # Set Headers
        labels = [get_class_name(self.dataset_type, i) for i in range(num_classes)]
        self.table.setHorizontalHeaderLabels(labels)
        self.table.setVerticalHeaderLabels(labels)
        
        # Populate Table
        max_val = 0
        for r in range(num_classes):
            for c in range(num_classes):
                val = self.matrix[r][c]
                if val > max_val: 
                    max_val = val
                    
        for r in range(num_classes):
            for c in range(num_classes):
                val = self.matrix[r][c]
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Color Logic
                # Diagonal = Good (Green intensity)
                # Off-diagonal = Bad (Red intensity)
                
                if r == c:
                    # Diagonal (Correct)
                    # Normalize against max value or row sum?
                    # Using global max for simplicity of heatmap
                    intensity = 0
                    if max_val > 0:
                        intensity = int((val / max_val) * 200) + 55 # 55-255
                    color = QColor(0, intensity, 0) # Green scales with count
                    # White text for dark backgrounds
                    if intensity > 128:
                        item.setForeground(QBrush(QColor(255, 255, 255)))
                else:
                    # Off-diagonal (Incorrect)
                    # Use a different logic for misclassifications to make them pop?
                    # Errors are usually much smaller than True Positives.
                    # Normalize against Max Error?
                    # Or against Row Sum to see probability of confusion?
                    # Let's keep it simple: Light to Dark Red.
                    intensity = 0
                    if max_val > 0:
                        # Amplify errors visually because they are small numbers
                        intensity = int((val / (max_val * 0.2 + 1)) * 255) 
                        if intensity > 255: intensity = 255
                    
                    if val > 0:
                        color = QColor(255, 255 - intensity, 255 - intensity) # White to Red
                    else:
                        color = QColor(255, 255, 255) # White for zero
                        
                item.setBackground(QBrush(color))
                self.table.setItem(r, c, item)
                
        # Adjust sizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        layout.addWidget(self.table)
