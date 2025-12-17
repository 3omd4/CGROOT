from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QSplitter, QGroupBox, 
                             QGridLayout, QLabel)
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtCore import Qt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QFont, QPen

class MetricsWidget(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.max_data_points = 1000
        
        self.init_ui()
        self.setup_charts()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Stats Group
        stats_group = QGroupBox("Current Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.curr_epoch_lbl = QLabel("Epoch: 0")
        self.curr_loss_lbl = QLabel("Loss: 0.0000")
        self.curr_acc_lbl = QLabel("Accuracy: 0.00%")
        self.best_loss_lbl = QLabel("Best Loss: N/A")
        self.best_acc_lbl = QLabel("Best Accuracy: N/A")
        
        stats_layout.addWidget(QLabel("Current Epoch:"), 0, 0)
        stats_layout.addWidget(self.curr_epoch_lbl, 0, 1)
        stats_layout.addWidget(QLabel("Current Loss:"), 0, 2)
        stats_layout.addWidget(self.curr_loss_lbl, 0, 3)
        
        stats_layout.addWidget(QLabel("Current Accuracy:"), 1, 0)
        stats_layout.addWidget(self.curr_acc_lbl, 1, 1)
        stats_layout.addWidget(QLabel("Best Loss:"), 1, 2)
        stats_layout.addWidget(self.best_loss_lbl, 1, 3)
        
        stats_layout.addWidget(QLabel("Best Accuracy:"), 2, 0)
        stats_layout.addWidget(self.best_acc_lbl, 2, 1)
        
        splitter.addWidget(stats_group)
        
        # Charts Splitter
        chart_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        loss_group = QGroupBox("Training Loss")
        loss_layout = QVBoxLayout(loss_group)
        self.loss_view = QChartView()
        self.loss_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        loss_layout.addWidget(self.loss_view)
        chart_splitter.addWidget(loss_group)
        
        acc_group = QGroupBox("Training Accuracy")
        acc_layout = QVBoxLayout(acc_group)
        self.acc_view = QChartView()
        self.acc_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        acc_layout.addWidget(self.acc_view)
        chart_splitter.addWidget(acc_group)
        
        splitter.addWidget(chart_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
    def setup_charts(self):
        # Shared Fonts
        title_font = QFont("Arial", 14, QFont.Weight.Bold)
        axis_title_font = QFont("Arial", 10, QFont.Weight.Bold)
        axis_label_font = QFont("Arial", 9)

        # Loss Chart
        self.loss_series = QLineSeries()
        self.loss_series.setName("Loss")
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(3)
        self.loss_series.setPen(pen)
        self.loss_series.setPointsVisible(True)
        self.loss_series.setPointLabelsVisible(False)
        
        self.loss_chart = QChart()
        self.loss_chart.addSeries(self.loss_series)
        self.loss_chart.setTitle("Training Loss Over Time")
        self.loss_chart.setTitleFont(title_font)
        self.loss_chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.loss_chart.legend().setFont(axis_label_font)
        self.loss_chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        self.loss_axis_x = QValueAxis()
        self.loss_axis_x.setTitleText("Epoch")
        self.loss_axis_x.setTitleFont(axis_title_font)
        self.loss_axis_x.setLabelsFont(axis_label_font)
        self.loss_axis_x.setLabelFormat("%d")
        
        self.loss_axis_y = QValueAxis()
        self.loss_axis_y.setTitleText("Loss")
        self.loss_axis_y.setTitleFont(axis_title_font)
        self.loss_axis_y.setLabelsFont(axis_label_font)
        self.loss_axis_y.setLabelFormat("%.4f")
        
        self.loss_chart.addAxis(self.loss_axis_x, Qt.AlignmentFlag.AlignBottom)
        self.loss_chart.addAxis(self.loss_axis_y, Qt.AlignmentFlag.AlignLeft)
        self.loss_series.attachAxis(self.loss_axis_x)
        self.loss_series.attachAxis(self.loss_axis_y)
        
        self.loss_view.setChart(self.loss_chart)
        
        # Accuracy Chart
        self.acc_series = QLineSeries()
        self.acc_series.setName("Accuracy")
        pen_acc = QPen(QColor(0, 150, 0))
        pen_acc.setWidth(3)
        self.acc_series.setPen(pen_acc)
        self.acc_series.setPointsVisible(True)
        
        self.acc_chart = QChart()
        self.acc_chart.addSeries(self.acc_series)
        self.acc_chart.setTitle("Training Accuracy Over Time")
        self.acc_chart.setTitleFont(title_font)
        self.acc_chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.acc_chart.legend().setFont(axis_label_font)
        self.acc_chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        self.acc_axis_x = QValueAxis()
        self.acc_axis_x.setTitleText("Epoch")
        self.acc_axis_x.setTitleFont(axis_title_font)
        self.acc_axis_x.setLabelsFont(axis_label_font)
        self.acc_axis_x.setLabelFormat("%d")
        
        self.acc_axis_y = QValueAxis()
        self.acc_axis_y.setTitleText("Accuracy")
        self.acc_axis_y.setTitleFont(axis_title_font)
        self.acc_axis_y.setLabelsFont(axis_label_font)
        self.acc_axis_y.setLabelFormat("%.2f")
        self.acc_axis_y.setRange(0.0, 1.0)
        
        self.acc_chart.addAxis(self.acc_axis_x, Qt.AlignmentFlag.AlignBottom)
        self.acc_chart.addAxis(self.acc_axis_y, Qt.AlignmentFlag.AlignLeft)
        self.acc_series.attachAxis(self.acc_axis_x)
        self.acc_series.attachAxis(self.acc_axis_y)
        
        self.acc_view.setChart(self.acc_chart)
    
    def set_animations(self, enabled):
        """Enable or disable chart animations for performance."""
        if enabled:
            self.loss_chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
            self.acc_chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        else:
            self.loss_chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
            self.acc_chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        
    def clear(self):
        self.loss_series.clear()
        self.acc_series.clear()
        
        # Start from 0,0
        self.loss_series.append(0, 0)
        self.acc_series.append(0, 0)
        
        self.curr_epoch_lbl.setText("Epoch: 0")
        self.curr_loss_lbl.setText("Loss: 0.0000")
        self.curr_acc_lbl.setText("Accuracy: 0.00%")
        self.best_loss_lbl.setText("Best Loss: N/A")
        self.best_acc_lbl.setText("Best Accuracy: N/A")
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        
        # Reset axes default (will be overridden by set_total_epochs)
        self.loss_axis_x.setRange(0, 10)
        self.loss_axis_y.setRange(0, 1)
        self.acc_axis_x.setRange(0, 10)

    def set_total_epochs(self, total_epochs):
        """Sets the X-axis range for charts to ensure integer alignment."""
        if total_epochs < 1:
            total_epochs = 1
            
        # Range should be exactly 0 to total_epochs to include the last epoch on the axis
        self.loss_axis_x.setRange(0, total_epochs)
        self.acc_axis_x.setRange(0, total_epochs)
        
        # Smart Tick Count for Integers
        # If we have a small number of epochs, show every integer
        if total_epochs <= 20:
            count = total_epochs + 1 # +1 for zero
            self.loss_axis_x.setTickCount(count)
            self.acc_axis_x.setTickCount(count)
        else:
            # For larger ranges, pick a number that divides somewhat cleanly or default to ~6-10
            # e.g. 50 epochs -> 0, 10, 20... (6 ticks: 0, 10, 20, 30, 40, 50)
            target_ticks = 6
            # Try to find a tick count that produces integer steps? 
            # Basic fallback is fine for large ranges
            self.loss_axis_x.setTickCount(target_ticks)
            self.acc_axis_x.setTickCount(target_ticks)

    def updateMetrics(self, loss, accuracy, epoch):
        self.loss_series.append(epoch, loss)
        self.acc_series.append(epoch, accuracy)
        
        if self.loss_series.count() > self.max_data_points:
            self.loss_series.removePoints(0, self.loss_series.count() - self.max_data_points)
        if self.acc_series.count() > self.max_data_points:
            self.acc_series.removePoints(0, self.acc_series.count() - self.max_data_points)
            
        if loss < self.best_loss: self.best_loss = loss
        if accuracy > self.best_accuracy: self.best_accuracy = accuracy
        
        self.curr_epoch_lbl.setText(f"Epoch: {epoch}")
        self.curr_loss_lbl.setText(f"Loss: {loss:.4f}")
        self.curr_acc_lbl.setText(f"Accuracy: {accuracy*100:.2f}%")
        self.best_loss_lbl.setText(f"Best Loss: {self.best_loss:.4f}")
        self.best_acc_lbl.setText(f"Best Accuracy: {self.best_accuracy*100:.2f}%")
        
        # Rescale X axis if we exceed initial estimate (just in case)
        current_max = self.loss_axis_x.max()
        if epoch > current_max:
             new_max = max(epoch, current_max * 1.5)
             self.set_total_epochs(int(new_max))

        
        # Auto scale loss Y
        if self.loss_series.count() > 0:
            points = self.loss_series.points()
            min_y = min(p.y() for p in points)
            max_y = max(p.y() for p in points)
            if max_y > min_y:
                self.loss_axis_y.setRange(max(0, min_y * 0.9), max_y * 1.1)
