from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QSplitter, QGroupBox, 
                             QGridLayout, QLabel)
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor

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
        stats_layout.addWidget(QLabel("Current Loss:"), 1, 0)
        stats_layout.addWidget(self.curr_loss_lbl, 1, 1)
        stats_layout.addWidget(QLabel("Current Accuracy:"), 2, 0)
        stats_layout.addWidget(self.curr_acc_lbl, 2, 1)
        stats_layout.addWidget(QLabel("Best Loss:"), 3, 0)
        stats_layout.addWidget(self.best_loss_lbl, 3, 1)
        stats_layout.addWidget(QLabel("Best Accuracy:"), 4, 0)
        stats_layout.addWidget(self.best_acc_lbl, 4, 1)
        
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
        # Loss Chart
        self.loss_series = QLineSeries()
        self.loss_series.setName("Loss")
        self.loss_series.setColor(QColor(255, 0, 0))
        
        self.loss_chart = QChart()
        self.loss_chart.addSeries(self.loss_series)
        self.loss_chart.setTitle("Training Loss Over Time")
        self.loss_chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        
        self.loss_axis_x = QValueAxis()
        self.loss_axis_x.setTitleText("Epoch")
        self.loss_axis_x.setLabelFormat("%d")
        
        self.loss_axis_y = QValueAxis()
        self.loss_axis_y.setTitleText("Loss")
        self.loss_axis_y.setLabelFormat("%.4f")
        
        self.loss_chart.addAxis(self.loss_axis_x, Qt.AlignmentFlag.AlignBottom)
        self.loss_chart.addAxis(self.loss_axis_y, Qt.AlignmentFlag.AlignLeft)
        self.loss_series.attachAxis(self.loss_axis_x)
        self.loss_series.attachAxis(self.loss_axis_y)
        
        self.loss_view.setChart(self.loss_chart)
        
        # Accuracy Chart
        self.acc_series = QLineSeries()
        self.acc_series.setName("Accuracy")
        self.acc_series.setColor(QColor(0, 150, 0))
        
        self.acc_chart = QChart()
        self.acc_chart.addSeries(self.acc_series)
        self.acc_chart.setTitle("Training Accuracy Over Time")
        self.acc_chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        
        self.acc_axis_x = QValueAxis()
        self.acc_axis_x.setTitleText("Epoch")
        self.acc_axis_x.setLabelFormat("%d")
        
        self.acc_axis_y = QValueAxis()
        self.acc_axis_y.setTitleText("Accuracy")
        self.acc_axis_y.setLabelFormat("%.2f")
        self.acc_axis_y.setRange(0.0, 1.0)
        
        self.acc_chart.addAxis(self.acc_axis_x, Qt.AlignmentFlag.AlignBottom)
        self.acc_chart.addAxis(self.acc_axis_y, Qt.AlignmentFlag.AlignLeft)
        self.acc_series.attachAxis(self.acc_axis_x)
        self.acc_series.attachAxis(self.acc_axis_y)
        
        self.acc_view.setChart(self.acc_chart)
        
    def clear(self):
        self.loss_series.clear()
        self.acc_series.clear()
        self.curr_epoch_lbl.setText("Epoch: 0")
        self.curr_loss_lbl.setText("Loss: 0.0000")
        self.curr_acc_lbl.setText("Accuracy: 0.00%")
        self.best_loss_lbl.setText("Best Loss: N/A")
        self.best_acc_lbl.setText("Best Accuracy: N/A")
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        
        # Reset axes
        self.loss_axis_x.setRange(0, 10)
        self.loss_axis_y.setRange(0, 1)
        self.acc_axis_x.setRange(0, 10)

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
        
        # Rescale axes
        if epoch > self.loss_axis_x.max():
             self.loss_axis_x.setMax(epoch * 1.2)
             self.acc_axis_x.setMax(epoch * 1.2)
        
        # Auto scale loss Y
        if self.loss_series.count() > 0:
            points = self.loss_series.points()
            min_y = min(p.y() for p in points)
            max_y = max(p.y() for p in points)
            if max_y > min_y:
                self.loss_axis_y.setRange(max(0, min_y * 0.9), max_y * 1.1)
