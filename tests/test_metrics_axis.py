import sys
import unittest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCharts import QValueAxis

# Mock controller since MetricsWidget needs one in init, 
# but doesn't use it for the tested functionality (except storing it)
class MockController:
    pass

# Import the widget to test
# Adjust path as needed; assuming run from root d:\CGROOT
sys.path.append('src/gui_py') 
from widgets.metricswidget import MetricsWidget

class TestMetricsWidgetAxis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # QApplication needed for QWidgets
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.controller = MockController()
        self.widget = MetricsWidget(self.controller)

    def test_set_total_epochs(self):
        # Test setting epochs to 50
        self.widget.set_total_epochs(50)
        
        # Verify Loss Axis X
        self.assertEqual(self.widget.loss_axis_x.min(), 0.0)
        self.assertEqual(self.widget.loss_axis_x.max(), 50.0)
        
        # Verify Accuracy Axis X
        self.assertEqual(self.widget.acc_axis_x.min(), 0.0)
        self.assertEqual(self.widget.acc_axis_x.max(), 50.0)

    def test_set_total_epochs_small(self):
        # Test edge case: ensure it doesn't crash on 0 or small numbers
        self.widget.set_total_epochs(0) # Logic sets min to 1
        self.assertEqual(self.widget.loss_axis_x.max(), 1.0)
        
        self.widget.set_total_epochs(5)
        self.assertEqual(self.widget.loss_axis_x.max(), 5.0)
        
    def test_update_metrics_rescale(self):
        # Set range to 10
        self.widget.set_total_epochs(10)
        
        # Update with epoch 5 (within range)
        self.widget.updateMetrics(0.5, 0.8, 5)
        self.assertEqual(self.widget.loss_axis_x.max(), 10.0)
        
        # Update with epoch 12 (exceeds range)
        self.widget.updateMetrics(0.4, 0.85, 12)
        # Should auto-scale to 12 * 1.2 = 14.4
        self.assertAlmostEqual(self.widget.loss_axis_x.max(), 14.4, delta=0.01)
        self.assertAlmostEqual(self.widget.acc_axis_x.max(), 14.4, delta=0.01)

if __name__ == '__main__':
    unittest.main()
