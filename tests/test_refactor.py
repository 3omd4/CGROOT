import sys
import unittest
# Add build output to path or ensure bindings are available
# Assuming running from d:\CGROOT
sys.path.append('build/lib') 
# Also try site-packages or wherever it builds
sys.path.append('src/gui_py')

try:
    import cgroot_core
except ImportError:
    # If not found, try to locate it or fail
    print("Could not import cgroot_core")
    # For now, we assume build puts it in a discoverable place or we might need to adjust
    pass

class TestBindingsRefactor(unittest.TestCase):
    def test_create_model_valid(self):
        config = {
            'num_conv_layers': 0,
            'num_fc_layers': 2,
            'neurons_per_fc_layer': [128, 10],
            'num_classes': 10,
            'image_height': 28,
            'image_width': 28,
            'optimizer': 'Adam',
            'learning_rate': 0.001
        }
        model = cgroot_core.create_model(config)
        self.assertIsNotNone(model)
        # Check if we can call a method
        probs = model.getProbabilities()
        self.assertTrue(len(probs) == 0 or len(probs) == 10) # initially might be empty or initialized

    def test_create_model_invalid(self):
        config = {
            'num_fc_layers': 2,
            'neurons_per_fc_layer': [128], # Mismatch
        }
        with self.assertRaises(RuntimeError):
            cgroot_core.create_model(config)

    def test_create_model_cnn(self):
        # Crash reproduction config
        config = {
            'num_conv_layers': 2,
            'kernels_per_layer': [6, 16],
            'kernel_dims': [(5, 5), (5, 5)],
            'pooling_intervals': [2, 2], 
            'pooling_type': 'Max',
            'num_fc_layers': 2,
            'neurons_per_fc_layer': [64, 10],
            'num_classes': 10,
            'image_height': 28,
            'image_width': 28,
            'optimizer': 'Adam',
            'learning_rate': 0.001
        }
        print(f"Testing config: {config}")
        model = cgroot_core.create_model(config)
        self.assertIsNotNone(model)
        print("Model created successfully")

        
    def test_classify_pixels(self):
        # Create a dummy model
        config = {'num_classes': 10, 'neurons_per_fc_layer': [64, 10]}
        model = cgroot_core.create_model(config)
        
        # Create dummy image buffer [28*28]
        width = 28
        height = 28
        stride = 28
        data = bytearray([0] * (width * height))
        
        # Set some pixels
        data[100] = 255
        
        # Classify
        result = cgroot_core.classify_pixels(model, data, width, height, stride)
        self.assertTrue(0 <= result < 10)

if __name__ == '__main__':
    # Need to make sure cgroot_core is imported successfully
    if 'cgroot_core' not in sys.modules:
        print("Skipping tests because cgroot_core module not found")
    else:
        unittest.main()
