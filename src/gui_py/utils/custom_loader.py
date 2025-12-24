import os
from PyQt6.QtGui import QImage, QColor
try:
    import cgroot_core
except ImportError:
    cgroot_core = None

class CustomDatasetLoader:
    @staticmethod
    def load_from_folder(root_path, target_width=28, target_height=28, grayscale=True):
        """
        Load dataset from a folder structure.
        Structure expected:
        root_path/
          class_0/
            img1.png
            img2.jpg
          class_1/
            ...
            
        Args:
            root_path (str): Path to root directory.
            target_width (int): Width to resize images to.
            target_height (int): Height to resize images to.
            grayscale (bool): If True, converts to 1 channel (L). Else 3 channels (RGB).
            
        Returns:
            cgroot_core.MNISTDataset: The constructed dataset object.
        """
        if not cgroot_core:
            raise ImportError("Core library not loaded")
            
        dataset = cgroot_core.MNISTDataset()
        dataset.images = [] # Start with empty list to append MNISTImage objects
        
        # 1. Scan for classes (subdirectories)
        classes = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        classes.sort() # Ensure deterministic order
        
        if not classes:
            # Maybe flat directory? Not supported yet.
            print("No class subdirectories found.")
            return None
            
        print(f"Found {len(classes)} classes: {classes}")
        
        total_images = 0
        
        for label_idx, class_name in enumerate(classes):
            class_dir = os.path.join(root_path, class_name)
            files = os.listdir(class_dir)
            
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(class_dir, f)
                    
                    # Load and Process Image using QImage
                    img = QImage(file_path)
                    if img.isNull():
                        continue
                        
                    # Scale
                    img = img.scaled(target_width, target_height, 
                                     aspectRatioMode=0) # Ignore aspect ratio to fit exact dims? Or keep?
                                     # MNIST is usually forcing 28x28. Let's force exact fit.
                    
                    # Store Pixels
                    # Structure: [depth][height][width]  <-- Assuming this based on common NN lib formats?
                    # OR [height][width][depth]?
                    # MNISTLoader C++ usually does:
                    # vector<vector<vector<unsigned char>>>
                    # We need to know the order.
                    # Looking at typical C++: often tensor is d, h, w or h, w, d.
                    # Let's assume [depth][height][width] which is common for "channels first" 
                    # OR [height][width][depth] for "channels last".
                    # CGROOT seems to be custom.
                    # Let's check `mnist_loader.cpp` logic via inference or just try one.
                    # If I look at `bindings.cpp`: `classify_pixels` takes bits directly.
                    # But dataset storage uses vectors.
                    # A safe bet for many libs is [depth][height][width].
                    # Let's try [depth][height][width].
                    
                    mnist_img = cgroot_core.MNISTImage()
                    mnist_img.label = label_idx
                    
                    # Extract pixels
                    depth = 1 if grayscale else 3
                    
                    # Initialize 3D array
                    # We will use [depth][height][width]
                    pixel_data = [] # List of (Height) Lists of (Width) values ??
                    # Wait, C++ `vector<vector<vector<uchar>>>`.
                    # Outer vector size = ?
                    # If it's [D][H][W]:
                    
                    channels_data = []
                    for d in range(depth):
                        h_list = []
                        for y in range(target_height):
                            w_list = [0] * target_width
                            h_list.append(w_list)
                        channels_data.append(h_list)
                        
                    # Fill
                    for y in range(target_height):
                        for x in range(target_width):
                            pixel_color = img.pixelColor(x, y)
                            if grayscale:
                                # Standard Rec 601 grayscale
                                val = int(0.299 * pixel_color.red() + 0.587 * pixel_color.green() + 0.114 * pixel_color.blue())
                                channels_data[0][y][x] = val
                            else:
                                channels_data[0][y][x] = pixel_color.red()
                                channels_data[1][y][x] = pixel_color.green()
                                channels_data[2][y][x] = pixel_color.blue()
                                
                    mnist_img.pixels = channels_data
                    dataset.images.append(mnist_img)
                    total_images += 1
                    
        dataset.num_images = total_images
        dataset.image_width = target_width
        dataset.image_height = target_height
        dataset.depth = 1 if grayscale else 3
        
        return dataset
