import sys
import os
import struct
import random
from PyQt6.QtGui import QImage, QColor, qRgb
from PyQt6.QtWidgets import QApplication

def generate_samples():
    # Setup paths
    base_dir = r"d:\CGROOT\src\data\datasets\fashion-mnist"
    images_path = os.path.join(base_dir, "t10k-images.idx3-ubyte")
    labels_path = os.path.join(base_dir, "t10k-labels.idx1-ubyte")
    output_dir = os.path.join(base_dir, "samples")
    
    if not os.path.exists(images_path):
        print(f"Error: Dataset not found at {images_path}")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Extracting samples from {images_path}...")
    
    # Read files
    with open(images_path, 'rb') as f_img, open(labels_path, 'rb') as f_lbl:
        # Parse headers
        magic_img, num_images, rows, cols = struct.unpack(">IIII", f_img.read(16))
        magic_lbl, num_labels = struct.unpack(">II", f_lbl.read(8))
        
        # Determine number of samples to extract
        num_samples = 20
        indices = sorted(random.sample(range(num_images), num_samples))
        
        current_idx = 0
        for target_idx in indices:
            # Skip to target
            skip = target_idx - current_idx
            if skip > 0:
                f_img.seek(skip * rows * cols, 1) # 1 = seek from current
                f_lbl.seek(skip, 1)
                current_idx = target_idx
            
            # Read data
            img_data = f_img.read(rows * cols)
            label = struct.unpack("B", f_lbl.read(1))[0]
            current_idx += 1
            
            # Create QImage
            image = QImage(cols, rows, QImage.Format.Format_Grayscale8)
            for y in range(rows):
                for x in range(cols):
                    pixel_val = img_data[y * cols + x]
                    # Invert colors? MNIST is white on black. 
                    # Usually for inference we might want to keep it valid.
                    # QImage Grayscale8: 0=black, 255=white.
                    # QImage.setPixelColor is slow, set via bytes if possible or per pixel
                    image.setPixel(x, y, qRgb(pixel_val, pixel_val, pixel_val))
            
            # Save
            filename = f"digit_{label}_idx{target_idx}.png"
            full_path = os.path.join(output_dir, filename)
            image.save(full_path)
            print(f"Saved {filename}")

if __name__ == "__main__":
    app = QApplication(sys.argv) # Needed for QImage sometimes or good practice
    generate_samples()
    print("Done!")
