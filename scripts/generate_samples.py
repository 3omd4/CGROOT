import sys
import os
import struct
import random
from PyQt6.QtGui import QImage, QColor, qRgb
from PyQt6.QtWidgets import QApplication

def generate_samples():
    # Setup paths
    base_dir = r"d:\CGROOT\src\data\datasets\cifar-10"
    # Use the new IDX4 file for images
    images_path = os.path.join(base_dir, "test-images.idx4-ubyte")
    labels_path = os.path.join(base_dir, "test-labels.idx1-ubyte")
    output_dir = os.path.join(base_dir, "samples")


    if not os.path.exists(images_path):
        print(f"Error: Dataset not found at {images_path}")
        print("Please run scripts/convert_cifar_to_idx.py first.")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Extracting samples from {images_path}...")
    
    # Read files
    with open(images_path, 'rb') as f_img, open(labels_path, 'rb') as f_lbl:
        # Parse headers
        # IDX formatted file
        # Magic: 4 bytes
        magic_img_bytes = f_img.read(4)
        magic_img = struct.unpack(">I", magic_img_bytes)[0]
        
        rows = 28
        cols = 28
        depth = 1
        
        if magic_img == 2051: # 0x0803: 3 Dims (N, H, W) -> Grayscale
            num_images, rows, cols = struct.unpack(">III", f_img.read(12))
            print(f"Detected IDX3 (Grayscale): {num_images} images, {rows}x{cols}")
        elif magic_img == 2052: # 0x0804: 4 Dims (N, D, H, W) -> Color/Volumetric
            num_images, depth, rows, cols = struct.unpack(">IIII", f_img.read(16))
            print(f"Detected IDX4 (Multi-channel): {num_images} images, {depth} channels, {rows}x{cols}")
            if depth != 3 and depth != 1:
                print(f"Warning: Depth is {depth}, expected 1 or 3 for standard display.")
        else:
            print(f"Unknown magic number: {magic_img}")
            return

        magic_lbl, num_labels = struct.unpack(">II", f_lbl.read(8))
        
        if num_images != num_labels:
            print(f"Error: Num images ({num_images}) != Num labels ({num_labels})")
            return

        # Determine number of samples to extract
        num_samples = 20
        indices = sorted(random.sample(range(num_images), num_samples))
        
        current_idx = 0
        img_size = rows * cols * depth
        
        for target_idx in indices:
            # Skip to target
            skip = target_idx - current_idx
            if skip > 0:
                f_img.seek(skip * img_size, 1) # 1 = seek from current
                f_lbl.seek(skip, 1)
                current_idx = target_idx
            
            # Read data
            img_data = f_img.read(img_size)
            label = struct.unpack("B", f_lbl.read(1))[0]
            current_idx += 1
            
            # Create QImage
            # CIFAR-10 is stored as Channels-First (CHW) in our IDX conversion
            # But QImage expects HWC (Interleaved) or usually specific formats.
            # Best to setPixel for correctness since format conversion is tricky.
            
            if depth == 1:
                image = QImage(cols, rows, QImage.Format.Format_Grayscale8)
                for y in range(rows):
                    for x in range(cols):
                        pixel_val = img_data[y * cols + x]
                        image.setPixel(x, y, qRgb(pixel_val, pixel_val, pixel_val))
            elif depth == 3:
                image = QImage(cols, rows, QImage.Format.Format_RGB888)
                # Data is CHW: [Red plane][Green plane][Blue plane]
                plane_size = rows * cols
                r_plane = img_data[0:plane_size]
                g_plane = img_data[plane_size:2*plane_size]
                b_plane = img_data[2*plane_size:3*plane_size]
                
                for y in range(rows):
                    for x in range(cols):
                        idx = y * cols + x
                        r = r_plane[idx]
                        g = g_plane[idx]
                        b = b_plane[idx]
                        image.setPixel(x, y, qRgb(r, g, b))
            
            # Mapping
            class_names = {
                0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 
                5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
            }
            class_name = class_names.get(label, "unknown")
            
            # Save
            filename = f"sample_{target_idx}_label_{class_name}.png"
            full_path = os.path.join(output_dir, filename)
            image.save(full_path)
            print(f"Saved {filename}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    generate_samples()
    print("Done!")
