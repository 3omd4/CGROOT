import sys
import os
import struct
import random
import argparse
from PyQt6.QtGui import QImage, QColor, qRgb
from PyQt6.QtWidgets import QApplication

# Class Mappings
MNIST_CLASSES = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9"
}

FASHION_MNIST_CLASSES = {
    0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

CIFAR_CLASSES = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

def get_class_name(label, dataset_type):
    if dataset_type == "fashion":
        return FASHION_MNIST_CLASSES.get(label, str(label))
    elif dataset_type == "cifar":
        return CIFAR_CLASSES.get(label, str(label))
    elif dataset_type == "mnist":
        return MNIST_CLASSES.get(label, str(label))
    return str(label)

def detect_dataset_type(path):
    lower_path = path.lower()
    if "fashion" in lower_path:
        return "fashion"
    elif "cifar" in lower_path:
        return "cifar"
    elif "mnist" in lower_path:
        return "mnist"
    return "unknown"

def generate_samples(images_path, labels_path, output_dir, num_samples):
    if not os.path.exists(images_path):
        print(f"Error: Images file not found at {images_path}")
        return

    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found at {labels_path}")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    dataset_type = detect_dataset_type(images_path)
    print(f"Detected dataset type: {dataset_type}")
    print(f"Extracting samples from {images_path}...")
    
    # Read files
    with open(images_path, 'rb') as f_img, open(labels_path, 'rb') as f_lbl:
        # Parse headers
        magic_img_bytes = f_img.read(4)
        magic_img = struct.unpack(">I", magic_img_bytes)[0]
        
        rows = 28
        cols = 28
        depth = 1
        num_images = 0
        
        if magic_img == 2051: # 0x0803: 3 Dims (N, H, W) -> Grayscale
            num_images, rows, cols = struct.unpack(">III", f_img.read(12))
            print(f"Detected IDX3 (Grayscale): {num_images} images, {rows}x{cols}")
        elif magic_img == 2052: # 0x0804: 4 Dims (N, D, H, W) -> Color/Volumetric
            num_images, depth, rows, cols = struct.unpack(">IIII", f_img.read(16))
            print(f"Detected IDX4 (Multi-channel): {num_images} images, {depth} channels, {rows}x{cols}")
        else:
            print(f"Unknown magic number: {magic_img}")
            return
            
        # Labels header
        magic_lbl, num_labels = struct.unpack(">II", f_lbl.read(8))
        
        if num_images != num_labels:
            print(f"Error: Num images ({num_images}) != Num labels ({num_labels})")
            return

        # Determine number of samples to extract
        indices = sorted(random.sample(range(num_images), min(num_samples, num_images)))
        
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
            image = None
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
            
            if image:
                class_name = get_class_name(label, dataset_type)
                filename = f"sample_{target_idx}_label_{class_name}.png"
                full_path = os.path.join(output_dir, filename)
                image.save(full_path)
                print(f"Saved {filename}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    parser = argparse.ArgumentParser(description="Generate sample images from IDX dataset files.")
    
    # Default paths (try to be smart about defaults or require args)
    # Keeping defaults for backward compat/ease of use if run from root
    default_img = r"d:\CGROOT\src\data\datasets\cifar-10\test-images.idx4-ubyte"
    default_lbl = r"d:\CGROOT\src\data\datasets\cifar-10\test-labels.idx1-ubyte"
    
    parser.add_argument("--images", default=default_img, help="Path to images IDX file")
    parser.add_argument("--labels", default=default_lbl, help="Path to labels IDX file")
    parser.add_argument("--output", default="samples", help="Output directory for sample images")
    parser.add_argument("--count", type=int, default=20, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Resolve output path relative to images file if it's just a name
    out_dir = args.output
    if not os.path.isabs(out_dir):
        # Place next to images file by default if not absolute
        parent = os.path.dirname(args.images)
        out_dir = os.path.join(parent, args.output)
        
    generate_samples(args.images, args.labels, out_dir, args.count)
    print("Done!")
