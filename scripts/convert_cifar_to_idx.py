import os
import struct
import numpy as np

def convert_cifar_to_idx(cifar_dir, output_dir):
    """
    Converts CIFAR-10 binary files to IDX format.
    Uses IDX4 (Magic 2052) for Images: [Magic, Num, Depth, Height, Width] ??
    Standard IDX is: [Magic, Dim0, Dim1, ... DimN]
    
    Magic 0x0803 = 2051 (3 dims): N, H, W (Grayscale)
    Magic 0x0804 = 2052 (4 dims): N, C, H, W (Color)
    
    CIFAR-10 binary format:
    <1 x label><3072 x pixel>
    3072 pixels are 32x32 red, 32x32 green, 32x32 blue.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configuration provided by user path
    # D:\CGROOT\src\data\datasets\cifar-10
    
    # ---------------- Processing Test Data ----------------
    test_batch = os.path.join(cifar_dir, "test_batch.bin")
    if os.path.exists(test_batch):
        print(f"Processing {test_batch}...")
        process_batch(test_batch, 
                      os.path.join(output_dir, "test-images.idx4-ubyte"),
                      os.path.join(output_dir, "test-labels.idx1-ubyte"),
                      10000)
    else:
        print(f"Warning: {test_batch} not found.")

    # ---------------- Processing Training Data ----------------
    # CIFAR-10 has 5 training batches
    train_files = [os.path.join(cifar_dir, f"data_batch_{i}.bin") for i in range(1, 6)]
    existing_train_files = [f for f in train_files if os.path.exists(f)]
    
    if existing_train_files:
        print(f"Processing {len(existing_train_files)} training batches...")
        # Combine all training batches into one IDX file
        process_multi_batches(existing_train_files,
                              os.path.join(output_dir, "train-images.idx4-ubyte"),
                              os.path.join(output_dir, "train-labels.idx1-ubyte"),
                              10000 * len(existing_train_files))
    else:
        print("Warning: No training batches found.")

def process_batch(input_file, img_out_path, lbl_out_path, num_images):
    # CIFAR-10: 1 byte label, 3072 bytes image (CHW)
    
    img_data = []
    lbl_data = []
    
    with open(input_file, 'rb') as f:
        # Read all data
        # Each "row" is 3073 bytes
        buffer = f.read()
        
    # Convert to numpy for easier manipulation
    # Total bytes = num_images * 3073
    data = np.frombuffer(buffer, dtype=np.uint8)
    data = data.reshape(num_images, 3073)
    
    labels = data[:, 0]
    images = data[:, 1:]
    
    # CIFAR images are stored as CHW (Channel, Height, Width)
    # 3 * 32 * 32
    # We will write them as valid IDX4 data: N, C, H, W
    
    write_idx4_images(img_out_path, images, num_images, 3, 32, 32)
    write_idx1_labels(lbl_out_path, labels, num_images)

def process_multi_batches(input_files, img_out_path, lbl_out_path, total_images):
    all_images = []
    all_labels = []
    
    for input_file in input_files:
        with open(input_file, 'rb') as f:
            buffer = f.read()
            data = np.frombuffer(buffer, dtype=np.uint8)
            num_in_batch = len(data) // 3073
            data = data.reshape(num_in_batch, 3073)
            
            all_labels.append(data[:, 0])
            all_images.append(data[:, 1:])
            
    # Concatenate
    final_labels = np.concatenate(all_labels)
    final_images = np.concatenate(all_images)
    
    write_idx4_images(img_out_path, final_images, total_images, 3, 32, 32)
    write_idx1_labels(lbl_out_path, final_labels, total_images)

def write_idx4_images(filepath, data_matrix, num, depth, rows, cols):
    """
    Writes image data to IDX4 format.
    Header:
    Magic: 0x00000804 (2052)
    Dim0: Number of images
    Dim1: Depth
    Dim2: Rows
    Dim3: Cols
    Data: unsigned bytes
    """
    with open(filepath, 'wb') as f:
        # Big-endian magic number (2052) and dims
        f.write(struct.pack('>IIIII', 2052, num, depth, rows, cols))
        f.write(data_matrix.tobytes())
    print(f"Wrote {filepath} (IDX4, {num}x{depth}x{rows}x{cols})")

def write_idx1_labels(filepath, labels, num):
    """
    Writes label data to IDX1 format.
    Header:
    Magic: 0x00000801 (2049)
    Dim0: Number of items
    Data: unsigned bytes
    """
    with open(filepath, 'wb') as f:
        # Big-endian magic number (2049) and num labels
        f.write(struct.pack('>II', 2049, num))
        f.write(labels.tobytes())
    print(f"Wrote {filepath} (IDX1, {num} labels)")

if __name__ == "__main__":
    # Assuming standard project structure
    base_dir = r"D:\Education\Elec3\1st term\Software Engineering\Project\Repo\CGROOT\cifar-10-batches-bin"
    # Using the raw folder inside 'cifar-10-batches-bin' if it exists there, 
    # or the user might have put them in base_dir directly.
    # From previous context, user used: D:\nasser\cifar-10-batches-bin in convert_bin_to_ubyte.py
    # But generate_samples.py pointed to d:\CGROOT\src\data\datasets\cifar-10
    
    # We will look in base_dir first.
    input_dir = base_dir
    if not os.path.exists(os.path.join(input_dir, "test_batch.bin")):
         # Fallback check, maybe subfolder
         if os.path.exists(os.path.join(input_dir, "cifar-10-batches-bin", "test_batch.bin")):
             input_dir = os.path.join(input_dir, "cifar-10-batches-bin")
    
    convert_cifar_to_idx(input_dir, base_dir)
