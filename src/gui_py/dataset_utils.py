
MNIST_CLASSES = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9"
}

FASHION_MNIST_CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

CIFAR_CLASSES = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

def get_class_name(dataset_type, index):
    if hasattr(dataset_type, 'lower'):
        dt = dataset_type.lower()
        if "fashion" in dt:
            return FASHION_MNIST_CLASSES.get(index, str(index))
        elif "mnist" in dt:  # Standard MNIST
            return MNIST_CLASSES.get(index, str(index))
        elif "cifar" in dt:  # Standard CIFAR-10
            return CIFAR_CLASSES.get(index, str(index))
    
    # Default fallback
    return str(index)
