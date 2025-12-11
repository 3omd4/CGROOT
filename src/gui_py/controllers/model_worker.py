from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread
from PyQt6.QtGui import QImage, qGray, qRgb
import sys
import time
import random
import math

try:
    import cgroot_core
except ImportError:
    cgroot_core = None

class ModelWorker(QObject):
    logMessage = pyqtSignal(str)
    metricsUpdated = pyqtSignal(float, float, int)  # loss, accuracy, epoch
    progressUpdated = pyqtSignal(int, int)
    imagePredicted = pyqtSignal(int, object, list) # int, QImage, list of floats
    trainingFinished = pyqtSignal()
    inferenceFinished = pyqtSignal()
    modelStatusChanged = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.dataset = None
        self.should_stop = False
        
    @pyqtSlot(str, str)
    def loadDataset(self, images_path, labels_path):
        if not cgroot_core:
            self.logMessage.emit("Error: Core library not loaded")
            return
            
        self.logMessage.emit(f"Loading dataset from: {images_path}")
        try:
            self.dataset = cgroot_core.MNISTLoader.load_training_data(images_path, labels_path)
            
            if self.dataset:
                self.logMessage.emit(f"Loaded {self.dataset.num_images} images")
            else:
                self.logMessage.emit("Failed to load dataset")
        except Exception as e:
            self.logMessage.emit(f"Exception loading dataset: {e}")
            import traceback
            traceback.print_exc()

    def _convert_mnist_to_image_format(self, flat_pixels, height=28, width=28):
        """Convert flat MNIST pixel array to [depth][height][width] format"""
        image_data = [[]]  # Single depth channel
        for y in range(height):
            row = []
            for x in range(width):
                idx = y * width + x
                if idx < len(flat_pixels):
                    row.append(int(flat_pixels[idx]))
                else:
                    row.append(0)
            image_data[0].append(row)
        return image_data

    def _calculate_loss_from_probs(self, probs, true_label):
        """Calculate cross-entropy loss from probabilities"""
        if not probs or len(probs) == 0:
            return 1.0
        
        # Cross-entropy loss: -log(p_true)
        true_prob = probs[true_label] if true_label < len(probs) else 0.0
        # Add small epsilon to avoid log(0)
        true_prob = max(true_prob, 1e-10)
        loss = -math.log(true_prob)
        return loss

    @pyqtSlot(dict)
    def trainModel(self, config):
        epochs = config.get('epochs', 10)
        
        if not self.dataset:
            self.logMessage.emit("No dataset loaded")
            self.trainingFinished.emit()
            return

        self.should_stop = False
        self.modelStatusChanged.emit(True)
        
        try:
            # Initialize model if not exists
            if not self.model:
                self.logMessage.emit("Initializing NNModel with Config...")
                arch = cgroot_core.architecture()
                
                # Clear vectors to ensure clean state
                arch.kernelsPerconvLayers = []
                arch.neuronsPerFCLayer = []
                arch.convLayerActivationFunc = []
                arch.FCLayerActivationFunc = []
                arch.convInitFunctionsType = []
                arch.FCInitFunctionsType = []
                arch.poolingLayersInterval = []
                arch.poolingtype = []
                arch.kernelsPerPoolingLayer = []
                
                # Dynamic Config
                arch.numOfConvLayers = config.get('num_conv_layers', 0)
                
                num_fc_layers = config.get('num_fc_layers', 2)
                neurons_list = config.get('neurons_per_fc_layer', [128, 10])
                
                # Sanity check: Ensure neurons list length matches num_fc_layers
                if len(neurons_list) != num_fc_layers:
                    self.logMessage.emit(f"Warning: Neurons list length ({len(neurons_list)}) does not match FC Layers count ({num_fc_layers}). Using list length.")
                    num_fc_layers = len(neurons_list)
                    
                arch.numOfFCLayers = num_fc_layers
                arch.neuronsPerFCLayer = neurons_list
                
                # Default activations and init functions
                # All FC layers use ReLU except output uses Softmax (handled by output layer)
                arch.FCLayerActivationFunc = [cgroot_core.activationFunction.RelU] * num_fc_layers
                arch.FCInitFunctionsType = [cgroot_core.initFunctions.Xavier] * num_fc_layers
                
                arch.distType = cgroot_core.distributionType.normalDistribution
                
                num_classes = config.get('num_classes', 10)
                img_h = config.get('image_height', 28)
                img_w = config.get('image_width', 28)
                
                # Optimizer Config
                opt_type_str = config.get('optimizer', 'Adam')
                lr = config.get('learning_rate', 0.001)
                wd = config.get('weight_decay', 0.0001)
                mom = config.get('momentum', 0.9)
                
                # Setup OptimizerConfig
                arch.optConfig.learningRate = lr
                arch.optConfig.weightDecay = wd
                arch.optConfig.momentum = mom
                arch.optConfig.beta1 = 0.9
                arch.optConfig.beta2 = 0.999
                arch.optConfig.epsilon = 1e-8
                
                if opt_type_str == "Adam":
                    arch.optConfig.type = cgroot_core.OptimizerType.Adam
                elif opt_type_str == "RMSprop" or opt_type_str == "RMSProp":
                    arch.optConfig.type = cgroot_core.OptimizerType.RMSprop
                else:
                    arch.optConfig.type = cgroot_core.OptimizerType.SGD

                # Set legacy learningRate for compatibility
                arch.learningRate = lr

                self.model = cgroot_core.NNModel(arch, num_classes, img_h, img_w, 1)
                self.logMessage.emit(f"Model initialized (MLP: Input->{neurons_list}) with {opt_type_str}, LR={lr}")

            # Training Loop
            num_images = self.dataset.num_images
            total_epochs = epochs
            batch_size = config.get('batch_size', 32)
            validation_split = config.get('validation_split', 0.2)
            use_validation = config.get('use_validation', True)
            
            # Split dataset if validation is enabled
            if use_validation and validation_split > 0:
                val_size = int(num_images * validation_split)
                train_size = num_images - val_size
                self.logMessage.emit(f"Using {train_size} training samples, {val_size} validation samples")
            else:
                train_size = num_images
                val_size = 0
            
            for epoch in range(total_epochs):
                if self.should_stop: 
                    break
                
                self.logMessage.emit(f"Epoch {epoch+1}/{total_epochs}")
                
                # Training phase
                train_correct = 0
                train_loss_sum = 0.0
                train_samples = 0
                
                # Create indices and shuffle them
                all_indices = list(range(num_images))
                random.shuffle(all_indices)
                
                train_indices = all_indices[:train_size]
                val_indices = all_indices[train_size:] if use_validation else []
                
                # Training loop
                batch_images = []
                batch_labels = []
                
                for i, idx in enumerate(train_indices):
                    if self.should_stop: 
                        break
                    
                    img_obj = self.dataset.images[idx]
                    flat = img_obj.pixels
                    
                    # Convert to proper image format [depth][height][width]
                    image_data = self._convert_mnist_to_image_format(flat, 28, 28)
                    label = int(img_obj.label)
                    
                    # Add to batch
                    batch_images.append(image_data)
                    batch_labels.append(label)
                    
                    # Train if batch full or last element
                    if len(batch_images) >= batch_size or i == len(train_indices) - 1:
                        if len(batch_images) > 0:
                            if len(batch_images) > 1:
                                self.model.train_batch(batch_images, batch_labels)
                            else:
                                self.model.train(batch_images[0], batch_labels[0])
                            
                            # Calculate accuracy and loss on this batch AFTER training
                            for img, lbl in zip(batch_images, batch_labels):
                                pred = self.model.classify(img)
                                if pred == lbl:
                                    train_correct += 1
                                
                                # Calculate loss from probabilities
                                probs = self.model.getProbabilities()
                                if probs:
                                    train_loss_sum += self._calculate_loss_from_probs(probs, lbl)
                                    train_samples += 1
                            
                            batch_images = []
                            batch_labels = []
                        
                    # Progress update every 1000 samples
                    if (i + 1) % 1000 == 0:
                        QThread.msleep(1)  # Allow GUI to update
                        progress = int((i + 1) / len(train_indices) * 100)
                        self.progressUpdated.emit(progress, 100)
                        
                        # Visualization
                        try:
                            q_img = QImage(28, 28, QImage.Format.Format_Grayscale8)
                            for y in range(28):
                                for x in range(28):
                                    val = flat[y*28 + x] if y*28 + x < len(flat) else 0
                                    q_img.setPixel(x, y, qRgb(val, val, val))
                            self.imagePredicted.emit(label, q_img, [])
                        except Exception as e:
                            pass  # Silently ignore visualization errors
                
                # Calculate training metrics
                train_acc = train_correct / train_size if train_size > 0 else 0.0
                train_loss = train_loss_sum / train_samples if train_samples > 0 else 1.0 - train_acc
                
                # Validation phase
                val_acc = 0.0
                val_loss = 0.0
                if use_validation and len(val_indices) > 0:
                    val_correct = 0
                    val_loss_sum = 0.0
                    val_samples = 0
                    
                    for idx in val_indices:
                        if self.should_stop:
                            break
                        
                        img_obj = self.dataset.images[idx]
                        flat = img_obj.pixels
                        image_data = self._convert_mnist_to_image_format(flat, 28, 28)
                        label = int(img_obj.label)
                        
                        # Predict without training
                        pred = self.model.classify(image_data)
                        if pred == label:
                            val_correct += 1
                        
                        # Calculate loss
                        probs = self.model.getProbabilities()
                        if probs:
                            val_loss_sum += self._calculate_loss_from_probs(probs, label)
                            val_samples += 1
                    
                    val_acc = val_correct / len(val_indices) if len(val_indices) > 0 else 0.0
                    val_loss = val_loss_sum / val_samples if val_samples > 0 else 1.0 - val_acc
                    
                    # Use validation metrics for display
                    current_acc = val_acc
                    current_loss = val_loss
                    self.logMessage.emit(f"Epoch {epoch+1} - Train: Acc={train_acc*100:.2f}%, Loss={train_loss:.4f} | Val: Acc={val_acc*100:.2f}%, Loss={val_loss:.4f}")
                else:
                    # Use training metrics
                    current_acc = train_acc
                    current_loss = train_loss
                    self.logMessage.emit(f"Epoch {epoch+1} - Train: Acc={train_acc*100:.2f}%, Loss={train_loss:.4f}")
                
                # Emit metrics (use validation if available, otherwise training)
                self.metricsUpdated.emit(current_loss, current_acc, epoch+1)
                self.progressUpdated.emit(epoch+1, total_epochs)
                
        except Exception as e:
            self.logMessage.emit(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            
        self.modelStatusChanged.emit(False)
        self.trainingFinished.emit()

    @pyqtSlot()
    def stopTraining(self):
        self.should_stop = True

    @pyqtSlot(object)
    def runInference(self, qimage_obj):
        if not self.model:
            self.logMessage.emit("No model available")
            self.inferenceFinished.emit()
            return

        if not qimage_obj:
            self.logMessage.emit("No image provided for inference")
            self.inferenceFinished.emit()
            return
            
        self.modelStatusChanged.emit(True)
        try:
            # Convert QImage to 3D vector [1][28][28]
            img = qimage_obj.scaled(28, 28).convertToFormat(QImage.Format.Format_Grayscale8)
            
            width = img.width()
            height = img.height()
            
            rows = []
            for y in range(height):
                row = []
                for x in range(width):
                    pixel_val = qGray(img.pixel(x, y))
                    row.append(pixel_val)
                rows.append(row)
            
            image_data = [rows]  # Depth 1
            
            # Run Classification
            self.logMessage.emit("Running inference on image...")
            predicted_class = self.model.classify(image_data)
            
            self.logMessage.emit(f"Inference Result: {predicted_class}")
            
            # Get probabilities from model
            probs = self.model.getProbabilities()
            if not probs or len(probs) == 0:
                probs = [0.0] * 10
                probs[predicted_class] = 1.0
            
            confidence = max(probs) if probs else 0.0
            self.logMessage.emit(f"Inference Confidence: {confidence:.4f}")
            
            self.imagePredicted.emit(predicted_class, qimage_obj, probs)
            
        except Exception as e:
            self.logMessage.emit(f"Inference error: {e}")
            import traceback
            traceback.print_exc()

        self.modelStatusChanged.emit(False)
        self.inferenceFinished.emit()
