#include "model.h"
#include "utils/store_and_load.h"
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace std;

// the NNModel constructor
// input:        -modelArch (contains all the necessary information about the
// architecture of the model)
//               -numOfClasses (the number of classes of the data set, used to
//               construct the output layer) -imageheight -imageWidth
//               -imageDepth
// output:       N/A
// side effect:  the model is constructed
// Note:         -the constructor makes an array of the layers by making used of
// class inheretence
//               -specify the initialization function, type of distribution and
//               the activation function of each layer (convolution or fully
//               connected) -the height and width of the image is constant for a
//               single model(object) and so is the architecture of a single
//               model (object)
NNModel::NNModel(architecture modelArch, size_t numOfClasses,
                 size_t imageHeight, size_t imageWidth, size_t imageDepth) {

  // ========== PARAMETER VALIDATION ==========
  // Validate basic parameters
  if (imageHeight == 0 || imageWidth == 0 || imageDepth == 0) {
    throw std::invalid_argument(
        "Image dimensions must be greater than 0. Got: " +
        std::to_string(imageHeight) + "x" + std::to_string(imageWidth) + "x" +
        std::to_string(imageDepth));
  }

  if (numOfClasses == 0) {
    throw std::invalid_argument("Number of classes must be greater than 0");
  }

  // Validate FC layer configuration
  if (modelArch.numOfFCLayers > 0) {
    if (modelArch.neuronsPerFCLayer.size() != modelArch.numOfFCLayers) {
      throw std::invalid_argument(
          "FC layer count mismatch: numOfFCLayers=" +
          std::to_string(modelArch.numOfFCLayers) +
          " but neuronsPerFCLayer.size()=" +
          std::to_string(modelArch.neuronsPerFCLayer.size()));
    }

    if (modelArch.FCLayerActivationFunc.size() != modelArch.numOfFCLayers) {
      throw std::invalid_argument(
          "FC activation function count mismatch: numOfFCLayers=" +
          std::to_string(modelArch.numOfFCLayers) +
          " but FCLayerActivationFunc.size()=" +
          std::to_string(modelArch.FCLayerActivationFunc.size()));
    }

    if (modelArch.FCInitFunctionsType.size() != modelArch.numOfFCLayers) {
      throw std::invalid_argument(
          "FC init function count mismatch: numOfFCLayers=" +
          std::to_string(modelArch.numOfFCLayers) +
          " but FCInitFunctionsType.size()=" +
          std::to_string(modelArch.FCInitFunctionsType.size()));
    }

    // Validate neuron counts are positive
    for (size_t i = 0; i < modelArch.neuronsPerFCLayer.size(); i++) {
      if (modelArch.neuronsPerFCLayer[i] == 0) {
        throw std::invalid_argument("FC layer " + std::to_string(i) +
                                    " has 0 neurons");
      }
    }
  }

  // Validate Conv layer configuration if present
  if (modelArch.numOfConvLayers > 0) {
    if (modelArch.kernelsPerconvLayers.size() != modelArch.numOfConvLayers) {
      throw std::invalid_argument(
          "Conv layer count mismatch: numOfConvLayers=" +
          std::to_string(modelArch.numOfConvLayers) +
          " but kernelsPerconvLayers.size()=" +
          std::to_string(modelArch.kernelsPerconvLayers.size()));
    }
  }
  // ========== END VALIDATION ==========

  // Debug logging to understand the architecture before model creation
  // // std::cout << "================ Architecture Debug Info ================"
  //           << std::endl;
  // // std::cout << "Number of Conv Layers: " << modelArch.numOfConvLayers
  //           << std::endl;
  // // std::cout << "Number of FC Layers: " << modelArch.numOfFCLayers <<
  // std::endl; // std::cout << "Neurons per FC Layer: " <<
  // modelArch.neuronsPerFCLayer.size()
  //           << std::endl;
  // // std::cout << "Convolutional layers kernels: "
  //           << modelArch.kernelsPerconvLayers.size() << std::endl;
  // // std::cout << "Fully connected layers neurons: "
  //           << modelArch.neuronsPerFCLayer.size() << std::endl;
  // // std::cout << "Convolutional layers activation functions: "
  //           << modelArch.convLayerActivationFunc.size() << std::endl;
  // // std::cout << "Fully connected layers activation functions: "
  //           << modelArch.FCLayerActivationFunc.size() << std::endl;
  // // std::cout << "Convolutional layers init functions: "
  //           << modelArch.convInitFunctionsType.size() << std::endl;
  // // std::cout << "Fully connected layers init functions: "
  //           << modelArch.FCInitFunctionsType.size() << std::endl;
  // // std::cout << "Pooling layers interval: "
  //           << modelArch.poolingLayersInterval.size() << std::endl;
  // // std::cout << "Pooling layers type: ";
  // for (size_t i = 0; i < modelArch.poolingtype.size(); i++) {
  //   // std::cout << static_cast<int>(modelArch.poolingtype[i]);
  //   if (i < modelArch.poolingtype.size() - 1)
  //     // std::cout << ", ";
  // }
  // // std::cout << std::endl;
  // // std::cout << "Pooling layers kernels: " << std::endl;
  // for (size_t i = 0; i < modelArch.kernelsPerPoolingLayer.size(); i++) {
  //   // std::cout << "depth X height X width: ";
  //   // std::cout << modelArch.kernelsPerPoolingLayer[i].filter_depth;
  //   // std::cout << "x";
  //   // std::cout << modelArch.kernelsPerPoolingLayer[i].filter_height;
  //   // std::cout << "x";
  //   // std::cout << modelArch.kernelsPerPoolingLayer[i].filter_width;
  //   if (i < modelArch.kernelsPerPoolingLayer.size() - 1)
  //     // std::cout << ",\n";
  // }
  // // std::cout << std::endl;
  // // std::cout << "Optimizer epsilon: " << modelArch.optConfig.epsilon
  //           << std::endl;
  // // std::cout <<
  // "============================================================="
  //           << std::endl;

  // // std::cout << "zxsaceq  wq wq q 123" << std::endl;

  try {
    // construct the input layer
    Layers.emplace_back(new inputLayer(imageHeight, imageWidth, imageDepth));
  } catch (const std::exception &e) {
    std::cerr << "ERROR creating input layer: " << e.what() << std::endl;
    throw;
  }

  // // std::cout << "cascasccs123" << std::endl;

  // Track current feature map dimensions
  size_t currentHeight = imageHeight;
  size_t currentWidth = imageWidth;
  size_t currentDepth = imageDepth;

  // poolCount is used to count the number of convolution layers after which
  // a pooling layer will be inserted
  size_t poolCount = 0;

  // an iterator that is used to iterate the pooling layers information vectors
  size_t poolIter = 0;

  // // std::cout << "cascasccs456" << std::endl;

  try {
    // start initializing the convolution and pooling layers
    // a pooling layer is inserted after a number of convolution layers
    for (size_t i = 0; i < modelArch.numOfConvLayers; i++) {
      // Calculate feature map dimensions for this conv layer

      // // std::cout << "cascasccs789" << std::endl;

      featureMapDim fmDim =
          calcFeatureMapDim(modelArch.kernelsPerconvLayers[i].kernel_height,
                            modelArch.kernelsPerconvLayers[i].kernel_width,
                            currentHeight, currentWidth);

      // Set kernel depth to match input depth
      modelArch.kernelsPerconvLayers[i].kernel_depth = currentDepth;
      fmDim.FM_depth = modelArch.kernelsPerconvLayers[i].numOfKerenels;

      // // std::cout << "cascasccs980" << std::endl;

      try {
        // Create convolution layer
        Layers.emplace_back(new convLayer(modelArch.kernelsPerconvLayers[i],
                                          modelArch.convLayerActivationFunc[i],
                                          modelArch.convInitFunctionsType[i],
                                          modelArch.distType, fmDim,
                                          modelArch.optConfig));
      } catch (const std::exception &e) {
        std::cerr << "ERROR creating convolution layer " << (i + 1) << ": "
                  << e.what() << std::endl;
        throw;
      }

      // // std::cout << "cascasccs981" << std::endl;

      // Update current dimensions
      currentHeight = fmDim.FM_height;
      currentWidth = fmDim.FM_width;
      currentDepth = fmDim.FM_depth;

      poolCount++;

      // Check if we need to insert a pooling layer
      if (poolIter < modelArch.poolingLayersInterval.size() &&
          poolCount >= modelArch.poolingLayersInterval[poolIter]) {
        // Calculate pooling output dimensions
        poolKernel &poolKern = modelArch.kernelsPerPoolingLayer[poolIter];
        poolKern.filter_depth = currentDepth;

        size_t poolOutHeight =
            (currentHeight - poolKern.filter_height) / poolKern.stride + 1;
        size_t poolOutWidth =
            (currentWidth - poolKern.filter_width) / poolKern.stride + 1;

        featureMapDim poolFMDim{poolOutHeight, poolOutWidth, currentDepth};

        try {
          // Create pooling layer
          Layers.emplace_back(new poolingLayer(
              poolKern, poolFMDim, modelArch.poolingtype[poolIter]));
        } catch (const std::exception &e) {
          std::cerr << "ERROR creating pooling layer " << (poolIter + 1) << ": "
                    << e.what() << std::endl;
          throw;
        }

        // Update current dimensions
        currentHeight = poolOutHeight;
        currentWidth = poolOutWidth;
        // Depth remains the same

        poolCount = 0;
        poolIter++;
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "ERROR creating convolution and pooling layers: " << e.what()
              << std::endl;
    throw;
  }

  // Determine if we need a flatten layer
  bool needsFlatten =
      (modelArch.numOfConvLayers > 0) ||
      (modelArch.numOfFCLayers > 0 &&
       (imageHeight != 1 || imageWidth != 1 || imageDepth != 1));

  if (needsFlatten) {
    try {
      Layers.emplace_back(
          new FlattenLayer(currentHeight, currentWidth, currentDepth));
    } catch (const std::exception &e) {
      std::cerr << "ERROR creating flatten layer: " << e.what() << std::endl;
      throw;
    }
  }

  // Build fully connected layers
  size_t fcInputSize = needsFlatten
                           ? currentHeight * currentWidth * currentDepth
                           : imageHeight * imageWidth * imageDepth;

  // // std::cout << "Building FC layers. Initial fcInputSize: " << fcInputSize
  //           << std::endl;
  // // std::cout << "Number of FC layers to create: " <<
  // modelArch.numOfFCLayers
  //           << std::endl;

  for (size_t i = 0; i < modelArch.numOfFCLayers; i++) {
    try {
      // // std::cout << "Creating FC layer " << (i + 1) << "/"
      //           << modelArch.numOfFCLayers << std::endl;
      // // std::cout << "  Input size: " << fcInputSize << std::endl;
      // // std::cout << "  Output size (neurons): " <<
      // modelArch.neuronsPerFCLayer[i]
      //           << std::endl;
      // // std::cout << "  Activation: " << modelArch.FCLayerActivationFunc[i]
      //           << std::endl;
      // // std::cout << "  Init function: " << modelArch.FCInitFunctionsType[i]
      //           << std::endl;

      Layers.emplace_back(new FullyConnected(
          modelArch.neuronsPerFCLayer[i], modelArch.FCLayerActivationFunc[i],
          modelArch.FCInitFunctionsType[i], modelArch.distType, fcInputSize,
          modelArch.optConfig));

      // // std::cout << "  FC layer " << (i + 1) << " created successfully"
      //           << std::endl;
      fcInputSize = modelArch.neuronsPerFCLayer[i];
    } catch (const std::exception &e) {
      std::cerr << "ERROR creating FC layer " << (i + 1) << ": " << e.what()
                << std::endl;
      throw;
    }
  }

  // Create output layer
  try {
    // // std::cout << "Creating output layer" << std::endl;
    // // std::cout << "  Output layer input size: " << fcInputSize <<
    // std::endl;
    // // std::cout << "  Number of classes: " << numOfClasses << std::endl;

    Layers.emplace_back(new outputLayer(
        numOfClasses, fcInputSize, modelArch.distType, modelArch.optConfig));

    // // std::cout << "  Output layer created successfully" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "ERROR creating output layer: " << e.what() << std::endl;
    std::cerr << "  fcInputSize was: " << fcInputSize << std::endl;
    std::cerr << "  numOfClasses was: " << numOfClasses << std::endl;
    throw;
  }
}

// Helper function to calculate cross-entropy loss from probabilities
static double calculate_loss_from_probs(const vector<double> &probs,
                                        int true_label) {
  if (probs.empty() || true_label >= static_cast<int>(probs.size())) {
    return 1.0;
  }

  // Cross-entropy loss: -log(p_true)
  double true_prob = probs[true_label];
  // Add small epsilon to avoid log(0)
  true_prob = std::max(true_prob, 1e-10);
  double loss = -std::log(true_prob);
  return loss;
}

// Helper function to convert MNIST flat pixels to image format
// [depth][height][width]
static image convert_mnist_to_image_format(const vector<uint8_t> &flat_pixels,
                                           size_t height = 28,
                                           size_t width = 28) {
  image image_data(1); // Single depth channel
  for (size_t y = 0; y < height; y++) {
    vector<unsigned char> row;
    for (size_t x = 0; x < width; x++) {
      size_t idx = y * width + x;
      if (idx < flat_pixels.size()) {
        row.push_back(static_cast<unsigned char>(flat_pixels[idx]));
      } else {
        row.push_back(0);
      }
    }
    image_data[0].push_back(row);
  }
  return image_data;
}

// NNModel constructor helper function:
// calculates the dimension of the output feature map of each convolution layer
// input:        -current layer kernel height (size_t kernelHeight)
//               -current layer kernel width (size_t kernelWidth)
//               -the input feature map height (size_t inputHeight)
//               -the input feature map width (size_t inputWidth)
// output:       -a featureMapDim struct that carries information about the
// dimensions of the current
//               output feature map (featureMapDim)
// side effect:  N/A
// Note:         the function also sets the data member featureMapDim.FM_depth
// to 0, so it must
//               setted later
featureMapDim NNModel::calcFeatureMapDim(size_t kernelHeight,
                                         size_t kernelWidth, size_t inputHeight,
                                         size_t inputWidth) {
  size_t outputHeight = inputHeight - kernelHeight + 1;
  size_t outputWidth = inputWidth - kernelWidth + 1;
  return featureMapDim{outputHeight, outputWidth, 0};
}

// train the model with a single image
// input:        -data (an image)
//               -trueOutput (the value for which the model compares its
//               output)
// output:       N/A
// side effect:  The model is trained by a single image and its paramters
// are updated Note:         N/A
// Updated train: Returns {loss, is_correct}
std::pair<double, int> NNModel::train(const image &imgData, int trueOutput) {
  this->data = imgData;

  // 1. Forward Pass
  int predictedClass = classify(imgData);

  // 2. Calculate Metrics (Zero overhead: uses existing forward pass result)
  int is_correct = (predictedClass == trueOutput) ? 1 : 0;
  double loss = 0.0;
  vector<double> probs = getProbabilities();
  if (!probs.empty()) {
    loss = calculate_loss_from_probs(probs, trueOutput);
  }

  // 3. Backward Prop
  for (size_t i = Layers.size() - 1; i > 0; i--) {
    switch (Layers[i]->getLayerType()) {
    case conv:
      switch (Layers[i - 1]->getLayerType()) {
      case input:
        switch (Layers[i + 1]->getLayerType()) {
        case conv:
          static_cast<convLayer *>(Layers[i])->backwardProp(
              static_cast<inputLayer *>(Layers[i - 1])->getOutput(),
              static_cast<convLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case pooling:
          static_cast<convLayer *>(Layers[i])->backwardProp(
              static_cast<inputLayer *>(Layers[i - 1])->getOutput(),
              static_cast<poolingLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case flatten:
          static_cast<convLayer *>(Layers[i])->backwardProp(
              static_cast<inputLayer *>(Layers[i - 1])->getOutput(),
              static_cast<FlattenLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        }
        break;
      case conv:
        switch (Layers[i + 1]->getLayerType()) {
        case conv:
          static_cast<convLayer *>(Layers[i])->backwardProp(
              static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps(),
              static_cast<convLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case pooling:
          static_cast<convLayer *>(Layers[i])->backwardProp(
              static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps(),
              static_cast<poolingLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case flatten:
          static_cast<convLayer *>(Layers[i])->backwardProp(
              static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps(),
              static_cast<FlattenLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        }
        break;
      case pooling:
        switch (Layers[i + 1]->getLayerType()) {
        case conv:
          static_cast<convLayer *>(Layers[i])->backwardProp(
              static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps(),
              static_cast<convLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case pooling:
          static_cast<convLayer *>(Layers[i])->backwardProp(
              static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps(),
              static_cast<poolingLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case flatten:
          static_cast<convLayer *>(Layers[i])->backwardProp(
              static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps(),
              static_cast<FlattenLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        }
        break;
      }
      break;

    case pooling:
      switch (Layers[i - 1]->getLayerType()) {
      case pooling:
        switch (Layers[i + 1]->getLayerType()) {
        case pooling:
          static_cast<poolingLayer *>(Layers[i])->backwardProp(
              static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps(),
              static_cast<poolingLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case conv:
          static_cast<poolingLayer *>(Layers[i])->backwardProp(
              static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps(),
              static_cast<convLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case flatten:
          static_cast<poolingLayer *>(Layers[i])->backwardProp(
              static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps(),
              static_cast<FlattenLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        }
        break;
      case conv:
        switch (Layers[i + 1]->getLayerType()) {
        case pooling:
          static_cast<poolingLayer *>(Layers[i])->backwardProp(
              static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps(),
              static_cast<poolingLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case conv:
          static_cast<poolingLayer *>(Layers[i])->backwardProp(
              static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps(),
              static_cast<convLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case flatten:
          static_cast<poolingLayer *>(Layers[i])->backwardProp(
              static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps(),
              static_cast<FlattenLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        }
        break;
      case input:
        switch (Layers[i + 1]->getLayerType()) {
        case pooling:
          static_cast<poolingLayer *>(Layers[i])->backwardProp(
              static_cast<inputLayer *>(Layers[i - 1])->getOutput(),
              static_cast<poolingLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case conv:
          static_cast<poolingLayer *>(Layers[i])->backwardProp(
              static_cast<inputLayer *>(Layers[i - 1])->getOutput(),
              static_cast<convLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case flatten:
          static_cast<poolingLayer *>(Layers[i])->backwardProp(
              static_cast<inputLayer *>(Layers[i - 1])->getOutput(),
              static_cast<FlattenLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        }
        break;
      }
      break;
    case flatten:
      switch (Layers[i + 1]->getLayerType()) {
      case output:
        static_cast<FlattenLayer *>(Layers[i])->backwardProp(
            static_cast<outputLayer *>(Layers[i + 1])->getPrevLayerGrad());
        break;
      case fullyConnected:
        static_cast<FlattenLayer *>(Layers[i])->backwardProp(
            static_cast<FullyConnected *>(Layers[i + 1])->getPrevLayerGrad());
        break;
      }
      break;
    case output:
      switch (Layers[i - 1]->getLayerType()) {
      case fullyConnected:
        static_cast<outputLayer *>(Layers[i])->backwardProp(
            static_cast<FullyConnected *>(Layers[i - 1])->getOutput(),
            trueOutput);
        break;
      case flatten:
        static_cast<outputLayer *>(Layers[i])->backwardProp(
            static_cast<FlattenLayer *>(Layers[i - 1])->getFlattenedArr(),
            trueOutput);
        break;
      }
      break;
    case fullyConnected:
      switch (Layers[i - 1]->getLayerType()) {
      case fullyConnected:
        switch (Layers[i + 1]->getLayerType()) {
        case output:
          static_cast<FullyConnected *>(Layers[i])->backwardProp(
              static_cast<FullyConnected *>(Layers[i - 1])->getOutput(),
              static_cast<outputLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case fullyConnected:
          static_cast<FullyConnected *>(Layers[i])->backwardProp(
              static_cast<FullyConnected *>(Layers[i - 1])->getOutput(),
              static_cast<FullyConnected *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        }
        break;
      case flatten:
        switch (Layers[i + 1]->getLayerType()) {
        case output:
          static_cast<FullyConnected *>(Layers[i])->backwardProp(
              static_cast<FlattenLayer *>(Layers[i - 1])->getFlattenedArr(),
              static_cast<outputLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case fullyConnected:
          static_cast<FullyConnected *>(Layers[i])->backwardProp(
              static_cast<FlattenLayer *>(Layers[i - 1])->getFlattenedArr(),
              static_cast<FullyConnected *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        }
        break;
      }
      break;
    }
  }

  // 4. Update Weights
  for (size_t i = Layers.size() - 1; i > 0; i--) {
    switch (Layers[i]->getLayerType()) {
    case conv:
      static_cast<convLayer *>(Layers[i])->update();
      break;
    case output:
      static_cast<outputLayer *>(Layers[i])->update();
      break;
    case fullyConnected:
      static_cast<FullyConnected *>(Layers[i])->update();
      break;
    }
  }

  return {loss, is_correct};
}

// train the model with a batch of images
// input:        -data (a vector of images)
//               -trueOutput (a vector of the true output values)
// output:       N/A
// side effect:  The model is trained by a batch of image and its paramters are
// updated Note:         N/A
// Updated train_batch: Returns {total_loss, total_correct}
std::pair<double, int> NNModel::train_batch(const vector<const image*> &batchData,
                                            const vector<int> &trueOutput) {
  double total_loss = 0.0;
  int total_correct = 0;

  for (size_t sample = 0; sample < batchData.size(); sample++) {
    // 1. Forward Pass (Classify)
    int predictedClass = classify(*batchData[sample]);

    // 2. Accumulate Metrics immediately
    if (predictedClass == trueOutput[sample]) {
      total_correct++;
    }
    vector<double> probs = getProbabilities();
    if (!probs.empty()) {
      total_loss += calculate_loss_from_probs(probs, trueOutput[sample]);
    }

    // 3. Backward Prop
    for (size_t i = Layers.size() - 1; i > 0; i--) {
      switch (Layers[i]->getLayerType()) {
      case conv:
        switch (Layers[i - 1]->getLayerType()) {
        case input:
          switch (Layers[i + 1]->getLayerType()) {
          case conv:
            static_cast<convLayer *>(Layers[i])->backwardProp_batch(
                static_cast<inputLayer *>(Layers[i - 1])->getOutput(),
                static_cast<convLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case pooling:
            static_cast<convLayer *>(Layers[i])->backwardProp_batch(
                static_cast<inputLayer *>(Layers[i - 1])->getOutput(),
                static_cast<poolingLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case flatten:
            static_cast<convLayer *>(Layers[i])->backwardProp_batch(
                static_cast<inputLayer *>(Layers[i - 1])->getOutput(),
                static_cast<FlattenLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          }
          break;
        case conv:
          switch (Layers[i + 1]->getLayerType()) {
          case conv:
            static_cast<convLayer *>(Layers[i])->backwardProp_batch(
                static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps(),
                static_cast<convLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case pooling:
            static_cast<convLayer *>(Layers[i])->backwardProp_batch(
                static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps(),
                static_cast<poolingLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case flatten:
            static_cast<convLayer *>(Layers[i])->backwardProp_batch(
                static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps(),
                static_cast<FlattenLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          }
          break;
        case pooling:
          switch (Layers[i + 1]->getLayerType()) {
          case conv:
            static_cast<convLayer *>(Layers[i])->backwardProp_batch(
                static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps(),
                static_cast<convLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case pooling:
            static_cast<convLayer *>(Layers[i])->backwardProp_batch(
                static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps(),
                static_cast<poolingLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case flatten:
            static_cast<convLayer *>(Layers[i])->backwardProp_batch(
                static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps(),
                static_cast<FlattenLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          }
          break;
        }
        break;

      case pooling:
        switch (Layers[i - 1]->getLayerType()) {
        case pooling:
          switch (Layers[i + 1]->getLayerType()) {
          case pooling:
            static_cast<poolingLayer *>(Layers[i])->backwardProp_batch(
                static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps(),
                static_cast<poolingLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case conv:
            static_cast<poolingLayer *>(Layers[i])->backwardProp_batch(
                static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps(),
                static_cast<convLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case flatten:
            static_cast<poolingLayer *>(Layers[i])->backwardProp_batch(
                static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps(),
                static_cast<FlattenLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          }
          break;
        case conv:
          switch (Layers[i + 1]->getLayerType()) {
          case pooling:
            static_cast<poolingLayer *>(Layers[i])->backwardProp_batch(
                static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps(),
                static_cast<poolingLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case conv:
            static_cast<poolingLayer *>(Layers[i])->backwardProp_batch(
                static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps(),
                static_cast<convLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case flatten:
            static_cast<poolingLayer *>(Layers[i])->backwardProp_batch(
                static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps(),
                static_cast<FlattenLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          }
          break;
        case input:
          switch (Layers[i + 1]->getLayerType()) {
          case pooling:
            static_cast<poolingLayer *>(Layers[i])->backwardProp_batch(
                static_cast<inputLayer *>(Layers[i - 1])->getOutput(),
                static_cast<poolingLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case conv:
            static_cast<poolingLayer *>(Layers[i])->backwardProp_batch(
                static_cast<inputLayer *>(Layers[i - 1])->getOutput(),
                static_cast<convLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case flatten:
            static_cast<poolingLayer *>(Layers[i])->backwardProp_batch(
                static_cast<inputLayer *>(Layers[i - 1])->getOutput(),
                static_cast<FlattenLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          }
          break;
        }
        break;
      case flatten:
        switch (Layers[i + 1]->getLayerType()) {
        case output:
          static_cast<FlattenLayer *>(Layers[i])->backwardProp_batch(
              static_cast<outputLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case fullyConnected:
          static_cast<FlattenLayer *>(Layers[i])->backwardProp_batch(
              static_cast<FullyConnected *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        }
        break;
      case output:
        switch (Layers[i - 1]->getLayerType()) {
        case fullyConnected:
          static_cast<outputLayer *>(Layers[i])->backwardProp_batch(
              static_cast<FullyConnected *>(Layers[i - 1])->getOutput(),
              trueOutput[sample]);
          break;
        case flatten:
          static_cast<outputLayer *>(Layers[i])->backwardProp_batch(
              static_cast<FlattenLayer *>(Layers[i - 1])->getFlattenedArr(),
              trueOutput[sample]);
          break;
        }
        break;
      case fullyConnected:
        switch (Layers[i - 1]->getLayerType()) {
        case fullyConnected:
          switch (Layers[i + 1]->getLayerType()) {
          case output:
            static_cast<FullyConnected *>(Layers[i])->backwardProp_batch(
                static_cast<FullyConnected *>(Layers[i - 1])->getOutput(),
                static_cast<outputLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case fullyConnected:
            static_cast<FullyConnected *>(Layers[i])->backwardProp_batch(
                static_cast<FullyConnected *>(Layers[i - 1])->getOutput(),
                static_cast<FullyConnected *>(Layers[i + 1])
                    ->getPrevLayerGrad());
            break;
          }
          break;
        case flatten:
          switch (Layers[i + 1]->getLayerType()) {
          case output:
            static_cast<FullyConnected *>(Layers[i])->backwardProp_batch(
                static_cast<FlattenLayer *>(Layers[i - 1])->getFlattenedArr(),
                static_cast<outputLayer *>(Layers[i + 1])->getPrevLayerGrad());
            break;
          case fullyConnected:
            static_cast<FullyConnected *>(Layers[i])->backwardProp_batch(
                static_cast<FlattenLayer *>(Layers[i - 1])->getFlattenedArr(),
                static_cast<FullyConnected *>(Layers[i + 1])
                    ->getPrevLayerGrad());
            break;
          }
          break;
        }
        break;
      }
    }
  }

  // 4. Update Weights
  for (size_t i = Layers.size() - 1; i > 0; i--) {
    switch (Layers[i]->getLayerType()) {
    case conv:
      static_cast<convLayer *>(Layers[i])->update_batch(
          static_cast<int>(batchData.size()));
      break;
    case output:
      static_cast<outputLayer *>(Layers[i])->update_batch(
          static_cast<int>(batchData.size()));
      break;
    case fullyConnected:
      static_cast<FullyConnected *>(Layers[i])->update_batch(
          static_cast<int>(batchData.size()));
      break;
    }
  }

  return {total_loss, total_correct};
}

// classify the image by applying the forward propagation on the image
// input:        data (the image)
// output:       int (the class of the image)
// side effect:  N/A
// Note:         This function is either called directly to get the image class
//               or by the train fucntion to train the model
int NNModel::classify(const image &imgData) {
  // make the data ready to be processed by different layers
  static_cast<inputLayer *>(Layers[0])->start(imgData);

  // Iterate over all the layers after the input layer and before the output
  // layer
  for (size_t i = 1; i < Layers.size() - 1; i++) {
    // see which layer this is to call the forward propagation function
    // and if needed, see which is the last layer to get the data from
    switch (Layers[i]->getLayerType()) {
    case conv:
      // for convolution layer check whether the last layer is the input
      // layer, pooling layer or another convoultion layer
      switch (Layers[i - 1]->getLayerType()) {
      case input:
        static_cast<convLayer *>(Layers[i])->forwardProp(
            static_cast<inputLayer *>(Layers[i - 1])->getOutput());
        break;
      case pooling:
        static_cast<convLayer *>(Layers[i])->forwardProp(
            static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps());
        break;
      default:
        static_cast<convLayer *>(Layers[i])->forwardProp(
            static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps());
        break;
      }

      break;
    case pooling:
      // for the pooling layer, the last layer is  a convolution or pooling or
      // input layer
      switch (Layers[i - 1]->getLayerType()) {
      case pooling:
        static_cast<poolingLayer *>(Layers[i])->forwardProp(
            static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps());
        break;
      case input:
        static_cast<poolingLayer *>(Layers[i])->forwardProp(
            static_cast<inputLayer *>(Layers[i - 1])->getOutput());
        break;
      case conv:
        static_cast<poolingLayer *>(Layers[i])->forwardProp(
            static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps());
        break;
      }
      break;
    case fullyConnected:
      // for the fully connected layer, the last layer is either the flatten
      // layer or another fully connected layer
      switch (Layers[i - 1]->getLayerType()) {
      case flatten:
        static_cast<FullyConnected *>(Layers[i])->forwardProp(
            static_cast<FlattenLayer *>(Layers[i - 1])->getFlattenedArr());
        break;
      case fullyConnected:
        static_cast<FullyConnected *>(Layers[i])->forwardProp(
            static_cast<FullyConnected *>(Layers[i - 1])->getOutput());
        break;
      }
      break;
    case flatten:
      // for the flatten layer, the last layer is either the input layer
      //(in Dense architecture) or a convolution layer(in CNN architecture)
      switch (Layers[i - 1]->getLayerType()) {
      case input:
        static_cast<FlattenLayer *>(Layers[i])->forwardProp(
            static_cast<inputLayer *>(Layers[i - 1])->getOutput());
        break;
      case conv:
        static_cast<FlattenLayer *>(Layers[i])->forwardProp(
            static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps());
        break;
      case pooling:
        static_cast<FlattenLayer *>(Layers[i])->forwardProp(
            static_cast<poolingLayer *>(Layers[i - 1])->getFeatureMaps());
        break;
      }

      break;
    default:
      break;
    }
  }

  // do the output layer forward propagtaiton
  switch (Layers[Layers.size() - 2]->getLayerType()) {
  case flatten:
    static_cast<outputLayer *>(Layers[Layers.size() - 1])
        ->forwardProp(static_cast<FlattenLayer *>(Layers[Layers.size() - 2])
                          ->getFlattenedArr());
    break;
  case fullyConnected:
    static_cast<outputLayer *>(Layers[Layers.size() - 1])
        ->forwardProp(static_cast<FullyConnected *>(Layers[Layers.size() - 2])
                          ->getOutput());
    break;
  default:
    break;
  }

  // get the image class
  int cls = static_cast<outputLayer *>(Layers[Layers.size() - 1])->getClass();
  return cls;
}

// Get the probability distribution for the last classification
vector<double> NNModel::getProbabilities() {
  if (Layers.empty())
    return {};

  // The last layer is the output layer
  return static_cast<outputLayer *>(Layers[Layers.size() - 1])->getOutput();
}

// Train the model for multiple epochs with dataset
vector<TrainingMetrics> NNModel::train_epochs(
    const cgroot::data::MNISTLoader::MNISTDataset &dataset,
    const TrainingConfig &config, ProgressCallback progress_callback,
    LogCallback log_callback, std::atomic<bool> *stop_requested) {

  vector<TrainingMetrics> history;

  // ... [Keep your existing validation checks for Layers, output layer, etc.]
  // ...
  if (Layers.empty() || Layers.back()->getLayerType() != output)
    return history;

  size_t num_images = dataset.num_images;
  if (num_images == 0)
    return history;

  // =========================================================
  // OPTIMIZATION 1: Cache Image Conversion
  // =========================================================
  if (log_callback)
    log_callback("Caching " + std::to_string(num_images) + " images...");

  vector<image> cached_images;
  cached_images.reserve(num_images);

  // Convert all images ONCE
  for (const auto &img_obj : dataset.images) {
    cached_images.push_back(
        convert_mnist_to_image_format(img_obj.pixels, 28, 28));
  }

  if (log_callback)
    log_callback("Image caching complete. Starting training...");
  // =========================================================

  // Split logic
  size_t train_size = num_images;
  size_t val_size = 0;
  if (config.use_validation && config.validation_split > 0) {
    val_size = static_cast<size_t>(num_images * config.validation_split);
    train_size = num_images - val_size;
  }

  std::mt19937 rng(config.random_seed);

  for (size_t epoch = 0; epoch < config.epochs; epoch++) {
    if (stop_requested && stop_requested->load())
      break;
    if (log_callback)
      log_callback("Epoch " + std::to_string(epoch + 1));

    // Shuffle indices
    vector<size_t> all_indices(num_images);
    for (size_t i = 0; i < num_images; i++)
      all_indices[i] = i;
    if (config.shuffle)
      std::shuffle(all_indices.begin(), all_indices.end(), rng);

    // Split
    vector<size_t> train_indices(all_indices.begin(),
                                 all_indices.begin() + train_size);
    vector<size_t> val_indices;
    if (config.use_validation && val_size > 0) {
      val_indices.assign(all_indices.begin() + train_size, all_indices.end());
    }

    // =========================================================
    // OPTIMIZATION 2: Integrated Metrics (No redundant classify)
    // =========================================================
    size_t train_correct = 0;
    double train_loss_sum = 0.0;
    size_t train_samples = 0; // Tracks actual samples processed

    vector<const image*> batch_images;
    vector<int> batch_labels;
    batch_images.reserve(config.batch_size);
    batch_labels.reserve(config.batch_size);

    for (size_t i = 0; i < train_indices.size(); i++) {
      if (stop_requested && (i % 100 == 0) && stop_requested->load())
        goto epoch_end;

      size_t idx = train_indices[i];

      // USE CACHED IMAGES - Store pointer to avoid copy
      batch_images.push_back(&cached_images[idx]);
      batch_labels.push_back(static_cast<int>(dataset.images[idx].label));

      // Train when batch full
      if (batch_images.size() >= config.batch_size ||
          i == train_indices.size() - 1) {
        if (!batch_images.empty()) {

          std::pair<double, int> result;

          if (batch_images.size() > 1) {
            result = train_batch(batch_images, batch_labels);
          } else {
            result = train(*batch_images[0], batch_labels[0]);
          }

          // Accumulate metrics DIRECTLY from training step
          train_loss_sum += result.first;
          train_correct += result.second;
          train_samples += batch_images.size();

          batch_images.clear();
          batch_labels.clear();

          double current_loss = 0.0;
          double current_acc = 0.0;
          if (train_samples > 0) {
            current_loss = train_loss_sum / train_samples;
            current_acc =
                static_cast<double>(train_correct) /
                train_samples; // Keep as ratio for consistency with epoch_acc
          }

          // Emit progress frequently (e.g. every sample or small batch to show
          // animation) The user wants "every sample".
          if (progress_callback) {
            progress_callback(epoch + 1, config.epochs, current_loss,
                              current_acc, static_cast<int>(idx));
          }
        }
      }
    }

  epoch_end:
    // Calculate final metrics from accumulation
    double train_acc =
        train_samples > 0 ? (double)train_correct / train_samples : 0.0;
    double train_loss =
        train_samples > 0 ? train_loss_sum / train_samples : 0.0;

    // Validation (Must still run classify here as this is unseen data)
    double val_acc = 0.0;
    double val_loss = 0.0;
    size_t val_samples = 0;

    if (config.use_validation && !val_indices.empty()) {
      size_t val_correct = 0;
      double val_loss_sum = 0.0;

      for (size_t idx : val_indices) {
        // USE CACHED IMAGES
        const image &image_data = cached_images[idx];
        int label = static_cast<int>(dataset.images[idx].label);

        int pred = classify(image_data);
        if (pred == label)
          val_correct++;

        vector<double> probs = getProbabilities();
        if (!probs.empty()) {
          val_loss_sum += calculate_loss_from_probs(probs, label);
          val_samples++;
        }
      }
      val_acc = (double)val_correct / val_indices.size();
      val_loss = val_samples > 0 ? val_loss_sum / val_samples : 0.0;
    }

    // Store & Log
    TrainingMetrics metrics;
    metrics.epoch = epoch + 1;
    metrics.train_loss = train_loss;
    metrics.train_accuracy = train_acc;
    metrics.val_loss = val_loss;
    metrics.val_accuracy = val_acc;
    history.push_back(metrics);

    if (log_callback) {
      std::ostringstream msg;
      msg << "Epoch " << (epoch + 1) << " - Train: Acc=" << std::fixed
          << std::setprecision(2) << (train_acc * 100) << "%"
          << " | Train: Loss=" << std::fixed << std::setprecision(4)
          << train_loss;
      if (config.use_validation)
        msg << " | Val: Acc=" << (val_acc * 100) << "%"
            << " | Val: Loss=" << std::fixed << std::setprecision(4)
            << val_loss;
      log_callback(msg.str());
    }

    if (progress_callback) {
      // Pass -1 or similar as index to indicate end of epoch or no specific
      // image
      progress_callback(epoch + 1, config.epochs,
                        config.use_validation ? val_loss : train_loss,
                        config.use_validation ? val_acc : train_acc, -1);
    }
  }

  return history;
}

vector<vector<vector<double>>> NNModel::getLayerFeatureMaps(size_t layerIndex) {
  if (layerIndex >= Layers.size()) {
    return {};
  }

  Layer *layer = Layers[layerIndex];
  LayerType type = layer->getLayerType();

  if (type == conv) {
    return static_cast<convLayer *>(layer)->getFeatureMaps();
  } else if (type == pooling) {
    return static_cast<poolingLayer *>(layer)->getFeatureMaps();
  } else if (type == input) {
    return static_cast<inputLayer *>(layer)->getOutput();
  }

  return {};
}

// Get the type of a specific layer
// input:        layerIndex
// output:       int (LayerType enum value)
int NNModel::getLayerType(size_t layerIndex) {
  if (layerIndex >= Layers.size()) {
    return -1; // Invalid or out of bounds
  }
  return (int)Layers[layerIndex]->getLayerType();
}


//store all the model parameters (kernels and weights)
//input:          folderPath (the path of the folder where the file
//                containing the model paramters will be created)
//output:         bool (true: operation successful, false: operation failed)
//side effects:   the model paramters are saved in file "model_param<number>.txt" in the folder path
//Note:           N/A
bool store(string &folderPath)
{
  string baseName = "\\model_param";
  string extention = ".txt";

  int counter = 0;

  string path = folderPath + baseName + to_string(counter) + extention;

  ifstream checker(path);
  while (checker.good())
  {
    checker.close();
    counter++;
    path = folderPath + baseName + to_string(counter) + extention;
    checker.open(path);
  }

  checker.close();

  ofstream newFile(path);
  newFile.close();

  for (size_t i = 1; i < Layers.size(); i++)
  {
    switch (Layers[i]->getLayerType())
    {
    case conv:
      if (!save4DVector<double>(static_cast<convLayer *>(Layers[i])->getKernels(), path))
      {
        return false;
      }
      break;
    case fullyConnected:
      if (!save2DVector<double>(static_cast<FullyConnected *>(Layers[i])->getNeurons(), path))
      {
        return false;
      }
      break;
    case output:
      if (!save2DVector<double>(static_cast<outputLayer *>(Layers[i])->getNeurons(), path))
      {
        return false;
      }
      break;
    }
  }

  return true;
}

//load the model parameters (kernels and weights)
//input:          filePath (the path of the file from which 
//                the model paramters will be loaded)
//output:         bool (true: operation successful, false: operation failed)
//side effects:   the model paramters are loaded into the weights and kernels
//Note:           N/A
bool load(string &filePath)
{
  std::streampos cursor = 0;
  for (size_t i = 1; i < Layers.size(); i++)
  {
    switch (Layers[i]->getLayerType())
    {
    case conv:
      if (!load4DVector<double>(static_cast<convLayer *>(Layers[i])->getKernels(), filePath, cursor))
      {
        return false;
      }
      break;
    case fullyConnected:
      if (!load2DVector<double>(static_cast<FullyConnected *>(Layers[i])->getNeurons(), filePath, cursor))
      {
        return false;
      }
      break;
    case output:
      if (!load2DVector<double>(static_cast<outputLayer *>(Layers[i])->getNeurons(), filePath, cursor))
      {
        return false;
      }
      break;
    }
  }

  return true;
}
