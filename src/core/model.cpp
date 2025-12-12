#include "model.h"
using namespace std;

//the NNModel constructor
//input:        -modelArch (contains all the necessary information about the architecture of the model)
//              -numOfClasses (the number of classes of the data set, used to construct the output layer)
//              -imageheight
//              -imageWidth
//              -imageDepth
//output:       N/A
//side effect:  the model is constructed
//Note:         -the constructor makes an array of the layers by making used of class inheretence
//              -specify the initialization function, type of distribution and the activation function 
//              of each layer (convolution or fully connected)
//              -the height and width of the image is constant for a single model(object)
//              and so is the architecture of a single model (object)
NNModel::NNModel(architecture modelArch, size_t numOfClasses,
                 size_t imageHeight, size_t imageWidth, size_t imageDepth)
    : optimizer(createOptimizer(modelArch.optConfig)) {
  // construct the input layer
  Layers.emplace_back(new inputLayer(imageHeight, imageWidth, imageDepth));

  // Track current feature map dimensions
  size_t currentHeight = imageHeight;
  size_t currentWidth = imageWidth;
  size_t currentDepth = imageDepth;

  // poolCount is used to count the number of convolution layers after which
  // a pooling layer will be inserted
  size_t poolCount = 0;

  // an iterator that is used to iterate the pooling layers information vectors
  size_t poolIter = 0;

  // start initializing the convolution and pooling layers
  // a pooling layer is inserted after a number of convolution layers
  for (size_t i = 0; i < modelArch.numOfConvLayers; i++) {
    // Calculate feature map dimensions for this conv layer
    featureMapDim fmDim =
        calcFeatureMapDim(modelArch.kernelsPerconvLayers[i].kernel_height,
                          modelArch.kernelsPerconvLayers[i].kernel_width,
                          currentHeight, currentWidth);

    // Set kernel depth to match input depth
    modelArch.kernelsPerconvLayers[i].kernel_depth = currentDepth;
    fmDim.FM_depth = modelArch.kernelsPerconvLayers[i].numOfKerenels;

    // Create convolution layer
    Layers.emplace_back(new convLayer(
        modelArch.kernelsPerconvLayers[i], modelArch.convLayerActivationFunc[i],
        modelArch.convInitFunctionsType[i], modelArch.distType, fmDim));

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

      // Create pooling layer
      Layers.emplace_back(new poolingLayer(poolKern, poolFMDim,
                                           modelArch.poolingtype[poolIter]));

      // Update current dimensions
      currentHeight = poolOutHeight;
      currentWidth = poolOutWidth;
      // Depth remains the same

      poolCount = 0;
      poolIter++;
    }
  }

  // Determine if we need a flatten layer
  bool needsFlatten =
      (modelArch.numOfConvLayers > 0) ||
      (modelArch.numOfFCLayers > 0 &&
       (imageHeight != 1 || imageWidth != 1 || imageDepth != 1));

  if (needsFlatten) {
    Layers.emplace_back(
        new FlattenLayer(currentHeight, currentWidth, currentDepth));
  }

  // Build fully connected layers
  size_t fcInputSize = needsFlatten
                           ? currentHeight * currentWidth * currentDepth
                           : imageHeight * imageWidth * imageDepth;

  for (size_t i = 0; i < modelArch.numOfFCLayers; i++) {
    Layers.emplace_back(new FullyConnected(
        modelArch.neuronsPerFCLayer[i], modelArch.FCLayerActivationFunc[i],
        modelArch.FCInitFunctionsType[i], modelArch.distType, fcInputSize));

    fcInputSize = modelArch.neuronsPerFCLayer[i];
  }

  // Create output layer
  Layers.emplace_back(new outputLayer(numOfClasses, fcInputSize,modelArch.distType));
                                      
}

//NNModel constructor helper function:
//calculates the dimension of the output feature map of each convolution layer
//input:        -current layer kernel height (size_t kernelHeight)
//              -current layer kernel width (size_t kernelWidth)
//              -the input feature map height (size_t inputHeight)
//              -the input feature map width (size_t inputWidth)
//output:       -a featureMapDim struct that carries information about the dimensions of the current
//              output feature map (featureMapDim)
//side effect:  N/A
//Note:         the function also sets the data member featureMapDim.FM_depth to 0, so it must 
//              setted later
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
void NNModel::train(const image &imgData, int trueOutput) {
  // store the last input image so classify() can use it
  this->data = imgData;

  // forward propagation
  classify(imgData);

  // iterate over each layer to call the backward propagation functions
  for (size_t i = Layers.size() - 1; i > 0; i--) {
    // see which layer this is
    switch (Layers[i]->getLayerType()) {
    case output:
      // see which layer is before this layer
      // for the output layer this can be either a fully connected or the
      // flatten layer
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
      // see which layer is before this layer
      // for the fully connected layer this can be either another fully
      // connected or the flatten layer
      switch (Layers[i - 1]->getLayerType()) {
      case fullyConnected:
        // also to get the error gradients from the next layer a checking is
        // needed the next layer for a fully connected layer is either
        // another fully connected or the output layer
        switch (Layers[i + 1]->getLayerType()) {
        case output:
          static_cast<FullyConnected *>(Layers[i])->backwardProp(
              static_cast<FullyConnected *>(Layers[i - 1])->getOutput(),
              static_cast<outputLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case fullyConnected:
          static_cast<FullyConnected *>(Layers[i])->backwardProp(
              static_cast<FullyConnected *>(Layers[i - 1])->getOutput(),
              static_cast<FullyConnected *>(Layers[i + 1])
                  ->getPrevLayerGrad());
          break;
        }
        break;
      case flatten:
        // also to get the error gradients from the next layer a checking is
        // needed the next layer for a fully connected layer is either
        // another fully connected or the output layer
        switch (Layers[i + 1]->getLayerType()) {
        case output:
          static_cast<FullyConnected *>(Layers[i])->backwardProp(
              static_cast<FlattenLayer *>(Layers[i - 1])->getFlattenedArr(),
              static_cast<outputLayer *>(Layers[i + 1])->getPrevLayerGrad());
          break;
        case fullyConnected:
          static_cast<FullyConnected *>(Layers[i])->backwardProp(
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

  // call the update functions of each layer
  for (size_t i = Layers.size() - 1; i > 0; i--) {
    switch (Layers[i]->getLayerType()) {
    case output:
      static_cast<outputLayer *>(Layers[i])->update(optimizer);
      break;
    case fullyConnected:
      static_cast<FullyConnected *>(Layers[i])->update(optimizer);
      break;
    }
  }
}

//train the model with a batch of images
//input:        -data (a vector of images)
//              -trueOutput (a vector of the true output values)
//output:       N/A
//side effect:  The model is trained by a batch of image and its paramters are updated
//Note:         N/A
void NNModel::train_batch(const vector<image> &batchData,
                          const vector<int> &trueOutput) {
  for (size_t sample = 0; sample < batchData.size(); sample++) {
    // store the last input image so classify() can use it
    this->data = batchData[sample];

    // forward propagation
    classify(batchData[sample]);

    // iterate over each layer to call the batch backward propagation
    // functions
    for (size_t i = Layers.size() - 1; i > 0; i--) {
      // see which layer this is
      switch (Layers[i]->getLayerType()) {
      case output:
        // see which layer is before this layer
        // for the output layer this can be either a fully connected or the
        // flatten layer
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
        // see which layer is before this layer
        // for the fully connected layer this can be either another fully
        // connected or the flatten layer
        switch (Layers[i - 1]->getLayerType()) {
        case fullyConnected:
          // also to get the error gradients from the next layer a checking
          // is needed the next layer for a fully connected layer is either
          // another fully connected or the output layer
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
          // also to get the error gradients from the next layer a checking
          // is needed the next layer for a fully connected layer is either
          // another fully connected or the output layer
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

  // call the update_batch functions of each layer
  for (size_t i = Layers.size() - 1; i > 0; i--) {
    switch (Layers[i]->getLayerType()) {
    case output:
      static_cast<outputLayer *>(Layers[i])->update_batch(optimizer,
          static_cast<int>(batchData.size()));
      break;
    case fullyConnected:
      static_cast<FullyConnected *>(Layers[i])->update_batch(optimizer,
          static_cast<int>(batchData.size()));
      break;
    }
  }
}

//classify the image by applying the forward propagation on the image
//input:        data (the image)
//output:       int (the class of the image)
//side effect:  N/A
//Note:         This function is either called directly to get the image class
//              or by the train fucntion to train the model 
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
      // for the pooling layer, the last layer is always a convolution layer
      static_cast<poolingLayer *>(Layers[i])->forwardProp(
          static_cast<convLayer *>(Layers[i - 1])->getFeatureMaps());
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
