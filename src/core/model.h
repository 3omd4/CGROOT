#ifndef _MODEL_H
#define _MODEL_H

#include "definitions.h"
#include "layers/layers.h"
#include "optimizers/optimizer.h"
#include "utils/mnist_loader.h"
#include <functional>
#include <random>
#include <vector>
#include <atomic>

using cgroot::data::MNISTLoader;
using std::vector;

// image typedef is defined in layers/layers.h to avoid circular dependency

// the struct that determine the arhitecture of the model
// which is used only to initialize the model
// contains information about the architecture
struct architecture {
  size_t numOfConvLayers; // the number of convolution layers

  size_t numOfFCLayers; // the number of fully connected layers

  vector<convKernels> kernelsPerconvLayers; // the specification of the kernels
                                            // of each convolution layer

  vector<size_t>
      neuronsPerFCLayer; // the number of neurons per layer, in other words
                         // the size of the output vector of each layer
                         // this is only for fully connected layers

  vector<activationFunction>
      convLayerActivationFunc; // the type of the activation function used in
                               // each convolution layer

  vector<activationFunction>
      FCLayerActivationFunc; // the type of the activation function used in each
                             // fully connected layer

  vector<initFunctions>
      convInitFunctionsType; // the type of the functions which will be used to
                             // initialize each convolution layer

  vector<initFunctions>
      FCInitFunctionsType; // the type of the functions which will be used to
                           // initialize each fully connected layer

  distributionType
      distType; // type of distribution to be used by all layers initializers

  vector<size_t>
      poolingLayersInterval; // the number of convolution layers after which a
                             // pooling layer is inserted

  vector<poolingLayerType>
      poolingtype; // the type of each pooling layer(max or average)

  vector<poolKernel>
      kernelsPerPoolingLayer; // the information about each kernel of  each
                              // pooling layer and the number of strides

  OptimizerConfig optConfig;
  // additional data used to initialize the data
};

// Training configuration for the train_epochs method
struct TrainingConfig {
  size_t epochs = 10;
  size_t batch_size = 32;
  float validation_split = 0.2f;
  bool use_validation = true;
  bool shuffle = true;
  unsigned int random_seed = 42;
};

// Training metrics returned after each epoch
struct TrainingMetrics {
  int epoch;
  double train_loss;
  double train_accuracy;
  double val_loss;
  double val_accuracy;

  TrainingMetrics()
      : epoch(0), train_loss(0.0), train_accuracy(0.0), val_loss(0.0),
        val_accuracy(0.0) {}
};

// Callback types for training progress updates
// Callback signature: (epoch, total_epochs, current_loss, current_accuracy)
using ProgressCallback = std::function<void(int, int, double, double)>;
// Callback signature: (message)
using LogCallback = std::function<void(const std::string &)>;

class NNModel {
private:
  vector<Layer *> Layers;
  image data;

public:
  // the NNModel constructor
  // input:        -modelArch (contains all the necessary information about the
  //               architecture of the model)
  //               -numOfClasses (the number of classes of the data set, used to
  //               construct the output layer)
  //               -imageheight
  //               -imageWidth
  //               -imageDepth
  // output:       N/A
  // side effect:  the model is constructed
  // Note:         -the constructor makes an array of the layers by making used
  // of class inheretence
  //               -specify the initialization function, type of distribution
  //               and the activation function of each layer (convolution or
  //               fully connected)
  //               -the height and width of the image is constant for a single
  //               model(object) and so is the architecture of a single model
  //               (object)
  NNModel(struct architecture, size_t numOfClasses, size_t imageVerDim,
          size_t imageHorDim, size_t imageDepDim);

  // NNModel constructor helper function:
  // calculates the dimension of the output feature map of each convolution
  // layer
  // input:         -current layer kernel height (size_t kernelHeight)
  //               -current layer kernel width (size_t kernelWidth)
  //               -the input feature map height (size_t inputHeight)
  //               -the input feature map width (size_t inputWidth)
  // output:       -a featureMapDim struct that carries information about the
  //               dimensions of the current output feature map (featureMapDim)
  // side effect:  N/A
  // Note:         the function also sets the data member featureMapDim.FM_depth
  //               to 0, so it must setted later
  featureMapDim calcFeatureMapDim(size_t kernelHeight, size_t kernelWidth,
                                  size_t inputHeight, size_t inputWidth);

  // train the model with a single image
  // input:        -data (an image)
  //               -trueOutput (the value for which the model compares its
  //               output)
  // output:       N/A
  // side effect:  The model is trained by a single image and its paramters are
  // updated Note:         N/A
  std::pair<double, int> train(const image &data, int trueOutput);

  std::pair<double, int> train_batch(const vector<const image*> &data,
                                     const vector<int> &trueOutput);

  // classify the image by applying the forward propagation on the image
  // input:        data (the image)
  // output:       int (the class of the image)
  // side effect:  N/A
  // Note:         This function is either called directly to get the image
  // class
  //               or by the train fucntion to train the model
  int classify(const image &data);

  ~NNModel() {
    for (Layer *l : Layers) {
      delete l;
    }
    Layers.clear();
  }
  // Get the probability distribution for the last classification
  vector<double> getProbabilities();

  // Train the model for multiple epochs with dataset
  // input:        -dataset (MNIST dataset with images and labels)
  //               -config (training configuration)
  //               -progress_callback (optional callback for progress updates)
  //               -log_callback (optional callback for log messages)
  //               -stop_requested (optional atomic flag for cancellation)
  // output:       vector of TrainingMetrics for each epoch
  // side effect:  The model is trained and parameters are updated
  vector<TrainingMetrics>
  train_epochs(const cgroot::data::MNISTLoader::MNISTDataset &dataset,
               const TrainingConfig &config,
               ProgressCallback progress_callback = nullptr,
               LogCallback log_callback = nullptr,
               std::atomic<bool> *stop_requested = nullptr);

  // additional functions
};

#endif
