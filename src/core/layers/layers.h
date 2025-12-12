// This file includes the layer abstract class declaration
// which every other layer class inherets from
// And the declaration of all other layers which include:
// input layer
// convolution layer
// fully connected layer
// pooling layer
// flatten layer
// output layer

#ifndef _LAYERS_H
#define _LAYERS_H

// #include "../model.h"
#include "../activation/activation.h"
#include "../definitions.h"
#include "../optimizers/optimizer.h"

#include <vector>
using namespace std;

// circular dependency stuff, probably needs more editing
enum distributionType;
enum activationFunction;
enum LayerType;
enum initFunctions;
struct convKernels;
// Forward declaration to avoid circular dependency
typedef vector<vector<vector<unsigned char>>> image;

// the data type of all vectors

// this is the base class which every other layer class inherets from
class Layer {

public:
  Layer();

  // get the layer type, this function must be implemented by all the child
  // classes
  virtual LayerType getLayerType() = 0;

  // these fucntions do forward and backward propagation
  // and must be implemented differently at every layer
  //(the declaration probably isn't correct)
  // virtual void forwardProp() = 0;
  // virtual void backwardProp() = 0;

  // get the layer output data(neurons output)
  // outType& getLayerOutput();

  virtual ~Layer() = default;
};

class inputLayer : public Layer {
public:
  typedef vector<vector<vector<double>>> imageType;

private:
  // any additional data
  LayerType type = input;
  imageType normalizedImage;

public:
  // input Layer constructor
  // input:        -imageHeight
  //               -imageWidth
  //               -imageDepth
  // output:       N/A
  // side effect:  The input layer is constructed
  // Note:         N/A
  inputLayer(size_t imageHeight, size_t imageWidth, size_t imageDepth);

  // start the process of training or classification by taking the input image,
  // normalizing it, and storing it in the normalizedImage matrix to be used by
  // the next layers
  // input:        -inputImage (3D unsigned char vector)
  // output:       N/A
  // side effect:  the normalizedImage matrix is initialzied by the image after
  // normalization Note:         N/A
  void start(const image &data);

  // get the layer type
  LayerType getLayerType() override { return type; }

  // get the normlized image
  imageType &getOutput() { return normalizedImage; }
};

class convLayer : public Layer {
public:
  typedef vector<vector<vector<double>>> kernelType;
  typedef vector<vector<double>> featureMapType;

private:
  vector<kernelType> kernels; // 4D: [Kernel][Depth][Row][Col]
  vector<kernelType> d_kernels; // Gradients for weights
  convKernels kernel_info;    // number and dimensions of each kernel
  LayerType type = conv;      // layer type

  vector<featureMapType>featureMaps;  // array of the different layers of the feature map
  vector<featureMapType> prevLayerGrad; // Gradients to pass to previous layer
  featureMapDim fm; // dimesions of the feature map
  activationFunction act_Funct; // activation function type

public:
  vector<vector<vector<Optimizer *>>> kernelOptimizers;
  // Optimizer *biasOpt; // Conv layer has no bias in this implementation

  // Destructor to clean up optimizers
  ~convLayer();

  // the convolution layer constructor
  convLayer(convKernels &kernelConfig, activationFunction actFunc,
            initFunctions initFunc, distributionType distType,
            featureMapDim &FM_Dim, OptimizerConfig optConfig);

  // initialize a kernel
  kernelType initKernel(convKernels &kernelConfig, initFunctions initFunc,
                        distributionType distType);
  
  // a set of getters to get the dimensions of the feature maps
  // mostly used in the model construction
  size_t getFeatureMapHeight() const { return fm.FM_height; } 
  size_t getFeatureMapWidth() const { return fm.FM_width; } 
  size_t getFeatureMapDepth() const { return fm.FM_depth; }

  // get the type of the activation function
  activationFunction getActivationFunctionType() const { return act_Funct; }




  // do the convolution operation by sweeping the kernels through
  // the input feature map and putin the result in the (output) feature map
  // essentially doing the forward propagation
  // input:        inputFeatureMaps (previous layer output feature maps)
  // output:       N/A
  // side effect:  this layer (output) feature maps is filled
  // Note:         N/A
  void convolute(vector<featureMapType> &inputFeatureMaps);

  // do the forward propagation of the convolution layer
  // by first applying the convolution and then the activation functions
  // input:        inputFeatureMaps
  // output:       N/A
  // side effect:  the feature maps are filled with the forward propagation
  // values note:         N/A
  void forwardProp(vector<featureMapType> &inputFeatureMaps);

  // backward propagate the error
  // input:                -inputFeatureMaps -thisLayerGrad
  // output:               N/A
  // side effect:          the prevLayerGrad is filled with the error to be propagated
  // Note:                 This function works with SGD or for updating after a single
  void backwardProp(vector<featureMapType> &inputFeatureMaps, vector<featureMapType> &thisLayerGrad);
  
  // backward propagate the error after a batch
  // input:                -inputFeatureMaps -thisLayerGrad
  // output:               N/A
  // side effect:          the prevLayerGrad is filled with the error to be propagated
  // Note:                 This function works with BGD or for updating after a whole batch
  void backwardProp_batch(vector<featureMapType> &inputFeatureMaps, vector<featureMapType> &thisLayerGrad);

  vector<featureMapType> &getFeatureMaps() { return featureMaps; } // get the output feature map
  vector<featureMapType> &getPrevLayerGrad() { return prevLayerGrad; } // get the previous layer gradient
  LayerType getLayerType() override { return type; }  // get the layer type
};

class poolingLayer : public Layer {
public:
  typedef vector<vector<vector<double>>> kernelType;
  typedef vector<vector<double>> featureMapType;

private:
  poolKernel kernel_info;   // dimensions of the kernel and number of strides
  LayerType type = pooling; // layer type
  poolingLayerType poolingType; // max or average
  vector<featureMapType> featureMaps;  // array of the different layers of the feature map
  vector<featureMapType> prevLayerGrad; // Gradients for previous layer
  featureMapDim fm; // dimesions of the feature map

public:
  // pooling layer constructor
  // input:                -kernelConfig (dimensions of the filter and number of
  // strides)
  //                       -FM_Dim (dimensions of the output feature map)
  //                       -poolType (max or average)
  // output:               N/A
  // side effect:          the pooling layer is constructed
  // Note:                 N/A
  poolingLayer(poolKernel &kernelConfig, featureMapDim &FM_Dim,
               poolingLayerType &poolType);

  // forward propagation of the pooling layer
  // done by applying max or average pooling to the feature maps
  // input:                inputFeatureMaps
  // output:               N/A
  // side effect:          the output feature map is filled with the result of
  // the pooling Note:                 N/A
  void forwardProp(vector<featureMapType> &inputFeatureMaps);

  // Backward Propagation
  void backwardProp(vector<featureMapType> &inputFeatureMaps, vector<featureMapType> &thisLayerGrad);
  void backwardProp_batch(vector<featureMapType> &inputFeatureMaps, vector<featureMapType> &thisLayerGrad);

  // a set of getters to get the dimensions of the feature maps
  // mostly used in the model construction
  size_t getFeatureMapHeight() const {
    return fm.FM_height;
  } // get the feature map height
  size_t getFeatureMapWidth() const {
    return fm.FM_width;
  } // get the feature map width
  size_t getFeatureMapDepth() const {
    return fm.FM_depth;
  } // get the feature map depth

  // get the output feature map
  vector<featureMapType> &getFeatureMaps() { return featureMaps; }
  // get the previous layer gradient
  vector<featureMapType> &getPrevLayerGrad() { return prevLayerGrad; } 
  // get the layer type
  LayerType getLayerType() override { return type; }
};

class FullyConnected : public Layer {
public:
  typedef vector<double> weights;

private:
  // any additional data
  vector<double> inputCache;    // Stores the input 'x' from forwardProp
  vector<double> preActivation; // Stores 'z' before ReLU/Sigmoid
  vector<weights> neurons;      // the vector of weights of each neuron
  vector<double> bias;          // the vector of biases
  vector<double> outputData;    // the vector of output data

  vector<vector<double>> d_weights; // store the gradients for weights
  vector<double> d_bias;            // store the gradients for biases
  vector<double> prevLayerGrad; // gradients to be used by the previous layer in
                                // backward propagation

  activationFunction act_Funct;    // the type of the activation function
  LayerType type = fullyConnected; // the type of the layer

public:
  vector<Optimizer *>
      neuronOptimizers; // One optimizer per neuron (weight vector)
  Optimizer *biasOptimizer;

  // Destructor to clean up optimizers
  ~FullyConnected();

  // the fully connected layer constructor
  FullyConnected(size_t numOfNeurons, activationFunction actFunc,
                 initFunctions initFunc, distributionType distType,
                 size_t numOfWeights, OptimizerConfig optConfig);

  // update the weights and biases of this fully connected layer
  void update();

  // update the weights and biases of this fully connected layer after a batch
  void update_batch(int numOfExamples);
  void forwardProp(vector<double> &inputData);

  // backward propagate the error
  // input:                -inputData
  //                       -thisLayerGrad
  // output:               N/A
  // side effect:          the d_bias and d_weights are filled with the
  // gradients
  //                       and prevLayerGrad is filled with the error to be
  //                       propagated
  // Note:                 This function works with SGD or for updating after a
  // single
  //                       example, if the update should happen after multiple
  //                       examples, then use bacwardProp_batch() instead
  void backwardProp(vector<double> &inputData, vector<double> &thisLayerGrad);

  // backward propagate the error
  // input:                -inputData
  //                       -thisLayerGrad
  // output:               N/A
  // side effect:          the d_bias and D_weights are filled with the
  // accumlated gradients
  //                       and prevLayerGrad is filled with the error to be
  //                       propagated
  // Note:                 This function works with BGD or for updating after a
  // whole batch
  //                       of examples, if the update should happen after a
  //                       single example, then use bacwardProp() instead
  void backwardProp_batch(vector<double> &inputData,
                          vector<double> &thisLayerGrad);

  // get the ouput data size (used by the constructor)
  size_t getOutputSize() { return neurons.size(); }

  // get the ouput data (used by the next fully connected layer)
  vector<double> &getOutput() { return outputData; }

  // get the previous layer gradient
  vector<double> &getPrevLayerGrad() { return prevLayerGrad; }

  // get the layer type
  LayerType getLayerType() override { return type; }
};

class FlattenLayer : public Layer {
private:
  vector<double> flattened_Arr; // the flattened array that contains the
                                // flattened image or feature maps
  LayerType type = flatten;     // the type of the layer

  size_t height, width, depth;  // dimensions of the input image or feature maps to return                           
  vector<convLayer::featureMapType> prevLayerGrad; // Gradients for previous layer
public:
  // the Flatten Layer constructor
  // input:        -imageHeight(or feature map height)
  //               -imageWidth(or feature map width)
  //               -imageDepth(or feature map depth)
  // output:       N/A
  // side effect:  the flatten layer is constructed
  // Notes:        N/A
  FlattenLayer(size_t imageHeight, size_t imageWidth, size_t imageDepth);

  // the flatten layer forward porpagation
  // input:        featureMaps
  // output:       N/A
  // side effect:  flat funciton is called so the input layer is flattened
  //               and put inside flattened_Arr
  // Note:         N/A
  void forwardProp(vector<convLayer::featureMapType> &featureMaps);

  void backwardProp(vector<double> &nextLayerGrad);
  void backwardProp_batch(vector<double> &nextLayerGrad);

  // flattens the incomming image or feature map
  // input:        -feature map or image
  // output:       N/A
  // side effect:  the flattened_Arr is filled with the incomming data
  // Note:         if the incomming is an image then it must be 3D
  //               or if 2D then the depth is 1
  void flat(vector<convLayer::featureMapType> &featureMaps);

  // get the flattened array size (used by the constructor)
  size_t getFlattenedArrSize() const { return flattened_Arr.size(); }

  // get the flattened array (used by the next fully connected layer)
  vector<double> &getFlattenedArr() { return flattened_Arr; }

  // get the layer type
  LayerType getLayerType() override { return type; }

  vector<convLayer::featureMapType> &getPrevLayerGrad() { return prevLayerGrad; }// get the previous layer gradient

  void applyOptimizer(Optimizer *opt);
};

class outputLayer : public Layer {
public:
  typedef vector<double> weights;

private:
  vector<weights> neurons;      // the vector of weights of each neuron
  vector<double> bias;          // the vector of biases
  vector<double> outputData;    // the vector of output data
  vector<weights> d_weights;    // store the gradients for weights
  vector<double> d_bias;        // store the gradients for biases
  vector<double> prevLayerGrad; // gradients to be used by the previous layer in
                                // backward propagation

  LayerType type = output;

public:
  vector<Optimizer *> neuronOptimizers;
  Optimizer *biasOptimizer;

  // Destructor to clean up optimizers
  ~outputLayer();

  // the output layer constructor
  outputLayer(size_t numOfClasses, size_t numOfWeights,
              distributionType distType, OptimizerConfig optConfig);

  // update the weights and biases of the output layer
  void update();

  // update the weights and biases of the output layer after a batch
  void update_batch(int numOfExamples);

  void forwardProp(vector<double> &featureMaps);

  void backwardProp(vector<double> &inputData, size_t correctClass);
  void backwardProp_batch(vector<double> &inputData, size_t correctClass);

  // get the layer type
  LayerType getLayerType() override { return type; }

  // get the previous layer gradient
  vector<double> &getPrevLayerGrad() { return prevLayerGrad; }

  // get the class num of the image
  // input:                N/A
  // output:               int value(the number/index of the class)
  // side effect:          N/A
  // note:                 the num of the class is in the same order
  //                       that is given to the backwardProp() function
  int getClass();

  // get the output probability vector
  vector<double> &getOutput() { return outputData; }
};

#endif
