#ifndef _DEFINITIONS_H
#define _DEFINITIONS_H


#include <vector>
using namespace std;

typedef vector<vector<vector<unsigned char>>> image;

//the types of different optimizers
enum OptimizerType {
    SGD,
    Adam,
};

//types of distribution to be used by all layers initializers
enum distributionType
{
    normalDistribution,
    uniformDistribution,
    maxNumOfDistributions,
};

//the type of the activation function used in activation layers
//for this model, the type of the activation function determines
//the type of the initialization function used
enum activationFunction{
    RelU,
    Sigmoid,
    Tanh,
    Softmax,
    maxNumOfActivationFunctionTypes,
};

//the types of different layers which the user will specify
//before constructing the model to be used for the model construction
enum LayerType{
    input,
    flatten,
    conv,
    pooling,
    fullyConnected,
    output,
    maxNumOfLayerTypes,
};

//the type of the initialization function of each layer
enum initFunctions
{
    Xavier,
    Kaiming,
    maxNumOfInitFuncs,
};

//the specifications of the kernels of each conv layer
//is specifies the number of kernels and its height and
//width of the layer, these are specified before the 
//model construction to be used in the model construction
//Note: kernel_depth is calculated at the model constructor
struct convKernels
{
    size_t numOfKerenels;
    size_t kernel_width;
    size_t kernel_height;
    size_t kernel_depth;
};


//The specification of the pooling layer kernel
//The dimensions of the kernel and the number of strides
//Used in the model construction
//Note: filter_depth is calculated at the model constructor
struct poolKernel
{
    size_t filter_width;
    size_t filter_height;
    size_t filter_depth;
    size_t stride;
};

//the dimensions of the feature maps of each convolutions layer
//the struct contains the dimension in compact form
//Note: the feature map dimensions are calculated at the model constructor
struct featureMapDim
{
    size_t FM_height;
    size_t FM_width;
    size_t FM_depth;
};


//the types of pooling layer
enum poolingLayerType{
    maxPooling,
    averagePooling,
    maxNumOfPoolingLayerType,
};

#endif