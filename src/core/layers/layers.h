//This file includes the layer abstract class declaration
//which every other layer class inherets from
//And the declaration of all other layers which include:
//input layer
//convolution layer
//fully connected layer
//pooling layer
//flatten layer
//output layer

#ifndef _LAYERS_H
#define _LAYERS_H



// #include "../model.h" // REMOVED to avoid circular dependency
#include "../definitions.h"
#include "../activation/activation.h"
#include "../optimizers/optimizer.h"

#include <vector>
using namespace std;

//circular dependency stuff, probably needs more editing
enum distributionType;
enum activationFunction;
enum LayerType;
enum initFunctions;
struct convKernels;
// Forward declaration to avoid circular dependency
// typedef vector<vector<vector<unsigned char>>> image; // MOVED TO definitions.h

//the data type of all vectors



//this is the base class which every other layer class inherets from
class Layer
{

    public:
    Layer();

    //get the layer type, this function must be implemented by all the child classes
    virtual LayerType getLayerType() = 0;


    //these fucntions do forward and backward propagation 
    //and must be implemented differently at every layer
    //(the declaration probably isn't correct)
    //virtual void forwardProp() = 0; 
    //virtual void backwardProp() = 0;

    //get the layer output data(neurons output)
    //outType& getLayerOutput();
 
    virtual ~Layer() = default;

};


class inputLayer : public Layer
{
    public:
    typedef vector<vector<vector<double>>> imageType;
    private:
    //any additional data
    LayerType type = input;
    imageType m_normalizedImage;

    public:
    //input Layer constructor
    //input:        -imageHeight
    //              -imageWidth
    //              -imageDepth
    //output:       N/A
    //side effect:  The input layer is constructed
    //Note:         N/A
    inputLayer(size_t imageHeight, size_t imageWidth, size_t imageDepth);

    //start the process of training or classification by taking the input image,
    //normalizing it, and storing it in the normalizedImage matrix to be used by 
    //the next layers   
    //input:        -inputImage (3D unsigned char vector)
    //output:       N/A
    //side effect:  the normalizedImage matrix is initialzied by the image after normalization
    //Note:         N/A
    void start(image& data);


    //get the layer type
    LayerType getLayerType() override {return type;}   

    //get the normlized image
    imageType& getOutput()  {return m_normalizedImage;}   
    
    vector<double> backwardProp(const vector<double>& outputError);
    void applyOptimizer(Optimizer* opt);

};

class convLayer : public Layer
{
    public:
    typedef vector<vector<vector<double>>> kernelType;
    typedef vector<vector<double>> featureMapType;

    private:
    vector<kernelType> kernels;    //Array of the layer kernels
    convKernels kernel_info;       //number and dimensions of each kernel
    LayerType type = conv;         //layer type

    vector<featureMapType> featureMaps; //array of the different layers of the feature map
    featureMapDim fm;   //dimesions of the feature map
    
    activationFunction act_Funct;   //activation function type


    public: 
    //the convolution layer constructor
    //input:        -kernelConfig (contains all the information about the kernel)
    //              -actFunc (activation function)
    //              -initFunc (initialization function)
    //              -distType (distribution type)
    //              -FM_Dim (the dimension of the output feature map)
    //ouput:        N/A
    //side effect:  the convolution layer is constructed
    //Note:         N/A
    convLayer(convKernels& kernelConfig, activationFunction actFunc, 
                initFunctions initFunc, distributionType distType
                , featureMapDim& FM_Dim);


    //initialize a kernel
    //input:        -kernelConfig (contains all the information about the kernel)
    //              -initFunc (the initialization function)
    //              -distType (the type of the distribution)
    //output:       kernelType (the initialized kernel)
    //side effect:  N/A
    //Note:         N/A
    kernelType initKernel(convKernels& kernelConfig, initFunctions initFunc,
                 distributionType distType);

    //a set of getters to get the dimensions of the feature maps
    //mostly used in the model construction
    size_t getFeatureMapHeight() const {return fm.FM_height;}   //get the feature map height
    size_t getFeatureMapWidth() const {return fm.FM_width;}     //get the feature map width
    size_t getFeatureMapDepth() const {return fm.FM_depth;}     //get the feature map depth

    
    //get the type of the activation function
    activationFunction getActivationFunctionType() const {return act_Funct;}

    //do the convolution operation by sweeping the kernels through 
    //the input feature map and putin the result in the (output) feature map
    //essentially doing the forward propagation
    //input:        inputFeatureMaps (previous layer output feature maps)
    //output:       N/A
    //side effect:  this layer (output) feature maps is filled
    //Note:         N/A
    void convolute(vector<featureMapType>& inputFeatureMaps);

    //do the forward propagation of the convolution layer
    //by first applying the convolution and then the activation functions
    //input:        inputFeatureMaps
    //output:       N/A
    //side effect:  the feature maps are filled with the forward propagation values 
    //note:         N/A
    void forwardProp(vector<featureMapType>& inputFeatureMaps);

    //get the output feature map
    vector<featureMapType>& getFeatureMaps() {return featureMaps;}

    //get the layer type
    LayerType getLayerType() override {return type;}

};



class poolingLayer : public Layer
{
    public:
    typedef vector<vector<vector<double>>> kernelType;
    typedef vector<vector<double>> featureMapType;

    private:
    poolKernel kernel_info;       //dimensions of the kernel and number of strides
    LayerType type = pooling;      //layer type
    poolingLayerType poolingType;   //max or average
    vector<featureMapType> featureMaps; //array of the different layers of the feature map
    featureMapDim fm;   //dimesions of the feature map

    public:
    //pooling layer constructor
    //input:                -kernelConfig (dimensions of the filter and number of strides)
    //                      -FM_Dim (dimensions of the output feature map)
    //                      -poolType (max or average)
    //output:               N/A
    //side effect:          the pooling layer is constructed
    //Note:                 N/A
    poolingLayer(poolKernel& kernelConfig, featureMapDim& FM_Dim, poolingLayerType& poolType);

    //forward propagation of the pooling layer
    //done by applying max or average pooling to the feature maps
    //input:                inputFeatureMaps
    //output:               N/A
    //side effect:          the output feature map is filled with the result of the pooling
    //Note:                 N/A
    void forwardProp(vector<featureMapType>& inputFeatureMaps);

    //a set of getters to get the dimensions of the feature maps
    //mostly used in the model construction
    size_t getFeatureMapHeight() const {return fm.FM_height;}   //get the feature map height
    size_t getFeatureMapWidth() const {return fm.FM_width;}     //get the feature map width
    size_t getFeatureMapDepth() const {return fm.FM_depth;}     //get the feature map depth

    //get the output feature map
    vector<featureMapType>& getFeatureMaps() {return featureMaps;}
    //get the layer type
    LayerType getLayerType() override {return type;}

};

class FullyConnected : public Layer
{
    public:
    typedef vector<double> weights;

    private:
    //any additional data
    vector<double> inputCache;      // Stores the input 'x' from forwardProp
    vector<double> preActivation;   // Stores 'z' before ReLU/Sigmoid
    vector<weights> neurons;    //the vector of weights of each neuron
    vector<double> bias;        //the vector of biases
    vector<double> outputData;  //the vector of output data
    vector<vector<double>> weightGradients; 
    vector<double> biasGradients;

    activationFunction act_Funct;   //the type of the activation function 
    LayerType type = fullyConnected;    //the type of the layer

    public:
    //the fully connected layer constructor
    //input:        -numOfNeurons 
    //              -actFunc (the layer activation function)
    //              -initFuc (the layer initializaiton function)
    //              -distType (the layer distribution used to initializet its weights)
    //              -numOfWeigths (the number of weights per layer)
    //output:       N/A
    //side effect:  a Fully connected layer is constructed
    //Note:         N/A
    FullyConnected(size_t numOfNeurons, activationFunction actFunc, 
                initFunctions initFunc, distributionType distType,
                        size_t numOfWeights);
    
    //forward propagate the input data to the output
    //input:        inputData
    //output:       N/A
    //side effects: the outputData vector is filled with the dot product 
    //              of the input data and each neuron weights
    //Note:         N/A
    void forwardProp(vector<double>& inputData);

    // Backward propagation function
    // input:   nextLayerGrad (gradient from the layer ahead)
    // output:  vector<double> (gradient to pass back to previous layer)
    vector<double> backwardProp(const vector<double>& outputError);

    void applyOptimizer(Optimizer* opt);
   

    //get the ouput data size (used by the constructor)
    size_t getOutputSize() {return outputData.size();}

    //get the ouput data (used by the next fully connected layer)
    vector<double>& getOutput() {return outputData;}

    //get the layer type
    LayerType getLayerType() override {return type;}
};

class FlattenLayer : public Layer
{
    private:
    vector<double> flattened_Arr;   //the flattened array that contains the flattened image or feature maps
    LayerType type = flatten;       //the type of the layer

    public:
    //the Flatten Layer constructor 
    //input:        -imageHeight(or feature map height)
    //              -imageWidth(or feature map width)
    //              -imageDepth(or feature map depth)
    //output:       N/A
    //side effect:  the flatten layer is constructed
    //Notes:        N/A
    FlattenLayer(size_t imageHeight, size_t imageWidth, size_t imageDepth);

    //the flatten layer forward porpagation
    //input:        featureMaps
    //output:       N/A
    //side effect:  flat funciton is called so the input layer is flattened
    //              and put inside flattened_Arr
    //Note:         N/A
    void forwardProp(vector<convLayer::featureMapType>& featureMaps)
    {flat(featureMaps);}

    //flattens the incomming image or feature map
    //input:        -feature map or image
    //output:       N/A
    //side effect:  the flattened_Arr is filled with the incomming data
    //Note:         if the incomming is an image then it must be 3D
    //              or if 2D then the depth is 1
    void flat(vector<convLayer::featureMapType>& featureMaps);

    //get the flattened array size (used by the constructor)
    size_t getFlattenedArrSize() const {return flattened_Arr.size();}

    //get the flattened array (used by the next fully connected layer)
    vector<double>& getFlattenedArr() {return flattened_Arr;}

    //get the layer type
    LayerType getLayerType() override {return type;}


    vector<double> backwardProp(const vector<double>& outputError);
    void applyOptimizer(Optimizer* opt);
};

class outputLayer : public Layer
{
    private:
    //any additional data
    LayerType type = output;


    public:
    outputLayer();

    //additional functions

    LayerType getLayerType() override {return type;}

};


#endif
