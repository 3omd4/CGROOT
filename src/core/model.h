#ifndef _MODEL_H
#define _MODEL_H

#include <vector>
#include <random>
#include "layers/layers.h"
#include "definitions.h"
#include "optimizers/optimizer.h"

using std::vector;

// image typedef is defined in layers/layers.h to avoid circular dependency


//the struct that determine the arhitecture of the model
//which is used only to initialize the model
//contains information about the architecture
struct architecture
{
    size_t numOfConvLayers;     //the number of convolution layers

    size_t numOfFCLayers;       //the number of fully connected layers

    vector<convKernels> kernelsPerconvLayers;  //the specification of the kernels of each convolution layer

    vector<size_t> neuronsPerFCLayer;      //the number of neurons per layer, in other words
                                            //the size of the output vector of each layer
                                            //this is only for fully connected layers

    vector<activationFunction> convLayerActivationFunc;    //the type of the activation function used in each convolution layer

    vector<activationFunction> FCLayerActivationFunc;    //the type of the activation function used in each fully connected layer

    vector<initFunctions> convInitFunctionsType;    //the type of the functions which will be used to
                                                    //initialize each convolution layer

    vector<initFunctions> FCInitFunctionsType;      //the type of the functions which will be used to
                                                    //initialize each fully connected layer

    distributionType distType; //type of distribution to be used by all layers initializers

    vector<size_t> poolingLayersInterval;   //the number of convolution layers after which a
                                            //pooling layer is inserted

    vector<poolingLayerType> poolingtype;   //the type of each pooling layer(max or average)

    vector<poolKernel> kernelsPerPoolingLayer; //the information about each kernel of  each
                                               //pooling layer and the number of strides

    double learningRate;        //the model learning rate

    //additional data used to initialize the data
};



class NNModel
{
    private:

        vector<Layer*> Layers;
        image data;


    double learningRate;        //the model learning rate

    public:
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
    NNModel(struct architecture, size_t numOfClasses, size_t imageVerDim,
                                                size_t imageHorDim, size_t imageDepDim);
    
    // NNModel constructor helper function:
    // calculates the dimension of the output feature map of each convolution layer
    // input:        -current layer kernel height (size_t kernelHeight)
    //               -current layer kernel width (size_t kernelWidth)
    //               -the input feature map height (size_t inputHeight)
    //               -the input feature map width (size_t inputWidth)
    // output:       -a featureMapDim struct that carries information about the dimensions of the current
    //               output feature map (featureMapDim)
    // side effect:  N/A
    // Note:         the function also sets the data member featureMapDim.FM_depth to 0, so it must
    //               setted later
    featureMapDim calcFeatureMapDim(size_t kernelHeight, size_t kernelWidth, size_t inputHeight,
                                        size_t inputWidth);


    //train the model with a single image
    //input:        -data (an image)
    //              -trueOutput (the value for which the model compares its output)
    //output:       N/A
    //side effect:  The model is trained by a single image and its paramters are updated
    //Note:         N/A
    void train(image data, int trueOutput);

    void train_batch(vector<image> data, vector<int> trueOutput);

    //classify the image by applying the forward propagation on the image
    //input:        data (the image)
    //output:       int (the class of the image)
    //side effect:  N/A
    //Note:         This function is either called directly to get the image class
    //              or by the train fucntion to train the model
    int classify(image data);

    ~NNModel() {
    for(Layer* l : Layers) {
        delete l;
    }
    Layers.clear();
}

};

#endif
