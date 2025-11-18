#ifndef _MODEL_H
#define _MODEL_H

#include <vector>
#include <random>
#include "layers/layers.h"

using std::vector;

// image typedef is defined in layers/layers.h to avoid circular dependency

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
    activation,
    conv,
    pooling,
    maxNumOfLayerTypes,
};

//the specifications of the kernels of each conv layer
//is specifies the number of kernels and its height and
//width of the layer, these are specified before the 
//model construction to be used in the model construction
struct convKernels
{
    unsigned int numOfKerenels;
    unsigned int kernel_width;
    unsigned int kernel_height;
};

//the struct that determine the arhitecture of the model
//which is used only to initialize the model
//contains information about the architecture
struct architecture
{
    unsigned int numOfLayers;
    vector<unsigned int*> neuronsPerActLayer;      //the number of neurons per layer, in other words
                                        //the size of the output vector of each layer
                                        //this is only for activation layers

    vector<activationFunction*> layerActivationFunc;    //the type of the activation function used in activation layers
                                                //for this model, the type of the activation function determines
                                                //the type of the initialization function used
    vector<LayerType> layersTypes;  //the type of each layer in order, these constitutes the 
                                    //model architecture layers

    vector<convKernels> convLayersKernels;  //the specification of the kernels of each layer

    //additional data used to initialize the data
};



class NNModel
{
    private:
    vector<Layer*> Layers;
    image data;

    public:
    //make an array of layers and specify the number of neurons of each layer and the type of each layer
    //specify the activation and initialization functions
    //initialize the kernel and weights of each layer(initalize/construct each layer)
    //assumptions made: the height and width of the image is constant for a single model(object)
                    //and so is the architecture of a single model (object)
    NNModel(struct architecture, unsigned int numOfClasses, unsigned int imageVerDim, 
                                                unsigned int imageHorDim);
    /*construction and initialization helper functions*/

    //function to calculate the number of neurons of the conv layer
    unsigned int calcNumOfNPConvLayer(unsigned int kernelDimension, unsigned int inputVerDim,
                                        unsigned int inputHorDim);
    /*******************end helpers*******************/

    //this function takes the input image to train the model
    void train(image data, int trueOutput);

    //this function is used after the model is trained 
    //to either validate the model or use it
    //returns an int based on the classification
    int classify(image data);

    //initialization function here

    /*He initialization function*/
    void intialization_He(vector<double>& arr, unsigned int numOfInputs);



    /*Xavier intialization function*/


    //additional functions
    
};

#endif