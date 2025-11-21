#include <iostream>
using namespace std;

#include "model.h"



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
NNModel::NNModel(architecture modelArch, size_t numOfClasses, size_t imageHeight, 
                                                size_t imageWidth, size_t imageDepth)
{
    //(optional) additional initialization code  

    //construct the input layer
    Layers.emplace_back(new inputLayer(imageHeight,imageWidth,imageDepth));
    

    //start initializaing the convolution and pooling layers
    //a pooling layer is inserted after every a number of convolution layers
    for(size_t i = 0; i < modelArch.numOfConvLayers; i++)
    {
        //insert a pooling layer if
        //1. it isn't the first layer after the input layer
        //2. poolingLayersInterval >= 1 , which means that pooling layer insertion is allowed
        //3. the interval of convolution layers is compeleted so pooling layer insertion is allowed
        //if poolingLayersInterval == 1, then a pooling layer is inserted after each convolution layer
        if((i) && modelArch.poolingLayersInterval && !(i%modelArch.poolingLayersInterval))
        {
            //insert a pooling layer
            //Layers.emplace_back(new poolingLayer());
        }



        featureMapDim FM_dims; //a struct to hold the dimensions of the feature map of the next convolution layer        //here the dimesions of the feature map of the next convolution layer is calculated
        
        //using calcFeatureMapDim function, which takes the current kernel 2D dimensions 
        //and the dimension of the image, the feature map of last pooling layer 
        //or the feature map of the last convolution layer
        if(i && (i%modelArch.poolingLayersInterval))
        {
        //enter the condition if:
        //1. this isn't the first iteration (otherwise use the image dimensions)
        //2. the last layer isn't a pooling layer
            FM_dims = calcFeatureMapDim(modelArch.kernelsPerconvLayers[i].kernel_height,
                 modelArch.kernelsPerconvLayers[i].kernel_width, dynamic_cast<convLayer*>(Layers[Layers.size()-1])->getFeatureMapHeight(),
                                    dynamic_cast<convLayer*>(Layers[Layers.size()-1])->getFeatureMapWidth());
        }
        else if(i)
        {
        //enter the condition if:
        //1. this isn't the first iteration (otherwise use the image dimensions)
        //this condition is entered in the case that the last layer is a pooling layer


            //FM_dims = calcFeatureMapDim(modelArch.kernelsPerconvLayers[i].kernel_height,
            //     modelArch.kernelsPerconvLayers[i].kernel_width, dynamic_cast<poolingLayer*>(Layers[Layers.size()-1])->getFeatureMapHeight(),
            //                        dynamic_cast<poolingLayer*>(Layers[Layers.size()-1])->getFeatureMapWidth());
        }
        else
        {
        //enter here if this is the first iteration so use the image dimensions

            FM_dims = calcFeatureMapDim(modelArch.kernelsPerconvLayers[i].kernel_height,
                 modelArch.kernelsPerconvLayers[i].kernel_width, imageHeight, imageWidth);
        }

        //get the kernel depth dimension either from the number of the kernels of 
        //the last convlution layer or the image
        modelArch.kernelsPerconvLayers[i].kernel_depth = (i)?modelArch.kernelsPerconvLayers[i-1].numOfKerenels:imageDepth;
        
        //get the feature map depth of the current conv layer from the current 
        //number of kernels
        FM_dims.FM_depth = modelArch.kernelsPerconvLayers[i].numOfKerenels;
        
        //consturct a convolution layer
        Layers.emplace_back(new convLayer(modelArch.kernelsPerconvLayers[i], 
                modelArch.convLayerActivationFunc[i], modelArch.convInitFunctionsType[i]
                    ,modelArch.distType, FM_dims));

    }

    //a pooling layer can be inserted after the last convolution layer..
    //but the flatten layer construction has to be handeled correctly

    //here the flatten layer is inserted to make the data ready for the 
    //fully connected layers
    if(modelArch.numOfConvLayers)
    {        
    //enter here if there were convolution layers before

        Layers.emplace_back(new FlattenLayer(modelArch.kernelsPerconvLayers[modelArch.numOfConvLayers-1].kernel_height,
                                modelArch.kernelsPerconvLayers[modelArch.numOfConvLayers-1].kernel_width,
                                modelArch.kernelsPerconvLayers[modelArch.numOfConvLayers-1].kernel_depth));
    }
    else
    {
    //enter here if there were no convolution layers before
        Layers.emplace_back(new FlattenLayer(imageHeight,imageWidth,imageDepth));
    }


    //start making fully connected layers
    for(size_t i = 0; i < modelArch.numOfFCLayers; i++)
    {
        size_t numOfWeights;
        //set the number of weights per layer
        if(i)
        {
        //if the last layer is fully connected
            numOfWeights = dynamic_cast<FullyConnected*>(Layers[Layers.size() -1])->getOutputSize();
        }
        else
        {
        //if the last layer is flatten layer, that is, the first iteration
            numOfWeights = dynamic_cast<FlattenLayer*>(Layers[Layers.size() -1])->getFlattenedArrSize();
        }

        //construct the fully connected layer
        Layers.emplace_back(new FullyConnected(modelArch.neuronsPerFCLayer[i],
            modelArch.FCLayerActivationFunc[i], modelArch.FCInitFunctionsType[i],
                    modelArch.distType,numOfWeights));
    }

    //construct the output layer
    //Layers.emplace_back(new outputLayer());

    //(optional) additional initialization code  
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
featureMapDim NNModel::calcFeatureMapDim(size_t kernelHeight, size_t kernelWidth, size_t inputHeight,
                                        size_t inputWidth)
{
    inputWidth = inputWidth - kernelWidth + 1;
    inputHeight = inputHeight - kernelHeight + 1;
    return featureMapDim{inputHeight, inputWidth, 0};
}

// -------------------------------------------------------------
// TEMPORARY IMPLEMENTATION TO FIX LINKER ERRORS
// These are placeholders until training logic is added
// -------------------------------------------------------------

void NNModel::train(image data, int trueOutput)
{
    // TODO: Add actual forward/backprop logic here

    // For now, simply store the last input image so classify() can use it
    this->data = data;

    // Debug print (optional)
    std::cout << "[NNModel] train() called. trueOutput = " << trueOutput << "\n";
}


int NNModel::classify(image data)
{
    // TODO: Add real forward pass here

    // For now return a dummy class = 0
    std::cout << "[NNModel] classify() called.\n";

    return 0;
}
