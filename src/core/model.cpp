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
    

    //poolCount is used to count the number of convolution layers after which 
    //a pooling layer will be inserted
    size_t poolCount = 0;
    
    //an iterator that is used to iterate the poolying layers information vectors
    size_t poolIter = 0;

    //start initializaing the convolution and pooling layers
    //a pooling layer is inserted after a number of convolution layers
    for(size_t i = 0; i < modelArch.numOfConvLayers; i++)
    {
        

        featureMapDim FM_dims; //a struct to hold the dimensions of the feature map of the next convolution layer        
        //here the dimesions of the feature map of the next convolution layer is calculated
        
        //using calcFeatureMapDim function, which takes the current kernel 2D dimensions 
        //and the dimension of the image, the feature map of last pooling layer 
        //or the feature map of the last convolution layer
        if(i && !(Layers[Layers.size()-1]->getLayerType() != pooling))
        {
        //enter the condition if:
        //1. this isn't the first iteration (otherwise use the image dimensions)
        //2. the last layer isn't a pooling layer
            FM_dims = calcFeatureMapDim(modelArch.kernelsPerconvLayers[i].kernel_height,
                 modelArch.kernelsPerconvLayers[i].kernel_width, static_cast<convLayer*>(Layers[Layers.size()-1])->getFeatureMapHeight(),
                                    static_cast<convLayer*>(Layers[Layers.size()-1])->getFeatureMapWidth());
        }
        else if(i)
        {
        //enter the condition if:
        //1. this isn't the first iteration (otherwise use the image dimensions)
        //this condition is entered in the case that the last layer is a pooling layer


            FM_dims = calcFeatureMapDim(modelArch.kernelsPerconvLayers[i].kernel_height,
                 modelArch.kernelsPerconvLayers[i].kernel_width, static_cast<poolingLayer*>(Layers[Layers.size()-1])->getFeatureMapHeight(),
                                    static_cast<poolingLayer*>(Layers[Layers.size()-1])->getFeatureMapWidth());
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

        //increment the counter (or the number elapsed convolution layers)
        poolCount++;

        //insert a pooling layer if:
        //1. there is a pooling layers to be inserted
        //2. the number of convolution layers before the pooling layer have been reached 
        if((poolIter < modelArch.poolingLayersInterval.size()) && (poolCount == modelArch.poolingLayersInterval[poolIter]))
        {
            //reset the counter
            poolCount = 0;
            
            //make the dimensions of the output feature map of the pooling layer
            featureMapDim FM_dims_pooling;

            //the depth is the same as the last convolution layer
            FM_dims_pooling.FM_depth = static_cast<convLayer*>(Layers[Layers.size()-1])->getFeatureMapDepth();

            //calculate the height according to (H_in - H_f)/S_f + 1
            //where:    H_in : input feature map height
            //          H_f  : filter height 
            //          S_f  : stride
            FM_dims_pooling.FM_height = static_cast<convLayer*>(Layers[Layers.size()-1])->getFeatureMapHeight() - modelArch.kernelsPerPoolingLayer[poolIter].filter_height;
            FM_dims_pooling.FM_height = (FM_dims_pooling.FM_height/modelArch.kernelsPerPoolingLayer[poolIter].stride) + 1;

            //calculate the height according to (W_in - W_f)/S_f + 1
            //where:    W_in : input feature map width
            //          W_f  : filter width 
            //          S_f  : stride
            FM_dims_pooling.FM_width = static_cast<convLayer*>(Layers[Layers.size()-1])->getFeatureMapWidth() - modelArch.kernelsPerPoolingLayer[poolIter].filter_width;
            FM_dims_pooling.FM_width = (FM_dims_pooling.FM_width/modelArch.kernelsPerPoolingLayer[poolIter].stride) + 1;

            //the filter depth is the same as the input feature map depth
            //this useless though
            modelArch.kernelsPerPoolingLayer[poolIter].filter_depth = FM_dims_pooling.FM_depth;

            //add the pooling layer
            Layers.emplace_back(new poolingLayer(modelArch.kernelsPerPoolingLayer[poolIter], FM_dims, modelArch.poolingtype[poolIter]));

            //increment the iterator
            poolIter++;
        }

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
            numOfWeights = static_cast<FullyConnected*>(Layers[Layers.size() -1])->getOutputSize();
        }
        else
        {
        //if the last layer is flatten layer, that is, the first iteration
            numOfWeights = static_cast<FlattenLayer*>(Layers[Layers.size() -1])->getFlattenedArrSize();
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
    //simply store the last input image so classify() can use it
    this->data = data;

    //forward propagation
    int result = classify(data);

    // TODO: Add actual backprop logic here
    

    // Debug print (optional)
    std::cout << "[NNModel] train() called. trueOutput = " << trueOutput << "\n";
}

//classify the image by applying the forward propagation on the image
//input:        data (the image)
//output:       int (the class of the image)
//side effect:  N/A
//Note:         This function is either called directly to get the image class
//              or by the train fucntion to train the model 
int NNModel::classify(image data)
{
    //make the data ready to be processed by different layers
    static_cast<inputLayer*>(Layers[0])->start(data);

    //Iterate over all the layers after the input layer and before the output layer
    for(size_t i = 0; i < Layers.size()-1 ; i++)
    {
        //see which layer this is to call the forward propagation function
        //and if needed, see which is the last layer to get the data from
        switch(Layers[i]->getLayerType())
        {
            case conv:
                //for convolution layer check whether the last layer is the input layer,
                //pooling layer or another convoultion layer 
                switch(Layers[i-1]->getLayerType())
                {
                    case input:
                        static_cast<convLayer*>(Layers[i])->forwardProp(static_cast<inputLayer*>(Layers[i-1])->getOutput());
                        break;
                    case pooling:
                        static_cast<convLayer*>(Layers[i])->forwardProp(static_cast<poolingLayer*>(Layers[i-1])->getFeatureMaps());
                        break;
                    default:
                        static_cast<convLayer*>(Layers[i])->forwardProp(static_cast<convLayer*>(Layers[i-1])->getFeatureMaps());
                    break;
                }

                break;
            case pooling:
                //for the pooling layer, the last layer is always a convolution layer
                static_cast<poolingLayer*>(Layers[i])->forwardProp(static_cast<convLayer*>(Layers[i-1])->getFeatureMaps());
                break;
            case fullyConnected:
                //for the fully connected layer, the last layer is either the flatten layer or another
                //fully connected layer
                switch(Layers[i]->getLayerType())
                {
                    case flatten:
                        static_cast<FullyConnected*>(Layers[i])->forwardProp(static_cast<FlattenLayer*>(Layers[i-1])->getFlattenedArr());
                        break;
                    case fullyConnected:
                        static_cast<FullyConnected*>(Layers[i])->forwardProp(static_cast<FullyConnected*>(Layers[i-1])->getOutput());
                        break;
                }
                break;
            case flatten:
                //for the flatten layer, the last layer is either the input layer
                //(in Dense architecture) or a convolution layer(in CNN architecture)
                switch(Layers[i-1]->getLayerType())
                {
                    case input:
                        static_cast<FlattenLayer*>(Layers[i])->forwardProp(static_cast<inputLayer*>(Layers[i-1])->getOutput());
                    break;
                    case conv:
                        static_cast<FlattenLayer*>(Layers[i])->forwardProp(static_cast<convLayer*>(Layers[i-1])->getFeatureMaps());
                    break;
                }
               
                break;
            default:
            break;
        }
    }

    //code about the getting the final result from the output layer

    // For now return a dummy class = 0
    std::cout << "[NNModel] classify() called.\n";

    return 0;
}
