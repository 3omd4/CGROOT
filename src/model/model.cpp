#include "model.h"
#include "layers.h"
#include <math.h>
#include <random>
#include <iostream>
using namespace std;

//serves the initialization functions


NNModel::NNModel(architecture modelArch, unsigned int numOfClasses, unsigned int imageVerDim, 
                                                unsigned int imageHorDim)
{
    //(optional) additional initialization code  
    Layers.emplace_back(new inputLayer(imageHorDim*imageVerDim));
    unsigned int iter_actLayers;    //an index used to iterate the neuronsPerActLayer vector
    unsigned int iter_convLayers;   //an index used to iterate the convLayersKernels vector

    for(unsigned int i = 0; i < modelArch.numOfLayers; i++)
    {
        switch(modelArch.layersTypes[i])
        {
        case activation:
            //Layers.emplace_back(new activationLayer(modelArch.neuronsPerActLayer[iter_actLayers]));
            break;
        case conv:
            break;
        case pooling:
            break;
        default:
            cout << "error not correct layer type\n";
        }
    }
    


    //(optional) additional initialization code  
}


void NNModel::intialization_He(vector<double>& arr, unsigned int numOfInputs)
{
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, sqrt(2.0/static_cast<double>(numOfInputs)));
    for(unsigned int i = 0; i < arr.size(); i++)
    {
        arr[i] = dist(gen);
    }
}


unsigned int NNModel::calcNumOfNPConvLayer(unsigned int kernelDimension, unsigned int inputVerDim,
                                        unsigned int inputHorDim)
{
    inputHorDim -= kernelDimension;
    inputVerDim -= kernelDimension;
    return inputHorDim*inputVerDim;
}