#include "layers.h"


Layer::Layer(unsigned int outputVectorSize)
{
    //resize the toNextLayer vector (the layer output vector) to the right size
     layerOutput.resize(outputVectorSize,0);
}

//get the layer output data(neurons output)
Layer::outType& Layer::getLayerOutput()
{
    return layerOutput;
}