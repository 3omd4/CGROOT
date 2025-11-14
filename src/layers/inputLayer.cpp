#include "layers.h"

inputLayer::inputLayer(unsigned int outputVectorSize) : Layer(outputVectorSize)
{
    //make any additional initialization needed
}

void inputLayer::start(image data)
{
    flatten(data);
}

void inputLayer::flatten(image& data)
{
    unsigned int size = data.size();
    unsigned int stepSize = data[0].size();
    for(unsigned int i = 0; i < size; i++)
    {
        for(unsigned int j = 0; j < stepSize; j++)
        {
            layerOutput[i*stepSize + j] = data[i][j];
        }
    }
}