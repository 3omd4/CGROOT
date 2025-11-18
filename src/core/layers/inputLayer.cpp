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
    size_t size = data.size();
    if (size == 0) return;
    size_t stepSize = data[0].size();
    for(size_t i = 0; i < size; i++)
    {
        for(size_t j = 0; j < stepSize; j++)
        {
            layerOutput[i*stepSize + j] = static_cast<double>(data[i][j]);
        }
    }
}