#include "layers.h"


Layer::Layer(unsigned int outputVectorSize)
{
    //resize the toNextLayer vector (the layer output vector) to the right size
     toNextLayer.resize(outputVectorSize,0);
}