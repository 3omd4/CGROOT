#include "layers.h"

convLayer::convLayer(unsigned int outputVectorSize, kernelType KernelInitValues)
    : Layer(outputVectorSize)
{
    kernels.push_back(KernelInitValues);
    //make any additional initialization needed
}