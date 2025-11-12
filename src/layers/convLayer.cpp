#include "layers.h"


convLayer::convLayer(unsigned int outputVectorSize, dataType KerenlInitValues)
    : Layer(outputVectorSize)
    {
        kernel.assign(KerenlInitValues.begin(), KerenlInitValues.end());
        //make any additional initialization needed
    }