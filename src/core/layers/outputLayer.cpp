#include "layers.h"


outputLayer::outputLayer()
{
        //make any additional initialization needed
}
vector<double> outputLayer::backwardProp(const vector<double>& outputError) {
    return outputError; // Just pass through
}

void outputLayer::applyOptimizer(Optimizer* opt) {
    // Empty
}
