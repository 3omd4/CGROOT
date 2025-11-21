
#include "layers.h"
#include "../Initialization/initialization.h"

//the fully connected layer constructor
//input:        -numOfNeurons 
//              -actFunc (the layer activation function)
//              -initFuc (the layer initializaiton function)
//              -distType (the layer distribution used to initializet its weights)
//              -numOfWeigths (the number of weights per layer)
//output:       N/A
//side effect:  a Fully connected layer is constructed
//Note:         N/A
FullyConnected::FullyConnected(size_t numOfNeurons, activationFunction actFunc, 
                initFunctions initFunc, distributionType distType,
                        size_t numOfWeights)
    :act_Funct(actFunc)
{
    //resize the neurons vector to the number of neurons
    neurons.resize(numOfNeurons);

    //initialize the weigths of each neuron
    for(size_t i = 0; i < numOfNeurons; i++)
    {
        //extend the weights vector of the neuron 
        neurons[i].assign(numOfWeights,0.0);

        //use the initialization function
        switch(initFunc)
        {
        case Kaiming:
            init_Kaiming(neurons[i], numOfWeights, distType);
            break;
        case Xavier:
            init_Xavier(neurons[i], numOfWeights, numOfNeurons, distType);
            break;
        }
    }
    
}