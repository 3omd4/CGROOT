
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


    //initialze the biases
    //because of the Dying ReLU problem, if the activation function is ReLU then 
    //initialize with 0.01, else initialize with zero
    if(actFunc == RelU)
    {
        bias.assign(numOfNeurons, 0.01);
    }
    else
    {
        bias.assign(numOfNeurons, 0.0);
    }
    
}

//forward propagate the input data to the output
//input:        inputData
//output:       N/A
//side effects: the outputData vector is filled with the dot product 
//              of the input data and each neuron weights
//Note:         N/A
void FullyConnected::forwardProp(vector<double>& inputData)
 {
    //do the dot product with input vector and the weights 
    //of each neurons and store the result in the corrisponding
    //entry in the outputData vector
    for(size_t i = 0; i < neurons.size(); i++)
    {
        //do the dot product
        for(size_t j = 0; j < inputData.size(); j++)
        {
            outputData[i] += neurons[i][j]*inputData[j];
        }
        //add the bias
        outputData[i] += bias[i];
    }
}