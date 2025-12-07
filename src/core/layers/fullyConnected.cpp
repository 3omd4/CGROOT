
#include "core/definitions.h"
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
                        size_t numOfWeights) : act_Funct(actFunc)
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
    this->inputCache = inputData; //SAVE INPUT FOR BACKPROP
    
    // Resize output and reset
    outputData.assign(neurons.size(), 0.0);
    preActivation.assign(neurons.size(), 0.0); // Storage for 'z'

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

    // 1. Handle Vector-wise Activations (Softmax)
    if (act_Funct == Softmax) 
    {
        softmax_Funct(outputData); // Run ONCE on the whole vector
    }
    // 2. Handle Element-wise Activations (ReLU, Sigmoid, Tanh)
    else 
    {
        for(size_t i = 0; i < outputData.size(); i++)
        {
            switch(act_Funct)
            {
            case RelU:
                reLU_Funct(outputData[i]);
                break;
            case Sigmoid:
                sigmoid_Funct(outputData[i]);
                break;
            case Tanh:
                tanh_Funct(outputData[i]);
                break;
            }
        }
    }
}



vector<double> FullyConnected::backwardProp(const vector<double>& outputError) {
    size_t inputSize = inputCache.size();
    size_t outputSize = neurons.size();
    
    vector<double> inputError(inputSize, 0.0); // Error to send to previous layer

    // Initialize gradients
    biasGradients.assign(outputSize, 0.0);
    weightGradients.resize(outputSize);

    for(size_t i = 0; i < outputSize; i++) {
        // 1. Calculate Activation Slope
        double derivative = 0.0;
        switch(act_Funct) {
            case RelU: derivative = reLU_Prime(preActivation[i]); break;
            case Sigmoid: derivative = sigmoid_Prime(preActivation[i]); break;
            case Tanh: derivative = tanh_Prime(preActivation[i]); break;
            // ...
        }

        // 2. Calculate Delta (Error * Slope)
        double delta = outputError[i] * derivative;

        // 3. Calculate Bias Gradient (dL/db = delta)
        biasGradients[i] = delta;

        // 4. Calculate Weight Gradients (dL/dw = delta * input)
        weightGradients[i].resize(inputSize);
        for(size_t j = 0; j < inputSize; j++) {
            weightGradients[i][j] = delta * inputCache[j];
            
            // 5. Accumulate Error for Previous Layer (dL/dx_prev = w * delta)
            inputError[j] += delta * neurons[i][j];
        }
    }

    return inputError; // Send this back to the previous layer
}

void FullyConnected::applyOptimizer(Optimizer* opt) {
    // 1. Update Weights for each neuron (Must be done in loop because it's a vector of vectors)
    for(size_t i = 0; i < neurons.size(); i++) {
        opt->update(neurons[i], weightGradients[i]);
    }

    // 2. Update ALL Biases at once
    opt->update(bias, biasGradients);
}
