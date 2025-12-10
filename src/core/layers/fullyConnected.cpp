
#include "core/definitions.h"
#include "layers.h"
#include "../Initialization/initialization.h"
#include <iostream>

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
    //resize the neurons and weights gradient vectors to the number of neurons
    neurons.resize(numOfNeurons);
    d_weights.resize(numOfNeurons);

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
        //resize the 2nd Dimension of the weights gradient vector
        //which corrisponds to the weights of each neuron
        d_weights[i].assign(numOfWeights, 0.0);
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

    //resize the bias gradient vector
    d_bias.assign(numOfNeurons, 0.0);

    //resize the gradient vector which will be used by the previous layer
    prevLayerGrad.assign(numOfWeights, 0.0);
    
    // Resize outputData to expected size
    outputData.resize(numOfNeurons, 0.0);
}

//forward propagate the input data to the output
//input:        inputData
//output:       N/A
//side effects:         the outputData vector is filled with the output of
//                      the activation function of the dot product
//                      of the input data and each neuron weights
//Note:         N/A
void FullyConnected::forwardProp(vector<double>& inputData) {

    //do the dot product with input vector and the weights 
    //of each neurons and store the result in the corrisponding
    //entry in the outputData vector

    // std::cout << "[DEBUG] FC forwardProp. Input: " << inputData.size() << " Neurons: " << neurons.size() << std::endl;

    // Debug Log (Optional, but requested)
    // // std::cout << "[DEBUG] FC forwardProp. Input: " << inputData.size() << " Neurons: " << neurons.size() << std::endl;

    for(size_t i = 0; i < neurons.size(); i++)
    {
        //reset the
        outputData[i] = 0.0;
        //do the dot product
        for(size_t j = 0; j < inputData.size(); j++)
        {
            outputData[i] += neurons[i][j]*inputData[j];
        }
        //add the bias
        outputData[i] += bias[i];

        //choose which activation function and apply it
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


//backward propagate the error
//input:                -inputData
//                      -thisLayerGrad
//output:               N/A
//side effect:          the d_bias and d_weights are filled with the gradients
//                      and prevLayerGrad is filled with the error to be propagated
//Note:                 This function works with SGD or for updating after a single
//                      example, if the update should happen after multiple examples,
//                      then use bacwardProp_batch() instead
void FullyConnected::backwardProp(vector<double>& inputData, vector<double>& thisLayerGrad)
{
    //fill prevLayerGrad with zeros to be filled with the new gradients
    fill(prevLayerGrad.begin(), prevLayerGrad.end(), 0.0);

    //iterate over every neuron in the output layer and apply the backward
    //propagation alorithm
    for(size_t i = 0; i < neurons.size(); i++)
    {

        //contains the value of the derivative of the activation function
        //contains the value of the derivative of the activation function
        double derivative = 0.0;

        //choose which activation function is used
        //the outputData[i] is used since its the result of the activation function
        switch(act_Funct)
        {
            case RelU:
                derivative = d_reLU_Funct(outputData[i]);
                break;
            case Sigmoid:
                derivative = d_sigmoid_Funct(outputData[i]);
                break;
            case Tanh:
                derivative = d_tanh_Funct(outputData[i]);
                break;
        }

        //calculate dZ_l = dA1 * g'(Zl)
        //as dA1 is thisLayerGrad, the error propagated from the next layer
        //g'(Z1) is g'(Z1)
        //l indicates that this error is of this layer
        double dZ_l = derivative * thisLayerGrad[i];

        //d_bias = dZ_n in the case of SGD
        d_bias[i] = dZ_l;


        //iterate over each weight of this neuron and calculate d_W
        //and the error to be used by the previous layer
        for(size_t j = 0; j < neurons[i].size(); j++)
        {
            //calculate the weights gradient
            d_weights[i][j] = dZ_l * inputData[j];

            //calculate the error of the previous layer
            //the error is calculated for each neuron and then sumed up
            prevLayerGrad[j] += dZ_l * neurons[i][j];
        }
    }
}

//backward propagate the error
//input:                -inputData
//                      -thisLayerGrad
//output:               N/A
//side effect:          the d_bias and D_weights are filled with the accumlated gradients
//                      and prevLayerGrad is filled with the error to be propagated
//Note:                 This function works with BGD or for updating after a whole batch
//                      of examples, if the update should happen after a single example,
//                      then use bacwardProp() instead
void FullyConnected::backwardProp_batch(vector<double>& inputData, vector<double>& thisLayerGrad)
{
    //fill prevLayerGrad with zeros to be filled with the new gradients
    fill(prevLayerGrad.begin(), prevLayerGrad.end(), 0.0);

    //iterate over every neuron in the output layer and apply the backward
    //propagation alorithm
    for(size_t i = 0; i < neurons.size(); i++)
    {

        //contains the value of the derivative of the activation function
        //contains the value of the derivative of the activation function
        double derivative = 0.0;

        //choose which activation function is used
        //the outputData[i] is used since its the result of the activation function
        switch(act_Funct)
        {
            case RelU:
                derivative = d_reLU_Funct(outputData[i]);
                break;
            case Sigmoid:
                derivative = d_sigmoid_Funct(outputData[i]);
                break;
            case Tanh:
                derivative = d_tanh_Funct(outputData[i]);
                break;
        }

        //calculate dZ_l = dA1 * g'(Zl)
        //as dA1 is thisLayerGrad, the error propagated from the next layer
        //g'(Z1) is g'(Z1)
        //l indicates that this error is of this layer
        double dZ_l = derivative * thisLayerGrad[i];

        //d_bias_new = d_bias_old dZ_n in the case of BGD
        //accumlates the bias gradient
        d_bias[i] += dZ_l;


        //iterate over each weight of this neuron and calculate d_W
        //and the error to be used by the previous layer
        for(size_t j = 0; j < neurons[i].size(); j++)
        {
            //accumlates the weights gradient
            d_weights[i][j] += dZ_l * inputData[j];

            //calculate the error of the previous layer
            //the error is calculated for each neuron and then sumed up
            prevLayerGrad[j] += dZ_l * neurons[i][j];
        }
    }
}


//update the weights and biases of this fully connected layer
//input:                learningRate
//output:               N/A
//side effect:          the weights and biases are updated
//Note:                 N/A
void FullyConnected::update(double learningRate)
{
    //iterate over each neuron and update its bias and weights
    for(size_t i = 0; i < neurons.size(); i++)
    {
        //update the bias
        bias[i] -= learningRate*d_bias[i];
        d_bias[i] = 0.0;        //reset the bias gradient

        //iterate over each weight and update it
        for(size_t j = 0; j < neurons[i].size(); j++)
        {
            //update the weights
            neurons[i][j] -= learningRate*d_weights[i][j];
            d_weights[i][j] = 0.0; //reset the weigth gradient
        }
    }
}


//update the weights and biases of this fully connected layer after a batch
//input:                -learningRate
//                      -numOfExamples
//output:               N/A
//side effect:          the weights and biases are updated
//Note:                 N/A
void FullyConnected::update_batch(double learningRate, int numOfExamples)
{
    //iterate over each neuron and update its bias and weights
    for(size_t i = 0; i < neurons.size(); i++)
    {
        //update the bias, with averaged bias gradient
        bias[i] -= learningRate*(d_bias[i]/static_cast<double>(numOfExamples));
        d_bias[i] = 0.0;        //reset the bias gradient

        //iterate over each weight and update it
        for(size_t j = 0; j < neurons[i].size(); j++)
        {
            //update the weights, with averaged weights gradient
            neurons[i][j] -= learningRate*(d_weights[i][j]/static_cast<double>(numOfExamples));
            d_weights[i][j] = 0.0; //reset the weigth gradient
        }
    }
}
