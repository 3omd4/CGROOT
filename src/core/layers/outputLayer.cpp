#include "../Initialization/initialization.h"
#include "layers.h"
#include <iostream>
#include <math.h>



//the output layer constructor
//input:                -numOfClasses
//                      -numOfWeights
//                      -distType
//output:               N/A
//side effect:          the output layer is constructed
//Note:                 N/A
outputLayer::outputLayer(size_t numOfClasses,  size_t numOfWeights, distributionType distType, OptimizerConfig optConfig)
{

        //resize the neurons and weights gradients vectors
        //to the number of neurons
        neurons.resize(numOfClasses);
        d_weights.resize(numOfClasses);
        neuronOptimizers.resize(numOfClasses);

        //initialize the weights of each neuron
        for(size_t i = 0; i < numOfClasses; i++)
        {
                //resize the weight vector inside each neuron
                neurons[i].assign(numOfWeights,0.0);

                //use the initialization function
                //Xavier is used since the activation function will
                //always be softmax
                init_Xavier(neurons[i], numOfWeights, numOfClasses, distType);

                //resize the 2nd Dimension of the weights gradient vector
                //which corrisponds to the weights of each neuron
                d_weights[i].assign(numOfWeights, 0.0);

                // Initialize Optimizer for this neuron
                neuronOptimizers[i] = createOptimizer(optConfig);
        }

        //resize bias and biase gradient vectors
        bias.assign(numOfClasses, 0.0);  //the initializtion is to zero since softmax will always be used
        d_bias.assign(numOfClasses, 0.0);
        biasOptimizer = createOptimizer(optConfig);
        //resize the gradient vector which will be used by the previous layer
        prevLayerGrad.assign(numOfWeights, 0.0);
}


//forward propagate the input into the corrisponding classes
//input:                inputData
//output:               N/A
//side effects:         the outputData vector is filled with the output of
//                      the activation function (softmax) of the dot product
//                      of the input data and each neuron weights
//Note:                 N/A
void outputLayer::forwardProp(vector<double> &inputData) {
  // Resize outputData to ensure it matches the number of neurons
  if (outputData.size() != neurons.size()) {
    outputData.assign(neurons.size(), 0.0);
  }

  // the sum of each entry of the output data (the output of the dot product)
  // and will be used in applying softmax

  double sum = 0.0;

  //parallize forward propagation
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < static_cast<int>(neurons.size()); i++) 
  {

    // do the dot product
    double result = 0.0;
    for (size_t j = 0; j < inputData.size(); j++) 
    {
      result += neurons[i][j] * inputData[j];
    }
    // add the bias
    result += bias[i];
    result = exp(result);
    outputData[i] = result;
    sum += result; // add the value of this entry to the sum
  }

  // apply the softmax activation function
  #pragma omp parallel for
  for (int i = 0; i < static_cast<int>(outputData.size()); i++) 
  {
    outputData[i] /= sum;
  }
}

//backward propagate the error
//input:                -inputData
//                      -correctClass
//output:               N/A
//side effect:          the d_bias and d_weights are filled with the gradients
//                      and prevLayerGrad is filled with the error to be propagated
//Note:                 This function works with SGD or for updating after a single
//                      example, if the update should happen after multiple examples,
//                      then use bacwardProp_batch() instead
void outputLayer::backwardProp(vector<double> &inputData, size_t correctClass) {
  // fill prevLayerGrad with zeros to be filled with the new gradients
  fill(prevLayerGrad.begin(), prevLayerGrad.end(), 0.0);

  // iterate over every neuron in the output layer and apply the backward
  // propagation alorithm

  #pragma omp parallel for
  for (int i = 0; i < static_cast<int>(neurons.size()); i++) {

    // calculate the error, dZ_n = A_n - Y
    // as Y is the correct value
    // A_n is the output of the activation function, softmax in this case
    // the n in here indicates that this is the last/output layer
    double dZ_n = (outputData[i] - ((i == correctClass) ? 1 : 0));

    // d_bias = dZ_n in the case of SGD
    d_bias[i] = dZ_n;

    // iterate over each weight of this neuron and calculate d_W
    for (size_t j = 0; j < neurons[i].size(); j++) 
    {
      // calculate the weights gradient
      d_weights[i][j] = dZ_n * inputData[j]; 
    }

    // iterate over each weight of this neuron and calculate 
    // the error to be used by the previous layer
    for (size_t j = 0; j < neurons[i].size(); j++) 
    {
      // calculate the error of the previous layer
      // the error is calculated for each neuron and then sumed up
      #pragma omp atomic
      prevLayerGrad[j] += dZ_n * neurons[i][j];
    }
  }
}

//backward propagate the error
//input:                -inputData
//                      -correctClass
//output:               N/A
//side effect:          the d_bias and D_weights are filled with the accumlated gradients
//                      and prevLayerGrad is filled with the error to be propagated
//Note:                 This function works with BGD or for updating after a whole batch
//                      of examples, if the update should happen after a single example,
//                      then use bacwardProp() instead
void outputLayer::backwardProp_batch(vector<double> &inputData,
                                     size_t correctClass) {
  // fill prevLayerGrad with zeros to be filled with the new gradients
  fill(prevLayerGrad.begin(), prevLayerGrad.end(), 0.0);

  // iterate over every neuron in the output layer and apply the backward
  // propagation alorithm

  #pragma omp parallel for
  for (int i = 0; i < static_cast<int>(neurons.size()); i++) {
    // calculate the error, dZ_n = A_n - Y
    // as Y is the correct value
    // A_n is the output of the activation function, softmax in this case
    // the n in here indicates that this is the last/output layer
    double dZ_n = outputData[i] - ((i == correctClass) ? 1 : 0);

    // d_bias_new = d_bias_old dZ_n in the case of BGD
    // accumlates the bias gradient
    d_bias[i] += dZ_n;

    // iterate over each weight of this neuron and calculate d_W
    for (size_t j = 0; j < neurons[i].size(); j++) 
    {
      // calculate the weights gradient
      d_weights[i][j] += dZ_n * inputData[j]; 
    }

    // iterate over each weight of this neuron and calculate 
    // the error to be used by the previous layer
    for (size_t j = 0; j < neurons[i].size(); j++) 
    {
      // calculate the error of the previous layer
      // the error is calculated for each neuron and then sumed up
      #pragma omp atomic
      prevLayerGrad[j] += dZ_n * neurons[i][j];
    }
  }
}

//update the weights and biases of the output layer
//input:                learningRate
//output:               N/A
//side effect:          the weights and biases are updated
//Note:                 N/A
void outputLayer::update() {
  // iterate over each neuron and update its bias and weights

  #pragma omp parallel for
  for (int i = 0; i < static_cast<int>(neurons.size()); i++) 
  {
    neuronOptimizers[i]->update(neurons[i], d_weights[i]);
    
    //clear gradients
    fill(d_weights[i].begin(), d_weights[i].end(), 0.0);
  }

  // Update biases
  biasOptimizer->update(bias, d_bias);
  fill(d_bias.begin(), d_bias.end(), 0.0);
}

//update the weights and biases of the output layer after a batch
//input:                -learningRate
//                      -numOfExamples
//output:               N/A
//side effect:          the weights and biases are updated
//Note:                 N/A
void outputLayer::update_batch(int numOfExamples) 
{
   //calculate the value to average the gradients
  double scale = 1.0 / static_cast<double>(numOfExamples);

  // iterate over each neuron and update its bias and weights
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(neurons.size()); i++) 
  {
    // average the weight gradients
    for (size_t j = 0; j < d_weights[i].size(); j++) 
    {
      d_weights[i][j] *= scale;
    }

  //average the bias gradients
    d_bias[i] *= scale;

    //update weights
    neuronOptimizers[i]->update(neurons[i], d_weights[i]);

    //reset gradients
    fill(d_weights[i].begin(), d_weights[i].end(), 0.0);
  }


  //updates biases
  biasOptimizer->update(bias, d_bias);
  fill(d_bias.begin(), d_bias.end(), 0.0);
}

// get the class num of the image
// input:                N/A
// output:               int value(the number/index of the class)
// side effect:          N/A
// note:                 the num of the class is in the same order
//                       that is given to the backwardProp() function
int outputLayer::getClass() {
  int maxValIndex = 0;
  for (size_t i = 0; i < outputData.size(); i++) {
    if (outputData[maxValIndex] < outputData[i]) {
      maxValIndex = (int)i;
    }
  }

  return maxValIndex;
}




outputLayer::~outputLayer() {
for (Optimizer *opt : neuronOptimizers) {
    delete opt;
  }
  neuronOptimizers.clear();
  delete biasOptimizer;
  biasOptimizer = nullptr;
}
