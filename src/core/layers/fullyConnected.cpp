

#include "../Initialization/initialization.h"
#include "layers.h"

// Destructor to clean up optimizers
FullyConnected::~FullyConnected() {
  for (Optimizer *opt : neuronOptimizers) {
    delete opt;
  }
  neuronOptimizers.clear();
  delete biasOptimizer;
  biasOptimizer = nullptr;
}

FullyConnected::FullyConnected(size_t numOfNeurons, activationFunction actFunc,
                               initFunctions initFunc,
                               distributionType distType, size_t numOfWeights,
                               OptimizerConfig optConfig)
    : act_Funct(actFunc) {
  // resize the neurons and weights gradient vectors to the number of neurons
  neurons.resize(numOfNeurons);
  d_weights.resize(numOfNeurons);
  neuronOptimizers.resize(numOfNeurons); // Resize optimizer vector

  // initialize the weigths of each neuron
  for (size_t i = 0; i < numOfNeurons; i++) {
    // extend the weights vector of the neuron
    neurons[i].assign(numOfWeights, 0.0);

    // use the initialization function
    switch (initFunc) {
    case Kaiming:
      init_Kaiming(neurons[i], numOfWeights, distType);
      break;
    case Xavier:
      init_Xavier(neurons[i], numOfWeights, numOfNeurons, distType);
      break;
    }
    // resize the 2nd Dimension of the weights gradient vector
    // which corrisponds to the weights of each neuron
    d_weights[i].assign(numOfWeights, 0.0);

    // Initialize Optimizer for this neuron
    neuronOptimizers[i] = createOptimizer(optConfig);
  }

  // initialze the biases
  // because of the Dying ReLU problem, if the activation function is ReLU then
  // initialize with 0.01, else initialize with zero
  if (actFunc == RelU) {
    bias.assign(numOfNeurons, 0.01);
  } else {
    bias.assign(numOfNeurons, 0.0);
  }

  // resize the bias gradient vector
  d_bias.assign(numOfNeurons, 0.0);

  // Initialize Bias Optimizer
  biasOptimizer = createOptimizer(optConfig);

  // resize the gradient vector which will be used by the previous layer
  prevLayerGrad.assign(numOfWeights, 0.0);

  // Resize outputData to expected size
  outputData.resize(numOfNeurons, 0.0);
}

// ... methods forwardProp, backwardProp ...
// (Skipping them as they don't change logic, but I need to be careful with
// replace_file_content not to delete them if I don't provide them) Wait, I am
// replacing the WHOLE file content from constructor downwards? No. I can only
// replace the Constructor and the Update methods. I will split this into
// chunks. Re-reading 'layers.h' implementation plan: I need to update
// constructor and update methods.

// update the weights and biases of this fully connected layer
void FullyConnected::update() {
// Parallelize neuron updates
#pragma omp parallel for
  for (int i = 0; i < (int)neurons.size(); i++) {
    // iterate over each weight and update it
    neuronOptimizers[i]->update(neurons[i], d_weights[i]);

    // Clear gradients
    fill(d_weights[i].begin(), d_weights[i].end(), 0.0);
  }

  // Update Biases (All at once)
  biasOptimizer->update(bias, d_bias);
  fill(d_bias.begin(), d_bias.end(), 0.0);
}

// update the weights and biases of this fully connected layer after a batch
void FullyConnected::update_batch(int numOfExamples) {
  double scale = 1.0 / static_cast<double>(numOfExamples);

// Parallelize neuron updates
#pragma omp parallel for
  for (int i = 0; i < (int)neurons.size(); i++) {
    // Scale gradients
    for (size_t j = 0; j < d_weights[i].size(); j++) {
      d_weights[i][j] *= scale;
    }

    // Update weights
    neuronOptimizers[i]->update(neurons[i], d_weights[i]);

    // Reset gradients
    fill(d_weights[i].begin(), d_weights[i].end(), 0.0);
  }

  // Scale bias gradients
  for (size_t i = 0; i < d_bias.size(); i++) {
    d_bias[i] *= scale;
  }

  // Update biases
  biasOptimizer->update(bias, d_bias);
  fill(d_bias.begin(), d_bias.end(), 0.0);
}

// forward propagate the input data to the output
// input:        inputData
// output:       N/A
// side effects:         the outputData vector is filled with the output of
//                       the activation function of the dot product
//                       of the input data and each neuron weights
// Note:         N/A
void FullyConnected::forwardProp(vector<double> &inputData) {
// Parallelize the outer loop over neurons
#pragma omp parallel for
  for (int i = 0; i < (int)neurons.size(); i++) {
    // reset and compute dot product
    outputData[i] = 0.0;
    for (size_t j = 0; j < inputData.size(); j++) {
      outputData[i] += neurons[i][j] * inputData[j];
    }
    // add the bias
    outputData[i] += bias[i];

    // choose which activation function and apply it
    switch (act_Funct) {
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

// backward propagate the error
// input:                -inputData
//                       -thisLayerGrad
// output:               N/A
// side effect:          the d_bias and d_weights are filled with the gradients
//                       and prevLayerGrad is filled with the error to be
//                       propagated
// Note:                 This function works with SGD or for updating after a
// single
//                       example, if the update should happen after multiple
//                       examples, then use bacwardProp_batch() instead
void FullyConnected::backwardProp(vector<double> &inputData,
                                  vector<double> &thisLayerGrad) {
  // fill prevLayerGrad with zeros to be filled with the new gradients
  fill(prevLayerGrad.begin(), prevLayerGrad.end(), 0.0);

// Parallelize over neurons with reduction for prevLayerGrad
#pragma omp parallel for
  for (int i = 0; i < (int)neurons.size(); i++) {
    // contains the value of the derivative of the activation function
    double derivative = 0.0;

    // choose which activation function is used
    switch (act_Funct) {
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

    // calculate dZ_l = dA1 * g'(Zl)
    double dZ_l = derivative * thisLayerGrad[i];

    // d_bias = dZ_n in the case of SGD
    d_bias[i] = dZ_l;

    // iterate over each weight of this neuron and calculate d_W
    // and the error to be used by the previous layer
    for (size_t j = 0; j < neurons[i].size(); j++) {
      // calculate the weights gradient
      d_weights[i][j] = dZ_l * inputData[j];

// calculate the error of the previous layer (thread-safe atomic update)
#pragma omp atomic
      prevLayerGrad[j] += dZ_l * neurons[i][j];
    }
  }
}

// backward propagate the error
// input:                -inputData
//                       -thisLayerGrad
// output:               N/A
// side effect:          the d_bias and D_weights are filled with the accumlated
// gradients
//                       and prevLayerGrad is filled with the error to be
//                       propagated
// Note:                 This function works with BGD or for updating after a
// whole batch
//                       of examples, if the update should happen after a single
//                       example, then use bacwardProp() instead
void FullyConnected::backwardProp_batch(vector<double> &inputData,
                                        vector<double> &thisLayerGrad) {
  // fill prevLayerGrad with zeros to be filled with the new gradients
  fill(prevLayerGrad.begin(), prevLayerGrad.end(), 0.0);

// Parallelize over neurons with reduction for prevLayerGrad
#pragma omp parallel for
  for (int i = 0; i < (int)neurons.size(); i++) {
    // contains the value of the derivative of the activation function
    double derivative = 0.0;

    // choose which activation function is used
    switch (act_Funct) {
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

    // calculate dZ_l = dA1 * g'(Zl)
    double dZ_l = derivative * thisLayerGrad[i];

// accumulate the bias gradient
#pragma omp atomic
    d_bias[i] += dZ_l;

    // iterate over each weight of this neuron and calculate d_W
    // and the error to be used by the previous layer
    for (size_t j = 0; j < neurons[i].size(); j++) {
      // accumulate the weights gradient
      d_weights[i][j] += dZ_l * inputData[j];

// calculate the error of the previous layer (thread-safe atomic update)
#pragma omp atomic
      prevLayerGrad[j] += dZ_l * neurons[i][j];
    }
  }
}
