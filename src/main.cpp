#include <iostream>

#include "core/model.h"
#include "core/utils/mnist_loader.h"



int main() {

  architecture modelArch;

  modelArch.numOfConvLayers = 0;
  modelArch.numOfFCLayers = 2;
  modelArch.neuronsPerFCLayer = {128, 10};
  modelArch.FCLayerActivationFunc = {RelU, RelU};
  modelArch.FCInitFunctionsType = {Xavier, Xavier};
  modelArch.distType = normalDistribution;
  size_t numOfClasses = 10;
  size_t imageHeight = 28;
  size_t imageWidth = 28;
  size_t imageDepth = 1;

  OptimizerConfig optConfig;

  optConfig.type = opt_SGD;
  optConfig.learningRate = 0.01;
  optConfig.weightDecay = 0.0005;

  Optimizer* opt = createOptimizer(optConfig);

  NNModel model(modelArch, numOfClasses, imageHeight, imageWidth, imageDepth);

  return 0;
}
