#include "core/model.h"
// #include <iostream>


int main() {

  architecture modelArch;

  modelArch.numOfConvLayers = 0;
  modelArch.numOfFCLayers = 2;
  modelArch.neuronsPerFCLayer = {128, 10};
  modelArch.FCLayerActivationFunc = {RelU, RelU};
  modelArch.FCInitFunctionsType = {Xavier, Xavier};
  modelArch.distType = normalDistribution;
  modelArch.learningRate = 0.001;
  size_t numOfClasses = 10;
  size_t imageHeight = 28;
  size_t imageWidth = 28;
  size_t imageDepth = 1;
  NNModel model(modelArch, numOfClasses, imageHeight, imageWidth, imageDepth);

  return 0;
}
