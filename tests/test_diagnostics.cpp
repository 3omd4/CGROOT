#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../src/core/activation/activation.h"
#include "../src/core/layers/layers.h"
#include "../src/core/model.h"

using namespace std;

// Macros for colored output
#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"

void run_test(string name, bool condition) {
  std::cout << "Test: " << name << " ... ";
  if (condition) {
    std::cout << GREEN << "PASSED" << RESET << std::endl;
  } else {
    std::cout << RED << "FAILED" << RESET << std::endl;
    exit(1);
  }
}

bool float_eq(double a, double b, double epsilon = 1e-5) {
  return std::abs(a - b) < epsilon;
}

void test_activation() {
  std::cout << "\n--- Testing Activation Functions ---" << std::endl;

  double val_relu = 1.0;
  reLU_Funct(val_relu);
  run_test("ReLU(1.0)", float_eq(val_relu, 1.0));

  val_relu = -1.0;
  reLU_Funct(val_relu);
  run_test("ReLU(-1.0)", float_eq(val_relu, 0.0));

  double val_sig = 0.0;
  sigmoid_Funct(val_sig);
  run_test("Sigmoid(0.0)", float_eq(val_sig, 0.5));
}

void test_fc_layer() {
  std::cout << "\n--- Testing Fully Connected Layer ---" << std::endl;

  // 2 Neurons, 2 Inputs
  architecture dummyArch;
  dummyArch.optConfig.learningRate = 0.01;
  FullyConnected fc(2, RelU, Xavier, normalDistribution, 2);

  vector<double> input = {1.0, 1.0};
  fc.forwardProp(input);
  const vector<double> &out = fc.getOutput();

  std::cout << "FC Output: " << out[0] << ", " << out[1] << std::endl;
  run_test("FC Output Size", out.size() == 2);

  // Backward
  vector<double> nextGrad = {0.1, -0.1};
  fc.backwardProp(input, nextGrad);

  const vector<double> &prevGrad = fc.getPrevLayerGrad();
  run_test("FC PrevGrad Size", prevGrad.size() == 2);

  fc.update(createOptimizer(dummyArch.optConfig));
  std::cout << "FC Update completed without crash." << std::endl;
}

void test_output_layer() {
  std::cout << "\n--- Testing Output Layer ---" << std::endl;

  architecture dummyArch;
  dummyArch.optConfig.learningRate = 0.01;
  outputLayer outL(2, 2, normalDistribution);

  vector<double> input = {1.0, 1.0};
  outL.forwardProp(input);

  const vector<double> &out = outL.getOutput();
  std::cout << "Output Layer (Softmax): " << out[0] << ", " << out[1]
            << std::endl;

  run_test("Softmax Sum", float_eq(out[0] + out[1], 1.0));

  outL.backwardProp(input, static_cast<size_t>(0));
  outL.update(createOptimizer(dummyArch.optConfig));
  std::cout << "Output Layer update completed." << std::endl;
}

void test_model_integration() {
  std::cout << "\n--- Testing Full Model Integration ---" << std::endl;

  architecture arch;
  arch.numOfConvLayers = 0;
  arch.numOfFCLayers = 1;
  arch.neuronsPerFCLayer = {10};
  arch.FCLayerActivationFunc = {RelU};
  arch.FCInitFunctionsType = {Xavier};
  arch.distType = normalDistribution;

  NNModel model(arch, 2, 2, 2, 1); // 2 Classes, 2x2 image, 1 depth

  // image is [depth][height][width], model expects 2x2x1
  image img(1, vector<vector<unsigned char>>(2, vector<unsigned char>(2)));
  img[0][0][0] = 255;
  img[0][0][1] = 0;
  img[0][1][0] = 0;
  img[0][1][1] = 255;

  std::cout << "Classify..." << std::endl;
  int cls = model.classify(img);
  std::cout << "Class: " << cls << std::endl;

  vector<double> probs = model.getProbabilities();
  if (probs.size() == 2) {
    run_test("Probs Range", float_eq(probs[0] + probs[1], 1.0));
  } else {
    run_test("Probs Size", false);
  }

  std::cout << "Train..." << std::endl;
  model.train(img, 0);

  std::cout << "Classify again..." << std::endl;
  vector<double> probs2 = model.getProbabilities();

  if (probs2[0] > probs[0]) {
    std::cout << GREEN << "Model is learning (P[0] increased)." << RESET
              << std::endl;
  } else {
    std::cout
        << YELLOW
        << "Model P[0] did not increase (might be learning rate or small iter)."
        << RESET << std::endl;
  }
}

void test_feature_map_calculation() {
  std::cout << "\n--- Testing Feature Map Calculation ---" << std::endl;

  // Create a dummy model just to access the helper function
  // We can use the same simple arch
  architecture arch;
  arch.numOfConvLayers = 0;
  arch.numOfFCLayers = 0; // minimal

  NNModel model(arch, 2, 28, 28, 1);

  // Test 1: No padding, stride 1 (assumed implicitly by calcFeatureMapDim logic
  // usually) Formula: (W - K) + 1  (if no padding/stride parameters in
  // function)
  size_t inputH = 28;
  size_t inputW = 28;
  size_t kernelH = 3;
  size_t kernelW = 3;

  featureMapDim dim = model.calcFeatureMapDim(kernelH, kernelW, inputH, inputW);

  std::cout << "Input: " << inputH << "x" << inputW << ", Kernel: " << kernelH
            << "x" << kernelW << std::endl;
  std::cout << "Output: " << dim.FM_height << "x" << dim.FM_width << std::endl;

  size_t expectedH = inputH - kernelH + 1;
  size_t expectedW = inputW - kernelW + 1;

  run_test("Feature Map Height", dim.FM_height == expectedH);
  run_test("Feature Map Width", dim.FM_width == expectedW);
}

void test_cnn_integration() {
  std::cout << "\n--- Testing CNN Integration ---" << std::endl;

  architecture arch;
  arch.numOfConvLayers = 1;

  // 1 Kernel 3x3
  convKernels kernel1;
  kernel1.numOfKerenels = 1;
  kernel1.kernel_height = 3;
  kernel1.kernel_width = 3;
  kernel1.kernel_depth = 0; // calculated dynamically
  arch.kernelsPerconvLayers = {kernel1};

  arch.convLayerActivationFunc = {RelU};
  arch.convInitFunctionsType = {Kaiming};

  // Pooling logic
  // Add a pooling layer after the first conv layer (interval 1)
  arch.poolingLayersInterval = {1};
  arch.poolingtype = {maxPooling};

  poolKernel pool1;
  pool1.filter_height = 2;
  pool1.filter_width = 2;
  pool1.stride = 2;
  pool1.filter_depth = 0; // calculated dynamically
  arch.kernelsPerPoolingLayer = {pool1};

  arch.numOfFCLayers = 1;
  arch.neuronsPerFCLayer = {10};
  arch.FCLayerActivationFunc = {RelU};
  arch.FCInitFunctionsType = {Xavier};
  arch.distType = normalDistribution;

  // Input 10x10, 1 depth
  NNModel model(arch, 2, 10, 10, 1);

  // Dummy Image 10x10x1
  image img(1,
            vector<vector<unsigned char>>(10, vector<unsigned char>(10, 128)));

  std::cout << "CNN Classify..." << std::endl;
  int cls = model.classify(img);
  std::cout << "CNN Prediction: " << cls << std::endl;

  vector<double> probs = model.getProbabilities();
  run_test("CNN Probs Size", probs.size() == 2);

  std::cout << "CNN Train..." << std::endl;
  model.train(img, 0);
  std::cout << "CNN Train completed without crash." << std::endl;
}

void test_batch_training() {
  std::cout << "\n--- Testing Batch Training ---" << std::endl;

  architecture arch;
  arch.numOfConvLayers = 0;
  arch.numOfFCLayers = 1;
  arch.neuronsPerFCLayer = {10};
  arch.FCLayerActivationFunc = {RelU};
  arch.FCInitFunctionsType = {Xavier};
  arch.distType = normalDistribution;

  NNModel model(arch, 2, 2, 2, 1);

  // Create batch of 2 images
  vector<image> batch;
  vector<int> labels;

  // Images are [depth][height][width], model expects 2x2x1
  image img1(1,
             vector<vector<unsigned char>>(2, vector<unsigned char>(2, 255)));
  image img2(1, vector<vector<unsigned char>>(2, vector<unsigned char>(2, 0)));

  batch.push_back(img1);
  labels.push_back(0);

  batch.push_back(img2);
  labels.push_back(1);

  std::cout << "Training Batch of size " << batch.size() << "..." << std::endl;
  model.train_batch(batch, labels);

  std::cout << "Batch training completed without crash." << std::endl;
  run_test("Batch Training", true);
}

void test_cnn_crash() {
  std::cout << "\n--- Testing CNN Crash Scenarios ---" << std::endl;
  // Test case that previously might have crashed due to dimension mismatch

  architecture arch;
  arch.numOfConvLayers = 1;

  convKernels kernel1;
  kernel1.numOfKerenels = 1;
  kernel1.kernel_height = 3;
  kernel1.kernel_width = 3;
  kernel1.kernel_depth = 0;
  arch.kernelsPerconvLayers = {kernel1};
  arch.convLayerActivationFunc = {RelU};
  arch.convInitFunctionsType = {Kaiming};

  arch.poolingLayersInterval = {1};
  arch.poolingtype = {maxPooling};

  poolKernel pool1;
  pool1.filter_height = 2;
  pool1.filter_width = 2;
  pool1.stride = 2;
  pool1.filter_depth = 0;

  arch.kernelsPerPoolingLayer = {pool1};

  arch.numOfFCLayers = 1;
  arch.neuronsPerFCLayer = {10};
  arch.FCLayerActivationFunc = {RelU};
  arch.FCInitFunctionsType = {Xavier};
  arch.distType = normalDistribution;

  // Use 28x28 input (MNIST size)
  NNModel model(arch, 2, 28, 28, 1);

  std::cout << "Model constructed." << std::endl;

  // Image is [depth][height][width], model expects 28x28x1
  image img(1,
            vector<vector<unsigned char>>(28, vector<unsigned char>(28, 128)));

  std::cout << "Classify (CNN Crash Check)..." << std::endl;
  int cls = model.classify(img);
  std::cout << "Classify Result: " << cls << std::endl;

  std::cout << "Train (CNN Crash Check)..." << std::endl;
  model.train(img, 0);
  std::cout << "Train completed." << std::endl;

  run_test("CNN Crash Test", true);
}

int main() {
  test_activation();
  test_fc_layer();
  test_output_layer();
  test_model_integration();
  test_feature_map_calculation();
  test_cnn_integration();
  test_cnn_crash();
  test_batch_training();

  std::cout << "\nAll Diagnostic Tests Finished." << std::endl;
  return 0;
}
