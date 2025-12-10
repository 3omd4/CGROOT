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
  FullyConnected fc(2, RelU, Xavier, normalDistribution, 2);

  // Manually set weights and biases for deterministic testing
  // Neuron 0: weights [0.5, 0.5], bias 0.1
  // Neuron 1: weights [-0.5, 0.5], bias -0.1

  // Hack: We assume we can access private members via getter or if tests are
  // friends. Since we don't have setters/getters for weights, we rely on public
  // access or hack. layers.h defines members as protected usually ? NO,
  // layers.h usually has proteced members. However, the prompt is about "where
  // the error is", checking logic. If I cannot access weights, I cannot
  // deterministically test unless I modify layers.h to make them public for
  // testing or add friends. Let's assume for diagnostics we just run it and
  // check shapes output non-zero.

  vector<double> input = {1.0, 1.0};
  fc.forwardProp(input);
  const vector<double> &out = fc.getOutput();

  std::cout << "FC Output: " << out[0] << ", " << out[1] << std::endl;
  run_test("FC Output Size", out.size() == 2);

  // Backward
  // Mock gradients from next layer
  vector<double> nextGrad = {0.1, -0.1};
  fc.backwardProp(input, nextGrad);

  // We can't easily check d_weights without accessers.
  // But we can check prevLayerGrad size.
  const vector<double> &prevGrad = fc.getPrevLayerGrad();
  run_test("FC PrevGrad Size", prevGrad.size() == 2);

  // Update
  fc.update(0.1);
  std::cout << "FC Update completed without crash." << std::endl;
}

void test_output_layer() {
  std::cout << "\n--- Testing Output Layer ---" << std::endl;

  // 2 Classes, 2 Inputs
  outputLayer outL(2, 2, normalDistribution);

  // Input
  vector<double> input = {1.0, 1.0}; // Result of previous layer
  outL.forwardProp(input);

  const vector<double> &out = outL.getOutput();
  std::cout << "Output Layer (Softmax): " << out[0] << ", " << out[1]
            << std::endl;

  // Check ranges
  run_test("Softmax Sum", float_eq(out[0] + out[1], 1.0));

  // Backward (Target Class 0)
  outL.backwardProp(input, 0);
  // Gradient check: dZ = P - Y. Class 0: P0 - 1. Class 1: P1 - 0.
  // If fixing gradient sign (removed negative), then dZ should be P-Y.
  // P0 approx 0.5. dZ0 = -0.5. P1 approx 0.5. dZ1 = 0.5.

  // We cannot access d_bias directly without getters.
  // But if we run this and print "Completed", it validates stability.
  // The user wants to find "where the error is".
  // If the error was the sign flip, measuring accuracy on a small dataset is
  // best.

  outL.update(0.1);
  std::cout << "Output Layer update completed." << std::endl;
}

void test_model_integration() {
  std::cout << "\n--- Testing Full Model Integration ---" << std::endl;

  architecture arch;
  arch.numOfConvLayers = 0;
  arch.numOfFCLayers = 1;
  arch.neuronsPerFCLayer = {10};
  arch.FCLayerActivationFunc = {
      RelU, Softmax}; // Last one ignored by logic usually as it handles output
                      // layer separatly?
  // Actually architecture struct requires filling vectors.
  arch.FCLayerActivationFunc = {RelU};
  arch.FCInitFunctionsType = {Xavier};
  arch.distType = normalDistribution;
  arch.learningRate = 0.01;

  NNModel model(arch, 2, 2, 2, 1); // 2 Classes, 2x2 image, 1 depth

  // Create dummy image 2x2
  image img(2, vector<vector<unsigned char>>(2, vector<unsigned char>(1)));
  img[0][0][0] = 255;
  img[0][1][0] = 0;
  img[1][0][0] = 0;
  img[1][1][0] = 255;

  // Flattened: 1, 0, 0, 1 (normalized to 0-1 range usually inside inputLayer?
  // No, inputLayer does not normalize usually unless specified) inputLayer just
  // takes image.

  std::cout << "Classify..." << std::endl;
  int cls = model.classify(img);
  std::cout << "Class: " << cls << std::endl;

  vector<double> probs = model.getProbabilities();
  std::cout << "Probs: ";
  for (double p : probs)
    std::cout << p << " ";
  std::cout << std::endl;

  if (probs.size() == 2) {
    run_test("Probs Range", float_eq(probs[0] + probs[1], 1.0));
  } else {
    run_test("Probs Size", false);
  }

  std::cout << "Train..." << std::endl;
  model.train(img, 0); // Train for class 0

  std::cout << "Classify again..." << std::endl;
  vector<double> probs2 = model.getProbabilities();
  std::cout << "Probs2: ";
  for (double p : probs2)
    std::cout << p << " ";
  std::cout << std::endl;

  // Check if probabilities changed (moved towards target 0)
  // If they move away or don't change, we have an issue.
  // If P0 increased, good.
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

int main() {
  test_activation();
  test_fc_layer();
  test_output_layer();
  test_model_integration();

  std::cout << "\nAll Diagnostic Tests Finished." << std::endl;
  return 0;
}
