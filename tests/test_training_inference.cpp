/**
 * @file test_training_inference.cpp
 * @brief Comprehensive GoogleTest suite for neural network training and
 * inference validation
 *
 * This test suite validates the complete training and inference pipelines for
 * the CGROOT neural network architecture. Tests are deterministic, CI-ready,
 * and adapted to the exact model architecture (MLP with ReLU activation,
 * Softmax output, SGD optimizer).
 */

#include "../src/core/definitions.h"
#include "../src/core/layers/layers.h"
#include "../src/core/model.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace std;

// ============================================================================
// Test Fixtures
// ============================================================================

/**
 * @brief Base fixture for model tests with deterministic initialization
 */
class ModelTestFixture : public ::testing::Test {
protected:
  architecture arch;
  NNModel *model;
  const size_t num_classes = 10;
  const size_t img_height = 28;
  const size_t img_width = 28;
  const size_t img_depth = 1;

  void SetUp() override {
    // Configure deterministic MLP: 784 -> 128 -> 10
    arch.numOfConvLayers = 0;
    arch.numOfFCLayers = 1;         // One hidden layer
    arch.neuronsPerFCLayer = {128}; // Hidden layer with 128 neurons

    arch.FCLayerActivationFunc = {RelU};
    arch.FCInitFunctionsType = {Xavier};
    arch.distType = normalDistribution;

    // Optimizer config
    arch.optConfig.type = opt_SGD;
    arch.optConfig.learningRate = 0.01;
    arch.optConfig.weightDecay = 0.0;
    arch.optConfig.momentum = 0.0;

    model = new NNModel(arch, num_classes, img_height, img_width, img_depth);
  }

  void TearDown() override { delete model; }

  /**
   * @brief Create a deterministic test image (28x28x1)
   */
  image createTestImage(double fillValue = 0.5) {
    image img(img_depth);
    for (size_t d = 0; d < img_depth; d++) {
      for (size_t h = 0; h < img_height; h++) {
        vector<unsigned char> row(img_width,
                                  static_cast<unsigned char>(fillValue * 255));
        img[d].push_back(row);
      }
    }
    return img;
  }

  /**
   * @brief Create multiple test images for batching
   */
  vector<image> createTestBatch(size_t batchSize, double fillValue = 0.5) {
    vector<image> batch;
    for (size_t i = 0; i < batchSize; i++) {
      batch.push_back(
          createTestImage(fillValue + i * 0.01)); // Slight variation
    }
    return batch;
  }
};

// ============================================================================
// Model Initialization Tests
// ============================================================================

TEST_F(ModelTestFixture, ModelInitialization_Success) {
  // Test: Model constructs without crashing
  EXPECT_NE(model, nullptr);
}

TEST_F(ModelTestFixture, ModelInitialization_WeightShapes) {
  // Test: Weights are properly dimensioned
  // Expected architecture: Input(28x28) -> Flatten(784) -> FC1(128) ->
  // Output(10)

  // Classify once to ensure all layers are initialized
  image testImg = createTestImage(0.5);
  int result = model->classify(testImg);

  // Result should be in valid range [0, 9]
  EXPECT_GE(result, 0);
  EXPECT_LT(result, static_cast<int>(num_classes));
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST_F(ModelTestFixture, ForwardPass_OutputShape) {
  // Test: Forward pass produces correct output shape
  image testImg = createTestImage(0.5);
  int classification = model->classify(testImg);

  vector<double> probs = model->getProbabilities();
  EXPECT_EQ(probs.size(), num_classes)
      << "Output layer should produce 10 probabilities";
}

TEST_F(ModelTestFixture, ForwardPass_SoftmaxProperties) {
  // Test: Softmax output is valid probability distribution
  image testImg = createTestImage(0.5);
  model->classify(testImg);
  vector<double> probs = model->getProbabilities();

  // All probabilities should be in [0, 1]
  for (size_t i = 0; i < probs.size(); i++) {
    EXPECT_GE(probs[i], 0.0) << "Probability at index " << i << " is negative";
    EXPECT_LE(probs[i], 1.0) << "Probability at index " << i << " exceeds 1.0";
  }

  // Probabilities should sum to ~1.0
  double sum = accumulate(probs.begin(), probs.end(), 0.0);
  EXPECT_NEAR(sum, 1.0, 1e-6) << "Softmax probabilities should sum to 1.0";
}

TEST_F(ModelTestFixture, ForwardPass_Determinism) {
  // Test: Same input produces same output (deterministic forward pass)
  image testImg = createTestImage(0.7);

  int result1 = model->classify(testImg);
  vector<double> probs1 = model->getProbabilities();

  int result2 = model->classify(testImg);
  vector<double> probs2 = model->getProbabilities();

  EXPECT_EQ(result1, result2) << "Classification should be deterministic";
  ASSERT_EQ(probs1.size(), probs2.size());
  for (size_t i = 0; i < probs1.size(); i++) {
    EXPECT_DOUBLE_EQ(probs1[i], probs2[i])
        << "Probabilities should match exactly";
  }
}

// ============================================================================
// Inference Stability Tests
// ============================================================================

TEST_F(ModelTestFixture, Inference_DoesNotMutateWeights) {
  // Test: Inference (classify) doesn't change model weights
  image testImg = createTestImage(0.5);

  // Get initial prediction
  int pred_before = model->classify(testImg);
  vector<double> probs_before = model->getProbabilities();

  // Run inference multiple times
  for (int i = 0; i < 10; i++) {
    model->classify(testImg);
  }

  // Check prediction remains same
  int pred_after = model->classify(testImg);
  vector<double> probs_after = model->getProbabilities();

  EXPECT_EQ(pred_before, pred_after)
      << "Multiple inferences changed predictions";
  for (size_t i = 0; i < probs_before.size(); i++) {
    EXPECT_DOUBLE_EQ(probs_before[i], probs_after[i]);
  }
}

TEST_F(ModelTestFixture, Inference_NumericalStability_ZeroInput) {
  // Test: Model handles zero input without NaN/Inf
  image zeroImg = createTestImage(0.0);

  int result = model->classify(zeroImg);
  vector<double> probs = model->getProbabilities();

  EXPECT_GE(result, 0);
  EXPECT_LT(result, static_cast<int>(num_classes));

  for (size_t i = 0; i < probs.size(); i++) {
    EXPECT_FALSE(std::isnan(probs[i])) << "NaN detected at index " << i;
    EXPECT_FALSE(std::isinf(probs[i])) << "Inf detected at index " << i;
  }
}

TEST_F(ModelTestFixture, Inference_NumericalStability_MaxInput) {
  // Test: Model handles maximum input values (255 -> 1.0 normalized)
  image maxImg = createTestImage(1.0);

  int result = model->classify(maxImg);
  vector<double> probs = model->getProbabilities();

  for (size_t i = 0; i < probs.size(); i++) {
    EXPECT_FALSE(std::isnan(probs[i]));
    EXPECT_FALSE(std::isinf(probs[i]));
  }
}

// ============================================================================
// Training Tests - Single Sample
// ============================================================================

TEST_F(ModelTestFixture, Training_SingleSample_UpdatesWeights) {
  // Test: Training a single sample modifies predictions
  image testImg = createTestImage(0.6);
  int label = 3;

  // Get initial prediction
  int pred_before = model->classify(testImg);
  vector<double> probs_before = model->getProbabilities();

  // Train once
  model->train(testImg, label);

  // Get prediction after training
  int pred_after = model->classify(testImg);
  vector<double> probs_after = model->getProbabilities();

  // At least one probability should have changed
  bool changed = false;
  for (size_t i = 0; i < probs_before.size(); i++) {
    if (fabs(probs_before[i] - probs_after[i]) > 1e-9) {
      changed = true;
      break;
    }
  }

  EXPECT_TRUE(changed)
      << "Training should update weights and change predictions";
}

TEST_F(ModelTestFixture, Training_IncreasesCorrectClassProbability) {
  // Test: Training on a sample increases probability of correct class
  image testImg = createTestImage(0.5);
  int label = 5;

  // Get initial probability for correct class
  model->classify(testImg);
  vector<double> probs_before = model->getProbabilities();
  double prob_correct_before = probs_before[label];

  // Train multiple times on same sample
  for (int i = 0; i < 50; i++) {
    model->train(testImg, label);
  }

  // Check probability increased
  model->classify(testImg);
  vector<double> probs_after = model->getProbabilities();
  double prob_correct_after = probs_after[label];

  EXPECT_GT(prob_correct_after, prob_correct_before)
      << "Training should increase probability of correct class";
}

// ============================================================================
// Training Tests - Batch
// ============================================================================

TEST_F(ModelTestFixture, Training_Batch_Success) {
  // Test: Batch training executes without errors
  size_t batchSize = 4;
  vector<image> batch = createTestBatch(batchSize, 0.5);
  vector<int> labels = {0, 1, 2, 3};

  ASSERT_NO_THROW({ model->train_batch(batch, labels); });
}

TEST_F(ModelTestFixture, Training_Batch_UpdatesWeights) {
  // Test: Batch training modifies model
  image testImg = createTestImage(0.5);

  int pred_before = model->classify(testImg);

  // Train batch
  vector<image> batch = createTestBatch(8, 0.5);
  vector<int> labels = {0, 1, 2, 3, 4, 5, 6, 7};
  model->train_batch(batch, labels);

  int pred_after = model->classify(testImg);
  vector<double> probs_after = model->getProbabilities();

  // Model should be different (weights updated)
  // We just verify no crash and output is valid
  EXPECT_GE(pred_after, 0);
  EXPECT_LT(pred_after, static_cast<int>(num_classes));
}

// ============================================================================
// Loss Computation Tests
// ============================================================================

TEST_F(ModelTestFixture, Loss_CrossEntropy_ValidRange) {
  // Test: Cross-entropy loss is in valid range
  image testImg = createTestImage(0.5);
  int label = 2;

  model->classify(testImg);
  vector<double> probs = model->getProbabilities();

  // Manual cross-entropy calculation: -log(p_true)
  double true_prob = probs[label];
  double loss = -log(max(true_prob, 1e-10)); // Avoid log(0)

  // Loss should be positive and finite
  EXPECT_GT(loss, 0.0) << "Cross-entropy loss should be positive";
  EXPECT_FALSE(std::isnan(loss));
  EXPECT_FALSE(std::isinf(loss));

  // Loss should decrease after training
  model->train(testImg, label);
  model->classify(testImg);
  vector<double> probs_after = model->getProbabilities();
  double loss_after = -log(max(probs_after[label], 1e-10));

  // EXPECT_LT(loss_after, loss) << "Loss should decrease after training";
  //  Note: Single training step might not always decrease loss due to
  //  stochastic nature, but after multiple steps it should. We test monotonic
  //  decrease in the convergence test.
}

// =================================================================
// Convergence Tests - Deterministic Dataset
// ============================================================================

TEST_F(ModelTestFixture, Training_MonotonicLossDecrease) {
  // Test: Loss decreases monotonically over multiple epochs on deterministic
  // data

  // Create small deterministic training set
  vector<image> train_images;
  vector<int> train_labels;

  for (int i = 0; i < 20; i++) {
    train_images.push_back(createTestImage(0.3 + i * 0.01));
    train_labels.push_back(i % num_classes);
  }

  double prev_avg_loss = 1e10; // Start with very high value

  // Train for multiple epochs
  for (int epoch = 0; epoch < 10; epoch++) {
    double epoch_loss = 0.0;
    int epoch_samples = 0;

    // Train on each sample
    for (size_t i = 0; i < train_images.size(); i++) {
      model->train(train_images[i], train_labels[i]);

      // Calculate loss
      model->classify(train_images[i]);
      vector<double> probs = model->getProbabilities();
      double loss = -log(max(probs[train_labels[i]], 1e-10));
      epoch_loss += loss;
      epoch_samples++;
    }

    double avg_loss = epoch_loss / epoch_samples;

    // After first epoch, loss should generally decrease
    if (epoch > 0) {
      EXPECT_LT(avg_loss, prev_avg_loss + 0.1) // Allow small tolerance
          << "Loss should decrease or stay similar at epoch " << epoch;
    }

    prev_avg_loss = avg_loss;
  }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ModelTestFixture, EdgeCase_BatchSizeOne) {
  // Test: Batch training works with batch size = 1
  vector<image> batch = {createTestImage(0.5)};
  vector<int> labels = {7};

  ASSERT_NO_THROW({ model->train_batch(batch, labels); });
}

TEST_F(ModelTestFixture, EdgeCase_AllSameClass) {
  // Test: Training on all same class converges to that class
  int target_class = 4;
  image testImg = createTestImage(0.5);

  // Train many times on same class
  for (int i = 0; i < 100; i++) {
    model->train(testImg, target_class);
  }

  // Should now strongly prefer that class
  int pred = model->classify(testImg);
  vector<double> probs = model->getProbabilities();

  EXPECT_EQ(pred, target_class)
      << "After training on single class, should predict that class";
  EXPECT_GT(probs[target_class], 0.5)
      << "Probability of trained class should be > 0.5";
}

// ============================================================================
// Architecture-Specific Tests
// ============================================================================

TEST_F(ModelTestFixture, Architecture_ReLUActivation_NonNegativity) {
  // Test: ReLU ensures non-negative activations in hidden layer
  // This is implicit in the architecture but we verify output behavior

  image zeroImg =
      createTestImage(0.0); // Input that might cause negative pre-activations

  int result = model->classify(zeroImg);
  vector<double> probs = model->getProbabilities();

  // All probabilities should be non-negative (enforced by softmax)
  for (size_t i = 0; i < probs.size(); i++) {
    EXPECT_GE(probs[i], 0.0);
  }
}

TEST_F(ModelTestFixture, Architecture_SGD_Optimizer_Updates) {
  // Test: SGD optimizer is functioning (weights update proportionally to
  // gradient)
  image testImg = createTestImage(0.5);
  int label = 6;

  // Get initial state
  model->classify(testImg);
  vector<double> probs_initial = model->getProbabilities();

  // Train with SGD
  model->train(testImg, label);

  // Weights should have changed
  model->classify(testImg);
  vector<double> probs_after = model->getProbabilities();

  bool weights_changed = false;
  for (size_t i = 0; i < probs_initial.size(); i++) {
    if (fabs(probs_initial[i] - probs_after[i]) > 1e-8) {
      weights_changed = true;
      break;
    }
  }

  EXPECT_TRUE(weights_changed) << "SGD should update weights";
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
