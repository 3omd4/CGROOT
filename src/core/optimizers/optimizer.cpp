#include "optimizer.h"
#include "../definitions.h"
#include <cmath> // For sqrt, pow
#include <iostream>

// Optimizer Base
Optimizer::Optimizer(double lr, double wd)
    : learning_rate(lr), weight_decay(wd) {}

// SGD Implementation
SGD::SGD(double lr, double mom, double wd) : Optimizer(lr, wd), momentum(mom) {}

void SGD::update(std::vector<double> &weights,
                 const std::vector<double> &grads) {
#ifndef NDEBUG
  if (weights.size() != grads.size()) {
    std::cerr << "Error: Optimizer size mismatch! Weights: " << weights.size()
              << " Grads: " << grads.size() << std::endl;
    return;
  }
#endif

  // Initialize velocity vector if empty (first update)
  if (v.empty()) {
    v.resize(weights.size(), 0.0);
  }

  for (size_t i = 0; i < weights.size(); ++i) {
    // Apply Weight Decay (L2 Regularization): grad = grad + wd * weight
    double g = grads[i] + weight_decay * weights[i];

    // Apply Momentum: v = v * momentum + g
    v[i] = momentum * v[i] + g;
    weights[i] -= learning_rate * v[i];
  }
}

// Adam Implementation
Adam::Adam(double lr, double b1, double b2, double eps, double wd)
    : Optimizer(lr, wd), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void Adam::update(std::vector<double> &weights,
                  const std::vector<double> &grads) {
#ifndef NDEBUG
  if (weights.size() != grads.size()) {
    std::cerr << "Error: Optimizer size mismatch! Weights: " << weights.size()
              << " Grads: " << grads.size() << std::endl;
    return;
  }
#endif

  if (m.empty()) {
    m.resize(weights.size(), 0.0);
    v.resize(weights.size(), 0.0);
  }

  t++;

  for (size_t i = 0; i < weights.size(); ++i) {
    // Apply Weight Decay
    double g = grads[i] + weight_decay * weights[i];

    m[i] = beta1 * m[i] + (1.0 - beta1) * g;
    v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g);

    double m_hat = m[i] / (1.0 - std::pow(beta1, t));
    double v_hat = v[i] / (1.0 - std::pow(beta2, t));

    weights[i] -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));
  }
}

// RMSprop Implementation
RMSprop::RMSprop(double lr, double b, double eps, double wd)
    : Optimizer(lr, wd), beta(b), epsilon(eps) {}

void RMSprop::update(std::vector<double> &weights,
                     const std::vector<double> &grads) {
#ifndef NDEBUG
  if (weights.size() != grads.size()) {
    std::cerr << "Error: Optimizer size mismatch! Weights: " << weights.size()
              << " Grads: " << grads.size() << std::endl;
    return;
  }
#endif

  if (s.empty()) {
    s.resize(weights.size(), 0.0);
  }

  for (size_t i = 0; i < weights.size(); ++i) {
    // Apply Weight Decay
    double g = grads[i] + weight_decay * weights[i];

    // s = beta * s + (1 - beta) * g^2
    s[i] = beta * s[i] + (1.0 - beta) * (g * g);

    // w = w - lr * g / sqrt(s + eps)
    weights[i] -= learning_rate * g / (std::sqrt(s[i]) + epsilon);
  }
}

// Factory Implementation
// Note: Using integer comparison to avoid name collision between enum values
// (Adam, RMSprop, SGD) and class names (Adam, RMSprop, SGD). Enum order: SGD=0,
// Adam=1, RMSprop=2
Optimizer *createOptimizer(const OptimizerConfig &config) {
  int type_val = static_cast<int>(config.type);
  if (type_val == 1) { // Adam
    return new class Adam(config.learningRate, config.beta1, config.beta2,
                          config.epsilon, config.weightDecay);
  } else if (type_val == 2) { // RMSprop
    return new class RMSprop(config.learningRate, config.beta1, config.epsilon,
                             config.weightDecay);
  } else { // SGD (0) or default
    return new class SGD(config.learningRate, config.momentum,
                         config.weightDecay);
  }
}
