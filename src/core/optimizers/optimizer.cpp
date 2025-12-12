#include "optimizer.h"
#include "../definitions.h"
#include <cmath> 
#include <iostream>

// Optimizer Base
Optimizer::Optimizer(double lr, double wd)
    : learning_rate(lr), weight_decay(wd) {}

// SGD Implementation
SGD::SGD(double lr, double wd) : Optimizer(lr, wd) {}

void SGD::update(std::vector<double>& weights, const std::vector<double>& grads){
#ifndef NDEBUG
  if (weights.size() != grads.size()) {
    std::cerr << "Error: Optimizer size mismatch! Weights: " << weights.size()
              << " Grads: " << grads.size() << std::endl;
    return;
  }
#endif

    for (size_t i = 0; i < weights.size(); ++i) {
        // Gradient descent with L2 regularization
        weights[i] -= learning_rate * (grads[i] + weight_decay * weights[i]);
    }
}

// SGD_Momentum Implementation
SGD_Momentum::SGD_Momentum(double lr, double mom, double wd) : Optimizer(lr, wd), momentum(mom) {}

void SGD_Momentum::update(std::vector<double> &weights,
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
    double g = grads[i] + weight_decay * weights[i];

    // s = beta * s + (1 - beta) * g^2
    s[i] = beta * s[i] + (1.0 - beta) * (g * g);

    weights[i] -= learning_rate * g / (std::sqrt(s[i]) + epsilon);
  }
}

// Factory Implementation
Optimizer *createOptimizer(const OptimizerConfig &config) {
  switch (config.type) {
    case opt_SGD_Momentum:
      return new SGD_Momentum(config.learningRate, config.momentum, config.weightDecay);
    case opt_Adam:
      return new Adam(config.learningRate, config.beta1, config.beta2,
                      config.epsilon, config.weightDecay);
    case opt_RMSprop:
      return new RMSprop(config.learningRate, config.beta1, config.epsilon,
                         config.weightDecay);
    case opt_SGD:
    default:
      return new SGD(config.learningRate, config.weightDecay);
  }
}
