#include "optimizer.h"
#include <iostream>
#include <cmath>      // For sqrt, pow

// Optimizer Base

Optimizer::Optimizer(double lr) : learning_rate(lr) {}

// SGD Implementation

SGD::SGD(double lr) : Optimizer(lr) {}

void SGD::update(std::vector<double>& weights, const std::vector<double>& grads) {
    if (weights.size() != grads.size()) {
    std::cerr << "Error: Optimizer size mismatch! Weights: " << weights.size() 
              << " Grads: " << grads.size() << std::endl;
    return;
}
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate * grads[i];
    }
}

// Adam Implementation

Adam::Adam(double lr, double b1, double b2, double eps)
    : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void Adam::update(std::vector<double>& weights, const std::vector<double>& grads) {
    if (weights.size() != grads.size()) {
    std::cerr << "Error: Optimizer size mismatch! Weights: " << weights.size() 
              << " Grads: " << grads.size() << std::endl;
    return;
    }

    // Initialize memory vectors on the very first update
    if (m.empty()) {
        m.resize(weights.size(), 0.0);
        v.resize(weights.size(), 0.0);
    }

    // Increment time step (needed for bias correction)
    t++;

    for (size_t i = 0; i < weights.size(); ++i) {
        double g = grads[i];

        // 1. Update Momentum (First Moment)
        // m = beta1 * m + (1 - beta1) * g
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;

        // 2. Update RMSprop/Variance (Second Moment)
        // v = beta2 * v + (1 - beta2) * g^2
        v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g);

        // 3. Bias Correction
        // (Fixes the issue where m and v start at 0 and are too small initially)
        double m_hat = m[i] / (1.0 - std::pow(beta1, t));
        double v_hat = v[i] / (1.0 - std::pow(beta2, t));

        // 4. Update Parameters
        // w = w - lr * ( m_hat / (sqrt(v_hat) + epsilon) )
        weights[i] -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));
    }
}
