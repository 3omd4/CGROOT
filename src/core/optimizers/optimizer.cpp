#include "optimizer.h"

// Optimizer Base

Optimizer::Optimizer(double lr) : learning_rate(lr) {}

// SGD Implementation

SGD::SGD(double lr) : Optimizer(lr) {}

void SGD::update(std::vector<double>& params, const std::vector<double>& grads) {
    for (size_t i = 0; i < params.size(); ++i) {
        params[i] -= learning_rate * grads[i];
    }
}
