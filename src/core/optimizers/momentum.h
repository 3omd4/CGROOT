#ifndef MOMENTUM_H
#define MOMENTUM_H

#include <vector>

namespace cgroot {
namespace core {
namespace optimizers {

class Momentum {
private:
    double learning_rate;
    double momentum;
    std::vector<std::vector<double>*> parameters;
    std::vector<std::vector<double>> gradients;
    std::vector<std::vector<double>> velocities;

public:
    Momentum(double lr = 0.01, double mom = 0.9) 
        : learning_rate(lr), momentum(mom) {}
    
    void set_parameters(std::vector<std::vector<double>*>& params) {
        parameters = params;
        gradients.resize(params.size());
        velocities.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            gradients[i].resize(params[i]->size(), 0.0);
            velocities[i].resize(params[i]->size(), 0.0);
        }
    }
    
    void step() {
        for (size_t i = 0; i < parameters.size(); ++i) {
            for (size_t j = 0; j < parameters[i]->size(); ++j) {
                velocities[i][j] = momentum * velocities[i][j] - learning_rate * gradients[i][j];
                (*parameters[i])[j] += velocities[i][j];
            }
        }
    }
    
    void zero_grad() {
        for (auto& grad : gradients) {
            std::fill(grad.begin(), grad.end(), 0.0);
        }
    }
};

} // namespace optimizers
} // namespace core
} // namespace cgroot

#endif // MOMENTUM_H

