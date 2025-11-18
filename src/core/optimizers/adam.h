#ifndef ADAM_H
#define ADAM_H

#include <vector>
#include <cmath>

namespace cgroot {
namespace core {
namespace optimizers {

class Adam {
private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int timestep;
    
    std::vector<std::vector<double>*> parameters;
    std::vector<std::vector<double>> gradients;
    std::vector<std::vector<double>> m;  // First moment
    std::vector<std::vector<double>> v;  // Second moment

public:
    Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), timestep(0) {}
    
    void set_parameters(std::vector<std::vector<double>*>& params) {
        parameters = params;
        gradients.resize(params.size());
        m.resize(params.size());
        v.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            gradients[i].resize(params[i]->size(), 0.0);
            m[i].resize(params[i]->size(), 0.0);
            v[i].resize(params[i]->size(), 0.0);
        }
    }
    
    void step() {
        ++timestep;
        double lr_t = learning_rate * std::sqrt(1.0 - std::pow(beta2, timestep)) 
                     / (1.0 - std::pow(beta1, timestep));
        
        for (size_t i = 0; i < parameters.size(); ++i) {
            for (size_t j = 0; j < parameters[i]->size(); ++j) {
                m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * gradients[i][j];
                v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * gradients[i][j] * gradients[i][j];
                
                (*parameters[i])[j] -= lr_t * m[i][j] / (std::sqrt(v[i][j]) + epsilon);
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

#endif // ADAM_H

