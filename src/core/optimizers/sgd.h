#ifndef SGD_H
#define SGD_H

#include <vector>

namespace cgroot {
namespace core {
namespace optimizers {

class SGD {
private:
    double learning_rate;
    std::vector<std::vector<double>*> parameters;
    std::vector<std::vector<double>> gradients;

public:
    SGD(double lr = 0.01) : learning_rate(lr) {}
    
    void set_parameters(std::vector<std::vector<double>*>& params) {
        parameters = params;
        gradients.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            gradients[i].resize(params[i]->size(), 0.0);
        }
    }
    
    void step() {
        for (size_t i = 0; i < parameters.size(); ++i) {
            for (size_t j = 0; j < parameters[i]->size(); ++j) {
                (*parameters[i])[j] -= learning_rate * gradients[i][j];
            }
        }
    }
    
    void zero_grad() {
        for (auto& grad : gradients) {
            std::fill(grad.begin(), grad.end(), 0.0);
        }
    }
    
    void set_learning_rate(double lr) {
        learning_rate = lr;
    }
};

} // namespace optimizers
} // namespace core
} // namespace cgroot

#endif // SGD_H

