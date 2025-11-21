#ifndef SGD_H
#define SGD_H

#include <vector>
#include <iostream>

namespace cgroot {
namespace core {
namespace optimizers {

class SGD {
private:
    double learning_rate;
    
    // A list of pairs, where each pair is a pointer to a parameter vector and its corresponding gradient vector
    std::vector<std::pair<std::vector<double>*, std::vector<double>*>> param_groups;

public:
    SGD(double lr = 0.01) : learning_rate(lr) {}

    // For 2D vectors like weights
    void add_parameters(std::vector<std::vector<double>>& params, std::vector<std::vector<double>>& grads) {
        for(size_t i = 0; i < params.size(); ++i) {
            param_groups.push_back({&params[i], &grads[i]});
        }
    }

    // For 1D vectors like biases
    void add_parameters(std::vector<double>& params, std::vector<double>& grads) {
        param_groups.push_back({&params, &grads});
    }
    
    void step() {
        for (size_t group_idx = 0; group_idx < param_groups.size(); ++group_idx) {
            std::vector<double>& params = *param_groups[group_idx].first;
            std::vector<double>& grads = *param_groups[group_idx].second;
            for (size_t i = 0; i < params.size(); ++i) {
                params[i] -= learning_rate * grads[i];
            }
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

