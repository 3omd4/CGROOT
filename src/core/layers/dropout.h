#ifndef DROPOUT_H
#define DROPOUT_H

#include "layers.h"
#include <vector>
#include <random>

namespace cgroot {
namespace core {
namespace layers {

class DropoutLayer : public Layer {
private:
    double dropout_rate;
    bool is_training;
    std::vector<bool> mask;
    std::mt19937 rng;

public:
    DropoutLayer(unsigned int size, double rate = 0.5)
        : Layer(size), dropout_rate(rate), is_training(true), rng(std::random_device{}()) {
        mask.resize(size);
    }
    
    void set_training(bool training) {
        is_training = training;
    }
    
    void forward(const std::vector<double>& input) {
        layerOutput.resize(input.size());
        
        if (is_training) {
            std::bernoulli_distribution dist(1.0 - dropout_rate);
            for (size_t i = 0; i < input.size(); ++i) {
                mask[i] = dist(rng);
                layerOutput[i] = input[i] * mask[i] / (1.0 - dropout_rate);
            }
        } else {
            for (size_t i = 0; i < input.size(); ++i) {
                layerOutput[i] = input[i];
            }
        }
    }
    
    void backward(const std::vector<double>& grad_output, std::vector<double>& grad_input) {
        grad_input.resize(grad_output.size());
        if (is_training) {
            for (size_t i = 0; i < grad_output.size(); ++i) {
                grad_input[i] = grad_output[i] * mask[i] / (1.0 - dropout_rate);
            }
        } else {
            grad_input = grad_output;
        }
    }
};

} // namespace layers
} // namespace core
} // namespace cgroot

#endif // DROPOUT_H

