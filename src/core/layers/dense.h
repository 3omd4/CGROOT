#ifndef DENSE_H
#define DENSE_H

#include "layers.h"
#include <vector>
#include <random>

namespace cgroot {
namespace core {
namespace layers {

class DenseLayer : public Layer {
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    unsigned int input_size;
    unsigned int output_size;
    
    void initialize_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = std::sqrt(6.0 / (input_size + output_size));
        std::uniform_real_distribution<double> dis(-limit, limit);
        
        for (auto& row : weights) {
            for (auto& w : row) {
                w = dis(gen);
            }
        }
        
        for (auto& b : biases) {
            b = 0.0;
        }
    }

public:
    DenseLayer(unsigned int input_sz, unsigned int output_sz)
        : Layer(output_sz), input_size(input_sz), output_size(output_sz) {
        weights.resize(output_size);
        for (auto& row : weights) {
            row.resize(input_size);
        }
        biases.resize(output_size);
        initialize_weights();
    }
    
    void forward(const std::vector<double>& input) {
        layerOutput.resize(output_size);
        for (unsigned int i = 0; i < output_size; ++i) {
            double sum = biases[i];
            for (unsigned int j = 0; j < input_size; ++j) {
                sum += weights[i][j] * input[j];
            }
            layerOutput[i] = sum;
        }
    }
    
    std::vector<std::vector<double>>& get_weights() { return weights; }
    std::vector<double>& get_biases() { return biases; }
};

} // namespace layers
} // namespace core
} // namespace cgroot

#endif // DENSE_H

