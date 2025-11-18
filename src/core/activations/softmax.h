#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <vector>
#include <cmath>
#include <algorithm>

namespace cgroot {
namespace core {
namespace activations {

class Softmax {
public:
    static void forward(std::vector<double>& output, const std::vector<double>& input) {
        output.resize(input.size());
        
        double max_val = *std::max_element(input.begin(), input.end());
        double sum = 0.0;
        
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }
        
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] /= sum;
        }
    }
    
    static void backward(std::vector<double>& grad_input,
                        const std::vector<double>& grad_output,
                        const std::vector<double>& output) {
        grad_input.resize(output.size());
        
        double sum = 0.0;
        for (size_t i = 0; i < output.size(); ++i) {
            sum += grad_output[i] * output[i];
        }
        
        for (size_t i = 0; i < output.size(); ++i) {
            grad_input[i] = output[i] * (grad_output[i] - sum);
        }
    }
};

} // namespace activations
} // namespace core
} // namespace cgroot

#endif // SOFTMAX_H

