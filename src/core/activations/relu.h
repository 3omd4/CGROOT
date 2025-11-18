#ifndef RELU_H
#define RELU_H

#include <vector>
#include <cmath>

namespace cgroot {
namespace core {
namespace activations {

class ReLU {
public:
    static void forward(std::vector<double>& output, const std::vector<double>& input) {
        output.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = input[i] > 0 ? input[i] : 0.0;
        }
    }
    
    static void backward(std::vector<double>& grad_input, 
                        const std::vector<double>& grad_output,
                        const std::vector<double>& input) {
        grad_input.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            grad_input[i] = input[i] > 0 ? grad_output[i] : 0.0;
        }
    }
};

} // namespace activations
} // namespace core
} // namespace cgroot

#endif // RELU_H

