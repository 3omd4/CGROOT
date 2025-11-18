#ifndef SIGMOID_H
#define SIGMOID_H

#include <vector>
#include <cmath>

namespace cgroot {
namespace core {
namespace activations {

class Sigmoid {
public:
    static void forward(std::vector<double>& output, const std::vector<double>& input) {
        output.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = 1.0 / (1.0 + std::exp(-input[i]));
        }
    }
    
    static void backward(std::vector<double>& grad_input,
                        const std::vector<double>& grad_output,
                        const std::vector<double>& output) {
        grad_input.resize(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            double s = output[i];
            grad_input[i] = grad_output[i] * s * (1.0 - s);
        }
    }
};

} // namespace activations
} // namespace core
} // namespace cgroot

#endif // SIGMOID_H

