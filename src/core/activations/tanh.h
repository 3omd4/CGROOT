#ifndef TANH_H
#define TANH_H

#include <vector>
#include <cmath>

namespace cgroot {
namespace core {
namespace activations {

class Tanh {
public:
    static void forward(std::vector<double>& output, const std::vector<double>& input) {
        output.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::tanh(input[i]);
        }
    }
    
    static void backward(std::vector<double>& grad_input,
                        const std::vector<double>& grad_output,
                        const std::vector<double>& output) {
        grad_input.resize(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            double t = output[i];
            grad_input[i] = grad_output[i] * (1.0 - t * t);
        }
    }
};

} // namespace activations
} // namespace core
} // namespace cgroot

#endif // TANH_H

