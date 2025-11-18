#ifndef BINARY_CROSSENTROPY_H
#define BINARY_CROSSENTROPY_H

#include <vector>
#include <cmath>
#include <algorithm>

namespace cgroot {
namespace core {
namespace losses {

class BinaryCrossEntropy {
public:
    static double compute(const std::vector<double>& predictions,
                         const std::vector<double>& targets) {
        if (predictions.size() != targets.size()) {
            return -1.0;
        }
        
        double loss = 0.0;
        const double epsilon = 1e-15;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            double p = std::max(epsilon, std::min(1.0 - epsilon, predictions[i]));
            double t = targets[i];
            loss -= t * std::log(p) + (1.0 - t) * std::log(1.0 - p);
        }
        
        return loss / predictions.size();
    }
    
    static void gradient(std::vector<double>& grad,
                        const std::vector<double>& predictions,
                        const std::vector<double>& targets) {
        grad.resize(predictions.size());
        const double epsilon = 1e-15;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            double p = std::max(epsilon, std::min(1.0 - epsilon, predictions[i]));
            grad[i] = (p - targets[i]) / (p * (1.0 - p)) / predictions.size();
        }
    }
};

} // namespace losses
} // namespace core
} // namespace cgroot

#endif // BINARY_CROSSENTROPY_H

