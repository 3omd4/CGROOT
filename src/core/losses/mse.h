#ifndef MSE_H
#define MSE_H

#include <vector>
#include <cmath>

namespace cgroot {
namespace core {
namespace losses {

class MSE {
public:
    static double compute(const std::vector<double>& predictions, 
                         const std::vector<double>& targets) {
        if (predictions.size() != targets.size()) {
            return -1.0;
        }
        
        double loss = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            double diff = predictions[i] - targets[i];
            loss += diff * diff;
        }
        
        return loss / predictions.size();
    }
    
    static void gradient(std::vector<double>& grad,
                        const std::vector<double>& predictions,
                        const std::vector<double>& targets) {
        grad.resize(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            grad[i] = 2.0 * (predictions[i] - targets[i]) / predictions.size();
        }
    }
};

} // namespace losses
} // namespace core
} // namespace cgroot

#endif // MSE_H

