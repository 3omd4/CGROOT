#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <cmath>
#include <algorithm>

namespace cgroot {
namespace core {
namespace utils {

class Metrics {
public:
    static double accuracy(const std::vector<int>& predictions, 
                         const std::vector<int>& targets) {
        if (predictions.size() != targets.size() || predictions.empty()) {
            return 0.0;
        }
        
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == targets[i]) {
                ++correct;
            }
        }
        
        return static_cast<double>(correct) / predictions.size();
    }
    
    static double precision(const std::vector<int>& predictions,
                           const std::vector<int>& targets,
                           int class_label) {
        int true_positives = 0;
        int predicted_positives = 0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == class_label) {
                ++predicted_positives;
                if (targets[i] == class_label) {
                    ++true_positives;
                }
            }
        }
        
        return predicted_positives > 0 ? 
               static_cast<double>(true_positives) / predicted_positives : 0.0;
    }
    
    static double recall(const std::vector<int>& predictions,
                        const std::vector<int>& targets,
                        int class_label) {
        int true_positives = 0;
        int actual_positives = 0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (targets[i] == class_label) {
                ++actual_positives;
                if (predictions[i] == class_label) {
                    ++true_positives;
                }
            }
        }
        
        return actual_positives > 0 ?
               static_cast<double>(true_positives) / actual_positives : 0.0;
    }
};

} // namespace utils
} // namespace core
} // namespace cgroot

#endif // METRICS_H

