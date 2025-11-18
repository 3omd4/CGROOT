#ifndef WEIGHT_INIT_H
#define WEIGHT_INIT_H

#include <vector>
#include <random>
#include <cmath>

namespace cgroot {
namespace core {
namespace utils {

class WeightInitializer {
public:
    enum class Type {
        XAVIER,
        HE,
        UNIFORM,
        NORMAL
    };
    
    static void xavier_init(std::vector<std::vector<double>>& weights, 
                           unsigned int input_size, 
                           unsigned int output_size) {
        double limit = std::sqrt(6.0 / (input_size + output_size));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-limit, limit);
        
        for (auto& row : weights) {
            for (auto& w : row) {
                w = dis(gen);
            }
        }
    }
    
    static void he_init(std::vector<std::vector<double>>& weights,
                       unsigned int input_size) {
        double stddev = std::sqrt(2.0 / input_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dis(0.0, stddev);
        
        for (auto& row : weights) {
            for (auto& w : row) {
                w = dis(gen);
            }
        }
    }
    
    static void uniform_init(std::vector<std::vector<double>>& weights,
                           double min_val = -0.1, double max_val = 0.1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(min_val, max_val);
        
        for (auto& row : weights) {
            for (auto& w : row) {
                w = dis(gen);
            }
        }
    }
};

} // namespace utils
} // namespace core
} // namespace cgroot

#endif // WEIGHT_INIT_H

