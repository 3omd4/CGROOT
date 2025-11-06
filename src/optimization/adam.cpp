#include "adam.h"
#include <cmath>

template <typename T>
Adam<T>::Adam(T lr, T beta1, T beta2, T eps)
    : lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(eps), t_(0) {}

template <typename T>
void Adam<T>::step(std::vector<Tensor<T>>& params,
                   const std::vector<Tensor<T>>& grads) {
    // takes a vector of parameters and their corresponding gradients
    // and updates the parameters in place using the Adam optimization algorithm
    if (m_.empty()) {
        m_.resize(params.size());
        v_.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            m_[i] = Tensor<T>(params[i].shape(), 0);
            v_[i] = Tensor<T>(params[i].shape(), 0);
        }
    }

    ++t_;
    for (size_t i = 0; i < params.size(); ++i) {
        auto& p = params[i];
        const auto& g = grads[i];
        auto& m = m_[i];
        auto& v = v_[i];

        for (size_t j = 0; j < p.size(); ++j) {
            m[j] = beta1_ * m[j] + (1 - beta1_) * g[j];
            v[j] = beta2_ * v[j] + (1 - beta2_) * g[j] * g[j];

            T m_hat = m[j] / (1 - std::pow(beta1_, t_));
            T v_hat = v[j] / (1 - std::pow(beta2_, t_));

            p[j] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}

// Explicit instantiation
template class Adam<float>;
template class Adam<double>;
