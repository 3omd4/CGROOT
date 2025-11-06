#ifndef CGOPTIMIZATION_ADAM_H
#define CGOPTIMIZATION_ADAM_H_
#include "optimizer.h"

template<typename T>
class Adam : public Optimizer<T> {
private:
    T beta1_;
    T beta2_;
    T epsilon_;
    int t_;
    std::vector<Tensor<T>> m_; // first moment estimates
    std::vector<Tensor<T>> v_; // second moment estimates

public:
    Adam(std::vector<Tensor<T>*>& params,
         T lr = static_cast<T>(0.001),
         T beta1 = static_cast<T>(0.9),
         T beta2 = static_cast<T>(0.999),
         T epsilon = static_cast<T>(1e-8))
        : Optimizer<T>(params, lr),
          beta1_(beta1),
          beta2_(beta2),
          epsilon_(epsilon),
          t_(0)
    {
        // initialize m_ and v_ to zero tensors of same shape as params
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor<T> zero_m(params[i]->shape(), static_cast<T>(0));
            Tensor<T> zero_v(params[i]->shape(), static_cast<T>(0));
            m_.push_back(zero_m);
            v_.push_back(zero_v);
        }
    }

    void step() override {
        t_ += 1;
        for (size_t i = 0; i < this->params_.size(); ++i) {
            Tensor<T>& p = *this->params_[i];
            Tensor<T>& grad = p.grad();
            Tensor<T>& m = m_[i];
            Tensor<T>& v = v_[i];

            // m = beta1 * m + (1 - beta1) * grad
            m = (m * beta1_) + (grad * (static_cast<T>(1) - beta1_));

            // v = beta2 * v + (1 - beta2) * grad^2
            v = (v * beta2_) + ((grad * grad) * (static_cast<T>(1) - beta2_));

            // compute bias-corrected values
            T bias_correction1 = static_cast<T>(1) - std::pow(beta1_, static_cast<T>(t_));
            T bias_correction2 = static_cast<T>(1) - std::pow(beta2_, static_cast<T>(t_));
            Tensor<T> m_hat = m / bias_correction1;
            Tensor<T> v_hat = v / bias_correction2;

            // parameter update: p = p - lr * m_hat / (sqrt(v_hat) + epsilon)
            Tensor<T> denom = v_hat.sqrt() + epsilon_;
            Tensor<T> update = (m_hat / denom) * this->lr_;
            p = p - update;
        }
    }
};
#endif
/*Purpose: Adam optimizer.

To-Do:

class Adam : public Optimizer.

Members: beta1_, beta2_, epsilon_, std::vector<Tensor<T>> m_ (1st moment), std::vector<Tensor<T>> v_ (2nd moment), int t_ (timestep).

adam.cpp (Constructor): Initialize m_ and v_ as vectors of zero-tensors, one for each parameter.

adam.cpp (step): Implement the Adam update logic for each parameter using m_, v_, and the betas.*/