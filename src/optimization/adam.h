#ifndef CGOPTIMIZATION_ADAM_H
#define CGOPTIMIZATION_ADAM_H_
#include "optimizer.h"

template <typename T>
class Adam : public Optimizer<T> {
private:
    T lr_, beta1_, beta2_, epsilon_;
    int t_;
    std::vector<Tensor<T>> m_;  // first moment
    std::vector<Tensor<T>> v_;  // second moment

public:
    Adam(T lr=0.001, T beta1=0.9, T beta2=0.999, T eps=1e-8);

    void step(std::vector<Tensor<T>>& params,
              const std::vector<Tensor<T>>& grads) override;
};
#endif
/*Purpose: Adam optimizer.

To-Do:

class Adam : public Optimizer.

Members: beta1_, beta2_, epsilon_, std::vector<Tensor<T>> m_ (1st moment), std::vector<Tensor<T>> v_ (2nd moment), int t_ (timestep).

adam.cpp (Constructor): Initialize m_ and v_ as vectors of zero-tensors, one for each parameter.

adam.cpp (step): Implement the Adam update logic for each parameter using m_, v_, and the betas.*/