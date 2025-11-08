# ifndef CGROOT_SRC_OPTIMIZATION_OPTIMIZER_H_
# define CGROOT_SRC_OPTIMIZATION_OPTIMIZER_H_
# include <vector>
#include "parameter.h"// A=Includes a simple wrapper for Tensors

template<typename T>
class Optimizer {
protected:
    std::vector<Tensor<T>*> params_;
    T lr_;

public:
    Optimizer(std::vector<Tensor<T>*>& params, T lr)
        : params_(params), lr_(lr) {}

    virtual void step() = 0;

    virtual void zero_grad() {
        for (size_t i = 0; i < params_.size(); ++i) {
            params_[i]->zero_grad();
        }
    }

    virtual ~Optimizer() {}
};
# endif
/*Purpose: Defines the base class for all optimizers.

To-Do:

class Optimizer.

Members: std::vector<Parameter<T>*> params_, float lr_.

Methods:

Optimizer(std::vector<Parameter<T>*> params, float lr).

virtual ~Optimizer() {}

void zero_grad(): Loops through params_ and calls p->zero_grad().

virtual void step() = 0; (Pure virtual).*/