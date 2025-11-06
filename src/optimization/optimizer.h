# ifndef CGROOT_SRC_OPTIMIZATION_OPTIMIZER_H_
# define CGROOT_SRC_OPTIMIZATION_OPTIMIZER_H_

# include <vector>
#include "parameter.h"// A=Includes a simple wrapper for Tensors

template <typename T>
class Optimizer {
    public:
    virtual void step(std::vector<Tensor<T>>& params,
                      const std::vector<Tensor<T>>& grads) = 0;
    virtual ~Optimizer() = default;
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