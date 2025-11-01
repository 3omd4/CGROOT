/*Purpose: Defines the base class for all optimizers.

To-Do:

class Optimizer.

Members: std::vector<Parameter<T>*> params_, float lr_.

Methods:

Optimizer(std::vector<Parameter<T>*> params, float lr).

virtual ~Optimizer() {}

void zero_grad(): Loops through params_ and calls p->zero_grad().

virtual void step() = 0; (Pure virtual).*/