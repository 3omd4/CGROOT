/*Purpose: Stochastic Gradient Descent optimizer.

To-Do:

class SGD : public Optimizer.

sgd.h: Declare constructor and void step() override;.

sgd.cpp (step):

Loop through all param in params_.

param->data() = param->data() - lr_ * param->grad();

Important: This update step must not build an autograd graph. You may need a Tensor::sub_inplace_no_grad method.*/