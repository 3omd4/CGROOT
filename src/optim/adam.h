/*Purpose: Adam optimizer.

To-Do:

class Adam : public Optimizer.

Members: beta1_, beta2_, epsilon_, std::vector<Tensor<T>> m_ (1st moment), std::vector<Tensor<T>> v_ (2nd moment), int t_ (timestep).

adam.cpp (Constructor): Initialize m_ and v_ as vectors of zero-tensors, one for each parameter.

adam.cpp (step): Implement the Adam update logic for each parameter using m_, v_, and the betas.*/