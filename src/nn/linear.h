/*Purpose: A fully connected linear layer.

To-Do:
class Linear : public Module.
Members: Parameter<T> weight_, Parameter<T> bias_.
linear.h: Declare constructor Linear(int in, int out) and Tensor<T> forward(const Tensor<T>& input) override;.
linear.cpp:
Constructor: Initialize weight_ and bias_ Parameters (e.g., Kaiming init). Call register_parameter(&weight_) and register_parameter(&bias_).
forward: return input.matmul(weight_.transpose()) + bias_;.*/