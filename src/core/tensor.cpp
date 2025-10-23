/*
Purpose: Implementation of all Tensor methods.

To-Do:
Implement all constructors. They must allocate memory for data_ and compute strides_.
Implement backward(). This method will call autograd::backward(*this).
Implement every Math Op (e.g., add):
Create a new Tensor (result) for the output.
Call the corresponding kernel (e.g., math::cpu_add(result.data(), this->data(), other.data(), ...)).
If requires_grad_ is true for either input:
Create an autograd node: result.grad_fn_ = std::make_shared<autograd::AddNode>(*this, other).
Set result.requires_grad_ = true.
Return result.
*/