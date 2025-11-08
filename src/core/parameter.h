
/*
Purpose: A simple wrapper for Tensor to identify it as a learnable model parameter.

To-Do:
Define template <typename T> class Parameter : public Tensor<T>.
Methods:
Implement constructors that simply call the Tensor base constructors but always set requires_grad = true.
Add a factory function (e.g., Parameter::create_kaiming(...)) to initialize weights with a specific distribution.
*/
// asdasdad