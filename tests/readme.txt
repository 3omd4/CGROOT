Contains all unit and integration tests. This is critical for success.

test_tensor.cpp: Unit tests for Tensor and cpu_kernels. Does matmul give the correct numerical result?

test_autograd.cpp: The most important test file. Uses numerical gradient checking (finite difference) to verify that the analytical gradients from your _backward() methods are correct for every single operation.

test_nn.cpp: Integration tests. Can you build a nn::Linear layer? Does optim::SGD correctly update its weights after a backward pass?