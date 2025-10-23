/*Purpose: Implementation of all cpu_... kernels.

To-Do:
Implement all declared functions using raw C++ loops.
cpu_matmul: Start with a naive 3-nested loop. This is the #1 candidate for optimization (tiling/blocking).
cpu_conv2d: This should be implemented by first calling cpu_im2col to unroll the input, then calling cpu_matmul.
Backward Kernels: Implement the derivatives (e.g., relu_backward is grad_in = (input > 0) ? grad_out : 0;).*/