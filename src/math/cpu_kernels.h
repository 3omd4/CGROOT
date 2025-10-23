/*
Purpose: Header file declaring all low-level CPU math functions.
To-Do:

Declare all functions within a namespace (e.g., cgroot::math).
Forward Kernels:
void cpu_add(T* out, const T* in1, const T* in2, size_t size);
void cpu_mul(T* out, const T* in1, const T* in2, size_t size);
void cpu_matmul(T* out, const T* in1, const T* in2, size_t M, size_t N, size_t K);
void cpu_im2col(T* out, const T* in, ...); // Helper for Conv2D
void cpu_relu(T* out, const T* in, size_t size);
void cpu_sigmoid(T* out, const T* in, size_t size);
...etc. for all ops.
Backward Kernels:
void cpu_relu_backward(T* grad_in, const T* grad_out, const T* input, size_t size);
void cpu_sigmoid_backward(T* grad_in, const T* grad_out, const T* input, size_t size);*/