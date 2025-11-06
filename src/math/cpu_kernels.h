#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H
namespace cgroot {
namespace math {

template <typename T>
void cpu_add(T* out, const T* in1, const T* in2, unsigned long long size);

template <typename T>
void cpu_mul(T* out, const T* in1, const T* in2, unsigned long long size);

template <typename T>
void cpu_matmul(T* out, const T* A, const T* B, unsigned long long M, unsigned long long N, unsigned long long K);

template <typename T>
void cpu_relu(T* out, const T* in, unsigned long long size);

template <typename T>
void cpu_sigmoid(T* out, const T* in, unsigned long long size);

template <typename T>
void cpu_relu_backward(T* grad_in, const T* grad_out, const T* input, unsigned long long size);

template <typename T>
void cpu_sigmoid_backward(T* grad_in, const T* grad_out, const T* input, unsigned long long size);

} // namespace math
} // namespace cgrootwhat 

#endif








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
void cpu_sigmoid_backward(T* grad_in, const T* grad_out, const T* input, size_t size);
*/