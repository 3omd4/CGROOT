#include "cpu_kernels.h"
#include <cmath>

namespace cgroot {
namespace math {
// note that all funtions ar of type void and operate on raw pointers
template <typename T>
void cpu_add(T* out, const T* in1, const T* in2, unsigned long long size) {
    // takes two input arrays in1 and in2 of given size, adds them element-wise, and stores the result in out
    for (unsigned long long i = 0; i < size; ++i)
        out[i] = in1[i] + in2[i];
}

template <typename T>
void cpu_mul(T* out, const T* in1, const T* in2, unsigned long long size) {
    // takes two input arrays in1 and in2 of given size, multiplies them element-wise, and stores the result in out
    for (unsigned long long i = 0; i < size; ++i)
        out[i] = in1[i] * in2[i];
}

template <typename T>
void cpu_matmul(T* out, const T* A, const T* B,
                unsigned long long M, unsigned long long N, unsigned long long K) {
    // takes matrices A (MxK) and B (KxN), performs matrix multiplication, and stores the result in out (MxN)                
    // A is MxK, B is KxN, out is MxN
    for (unsigned long long i = 0; i < M; ++i) {
        for (unsigned long long j = 0; j < N; ++j) {
            T sum = 0;
            for (unsigned long long k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            out[i * N + j] = sum;
        }
    }
}

template <typename T>
void cpu_relu(T* out, const T* in, unsigned long long size) {
    // takes input array in of given size, applies ReLU activation, and stores the result in out
    for (unsigned long long i = 0; i < size; ++i)
        out[i] = in[i] > 0 ? in[i] : 0;
}

template <typename T>
void cpu_sigmoid(T* out, const T* in, unsigned long long size) {
    // takes input array in of given size, applies Sigmoid activation, and stores the result in out
    for (unsigned long long i = 0; i < size; ++i)
        out[i] = 1 / (1 + std::exp(-in[i]));
}

template <typename T>
void cpu_relu_backward(T* grad_in, const T* grad_out, const T* input, unsigned long long size) {
    // takes gradient output grad_out and input, computes gradient input for ReLU, and stores it in grad_in
    for (unsigned long long i = 0; i < size; ++i)
        grad_in[i] = input[i] > 0 ? grad_out[i] : 0;
}

template <typename T>
void cpu_sigmoid_backward(T* grad_in, const T* grad_out, const T* input, unsigned long long size) {
    // takes gradient output grad_out and input, computes gradient input for Sigmoid, and stores it in grad_in
    for (unsigned long long i = 0; i < size; ++i) {
        T s = 1 / (1 + std::exp(-input[i]));
        grad_in[i] = grad_out[i] * s * (1 - s);
    }
}

// Explicit template instantiation
// it forces the compiler to compile and emit code for cpu_add<double> so that other .cpp files can link against it.
template void cpu_add<double>(double*, const double*, const double*, unsigned long long);
template void cpu_mul<double>(double*, const double*, const double*, unsigned long long);
template void cpu_matmul<double>(double*, const double*, const double*, unsigned long long, unsigned long long, unsigned long long);
template void cpu_relu<double>(double*, const double*, unsigned long long);
template void cpu_sigmoid<double>(double*, const double*, unsigned long long);
template void cpu_relu_backward<double>(double*, const double*, const double*, unsigned long long);
template void cpu_sigmoid_backward<double>(double*, const double*, const double*, unsigned long long);

} // namespace math
} // namespace cgroot
/*Purpose: Implementation of all cpu_... kernels.
To-Do:
Implement all declared functions using raw C++ loops.
cpu_matmul: Start with a naive 3-nested loop. This is the #1 candidate for optimization (tiling/blocking).
cpu_conv2d: This should be implemented by first calling cpu_im2col to unroll the input, then calling cpu_matmul.
Backward Kernels: Implement the derivatives (e.g., relu_backward is grad_in = (input > 0) ? grad_out : 0;).*/