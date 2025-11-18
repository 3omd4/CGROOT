// cpu_kernels.cpp

#include "cpu_kernels.h"
#include <cmath>
#include <vector>    // for mat fallback if not declared in header
#include <cstddef>   // for std::size_t

using namespace std;

namespace cgroot {
namespace math {

// If your header already defines `mat` (e.g. using mat = std::vector<std::vector<double>>;)
// then this alias will be redundant but harmless. If header uses a different type:
// remove or adjust this alias to match the header definition.
#ifndef CGROOT_MAT_ALIAS_PROVIDED
using mat = std::vector<std::vector<double>>;
#endif

// note: all functions operate on raw pointers / simple containers

template <typename T>
void cpu_add(T* out, const T* in1, const T* in2, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) out[i] = in1[i] + in2[i];
}

template <typename T>
void cpu_mul(T* out, const T* in1, const T* in2, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) out[i] = in1[i] * in2[i];
}

template <typename T>
void cpu_matmul(T* out, const T* A, const T* B,
                std::size_t M, std::size_t N, std::size_t K) {
    // A is MxK, B is KxN, out is MxN (row-major)
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            T sum = T(0);
            for (std::size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}

template <typename T>
void cpu_relu(T* out, const T* in, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) out[i] = in[i] > T(0) ? in[i] : T(0);
}

template <typename T>
void cpu_sigmoid(T* out, const T* in, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        double val = static_cast<double>(in[i]);
        double s = 1.0 / (1.0 + std::exp(-val));
        out[i] = static_cast<T>(s);
    }
}

template <typename T>
void cpu_relu_backward(T* grad_in, const T* grad_out, const T* input, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) grad_in[i] = input[i] > T(0) ? grad_out[i] : T(0);
}

template <typename T>
void cpu_sigmoid_backward(T* grad_in, const T* grad_out, const T* input, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        double x = static_cast<double>(input[i]);
        double s = 1.0 / (1.0 + std::exp(-x));
        double deriv = s * (1.0 - s);
        grad_in[i] = static_cast<T>(grad_out[i] * deriv);
    }
}

// Explicit template instantiation for double
template void cpu_add<double>(double*, const double*, const double*, std::size_t);
template void cpu_mul<double>(double*, const double*, const double*, std::size_t);
template void cpu_matmul<double>(double*, const double*, const double*, std::size_t, std::size_t, std::size_t);
template void cpu_relu<double>(double*, const double*, std::size_t);
template void cpu_sigmoid<double>(double*, const double*, std::size_t);
template void cpu_relu_backward<double>(double*, const double*, const double*, std::size_t);
template void cpu_sigmoid_backward<double>(double*, const double*, const double*, std::size_t);

// ---------------------------
// Matrix helpers using mat
// ---------------------------

void mat_add(const mat& A, const mat& B, mat& C)
{
    std::size_t rows = C.size();
    if (rows == 0) return;
    std::size_t cols = C[0].size();

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void mat_sub(const mat& A, const mat& B, mat& C)
{
    std::size_t rows = C.size();
    if (rows == 0) return;
    std::size_t cols = C[0].size();

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

void mat_mul(const mat& A, const mat& B, mat& C)
{
    std::size_t rows = C.size();
    if (rows == 0) return;
    std::size_t cols = C[0].size();
    std::size_t dotLen = B.size(); // B rows = K

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < dotLen; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void mat_scaler_mul(const mat& A, const double v, mat& C)
{
    std::size_t rows = C.size();
    if (rows == 0) return;
    std::size_t cols = C[0].size();

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            C[i][j] = v * A[i][j];
        }
    }
}

void mat_scaler_mul_inPlace(mat& A, const double v)
{
    std::size_t rows = A.size();
    if (rows == 0) return;
    std::size_t cols = A[0].size();

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            A[i][j] *= v;
        }
    }
}

void mat_transpose(const mat& A, mat& C)
{
    std::size_t rows = C.size();
    if (rows == 0) return;
    std::size_t cols = C[0].size();

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            C[i][j] = A[j][i];
        }
    }
}

void mat_mul_element_wise(const mat& A, const mat& B, mat& C)
{
    std::size_t rows = C.size();
    if (rows == 0) return;
    std::size_t cols = C[0].size();

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            C[i][j] = A[i][j] * B[i][j];
        }
    }
}

} // namespace math
} // namespace cgroot
