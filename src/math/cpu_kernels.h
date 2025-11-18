#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

// Matrix operations using vector<vector<double>>
#include <vector>
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

// mat is a 2-dimensional matrix of type vector<vector<double>>
typedef std::vector<std::vector<double>> mat;

/***************************************************WARNING***********************************************/
/*ALL FUNCTIONS in this file expect the Output matrix(C) to be of CORRECT DIMENSIONS AND WELL INITIALIZED*/

//Add two Matrices A and B element-wise and returns the Output matrix C = A + B
void mat_add(const mat& A, const mat& B, mat& C);

//Subtract two Matrices A and B element-wise and returns the Output matrix C = A - B
void mat_sub(const mat& A, const mat& B, mat& C);

//Multiply two Matrices A and B and returns the new matrix C = A * B
void mat_mul(const mat& A, const mat& B, mat& C);

//Multiply a Matrix A with a scalar v and returns the new matrix C = v * A
void mat_scaler_mul(const mat& A, const double v, mat& C);

//Multiply a Matrix A with a scalar v and returns A 
void mat_scaler_mul_inPlace(mat& A, const double v);

//Transpose the Matrix A and return the new matrix C = A(transpose)
void mat_transpose(const mat& A, mat& C);

//Multiply A and B element-wise and return the new matrix C = A .* B
void mat_mul_element_wise(const mat& A, const mat& B, mat& C);

} // namespace math
} // namespace cgroot

#endif
