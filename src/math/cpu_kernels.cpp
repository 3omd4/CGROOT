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
Backward Kernels: Implement the derivatives (e.g., relu_backward is grad_in = (input > 0) ? grad_C : 0;).*/



#include <algorithm>
#include "cpu_kernels.h"


//Add two Matrices A and B element-wise and returns the Output matrix C  = A + B
void mat_add(const mat& A, const mat& B, mat& C)
{
    unsigned int rows = C.size();
    unsigned int columns = C[0].size();

    for(unsigned int i = 0; i < rows; i++)
    {
        for(unsigned int j = 0; j < columns; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

//Subtract two Matrices A and B element-wise and returns the Output matrix C = A - B
void mat_sub(const mat& A, const mat& B, mat& C)
{
    unsigned int rows = C.size();
    unsigned int columns = C[0].size();

    for(unsigned int i = 0; i < rows; i++)
    {
        for(unsigned int j = 0; j < columns; j++)
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

//Multiply two Matrices A and B and returns the new matrix C = A * B
void mat_mul(const mat& A, const mat& B, mat& C)
{   
    unsigned int rows = C.size();
    unsigned int columns = C[0].size();
    unsigned int dotProductLenght = B.size();

    for(unsigned int i = 0; i < rows; i++)
    {
        for(unsigned int j = 0; j < columns; j++)
        {
            C[i][j] = 0;
            for(unsigned int k = 0; k < dotProductLenght; k++)
            {
                C[i][j] += A[i][k] * B[k][j]; 
            }
        }
    }
}

//Multiply a Matrix A with a scalar v and returns the new matrix C = v * A
void mat_scaler_mul(const mat& A, const double v, mat& C)
{
    unsigned int rows = C.size();
    unsigned int columns = C[0].size();

    for(unsigned int i = 0; i < rows; i++)
    {
        for(unsigned int j = 0; j < columns; j++)
        {
            C[i][j] = v*A[i][j];
        }
    }
}

//Multiply a Matrix A with a scalar v and returns A 
void mat_scaler_mul_inPlace(mat& A, const double v)
{
    unsigned int rows = A.size();
    unsigned int columns = A[0].size();

    for(unsigned int i = 0; i < rows; i++)
    {
        for(unsigned int j = 0; j < columns; j++)
        {
            A[i][j] *= v;
        }
    }
}


//Transpose the Matrix A and return the new matrix C = A(transpose)
void mat_transpose(const mat& A, mat& C)
{
    unsigned int rows = C.size();
    unsigned int columns = C[0].size();

    for(unsigned int i = 0; i < rows; i++)
    {
        for(unsigned int j = 0; j < columns; j++)
        {
            C[i][j] = A[j][i];
        }
    }
}

//Multiply A and B element-wise and return the new matrix C = A .* B
void mat_mul_element_wise(const mat& A, const mat& B, mat& C)
{
    unsigned int rows = C.size();
    unsigned int columns = C[0].size();

    for(unsigned int i = 0; i < rows; i++)
    {
        for(unsigned int j = 0; j < columns; j++)
        {
            C[i][j] = A[i][j] * A[i][j];
        }
    }
}


void printMat(const mat& matrix)
{
    for(int i = 0; i < matrix.size(); i++)
    {
        for(int j = 0; j < matrix[0].size(); j++)
        {
            cout << matrix[i][j] << ' ';
        }
        cout << '\n';
    }
}

int main()
{
    mat A = {
    {1.1, 4.5, 0.2, 7.3, 8.0, 5.5, 6.9, 2.4, 3.6, 9.7},
    {0.8, 6.2, 3.3, 9.1, 4.7, 2.0, 8.5, 1.9, 7.4, 5.0},
    {5.8, 2.9, 7.7, 1.4, 6.3, 9.5, 0.6, 4.1, 8.8, 3.0},
    {4.2, 8.6, 1.0, 5.3, 2.7, 7.1, 9.9, 3.8, 6.4, 0.3},
    {7.0, 3.1, 9.4, 6.8, 1.5, 8.2, 4.0, 0.7, 5.9, 2.2},
    {2.5, 5.7, 8.3, 0.9, 4.4, 1.8, 6.0, 9.6, 3.2, 7.9},
    {9.2, 6.6, 3.9, 7.5, 0.1, 5.1, 2.8, 8.4, 1.3, 4.8},
    {1.7, 8.9, 4.6, 2.3, 9.0, 3.7, 7.2, 5.2, 0.4, 6.1},
    {6.5, 0.0, 5.4, 8.1, 3.5, 4.9, 1.2, 2.6, 9.3, 7.8},
    {3.4, 7.6, 2.1, 4.3, 5.6, 6.7, 8.7, 9.8, 1.6, 0.5}
};

    mat B = {
    {9.9, 2.8, 5.1, 0.3, 8.4, 1.7, 4.6, 6.0, 7.2, 3.5},
    {1.2, 8.8, 3.7, 6.4, 0.9, 5.0, 2.3, 9.5, 4.1, 7.6},
    {7.0, 4.3, 9.1, 2.2, 6.8, 3.1, 0.5, 8.7, 5.4, 1.9},
    {5.7, 0.1, 8.2, 4.9, 1.4, 7.3, 3.8, 6.6, 9.0, 2.4},
    {3.3, 6.9, 1.6, 7.8, 5.2, 9.4, 4.0, 0.7, 8.5, 2.1},
    {8.0, 2.7, 6.2, 1.1, 4.5, 0.4, 7.9, 3.6, 5.8, 9.3},
    {0.6, 5.3, 2.0, 8.9, 3.4, 6.1, 9.7, 1.3, 4.8, 7.5},
    {4.2, 7.1, 0.8, 3.0, 9.6, 5.9, 2.5, 8.1, 1.8, 6.7},
    {6.5, 1.0, 7.4, 5.5, 2.9, 8.6, 3.2, 9.8, 0.2, 4.7},
    {2.6, 9.2, 4.4, 7.7, 1.5, 3.9, 6.3, 0.0, 8.3, 5.6}};

    mat C(10);
    for(int i = 0; i < 10; i++)
    {
        C[i].resize(10);
    }

    cout << "\n\nadd matrices:\n\n";
    mat_add(A, B, C);
    printMat(C);

    cout << "\n\nsubtract matrices:\n\n";
    mat_sub(A, B, C);
    printMat(C);

    for(int i = 0; i < 10; i++)
    {
        fill(C[i].begin(), C[i].end(), 0);
    }

    cout << "\n\nmultiply matrices:\n\n";
    mat_mul(A, B, C);
    printMat(C);

    return 0;
}