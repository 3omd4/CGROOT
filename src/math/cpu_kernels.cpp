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