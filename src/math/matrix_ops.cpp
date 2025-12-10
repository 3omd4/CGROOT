#include "matrix_ops.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <string>


using namespace std;

namespace cgroot {
namespace math {

Matrix::Matrix(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {
}

Matrix::Matrix(const std::vector<std::vector<double>>& data) {
    if (data.empty() || data[0].empty()) {
        throw std::invalid_argument("Matrix cannot be empty");
    }
    
    rows_ = data.size();
    cols_ = data[0].size();
    data_.resize(rows_ * cols_);
    
    for (size_t i = 0; i < rows_; ++i) {
        if (data[i].size() != cols_) {
            throw std::invalid_argument("All rows must have the same size");
        }
        for (size_t j = 0; j < cols_; ++j) {
            data_[i * cols_ + j] = data[i][j];
        }
    }
}

double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[row * cols_ + col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[row * cols_ + col];
}

void Matrix::print() const {
    for (size_t i = 0; i < rows_; ++i) {
        std::cout << "[ ";
        for (size_t j = 0; j < cols_; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                      << (*this)(i, j);
            if (j < cols_ - 1) std::cout << ", ";
        }
        std::cout << " ]" << std::endl;
    }
}

std::vector<std::vector<double>> Matrix::toVector() const {
    std::vector<std::vector<double>> result(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        result[i].resize(cols_);
        for (size_t j = 0; j < cols_; ++j) {
            result[i][j] = (*this)(i, j);
        }
    }
    return result;
}

Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    if (a.cols_ != b.rows_) {
        throw std::invalid_argument(
            "Matrix dimensions incompatible for multiplication: "
            "A.cols (" + std::to_string(a.cols_) + ") must equal "
            "B.rows (" + std::to_string(b.rows_) + ")"
        );
    }
    
    Matrix result(a.rows_, b.cols_);
    
    for (size_t i = 0; i < a.rows_; ++i) {
        for (size_t j = 0; j < b.cols_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < a.cols_; ++k) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
    
    return result;
}

Matrix Matrix::elementwiseMultiply(const Matrix& a, const Matrix& b) {
    if (a.rows_ != b.rows_ || a.cols_ != b.cols_) {
        throw std::invalid_argument(
            "Matrices must have the same dimensions for element-wise multiplication"
        );
    }
    
    Matrix result(a.rows_, a.cols_);
    
    for (size_t i = 0; i < a.rows_; ++i) {
        for (size_t j = 0; j < a.cols_; ++j) {
            result(i, j) = a(i, j) * b(i, j);
        }
    }
    
    return result;
}

double Matrix::determinant(const Matrix& m) {
    if (m.rows_ != m.cols_) {
        throw std::invalid_argument("Determinant only defined for square matrices");
    }
    
    size_t n = m.rows_;
    
    if (n == 0) {
        return 1.0;
    }
    
    if (n == 1) {
        return m(0, 0);
    }
    
    if (n == 2) {
        return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    }
    
    double det = 0.0;
    for (size_t j = 0; j < n; ++j) {
        Matrix sub = submatrix(m, 0, j);
        double sign = (j % 2 == 0) ? 1.0 : -1.0;
        det += sign * m(0, j) * determinant(sub);
    }
    
    return det;
}

Matrix Matrix::submatrix(const Matrix& m, size_t excludeRow, size_t excludeCol) {
    Matrix result(m.rows_ - 1, m.cols_ - 1);
    
    size_t resultRow = 0;
    for (size_t i = 0; i < m.rows_; ++i) {
        if (i == excludeRow) continue;
        
        size_t resultCol = 0;
        for (size_t j = 0; j < m.cols_; ++j) {
            if (j == excludeCol) continue;
            result(resultRow, resultCol) = m(i, j);
            ++resultCol;
        }
        ++resultRow;
    }
    
    return result;
}

Matrix Matrix::adjugate(const Matrix& m) {
    if (m.rows_ != m.cols_) {
        throw std::invalid_argument("Adjugate only defined for square matrices");
    }
    
    size_t n = m.rows_;
    Matrix adj(n, n);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Matrix sub = submatrix(m, j, i);
            double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            adj(i, j) = sign * determinant(sub);
        }
    }
    
    return adj;
}

Matrix Matrix::invert(const Matrix& a) {
    if (a.rows_ != a.cols_) {
        throw std::invalid_argument("Inverse only defined for square matrices");
    }
    
    if (a.rows_ == 0) {
        throw std::invalid_argument("Cannot invert empty matrix");
    }
    
    double det = determinant(a);
    if (std::abs(det) < 1e-10) {
        throw std::runtime_error("Matrix is singular (determinant is zero), cannot invert");
    }
    
    Matrix adj = adjugate(a);
    Matrix result(a.rows_, a.cols_);
    
    for (size_t i = 0; i < a.rows_; ++i) {
        for (size_t j = 0; j < a.cols_; ++j) {
            result(i, j) = adj(i, j) / det;
        }
    }
    
    return result;
}

}
}

