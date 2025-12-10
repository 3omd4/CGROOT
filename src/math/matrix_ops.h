#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <vector>

namespace cgroot {
namespace math {

class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    Matrix(const std::vector<std::vector<double>>& data);
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    double& at(size_t row, size_t col) { return (*this)(row, col); }
    const double& at(size_t row, size_t col) const { return (*this)(row, col); }
    
    void print() const;
    std::vector<std::vector<double>> toVector() const;
    
    static Matrix multiply(const Matrix& a, const Matrix& b);
    static Matrix invert(const Matrix& a);
    static Matrix elementwiseMultiply(const Matrix& a, const Matrix& b);
    
    static Matrix inverse(const Matrix& a) { return invert(a); }
    static Matrix elementWiseMultiply(const Matrix& a, const Matrix& b) { return elementwiseMultiply(a, b); }

private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;
    
    static double determinant(const Matrix& m);
    static Matrix adjugate(const Matrix& m);
    static Matrix submatrix(const Matrix& m, size_t excludeRow, size_t excludeCol);
};

}
}

#endif

