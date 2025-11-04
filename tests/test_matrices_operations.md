# 🧮 Matrix Operations Library - Complete Documentation

**Project:** CGroot++  
**Location:** `src/math/`  
**Test Suite:** `tests/test_matrices_operations.cpp`

---

## 📘 Overview

This C++ project implements and tests three fundamental matrix operations:

1. **Matrix Multiplication** - Standard matrix product `C = A × B`
2. **Matrix Inversion** - Inverse matrix computation `B = A^(-1)`
3. **Element-wise Multiplication** - Hadamard product `C[i,j] = A[i,j] × B[i,j]`

The implementation provides clean, human-readable code with comprehensive unit tests covering normal cases, edge cases, and error handling.

---

## 📁 Project Structure

```
src/math/
├── matrix_ops.h      # Matrix class declaration
└── matrix_ops.cpp    # Matrix operations implementation

tests/
├── test_matrices_operations.cpp  # Comprehensive test suite (31+ tests)
└── test_matrices_operations.md   # This documentation

examples/
└── matrix_ops_example.cpp        # Usage examples with I/O
```

---

## 🔧 API Reference

### Matrix Class

```cpp
namespace cgroot::math {

class Matrix {
public:
    // Constructors
    Matrix(size_t rows, size_t cols);
    Matrix(const std::vector<std::vector<double>>& data);
    
    // Accessors
    size_t rows() const;
    size_t cols() const;
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    double& at(size_t row, size_t col);  // Compatibility method
    const double& at(size_t row, size_t col) const;
    
    // Operations
    static Matrix multiply(const Matrix& a, const Matrix& b);
    static Matrix invert(const Matrix& a);
    static Matrix elementwiseMultiply(const Matrix& a, const Matrix& b);
    
    // Aliases for compatibility
    static Matrix inverse(const Matrix& a);
    static Matrix elementWiseMultiply(const Matrix& a, const Matrix& b);
    
    // Utilities
    void print() const;
    std::vector<std::vector<double>> toVector() const;
};

}
```

---

## ⚙️ 1️⃣ Matrix Multiplication

**Function:**
```cpp
static Matrix Matrix::multiply(const Matrix& A, const Matrix& B);
```

**Description:**
Computes the product `C = A × B` using standard matrix multiplication algorithm.

**Requirements:**
- Number of columns in `A` must equal the number of rows in `B`
- Result matrix dimensions: `C.rows = A.rows`, `C.cols = B.cols`

**Algorithm:**
```
For each row i in A:
    For each column j in B:
        C[i,j] = Σ(k=0 to A.cols-1) A[i,k] × B[k,j]
```

**Throws:**
- `std::invalid_argument` if dimensions are incompatible

---

### ✅ Test Cases (10 tests)

| #  | Type           | Input A                | Input B                   | Expected Output / Behavior             |
| -- | -------------- | ---------------------- | ------------------------- | -------------------------------------- |
| 1  | Basic          | `[[1,2],[3,4]]`        | `[[5,6],[7,8]]`           | `[[19,22],[43,50]]`                    |
| 2  | Single element | `[[2]]`                | `[[3]]`                   | `[[6]]`                                |
| 3  | Non-square     | `[[1,0,2],[-1,3,1]]`   | `[[3,1],[2,1],[1,0]]`     | `[[5,1],[4,2]]`                        |
| 4  | Rectangular    | `[[1,2,3],[4,5,6]]`    | `[[7,8],[9,10],[11,12]]`  | `[[58,64],[139,154]]`                  |
| 5  | Rectangular    | `[[1,2],[3,4],[5,6]]`  | `[[7,8,9],[10,11,12]]`    | `[[27,30,33],[61,68,75],[95,106,117]]` |
| 6  | Invalid        | A: 2×2, B: 3×3         | Throws `invalid_argument` |                                        |
| 7  | Empty          | A: 0×0, B: 0×0         | Throws `invalid_argument` |                                        |
| 8  | Identity       | `I * A`                 | Returns A                 |                                        |
| 9  | Zero           | Zero * A                | Zero                      |                                        |
| 10 | Scalar check   | `[[1000]] * [[0.001]]` | `[[1]]`                   |                                        |

**Example:**
```cpp
Matrix A({{1, 2}, {3, 4}});
Matrix B({{5, 6}, {7, 8}});
Matrix C = Matrix::multiply(A, B);
// C = [[19, 22], [43, 50]]
```

---

## 🔄 2️⃣ Matrix Inversion

**Function:**
```cpp
static Matrix Matrix::invert(const Matrix& A);
// Alias: static Matrix Matrix::inverse(const Matrix& A);
```

**Description:**
Computes the inverse of a square matrix using the **adjugate method**:
1. Calculate the determinant
2. Compute the adjugate matrix
3. Divide adjugate by determinant: `A^(-1) = adj(A) / det(A)`

**Requirements:**
- Matrix must be square (`A.rows == A.cols`)
- Matrix must be non-singular (`det(A) ≠ 0`)

**Throws:**
- `std::invalid_argument` → if matrix is not square
- `std::runtime_error` → if matrix is singular (determinant is zero)

---

### ✅ Test Cases (11 tests)

| #  | Type            | Input                         | Expected Output / Behavior          |
| -- | --------------- | ----------------------------- | ----------------------------------- |
| 1  | Basic           | `[[1,2],[3,4]]`               | `[[-2,1],[1.5,-0.5]]`               |
| 2  | Basic           | `[[4,7],[2,6]]`               | `[[0.6,-0.7],[-0.2,0.4]]`           |
| 3  | Single          | `[[2]]`                       | `[[0.5]]`                           |
| 4  | 3×3             | `[[1,2,3],[0,1,4],[5,6,0]]`   | `[[−24,18,5],[20,−15,−4],[−5,4,1]]` |
| 5  | Identity        | `I`                           | `I`                                 |
| 6  | Non-square      | `2×3`                         | Throws `invalid_argument`           |
| 7  | Singular        | `[[1,2],[2,4]]`               | Throws `runtime_error`              |
| 8  | Empty           | `0×0`                         | Throws `invalid_argument`           |
| 9  | Diagonal        | `[[2,0],[0,4]]`               | `[[0.5,0],[0,0.25]]`                |
| 10 | Nearly singular | `[[1,1.000001],[1,1.000002]]` | Valid with tolerance check          |
| 11 | Large values    | `[[1000,2000],[3000,4000]]`   | May throw or succeed with tolerance |

**Example:**
```cpp
Matrix A({{4, 7}, {2, 6}});
Matrix Inv = Matrix::invert(A);
Matrix Verify = Matrix::multiply(A, Inv);
// Verify should be identity matrix [[1,0],[0,1]]
```

**Verification:**
After inversion, `A × A^(-1) = I` (identity matrix) should hold true.

---

## ✴️ 3️⃣ Element-wise Multiplication

**Function:**
```cpp
static Matrix Matrix::elementwiseMultiply(const Matrix& A, const Matrix& B);
// Alias: static Matrix Matrix::elementWiseMultiply(const Matrix& A, const Matrix& B);
```

**Description:**
Computes the Hadamard product (element-wise multiplication) where each element of the result is the product of corresponding elements from the input matrices.

**Formula:**
```
C[i,j] = A[i,j] × B[i,j]
```

**Requirements:**
- Both matrices must have identical dimensions
- `A.rows == B.rows` AND `A.cols == B.cols`

**Throws:**
- `std::invalid_argument` if matrix dimensions differ

---

### ✅ Test Cases (10 tests)

| #  | Type            | Input A                               | Input B                   | Expected Output / Behavior |
| -- | --------------- | ------------------------------------- | ------------------------- | -------------------------- |
| 1  | Basic           | `[[1,2],[3,4]]`                       | `[[5,6],[7,8]]`           | `[[5,12],[21,32]]`         |
| 2  | Single          | `[[2]]`                               | `[[3]]`                   | `[[6]]`                    |
| 3  | Row vectors     | `[[1,2,3]]`                           | `[[4,5,6]]`               | `[[4,10,18]]`              |
| 4  | Rectangular     | `[[1,2,3],[4,5,6]]`                   | `[[6,5,4],[3,2,1]]`       | `[[6,10,12],[12,10,6]]`    |
| 5  | Invalid         | A: 2×2, B: 3×3                        | Throws `invalid_argument` |                            |
| 6  | Invalid         | A: 2×3, B: 2×2                        | Throws `invalid_argument` |                            |
| 7  | Zero            | Zero matrices                         | Zero matrix               |                            |
| 8  | All ones        | Ones * arbitrary                      | Equals arbitrary matrix   |                            |
| 9  | Mixed signs     | `[[−1,2],[−3,4]]` × `[[2,−2],[3,−4]]` | `[[-2,−4],[-9,−16]]`      |                            |
| 10 | Floating values | `[[0.1,0.2]]` × `[[10,5]]`            | `[[1,1]]`                 |                            |

**Example:**
```cpp
Matrix A({{1, 2}, {3, 4}});
Matrix B({{5, 6}, {7, 8}});
Matrix C = Matrix::elementwiseMultiply(A, B);
// C = [[5, 12], [21, 32]]
```

---

## 🧪 Running the Tests

### 🧱 Compilation

**Windows (PowerShell):**
```powershell
cd "D:\Faculty Of Engineering\Year Three Eng\Software Engineering Project\CGROOT"
g++ -std=c++17 -I. src/math/matrix_ops.cpp tests/test_matrices_operations.cpp -o test_matrices_operations.exe
```

**Linux/Mac:**
```bash
g++ -std=c++17 -I. src/math/matrix_ops.cpp tests/test_matrices_operations.cpp -o test_matrices_operations
```

### ▶️ Execution

**Windows:**
```powershell
.\test_matrices_operations.exe
```

**Linux/Mac:**
```bash
./test_matrices_operations
```

### 🖥️ Expected Output

```
============================================================
  MATRIX OPERATIONS - COMPREHENSIVE TEST SUITE
============================================================

============================================================
  Testing Matrix Multiplication
============================================================

  ✓ Test 1: Basic 2x2 PASSED
  ✓ Test 2: Single element PASSED
  ✓ Test 3: Non-square (2x3 * 3x2) PASSED
  ...
------------------------------------------------------------
Matrix Multiplication: 10/10 tests passed

============================================================
  Testing Matrix Inversion
============================================================
  ...
------------------------------------------------------------
Matrix Inversion: 11/11 tests passed

============================================================
  Testing Element-wise Multiplication
============================================================
  ...
------------------------------------------------------------
Element-wise Multiplication: 10/10 tests passed

============================================================
  ✅ All matrix operation tests completed successfully!
============================================================
```

---

## 📊 Test Summary

| Operation              | Normal Tests | Edge Cases | Error Tests | **Total**      |
| ---------------------- | ------------ | ---------- | ----------- | -------------- |
| Multiplication         | 5            | 3          | 2           | **10**         |
| Inversion              | 5            | 4          | 3           | **12**         |
| Element-wise           | 4            | 3          | 2           | **9**          |
| **Grand Total**        | **14**       | **10**     | **7**       | **31 Tests ✅** |

---

## 💡 Implementation Notes

### Floating-Point Precision

- All floating-point comparisons use tolerance-based checking (`1e-6` default)
- Helper function: `bool nearlyEqual(double a, double b, double eps = 1e-6)`
- Matrix inversion verification uses relaxed tolerance for near-singular cases

### Error Handling

- **Dimension mismatches**: `std::invalid_argument` exception
- **Singular matrices**: `std::runtime_error` exception
- **Empty matrices**: `std::invalid_argument` exception
- **Index out of bounds**: `std::out_of_range` exception

### Memory Management

- Uses `std::vector<double>` for efficient storage
- Row-major storage layout: `data_[row * cols_ + col]`
- Zero-copy operations where possible

### Performance Considerations

- Matrix multiplication: O(n³) for n×n matrices
- Matrix inversion: O(n³) for n×n matrices (determinant + adjugate)
- Element-wise operations: O(n²) for n×n matrices
- Future optimization opportunities: tiling, SIMD, BLAS integration

---

## 🔗 Related Files

- **Implementation**: `src/math/matrix_ops.h`, `src/math/matrix_ops.cpp`
- **Test Suite**: `tests/test_matrices_operations.cpp`
- **Examples**: `examples/matrix_ops_example.cpp`
- **Documentation**: This file (`tests/test_matrices_operations.md`)

---

## 📝 Usage Example

```cpp
#include "../src/math/matrix_ops.h"
using namespace cgroot::math;

int main() {
    // Create matrices
    Matrix A({{1, 2}, {3, 4}});
    Matrix B({{5, 6}, {7, 8}});
    
    // Matrix multiplication
    Matrix C = Matrix::multiply(A, B);
    C.print();
    
    // Matrix inversion
    Matrix Inv = Matrix::invert(A);
    Inv.print();
    
    // Element-wise multiplication
    Matrix D = Matrix::elementwiseMultiply(A, B);
    D.print();
    
    return 0;
}
```

---

## ✅ Code Quality

- **Clean Code**: Minimal comments, self-documenting function names
- **Human-Readable**: Clear structure and logical flow
- **Comprehensive Tests**: 31+ test cases covering all scenarios
- **Error Handling**: Proper exception handling with descriptive messages
- **Standards Compliant**: C++17 standard, follows best practices

---

**Last Updated**: 2025
**Version**: 1.0  
**Status**: ✅ Production Ready
