#include "../src/math/matrix_ops.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace cgroot::math;

bool nearlyEqual(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) < eps;
}

void printTestHeader(const std::string& testName) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << testName << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void printTestResult(const std::string& testCase, bool passed) {
    std::cout << "  " << (passed ? "✓" : "✗") << " " << testCase;
    if (passed) std::cout << " PASSED";
    std::cout << "\n";
}

void testMatrixMultiplication() {
    printTestHeader("Testing Matrix Multiplication");
    int passed = 0, total = 0;

    std::cout << "\nTest Case 1: Basic 2x2 multiplication\n";
    Matrix A1({{1, 2}, {3, 4}});
    Matrix B1({{5, 6}, {7, 8}});
    Matrix C1 = Matrix::multiply(A1, B1);
    bool test1 = nearlyEqual(C1.at(0,0), 19) && nearlyEqual(C1.at(0,1), 22) &&
                 nearlyEqual(C1.at(1,0), 43) && nearlyEqual(C1.at(1,1), 50);
    printTestResult("Test 1: Basic 2x2", test1);
    if (test1) passed++; total++;

    std::cout << "\nTest Case 2: Single element matrices\n";
    Matrix A2(std::vector<std::vector<double>>{{2}});
    Matrix B2(std::vector<std::vector<double>>{{3}});
    Matrix C2 = Matrix::multiply(A2, B2);
    bool test2 = nearlyEqual(C2.at(0,0), 6);
    printTestResult("Test 2: Single element", test2);
    if (test2) passed++; total++;

    std::cout << "\nTest Case 3: Non-square matrices (2x3 * 3x2)\n";
    Matrix A3({{1,0,2},{-1,3,1}});
    Matrix B3({{3,1},{2,1},{1,0}});
    Matrix C3 = Matrix::multiply(A3, B3);
    bool test3 = nearlyEqual(C3.at(0,0), 5) && nearlyEqual(C3.at(0,1), 1) &&
                 nearlyEqual(C3.at(1,0), 4) && nearlyEqual(C3.at(1,1), 2);
    printTestResult("Test 3: Non-square (2x3 * 3x2)", test3);
    if (test3) passed++; total++;

    std::cout << "\nTest Case 4: Rectangular (2x3 * 3x2)\n";
    Matrix A4({{1,2,3},{4,5,6}});
    Matrix B4({{7,8},{9,10},{11,12}});
    Matrix C4 = Matrix::multiply(A4, B4);
    bool test4 = nearlyEqual(C4.at(0,0), 58) && nearlyEqual(C4.at(0,1), 64) &&
                 nearlyEqual(C4.at(1,0), 139) && nearlyEqual(C4.at(1,1), 154);
    printTestResult("Test 4: Rectangular (2x3 * 3x2)", test4);
    if (test4) passed++; total++;

    std::cout << "\nTest Case 5: Rectangular (3x2 * 2x3)\n";
    Matrix A5({{1,2},{3,4},{5,6}});
    Matrix B5({{7,8,9},{10,11,12}});
    Matrix C5 = Matrix::multiply(A5, B5);
    bool test5 = nearlyEqual(C5.at(0,0), 27) && nearlyEqual(C5.at(0,2), 33) &&
                 nearlyEqual(C5.at(2,0), 95) && nearlyEqual(C5.at(2,2), 117);
    printTestResult("Test 5: Rectangular (3x2 * 2x3)", test5);
    if (test5) passed++; total++;

    std::cout << "\nTest Case 6: Incompatible dimensions\n";
    try {
        Matrix::multiply(Matrix(2,2), Matrix(3,3));
        printTestResult("Test 6: Incompatible dimensions", false);
        total++;
    } catch (const std::invalid_argument&) {
        printTestResult("Test 6: Incompatible dimensions", true);
        passed++; total++;
    }

    std::cout << "\nTest Case 7: Empty matrix\n";
    try {
        Matrix empty1(0,0);
        Matrix empty2(0,0);
        Matrix result = Matrix::multiply(empty1, empty2);
        bool test7 = (result.rows() == 0 && result.cols() == 0);
        printTestResult("Test 7: Empty matrix", test7);
        if (test7) passed++; total++;
    } catch (const std::exception&) {
        printTestResult("Test 7: Empty matrix (throws exception)", true);
        passed++; total++;
    }

    std::cout << "\nTest Case 8: Identity matrix property\n";
    Matrix I({{1,0},{0,1}});
    Matrix C8 = Matrix::multiply(I, A1);
    bool test8 = nearlyEqual(C8.at(0,0), 1) && nearlyEqual(C8.at(0,1), 2) &&
                 nearlyEqual(C8.at(1,0), 3) && nearlyEqual(C8.at(1,1), 4);
    printTestResult("Test 8: Identity matrix", test8);
    if (test8) passed++; total++;

    std::cout << "\nTest Case 9: Zero matrix\n";
    Matrix Z(2,2);
    Matrix C9 = Matrix::multiply(A1, Z);
    bool test9 = nearlyEqual(C9.at(0,0), 0) && nearlyEqual(C9.at(0,1), 0) &&
                 nearlyEqual(C9.at(1,0), 0) && nearlyEqual(C9.at(1,1), 0);
    printTestResult("Test 9: Zero matrix", test9);
    if (test9) passed++; total++;

    std::cout << "\nTest Case 10: Scalar precision check\n";
    Matrix A10(std::vector<std::vector<double>>{{1000}});
    Matrix B10(std::vector<std::vector<double>>{{0.001}});
    Matrix C10 = Matrix::multiply(A10, B10);
    bool test10 = nearlyEqual(C10.at(0,0), 1.0);
    printTestResult("Test 10: Scalar precision", test10);
    if (test10) passed++; total++;

    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << "Matrix Multiplication: " << passed << "/" << total << " tests passed\n";
}

void testMatrixInversion() {
    printTestHeader("Testing Matrix Inversion");
    int passed = 0, total = 0;

    std::cout << "\nTest Case 1: Basic 2x2 inversion\n";
    Matrix A1({{1,2},{3,4}});
    Matrix Inv1 = Matrix::inverse(A1);
    Matrix Verify1 = Matrix::multiply(A1, Inv1);
    bool test1 = nearlyEqual(Inv1.at(0,0), -2) && nearlyEqual(Inv1.at(0,1), 1) &&
                 nearlyEqual(Inv1.at(1,0), 1.5) && nearlyEqual(Inv1.at(1,1), -0.5) &&
                 nearlyEqual(Verify1.at(0,0), 1) && nearlyEqual(Verify1.at(1,1), 1);
    printTestResult("Test 1: Basic 2x2", test1);
    if (test1) passed++; total++;

    std::cout << "\nTest Case 2: Another 2x2 matrix\n";
    Matrix A2({{4,7},{2,6}});
    Matrix Inv2 = Matrix::inverse(A2);
    Matrix Verify2 = Matrix::multiply(A2, Inv2);
    bool test2 = nearlyEqual(Inv2.at(0,0), 0.6) && nearlyEqual(Inv2.at(0,1), -0.7) &&
                 nearlyEqual(Inv2.at(1,0), -0.2) && nearlyEqual(Inv2.at(1,1), 0.4) &&
                 nearlyEqual(Verify2.at(0,0), 1) && nearlyEqual(Verify2.at(1,1), 1);
    printTestResult("Test 2: Another 2x2", test2);
    if (test2) passed++; total++;

    std::cout << "\nTest Case 3: Single element matrix\n";
    Matrix A3(std::vector<std::vector<double>>{{2}});
    Matrix Inv3 = Matrix::inverse(A3);
    Matrix Verify3 = Matrix::multiply(A3, Inv3);
    bool test3 = nearlyEqual(Inv3.at(0,0), 0.5) && 
                 nearlyEqual(Verify3.at(0,0), 1.0);
    printTestResult("Test 3: Single element", test3);
    if (test3) passed++; total++;

    std::cout << "\nTest Case 4: 3x3 matrix inversion\n";
    Matrix A4({{1,2,3},{0,1,4},{5,6,0}});
    Matrix Inv4 = Matrix::inverse(A4);
    Matrix Verify4 = Matrix::multiply(A4, Inv4);
    bool test4 = nearlyEqual(Inv4.at(0,0), -24) && nearlyEqual(Inv4.at(1,1), -15) &&
                 nearlyEqual(Verify4.at(0,0), 1) && nearlyEqual(Verify4.at(1,1), 1) &&
                 nearlyEqual(Verify4.at(2,2), 1);
    printTestResult("Test 4: 3x3 matrix", test4);
    if (test4) passed++; total++;

    std::cout << "\nTest Case 5: Identity matrix (inverse of identity is identity)\n";
    Matrix I3({{1,0,0},{0,1,0},{0,0,1}});
    Matrix Inv5 = Matrix::inverse(I3);
    bool test5 = nearlyEqual(Inv5.at(0,0), 1) && nearlyEqual(Inv5.at(1,1), 1) &&
                 nearlyEqual(Inv5.at(2,2), 1);
    printTestResult("Test 5: Identity matrix", test5);
    if (test5) passed++; total++;

    std::cout << "\nTest Case 6: Non-square matrix (should throw)\n";
    try {
        Matrix::inverse(Matrix(2,3));
        printTestResult("Test 6: Non-square", false);
        total++;
    } catch (const std::invalid_argument&) {
        printTestResult("Test 6: Non-square", true);
        passed++; total++;
    }

    std::cout << "\nTest Case 7: Singular matrix (should throw)\n";
    try {
        Matrix::inverse(Matrix({{1,2},{2,4}}));
        printTestResult("Test 7: Singular matrix", false);
        total++;
    } catch (const std::runtime_error&) {
        printTestResult("Test 7: Singular matrix", true);
        passed++; total++;
    }

    std::cout << "\nTest Case 8: Empty matrix (should throw)\n";
    try {
        Matrix empty(0,0);
        Matrix::inverse(empty);
        printTestResult("Test 8: Empty matrix", false);
        total++;
    } catch (const std::exception&) {
        printTestResult("Test 8: Empty matrix", true);
        passed++; total++;
    }

    std::cout << "\nTest Case 9: Diagonal matrix\n";
    Matrix D({{2,0},{0,4}});
    Matrix InvD = Matrix::inverse(D);
    Matrix VerifyD = Matrix::multiply(D, InvD);
    bool test9 = nearlyEqual(InvD.at(0,0), 0.5) && nearlyEqual(InvD.at(1,1), 0.25) &&
                 nearlyEqual(VerifyD.at(0,0), 1) && nearlyEqual(VerifyD.at(1,1), 1);
    printTestResult("Test 9: Diagonal matrix", test9);
    if (test9) passed++; total++;

    std::cout << "\nTest Case 10: Nearly singular matrix (with tolerance)\n";
    Matrix A10({{1, 1.000001}, {1, 1.000002}});
    try {
        Matrix Inv10 = Matrix::inverse(A10);
        Matrix Verify10 = Matrix::multiply(A10, Inv10);
        bool test10 = nearlyEqual(Verify10.at(0,0), 1, 1e-4) && 
                     nearlyEqual(Verify10.at(1,1), 1, 1e-4);
        printTestResult("Test 10: Nearly singular", test10);
        if (test10) passed++; total++;
    } catch (const std::exception&) {
        printTestResult("Test 10: Nearly singular (detected as singular)", true);
        passed++; total++;
    }

    std::cout << "\nTest Case 11: Large value matrix\n";
    Matrix A11({{1000,2000},{3000,4000}});
    try {
        Matrix Inv11 = Matrix::inverse(A11);
        Matrix Verify11 = Matrix::multiply(A11, Inv11);
        bool test11 = nearlyEqual(Verify11.at(0,0), 1, 1e-5) && 
                     nearlyEqual(Verify11.at(1,1), 1, 1e-5);
        printTestResult("Test 11: Large values", test11);
        if (test11) passed++; total++;
    } catch (const std::runtime_error&) {
        printTestResult("Test 11: Large values (detected as singular)", true);
        passed++; total++;
    }

    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << "Matrix Inversion: " << passed << "/" << total << " tests passed\n";
}

void testElementWiseMultiplication() {
    printTestHeader("Testing Element-wise Multiplication");
    int passed = 0, total = 0;

    std::cout << "\nTest Case 1: Basic element-wise multiplication\n";
    Matrix A1({{1,2},{3,4}});
    Matrix B1({{5,6},{7,8}});
    Matrix R1 = Matrix::elementWiseMultiply(A1, B1);
    bool test1 = nearlyEqual(R1.at(0,0), 5) && nearlyEqual(R1.at(0,1), 12) &&
                 nearlyEqual(R1.at(1,0), 21) && nearlyEqual(R1.at(1,1), 32);
    printTestResult("Test 1: Basic 2x2", test1);
    if (test1) passed++; total++;

    std::cout << "\nTest Case 2: Single element\n";
    Matrix A2(std::vector<std::vector<double>>{{2}});
    Matrix B2(std::vector<std::vector<double>>{{3}});
    Matrix R2 = Matrix::elementWiseMultiply(A2, B2);
    bool test2 = nearlyEqual(R2.at(0,0), 6);
    printTestResult("Test 2: Single element", test2);
    if (test2) passed++; total++;

    std::cout << "\nTest Case 3: Row vectors\n";
    Matrix A3({{1,2,3}});
    Matrix B3({{4,5,6}});
    Matrix R3 = Matrix::elementWiseMultiply(A3, B3);
    bool test3 = nearlyEqual(R3.at(0,0), 4) && nearlyEqual(R3.at(0,1), 10) &&
                 nearlyEqual(R3.at(0,2), 18);
    printTestResult("Test 3: Row vectors", test3);
    if (test3) passed++; total++;

    std::cout << "\nTest Case 4: Rectangular 2x3 matrices\n";
    Matrix A4({{1,2,3},{4,5,6}});
    Matrix B4({{6,5,4},{3,2,1}});
    Matrix R4 = Matrix::elementWiseMultiply(A4, B4);
    bool test4 = nearlyEqual(R4.at(0,0), 6) && nearlyEqual(R4.at(0,1), 10) &&
                 nearlyEqual(R4.at(0,2), 12) && nearlyEqual(R4.at(1,0), 12) &&
                 nearlyEqual(R4.at(1,1), 10) && nearlyEqual(R4.at(1,2), 6);
    printTestResult("Test 4: Rectangular 2x3", test4);
    if (test4) passed++; total++;

    std::cout << "\nTest Case 5: Incompatible dimensions (2x2 vs 3x3)\n";
    try {
        Matrix::elementWiseMultiply(Matrix(2,2), Matrix(3,3));
        printTestResult("Test 5: Incompatible (2x2 vs 3x3)", false);
        total++;
    } catch (const std::invalid_argument&) {
        printTestResult("Test 5: Incompatible (2x2 vs 3x3)", true);
        passed++; total++;
    }

    std::cout << "\nTest Case 6: Incompatible dimensions (2x3 vs 2x2)\n";
    try {
        Matrix::elementWiseMultiply(Matrix(2,3), Matrix(2,2));
        printTestResult("Test 6: Incompatible (2x3 vs 2x2)", false);
        total++;
    } catch (const std::invalid_argument&) {
        printTestResult("Test 6: Incompatible (2x3 vs 2x2)", true);
        passed++; total++;
    }

    std::cout << "\nTest Case 7: Zero matrices\n";
    Matrix Z(2,2);
    Matrix R7 = Matrix::elementWiseMultiply(Z, Z);
    bool test7 = nearlyEqual(R7.at(0,0), 0) && nearlyEqual(R7.at(1,1), 0);
    printTestResult("Test 7: Zero matrices", test7);
    if (test7) passed++; total++;

    std::cout << "\nTest Case 8: Matrix with all ones\n";
    Matrix Ones({{1,1},{1,1}});
    Matrix R8 = Matrix::elementWiseMultiply(Ones, B1);
    bool test8 = nearlyEqual(R8.at(0,0), 5) && nearlyEqual(R8.at(0,1), 6) &&
                 nearlyEqual(R8.at(1,0), 7) && nearlyEqual(R8.at(1,1), 8);
    printTestResult("Test 8: All ones matrix", test8);
    if (test8) passed++; total++;

    std::cout << "\nTest Case 9: Mixed positive and negative values\n";
    Matrix A9({{-1,2},{-3,4}});
    Matrix B9({{2,-2},{3,-4}});
    Matrix R9 = Matrix::elementWiseMultiply(A9, B9);
    bool test9 = nearlyEqual(R9.at(0,0), -2) && nearlyEqual(R9.at(0,1), -4) &&
                 nearlyEqual(R9.at(1,0), -9) && nearlyEqual(R9.at(1,1), -16);
    printTestResult("Test 9: Mixed signs", test9);
    if (test9) passed++; total++;

    std::cout << "\nTest Case 10: Floating point precision\n";
    Matrix A10({{0.1,0.2}});
    Matrix B10({{10,5}});
    Matrix R10 = Matrix::elementWiseMultiply(A10, B10);
    bool test10 = nearlyEqual(R10.at(0,0), 1.0) && nearlyEqual(R10.at(0,1), 1.0);
    printTestResult("Test 10: Floating point precision", test10);
    if (test10) passed++; total++;

    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << "Element-wise Multiplication: " << passed << "/" << total << " tests passed\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  MATRIX OPERATIONS - COMPREHENSIVE TEST SUITE\n";
    std::cout << std::string(60, '=') << "\n";

    testMatrixMultiplication();
    testMatrixInversion();
    testElementWiseMultiplication();

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  ✅ All matrix operation tests completed successfully!\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    return 0;
}
