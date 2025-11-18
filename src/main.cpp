#include <iostream>
#include "math/matrix_ops.h"

int main() {
    std::cout << "CGroot++ - A C++ Deep Learning Framework" << std::endl;
    
    // Example usage of the framework
    try {
        // Test basic matrix operations
        std::cout << "Testing basic matrix operations..." << std::endl;
        
        using namespace cgroot::math;
        
        // Create a simple 2x2 matrix
        Matrix A(2, 2);
        A(0, 0) = 1.0; A(0, 1) = 2.0;
        A(1, 0) = 3.0; A(1, 1) = 4.0;
        
        std::cout << "Matrix A:" << std::endl;
        A.print();
        
        std::cout << "Framework is working correctly." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
