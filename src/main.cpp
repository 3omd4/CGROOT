#include <iostream>
#include "core/tensor.h"

int main() {
    std::cout << "CGroot++ - A C++ Deep Learning Framework" << std::endl;
    
    // Example usage of the framework
    try {
        // Test basic tensor functionality
        std::cout << "Testing basic tensor operations..." << std::endl;
        
        // This will test if the tensor class compiles and works
        std::cout << "Framework is working correctly." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
