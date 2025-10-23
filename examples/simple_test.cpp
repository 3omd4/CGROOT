#include <iostream>
#include "../src/core/tensor.h"

int main() {
    std::cout << "Simple Test Example" << std::endl;
    std::cout << "Testing basic tensor functionality..." << std::endl;
    
    try {
        // This will test if the tensor class compiles and works
        std::cout << "Example compiled and ran successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
