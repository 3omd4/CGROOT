/*Purpose: Unit test math kernels. (Use GoogleTest).

To-Do:

TEST(TensorMath, Add): Check [1, 2] + [3, 4] == [4, 6].

TEST(TensorMath, Matmul): Check [2x2] @ [2x2] against a hand-calculated result.*/

/** 

#include <iostream>
#include "../src/core/tensor.h"


int main() {
    std::cout << "Tensor Tests" << std::endl;
    std::cout << "Testing tensor functionality..." << std::endl;
    
    try {
        // TODO: Add actual tensor tests here
        // This is a placeholder until the tensor class is implemented
        std::cout << "Tensor tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

*/