Contains the raw, low-level CPU math functions. These functions operate on C-style pointers and are totally unaware of Tensor objects or gradients.

cpu_kernels.h: Header file declaring all the math kernels (e.g., cpu_add, cpu_matmul, cpu_im2col).

cpu_kernels.cpp: The implementation of your kernels. This is where you'll write your nested loops for matmul, your im2col logic, and all element-wise operations. This is also where all performance optimizations (tiling, SIMD) will live.