cmake_minimum_required(VERSION 3.10)
project(CGroot++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(src)

# Collect all source files
set(CORE_SOURCES
    src/core/tensor.cpp
    src/core/parameter.h
    src/core/shape.h
)

set(AUTOGRAD_SOURCES
    src/autograd/graph.cpp
    src/autograd/op_nodes.cpp
    src/autograd/graph.h
    src/autograd/op_nodes.h
)

set(MATH_SOURCES
    src/math/cpu_kernels.cpp
    src/math/cpu_kernels.h
)

set(NN_SOURCES
    src/nn/conv2d.h
    src/nn/cross_entropy_loss.h
    src/nn/linear.h
    src/nn/module.h
    src/nn/mse_loss.h
    src/nn/relu.h
    src/nn/sequential.h
    src/nn/sigmoid.h
)

set(OPTIM_SOURCES
    src/optim/adam.h
    src/optim/optimizer.h
    src/optim/sgd.h
)

# Combine all sources
set(ALL_SOURCES
    ${CORE_SOURCES}
    ${AUTOGRAD_SOURCES}
    ${MATH_SOURCES}
    ${NN_SOURCES}
    ${OPTIM_SOURCES}
)

# Create the main executable
add_executable(cgrunner src/main.cpp ${ALL_SOURCES})

# Create example executable
add_executable(simple_test examples/simple_test.cpp ${ALL_SOURCES})

# Set output directory
set_target_properties(cgrunner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(simple_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
