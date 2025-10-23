cmake_minimum_required(VERSION 3.10)
project(CGroot++ VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include directories
include_directories(src)

# Find required packages
find_package(GTest QUIET)

# Core library sources
set(CORE_SOURCES
    src/core/tensor.cpp
    src/core/tensor.h
    src/core/parameter.h
    src/core/shape.h
)

set(AUTOGRAD_SOURCES
    src/autograd/graph.cpp
    src/autograd/graph.h
    src/autograd/op_nodes.cpp
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

# Data I/O sources (to be created)
set(DATA_SOURCES
    src/data/mnist_loader.cpp
    src/data/mnist_loader.h
    src/data/data_utils.cpp
    src/data/data_utils.h
)

# Combine all library sources
set(LIBRARY_SOURCES
    ${CORE_SOURCES}
    ${AUTOGRAD_SOURCES}
    ${MATH_SOURCES}
    ${NN_SOURCES}
    ${OPTIM_SOURCES}
    ${DATA_SOURCES}
)

# Create the main library
add_library(cgroot_lib STATIC ${LIBRARY_SOURCES})

# Set library properties
set_target_properties(cgroot_lib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Create the main executable
add_executable(cgrunner src/main.cpp)
target_link_libraries(cgrunner cgroot_lib)

# Create example executables
add_executable(simple_test examples/simple_test.cpp)
target_link_libraries(simple_test cgroot_lib)

add_executable(xor_solver examples/xor_solver.cpp)
target_link_libraries(xor_solver cgroot_lib)

# Create test executables
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor cgroot_lib)

add_executable(test_autograd tests/test_autograd.cpp)
target_link_libraries(test_autograd cgroot_lib)

# If GoogleTest is available, use it for testing
if(GTest_FOUND)
    target_link_libraries(test_tensor GTest::GTest GTest::Main)
    target_link_libraries(test_autograd GTest::GTest GTest::Main)
    
    # Enable testing
    enable_testing()
    add_test(NAME TensorTests COMMAND test_tensor)
    add_test(NAME AutogradTests COMMAND test_autograd)
endif()

# Set output directories
set_target_properties(cgrunner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(simple_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(xor_solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(test_tensor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(test_autograd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installation
install(TARGETS cgroot_lib cgrunner simple_test xor_solver
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY src/ DESTINATION include/cgroot
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration summary
message(STATUS "CGroot++ Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  GoogleTest found: ${GTest_FOUND}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")