cmake_minimum_required(VERSION 3.10)
project(CGroot++)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable and specify its source files
add_executable(cgrunner src/cpp/main.cpp)
