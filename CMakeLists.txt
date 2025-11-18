cmake_minimum_required(VERSION 3.10)
project(CGroot++ VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set output directories to keep build artifacts organized
# All executables go to build/bin, libraries to build/lib
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Warn if building in-source (not recommended)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(WARNING "Building in-source is not recommended!")
    message(WARNING "Please use an out-of-source build:")
    message(WARNING "  mkdir build && cd build && cmake ..")
    message(WARNING "  OR: cmake -B build -S .")
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

# Find Qt6 (optional - only needed for GUI)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Try to find Qt6 - allow user to set Qt6_DIR or CMAKE_PREFIX_PATH
find_package(Qt6 QUIET COMPONENTS Core Widgets Charts)

# If Qt6 not found and on Windows, try common installation path
if(NOT Qt6_FOUND AND WIN32)
    if("${CMAKE_PREFIX_PATH}" STREQUAL "")
        set(CMAKE_PREFIX_PATH "C:/Qt/6.10.0/msvc2022_64")
        find_package(Qt6 QUIET COMPONENTS Core Widgets Charts)
        # Try common Qt6 installation paths
        # set(QT_PATHS
        #     "C:/Qt/6.10.0/msvc2022_64"
        #     "C:/Qt/6.10.0/msvc2019_64"
        #     "C:/Qt/6.9.0/msvc2022_64"
        #     "C:/Qt/6.9.0/msvc2019_64"
        # )
        # foreach(QT_PATH ${QT_PATHS})
        #     if(EXISTS ${QT_PATH})
        #         set(CMAKE_PREFIX_PATH ${QT_PATH})
        #         find_package(Qt6 QUIET COMPONENTS Core Widgets Charts)
        #         if(Qt6_FOUND)
        #             break()
        #         endif()
        #     endif()
        # endforeach()
    endif()
endif()

# Provide helpful message if Qt6 not found
if(NOT Qt6_FOUND)
    message(STATUS "Qt6 not found - GUI will not be built")
    message(STATUS "To enable GUI, set CMAKE_PREFIX_PATH to your Qt6 installation:")
    message(STATUS "  cmake .. -DCMAKE_PREFIX_PATH=C:/Qt/6.x.x/msvc2022_64")
else()
    message(STATUS "Qt6 found: ${Qt6_VERSION}")
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Installation
install(TARGETS cgroot_lib
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
message(STATUS "  Qt6 found: ${Qt6_FOUND}")
if(Qt6_FOUND)
    message(STATUS "  Qt6 version: ${Qt6_VERSION}")
    message(STATUS "  Qt6 Widgets: ${Qt6Widgets_FOUND}")
    message(STATUS "  Qt6 Charts: ${Qt6Charts_FOUND}")
endif()
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
