cmake_minimum_required(VERSION 3.22)
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
    # For full logs
    # add_compile_options(/W4 /permissive-)

    # For no warnings
    add_compile_options(/w)

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /Oi /Ot /GL /Qpar-report:2 /fp:fast /DNDEBUG /openmp /arch:AVX2)
        add_link_options(/LTCG)
    endif()
else()
    # For full logs
    # add_compile_options(-Wall -Wextra -Wpedantic)

    # For no warnings
    add_compile_options(-w)

    # Release optimizations for GCC/Clang
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -ffast-math -fopenmp -finline-functions -DNDEBUG)
    endif()
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


# Forcing PDB processing on MSVC
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FS")
endif()


# If Qt6 not found and on Windows, try common installation path
if(NOT Qt6_FOUND AND WIN32)
    if("${CMAKE_PREFIX_PATH}" STREQUAL "")
        set(CMAKE_PREFIX_PATH "C:/Qt/6.10.0/msvc2022_64") # Default fallback
        find_package(Qt6 QUIET COMPONENTS Core Widgets Charts)
        
        # Dynamic Discovery of Qt paths
        set(QT_SEARCH_ROOTS "C:/Qt" "$ENV{USERPROFILE}/Qt")
        set(QT_CANDIDATES "")

        foreach(ROOT ${QT_SEARCH_ROOTS})
            if(EXISTS ${ROOT})
                # Find all version directories (e.g. 6.x.x)
                file(GLOB VERSION_DIRS "${ROOT}/6.*.*")
                # Sort descending to try latest first
                list(SORT VERSION_DIRS COMPARE STRING ORDER DESCENDING)
                
                foreach(VER_DIR ${VERSION_DIRS})
                    if(IS_DIRECTORY ${VER_DIR})
                        # Find compiler subdirectories (msvc*, mingw*)
                        file(GLOB COMPILER_DIRS "${VER_DIR}/*_64")
                        
                        # Prioritize MSVC if on Windows and building with MSVC
                        if(MSVC)
                             list(FILTER COMPILER_DIRS INCLUDE REGEX "msvc")
                        endif()

                        foreach(COMP_DIR ${COMPILER_DIRS})
                            if(IS_DIRECTORY ${COMP_DIR})
                                list(APPEND QT_CANDIDATES ${COMP_DIR})
                            endif()
                        endforeach()
                    endif()
                endforeach()
            endif()
        endforeach()

        # Iterate through dynamically found paths
        foreach(QT_PATH ${QT_CANDIDATES})
            message(STATUS "Checking Qt path: ${QT_PATH}")
            if(EXISTS ${QT_PATH})
                set(CMAKE_PREFIX_PATH ${QT_PATH})
                find_package(Qt6 QUIET COMPONENTS Core Widgets Charts)
                if(Qt6_FOUND)
                    message(STATUS "Qt6 found at: ${QT_PATH}")
                    break()
                endif()
            endif()
        endforeach()
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

# ---- Speed up build dramatically ----
set(CMAKE_UNITY_BUILD ON)                # Combine source files into mega translation units
set(CMAKE_UNITY_BUILD_BATCH_SIZE 16)

if(MSVC)
    if(NOT CMAKE_GENERATOR MATCHES "Ninja")
        add_compile_options(/MP)  # Multi-core compilation (let Ninja handle parallel jobs if using Ninja)
    endif()
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Using ccache for fast rebuilds")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# Reduce reconfigure time of FetchContent dependencies
set(FETCHCONTENT_QUIET ON)
set(FETCHCONTENT_FULLY_DISCONNECTED OFF)  # Set ON after first successful fetch

# Avoid building pybind11 tests/docs/examples
set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
set(PYBIND11_INSTALL OFF CACHE BOOL "" FORCE)
set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)

# Speed up Qt discovery
set(QT_NO_CREATE_VERSIONLESS_FUNCTIONS ON)
set(QT_NO_CREATE_TARGETS ON)
set(QT_DEFAULT_MAJOR_VERSION 6)


# Prevent CMake scanning .git folders (huge speed win)
set(CMAKE_SUPPRESS_REGENERATION ON)
file(GLOB_RECURSE GIT_DIRS "${PROJECT_SOURCE_DIR}/**/.git")
set_property(SOURCE ${GIT_DIRS} PROPERTY HEADER_FILE_ONLY TRUE)


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

# ==========================================
# PyBind11 Bindings
# ==========================================
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(cgroot_core src/bindings/bindings.cpp)
target_link_libraries(cgroot_core PRIVATE cgroot_lib)
target_include_directories(cgroot_core PRIVATE src)

# Ensure the library is built as an extension
set_target_properties(cgroot_core PROPERTIES 
    CXX_VISIBILITY_PRESET "default"
    VISIBILITY_INLINES_HIDDEN OFF
)

# Copy the extension to the bin directory (so Python can find it easily)
add_custom_command(TARGET cgroot_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:cgroot_core>
    ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/$<TARGET_FILE_NAME:cgroot_core>
)
