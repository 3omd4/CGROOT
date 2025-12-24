# Core library sources


# Core layers
set(CORE_LAYERS_SOURCES
    core/layers/layer.cpp
    core/layers/layers.h
    core/layers/convLayer.cpp
    core/layers/poolingLayer.cpp
    core/layers/flatten.cpp
    core/layers/fullyConnected.cpp
    core/layers/inputLayer.cpp
    core/layers/outputLayer.cpp
    # core/layers/dense.h
    # core/layers/dropout.h
)

# Core activations
set(CORE_ACTIVATIONS_SOURCES
    core/activation/activation.cpp
    core/activation/activation.h
)

# Core losses
set(CORE_LOSSES_SOURCES
    core/losses/mse.h
    core/losses/binary_crossentropy.h
    core/losses/categorical_crossentropy.h
)

# Core optimizers
set(CORE_OPTIMIZERS_SOURCES
    core/optimizers/optimizer.cpp
    core/optimizers/optimizer.h
)

# Core initializations
set(CORE_INITIALIZATION_SOURCES
    core/Initialization/initialization.cpp
    core/Initialization/initialization.h
)

# Core utils
set(CORE_UTILS_SOURCES
    core/utils/mnist_loader.cpp
    core/utils/mnist_loader.h
    core/utils/data_utils.cpp
    core/utils/data_utils.h
)

# Core model
set(CORE_MODEL_SOURCES
    core/model.cpp
    core/model.h
)

# Combine all core library sources
set(LIBRARY_SOURCES
    ${CORE_LAYERS_SOURCES}
    ${CORE_ACTIVATIONS_SOURCES}
    ${CORE_OPTIMIZERS_SOURCES}
    ${CORE_INITIALIZATION_SOURCES}
    ${CORE_UTILS_SOURCES}
    ${CORE_MODEL_SOURCES}
)

# Filter out non-existent .cpp files (keep all headers)
set(FILTERED_SOURCES "")
foreach(SOURCE_FILE ${LIBRARY_SOURCES})
    set(FULL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE})
    if(${SOURCE_FILE} MATCHES "\\.h$")
        list(APPEND FILTERED_SOURCES ${FULL_PATH})
    elseif(EXISTS ${FULL_PATH})
        list(APPEND FILTERED_SOURCES ${FULL_PATH})
    endif()
endforeach()

# Create the main library
add_library(cgroot_lib STATIC ${FILTERED_SOURCES})

# Set library properties
set_target_properties(cgroot_lib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Enable OpenMP
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: enabling parallelization")

    if (MSVC)
        target_compile_options(cgroot_lib PRIVATE /openmp)
    else()
        target_compile_options(cgroot_lib PRIVATE ${OpenMP_CXX_FLAGS})
    endif()

    target_link_libraries(cgroot_lib PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP NOT found — parallelization disabled")
endif()

# Link filesystem library (required for C++17 <filesystem> on some compilers)
# On MSVC, this is typically not needed, but on older GCC/Clang it might be stdc++fs
if(NOT MSVC)
    target_link_libraries(cgroot_lib PRIVATE stdc++fs)
endif()


# Create the main executable (only if main.cpp exists)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
    add_executable(cgrunner main.cpp)
    target_link_libraries(cgrunner cgroot_lib)
endif()

# Add subdirectories
add_subdirectory(examples)
# if(Qt6_FOUND AND Qt6Core_FOUND AND Qt6Widgets_FOUND AND Qt6Charts_FOUND)
#     add_subdirectory(gui)
# endif()

