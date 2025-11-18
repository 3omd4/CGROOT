# Core library sources
set(MATH_SOURCES
    math/cpu_kernels.cpp
    math/cpu_kernels.h
    math/matrix_ops.cpp
    math/matrix_ops.h
)

# Core layers
set(CORE_LAYERS_SOURCES
    core/layers/layer.cpp
    core/layers/layers.h
    core/layers/dense.cpp
    core/layers/dense.h
    core/layers/dropout.h
    core/layers/conv2d.cpp
    core/layers/pooling.cpp
    core/layers/ActivationLayer.cpp
    core/layers/inputLayer.cpp
    core/layers/outputLayer.cpp
)

# Core activations
set(CORE_ACTIVATIONS_SOURCES
    core/activations/relu.h
    core/activations/sigmoid.h
    core/activations/tanh.h
    core/activations/softmax.h
)

# Core losses
set(CORE_LOSSES_SOURCES
    core/losses/mse.h
    core/losses/binary_crossentropy.h
    core/losses/categorical_crossentropy.h
)

# Core optimizers
set(CORE_OPTIMIZERS_SOURCES
    core/optimizers/sgd.h
    core/optimizers/momentum.h
    core/optimizers/adam.h
)

# Core utils
set(CORE_UTILS_SOURCES
    core/utils/mnist_loader.cpp
    core/utils/mnist_loader.h
    core/utils/data_utils.cpp
    core/utils/data_utils.h
    core/utils/weight_init.h
    core/utils/metrics.h
)

# Core model
set(CORE_MODEL_SOURCES
    core/model.cpp
    core/model.h
)

# Combine all core library sources
set(LIBRARY_SOURCES
    ${MATH_SOURCES}
    ${CORE_LAYERS_SOURCES}
    ${CORE_ACTIVATIONS_SOURCES}
    ${CORE_LOSSES_SOURCES}
    ${CORE_OPTIMIZERS_SOURCES}
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

# Create the main executable (only if main.cpp exists)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
    add_executable(cgrunner main.cpp)
    target_link_libraries(cgrunner cgroot_lib)
endif()

# Add subdirectories
add_subdirectory(examples)
if(Qt6_FOUND AND Qt6Core_FOUND AND Qt6Widgets_FOUND AND Qt6Charts_FOUND)
    add_subdirectory(gui)
endif()

