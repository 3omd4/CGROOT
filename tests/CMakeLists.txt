# Test executables (only build if files exist)

# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor.cpp)
#     add_executable(test_tensor test_tensor.cpp)
#     target_link_libraries(test_tensor cgroot_lib)
# endif()

# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_autograd.cpp)
#     add_executable(test_autograd test_autograd.cpp)
#     target_link_libraries(test_autograd cgroot_lib)
# endif()

# If GoogleTest is available, use it for testing
add_executable(test_diagnostics test_diagnostics.cpp)
target_link_libraries(test_diagnostics cgroot_lib)

if(GTest_FOUND)
    if(TARGET test_tensor)
        target_link_libraries(test_tensor GTest::GTest GTest::Main)
    endif()
    if(TARGET test_autograd)
        target_link_libraries(test_autograd GTest::GTest GTest::Main)
    endif()
    
    # Enable testing
    enable_testing()
    if(TARGET test_tensor)
        add_test(NAME TensorTests COMMAND test_tensor)
    endif()
    if(TARGET test_autograd)
        add_test(NAME AutogradTests COMMAND test_autograd)
    endif()
endif()

