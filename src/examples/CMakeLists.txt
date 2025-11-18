# Example executables (only build if files exist)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/simple_test.cpp)
    add_executable(simple_test simple_test.cpp)
    target_link_libraries(simple_test cgroot_lib)
endif()

# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/xor_solver.cpp)
#     add_executable(xor_solver xor_solver.cpp)
#     target_link_libraries(xor_solver cgroot_lib)
# endif()

