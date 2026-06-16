# tests/CMakeLists.txt

add_executable(test_status test_status.c)
target_link_libraries(test_status PRIVATE mlxc)
add_test(NAME test_status COMMAND test_status)

add_executable(test_context test_context.c)
target_link_libraries(test_context PRIVATE mlxc)
add_test(NAME test_context COMMAND test_context)

add_executable(test_array test_array.c)
target_link_libraries(test_array PRIVATE mlxc)
add_test(NAME test_array COMMAND test_array)

add_executable(test_pure_c test_pure_c.c)
target_link_libraries(test_pure_c PRIVATE mlxc)
add_test(NAME test_pure_c COMMAND test_pure_c)

add_executable(test_ops test_ops.c)
target_link_libraries(test_ops PRIVATE mlxc)
add_test(NAME test_ops COMMAND test_ops)
