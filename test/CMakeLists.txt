#Vector test
include_directories(../src)
add_executable(bd_tree_test bd_tree_test.cpp)
target_link_libraries(bd_tree_test gtest gtest_main)
