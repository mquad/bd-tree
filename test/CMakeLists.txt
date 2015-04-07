#Vector test
include_directories(../src)
add_executable(bd_tree_test bd_tree_test.cpp)
target_link_libraries(bd_tree_test gtest gtest_main)
add_executable(metrics_test metrics_test.cpp)
target_link_libraries(metrics_test gtest gtest_main)
