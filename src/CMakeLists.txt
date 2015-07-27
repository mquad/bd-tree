include_directories(../util)
add_executable(bdtree_err main_err.cpp)
target_link_libraries(bdtree_err ${Boost_IOSTREAMS_LIBRARY})

add_executable(bdtree_rank main_rank.cpp)
target_link_libraries(bdtree_rank ${Boost_IOSTREAMS_LIBRARY})
