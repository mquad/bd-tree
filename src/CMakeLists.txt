include_directories(../util)
add_executable(bd-tree main.cpp)
target_link_libraries(bd-tree ${Boost_IOSTREAMS_LIBRARY})
