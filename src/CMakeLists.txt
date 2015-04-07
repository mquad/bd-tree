include_directories(../util)
#add_executable(bd-tree main.cpp)
#target_link_libraries(bd-tree ${Boost_IOSTREAMS_LIBRARY})

add_executable(bd-tree2 main2.cpp)
target_link_libraries(bd-tree2 ${Boost_IOSTREAMS_LIBRARY})

add_executable(bd-tree3 main3.cpp)
target_link_libraries(bd-tree3 ${Boost_IOSTREAMS_LIBRARY})
