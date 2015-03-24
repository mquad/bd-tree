project(bootstrap-trees)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(./src SRC_LIST)

#configure the compiler
set (CMAKE_CXX_COMPILER "g++")
set (CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Werror -pedantic")
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -pg") # enable gprof during compilation
set (CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -march=native -DNDEBUG")
set (CMAKE_EXE_LINKER_FLAGS_DEBUG "-pg") # enable gprof during linking

#add subdirectories
add_subdirectory(src)       #directory for application sources
