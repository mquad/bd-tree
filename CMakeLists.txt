project(bd-tree)
cmake_minimum_required(VERSION 2.8)

# Import and build Google Test
include(ExternalProject)
set_directory_properties(properties EP_PREFIX "${CMAKE_BINARY_DIR}/third_party")
ExternalProject_Add(googletest
    URL "https://googletest.googlecode.com/files/gtest-1.7.0.zip"
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third_party/gtest"
    INSTALL_COMMAND "")
ExternalProject_Get_Property(googletest source_dir)
include_directories(${source_dir}/include)
ExternalProject_Get_Property(googletest binary_dir)
link_directories(${binary_dir})

find_package(Boost COMPONENTS iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIR})


#configure the compiler
if(APPLE)
    set (CMAKE_CXX_COMPILER "g++-5")
else()
    set (CMAKE_CXX_COMPILER "g++")
endif()

set (CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -pedantic -fopenmp")
set (CMAKE_CXX_FLAGS_DEBUG "-Ofast -g")
set (CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")

#add subdirectories
add_subdirectory(src)       #application sources
#add_subdirectory(test)      #tests

#add test target (make test)
#enable_testing()
#add_test(NAME BDTreeTest COMMAND bd_tree_test)
#add_test(NAME MetricsTest COMMAND metrics_test)
