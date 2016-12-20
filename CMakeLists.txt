cmake_minimum_required(VERSION 3.7.1)

enable_language(CXX)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-g -Wall")
set (CMAKE_MACOSX_RPATH 1)
set (SOURCE src/wrapper.cpp)

find_package(fann)

project (FANN_WRAPPER)
add_library (FANN_WRAPPER ${SOURCE})
target_include_directories (FANN_WRAPPER PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} include)
