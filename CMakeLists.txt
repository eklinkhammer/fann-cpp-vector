cmake_minimum_required(VERSION 3.7.1)

enable_language(CXX)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-g -Wall")
set (CMAKE_MACOSX_RPATH 1)
set (SOURCE src/wrapper.cpp)
set (TEST_SOURCE test/wrapper_test.cpp)
set (TEST_EXEC runUnitTests)

find_package(fann)

project (FANN_WRAPPER)
add_library (FANN_WRAPPER ${SOURCE})
target_include_directories (FANN_WRAPPER PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} include)

configure_file(CMakeLists.txt.in
               googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
 
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build)
enable_testing()
add_executable(${TEST_EXEC} ${TEST_SOURCE})
target_link_libraries(${TEST_EXEC} gtest gtest_main)
target_link_libraries(${TEST_EXEC} fann)
target_link_libraries(${TEST_EXEC} FANN_WRAPPER)
add_test(NAME gtest-models COMMAND ${TEST_EXEC})
