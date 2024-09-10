cmake_minimum_required(VERSION 3.10)

project(msml LANGUAGES C CXX)

message(WARNING "The CMake project is not used for unti testing of MSML itself. Use the single header file instead.")

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)

enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE TEST_SOURCES test/*.hpp test/*.cpp)
add_executable(pluto_test ${TEST_SOURCES})

target_link_libraries(
  pluto_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(pluto_test)
