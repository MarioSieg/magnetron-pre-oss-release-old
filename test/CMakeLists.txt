# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(googletest)

file(GLOB_RECURSE TEST_SOURCES *.cpp)
add_executable(msml_test ${TEST_SOURCES} ../msml/msml.hpp)
target_link_libraries(msml_test msml)
target_include_directories(msml_test PRIVATE ../msml)
target_link_libraries(
    msml_test
    GTest::gtest_main
)
