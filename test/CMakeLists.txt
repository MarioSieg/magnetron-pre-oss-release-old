# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE TEST_SOURCES *.cpp)
add_executable(msml_test ${TEST_SOURCES})
target_link_libraries(msml_test msml)
target_include_directories(msml_test PRIVATE ../msml)
target_link_libraries(
    msml_test
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(msml_test)
