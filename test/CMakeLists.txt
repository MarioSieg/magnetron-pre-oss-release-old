# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(googletest)

file(GLOB_RECURSE TEST_SOURCES *.cpp)
add_executable(wavelet_test ${TEST_SOURCES} ../wavelet/wavelet.hpp)
target_link_libraries(wavelet_test wavelet)
target_include_directories(wavelet_test PRIVATE ../wavelet)
target_link_libraries(
    wavelet_test
    GTest::gtest_main
)

if (${WAVELET_ENABLE_CUDA})
    target_link_libraries(wavelet_test wavelet_cuda)
    target_compile_definitions(wavelet_test PRIVATE WL_ENABLE_CUDA)
endif()
