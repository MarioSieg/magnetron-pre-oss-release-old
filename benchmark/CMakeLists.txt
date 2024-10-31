# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE BENCHMARK_SOURCES *.cpp)
add_executable(wavelet_benchmark ${BENCHMARK_SOURCES})
target_link_libraries(wavelet_benchmark wavelet)
target_include_directories(wavelet_benchmark PRIVATE ../wavelet)
target_include_directories(wavelet_benchmark PRIVATE nanobench)
