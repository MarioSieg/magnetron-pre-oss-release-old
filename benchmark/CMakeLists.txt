# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE BENCHMARK_SOURCES *.cpp)
add_executable(msml_benchmark ${BENCHMARK_SOURCES})
target_link_libraries(msml_benchmark msml)
target_include_directories(msml_benchmark PRIVATE ../msml)
target_include_directories(msml_benchmark PRIVATE nanobench)
