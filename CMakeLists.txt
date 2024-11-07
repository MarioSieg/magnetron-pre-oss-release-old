# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

cmake_minimum_required(VERSION 3.10)

project(wavelet LANGUAGES C)

set(CMAKE_C_STANDARD 99)
option(WAVELET_BUILD_TESTS "Build tests" ON)
option(WAVELET_BUILD_BENCHMARKS "Build benchmarks" ON)
option(WAVELET_BUILD_FUZZERS "Build fuzzers" OFF)
option(WAVELET_ENABLE_IMAGE_SUPPORT "Enable creating tensor from images support" ON)
if (APPLE)
    option(WAVELET_USE_ACCELERATE "Use Apple's Accelerate framework" ON)
endif()

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

if (${WAVELET_BUILD_TESTS})
    enable_testing()
    add_subdirectory(test)
endif()

if (${WAVELET_BUILD_FUZZERS})
    add_subdirectory(fuzzer)
endif()

if (${WAVELET_BUILD_BENCHMARKS})
    add_subdirectory(benchmark)
endif()

add_library(wavelet SHARED wavelet/wavelet.h wavelet/wavelet.c)
target_include_directories(wavelet PRIVATE wavelet/optional)
if (WIN32)
    target_compile_options(wavelet PRIVATE /W3 /Oi)
    # Enable optimizations only for release builds
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        target_compile_options(wavelet PRIVATE /O2 /Oy)
    endif()
else ()
    target_link_libraries(wavelet m) # link math library
    target_compile_options(wavelet PRIVATE
        -Wall
        -Werror
        -std=c99
        -Wno-gnu-zero-variadic-macro-arguments
        -Wno-error=overflow
        -Wno-error=unused-function
        -fvisibility=hidden
        -std=gnu99
    )
    if (APPLE)
        if (${WAVELET_USE_ACCELERATE})
            find_library(ACCELERATE_FRAMEWORK Accelerate)
            if (ACCELERATE_FRAMEWORK)
                message(STATUS "Accelerate framework found")
                target_compile_definitions(wavelet PRIVATE WL_USE_ACCELERATE)
                target_compile_definitions(wavelet PRIVATE ACCELERATE_NEW_LAPACK)
                target_compile_definitions(wavelet PRIVATE ACCELERATE_LAPACK_ILP64)
                target_link_libraries(wavelet ${ACCELERATE_FRAMEWORK})
            else()
                message(WARNING "Accelerate framework not found, using fallback")
            endif()
        endif()
    else() # todo x86
        target_compile_options(wavelet PRIVATE -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mpclmul)
    endif()
    # Enable optimizations only for release builds
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        target_compile_options(wavelet PRIVATE -O3 -flto)
        target_link_options(wavelet PRIVATE -flto)
    endif()
endif()
if (${WAVELET_ENABLE_IMAGE_SUPPORT})
    target_compile_definitions(wavelet PRIVATE WL_ENABLE_IMAGE_SUPPORT)
endif()
