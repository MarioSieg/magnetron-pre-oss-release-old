# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

cmake_minimum_required(VERSION 3.18)

project(wavelet LANGUAGES C)

message("Configuring wavelet project for ${CMAKE_SYSTEM_PROCESSOR}...")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(IS_AMD64 TRUE)
else()
    set(IS_AMD64 FALSE)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
    set(IS_ARM64 TRUE)
else()
    set(IS_ARM64 FALSE)
endif()

set(CMAKE_C_STANDARD 99)
option(WAVELET_BUILD_TESTS "Build tests" ON)
option(WAVELET_BUILD_BENCHMARKS "Build benchmarks" ON)
option(WAVELET_ENABLE_CUDA "Enable CUDA support" ON)
option(WAVELET_BUILD_FUZZERS "Build fuzzers" OFF)
option(WAVELET_ENABLE_IMAGE_SUPPORT "Enable loading/saving tensors from/to images (includes stb_image)" ON)
if (APPLE)
    option(WAVELET_USE_ACCELERATE "Use Apple's Accelerate framework" ON)
endif()

set(WAVELET_CUDA_COMPILER "/usr/bin/nvcc")

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

file(GLOB_RECURSE WAVELET_SOURCES wavelet/*.h wavelet/*.c wavelet/*.m)
add_library(wavelet SHARED ${WAVELET_SOURCES})
target_include_directories(wavelet PRIVATE wavelet/optional)
if(WIN32) # Windows specific config
    target_compile_options(wavelet PRIVATE /W3 /Oi)
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")     # Enable optimizations for release builds
        target_compile_options(wavelet PRIVATE /O2 /Oy)
    endif()
else() # POSIX specific config
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
    if (APPLE) # Apple specific configuration
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
    endif()
    if(${IS_AMD64}) # x86-64 specific compilation options
        target_compile_options(wavelet PRIVATE -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mpclmul)
    elseif(${IS_ARM64})
        target_compile_options(wavelet PRIVATE -march=armv8-a+simd)
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")     # Enable optimizations only for release builds
        target_compile_options(wavelet PRIVATE -O3 -flto)
        target_link_options(wavelet PRIVATE -flto)
    endif()
endif()
if(${WAVELET_ENABLE_IMAGE_SUPPORT})
    target_compile_definitions(wavelet PRIVATE WL_ENABLE_IMAGE_SUPPORT)
endif()

if (${WAVELET_ENABLE_CUDA})
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            if (GGML_NATIVE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
                #set(CMAKE_CUDA_ARCHITECTURES "native")
            # else() TODO: F16 support
                #set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75")
            else()
                set(CMAKE_CUDA_ARCHITECTURES "52;61;70;75")
            endif()
        endif()
        if (NOT DEFINED CMAKE_CUDA_COMPILER)
            set(CMAKE_CUDA_COMPILER "${WAVELET_CUDA_COMPILER}")
        endif()
        enable_language(CUDA)
        file(GLOB_RECURSE WAVELET_CUDA_SOURCES wavelet/*.cu wavelet/*.cuh)
        add_library(wavelet_cuda SHARED ${WAVELET_CUDA_SOURCES})
        target_include_directories(wavelet_cuda PRIVATE wavelet/optional)
        target_link_libraries(wavelet_cuda wavelet)
        target_compile_definitions(wavelet_cuda PRIVATE WL_ENABLE_CUDA)
        target_compile_options(wavelet_cuda PRIVATE -arch=sm_30)
    else()
        message(WARNING "CUDA not found, disabling CUDA support")
    endif()
endif()
