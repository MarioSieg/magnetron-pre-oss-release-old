# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

cmake_minimum_required(VERSION 3.10)

project(msml LANGUAGES C)

set(CMAKE_C_STANDARD 99)
option (MSML_BUILD_TESTS "Build tests" ON)
option (MSML_ENABLE_IMAGE_SUPPORT "Enable creating tensor from images support" ON)

if (${MSML_BUILD_TESTS})
    enable_testing()
    add_subdirectory(test)
endif()

add_library(msml SHARED msml/msml.h msml/msml.c)
target_include_directories(msml PRIVATE msml/optional)
if (NOT WIN32)
    target_link_libraries(msml m) # link math library
    target_compile_options(msml PRIVATE
        -Wall
        -Werror
        -std=c99
        -Wno-gnu-zero-variadic-macro-arguments
        -Wno-error=overflow
        -fvisibility=hidden
        -std=gnu99
    )
    # Enable optimizations only for release builds
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        target_compile_options(msml PRIVATE -O3 -flto)
        target_link_options(msml PRIVATE -flto)
    endif()
endif()
if (${MSML_ENABLE_IMAGE_SUPPORT})
    target_compile_definitions(msml PRIVATE MSML_ENABLE_IMAGE_SUPPORT)
endif()
