cmake_minimum_required(VERSION 3.10)

project(msml LANGUAGES C)

set(CMAKE_C_STANDARD 99)
option (MSML_BUILD_TESTS "Build tests" ON)

if (${MSML_BUILD_TESTS})
    enable_testing()
    add_subdirectory(test)
endif()

add_library(msml SHARED msml.h msml.c)
