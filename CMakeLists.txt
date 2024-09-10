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
if (${MSML_ENABLE_IMAGE_SUPPORT})
    target_compile_definitions(msml PRIVATE MSML_ENABLE_IMAGE_SUPPORT)
endif()
