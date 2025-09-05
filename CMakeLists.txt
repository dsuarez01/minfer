cmake_minimum_required(VERSION 3.18)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# This logic is specific to enabling OpenMP support via brew on macos systems, and can be removed if needed
if(APPLE)
    # sets the paths for find_package to locate libomp
    set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp" ${CMAKE_PREFIX_PATH})
endif()

project(
    minfer 
    VERSION 1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# see src/CMakeLists.txt for more info on how OpenMP is enabled
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - parallel support enabled")
else()
    message(WARNING "OpenMP not found.")
endif()

function(apply_arch_flags target_name)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        target_compile_options(${target_name} PRIVATE -march=armv8.6-a+fp16)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(${target_name} PRIVATE -march=native -mf16c)
    endif()
endfunction()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# PCRE2 as a submodule in extern/
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/pcre2/CMakeLists.txt")
    message(FATAL_ERROR 
        "PCRE2 submodule not found. Please run:\n"
        "(if not cloned)             git clone --recurse-submodules -b feature/cpu https://github.com/dsuarez01/minfer\n"
        "(if not cloned, cont.)      git submodule update --init --recursive\n"
        "(if already cloned)         git submodule update --init --recursive\n"
    )
endif()

set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "")
set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "")
set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "")
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "")
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "")
add_subdirectory(extern/pcre2)

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tests)