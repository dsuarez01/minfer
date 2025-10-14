cmake_minimum_required(VERSION 3.18)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# logic specific to enabling OpenMP support via brew on macos
if(APPLE)
    set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp" ${CMAKE_PREFIX_PATH})
endif()

project(
    minfer 
    VERSION 1.0
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# build types
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")

option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ENABLE_THREADING "Enable OpenMP threading support" ON)

if(ENABLE_SANITIZERS)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(WARNING "Sanitizers typically used w/ debug builds. Consider using: -DCMAKE_BUILD_TYPE=Debug")
    endif()

    message(STATUS "Sanitizers enabled")
    set(SANITIZER_FLAGS "-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZER_FLAGS}")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")

# see src/CMakeLists.txt
if(ENABLE_THREADING)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found - parallel support enabled")
    else()
        message(WARNING "OpenMP not found.")
    endif()
else()
    message(STATUS "Threading disabled by user")
endif()


# for now, only arm64 procs are supported (TO-DO: improve exception handling)
function(apply_arch_flags target_name)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        target_compile_definitions(${target_name} PRIVATE USE_ARM64=1)    
        target_compile_options(${target_name} PRIVATE -march=armv8.6-a+fp16+bf16 -mtune=apple-m2)
        target_compile_definitions(${target_name} PRIVATE
            USE_FP16=1
            USE_BF16=1
        )
    endif()
endfunction()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# PCRE2 as a submodule in extern/
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/pcre2/CMakeLists.txt")
    message(FATAL_ERROR 
        "PCRE2 submodule not found. (Hint) run: git submodule update --init --recursive\n"
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