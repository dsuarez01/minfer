cmake_minimum_required(VERSION 3.16)

# build type set BEFORE project() command  
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(minfer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(src external includes)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(ARCH_FLAGS "-march=armv8.6-a+fp16") # to enable BF16 opn support as well as FP16 opn support
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_FLAGS "-march=native -mf16c")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# PCRE2: build from source, copy header to external/
include(FetchContent)
FetchContent_Declare(
    pcre2
    GIT_REPOSITORY https://github.com/PCRE2Project/pcre2.git
    GIT_TAG pcre2-10.44
    GIT_SHALLOW TRUE
)
set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "Build 8-bit PCRE2 library")
set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "Build 16-bit PCRE2 library")  
set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "Build 32-bit PCRE2 library")
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "Build pcre2grep")
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "Build tests")
FetchContent_MakeAvailable(pcre2)

# copy header to external/
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${pcre2_BINARY_DIR}/pcre2.h
            ${CMAKE_SOURCE_DIR}/external/pcre2.h
)

# source files
set(COMMON_SOURCES
    src/common/config/config.cpp
    src/common/parsing/gguf.cpp
    src/common/base/model.cpp
    src/common/base/module.cpp
    src/common/base/sampler.cpp
    src/common/ops/cpu_ops.cpp
)

set(QWEN3_SOURCES
    src/common/models/qwen3/model.cpp
    src/common/models/qwen3/module.cpp
)

set(TOKENIZER_SOURCES
    src/common/models/qwen3/tokenizer.cpp
)

set(TEST_BASE_SOURCES
    src/tests/base/test_base.cpp
)

set(TEST_OPS_SOURCES
    src/tests/ops/test_matmul.cpp
    src/tests/ops/test_silu.cpp
    src/tests/ops/test_softmax.cpp
    src/tests/ops/test_rmsnorm.cpp
    src/tests/ops/test_rope.cpp
    src/tests/ops/test_attn.cpp
    src/tests/ops/test_swiglu.cpp
    src/tests/ops/test_route.cpp
)

# executables
add_executable(test_ops_runner
    src/tests/ops/test_ops_main.cpp
    ${TEST_OPS_SOURCES}
    ${TEST_BASE_SOURCES}
    ${COMMON_SOURCES}
    ${QWEN3_SOURCES}
    ${TOKENIZER_SOURCES}
)
target_link_libraries(test_ops_runner pcre2-8 pthread)

option(ENABLE_INLINE_REPORTS "Enable compiler inline reports for generate target" OFF)

add_executable(generate
    src/tests/generate/generate.cpp
    ${COMMON_SOURCES}
    ${QWEN3_SOURCES}
    ${TOKENIZER_SOURCES}
)

set_target_properties(generate PROPERTIES 
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
)

if(ENABLE_INLINE_REPORTS)
    target_compile_options(generate PRIVATE 
        $<$<CONFIG:Release>:-Rpass=inline>
        $<$<CONFIG:Release>:-Rpass-missed=inline>
    )
endif()

target_link_libraries(generate pcre2-8 pthread)

add_executable(test_tokenizer
    src/tests/tokenizer/test_tokenizer_main.cpp
    src/tests/tokenizer/test_tokenizer.cpp
    ${TEST_BASE_SOURCES}
    ${COMMON_SOURCES}
    ${TOKENIZER_SOURCES}
)
target_link_libraries(test_tokenizer pcre2-8 pthread)