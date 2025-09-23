add_library(minfer
    base/model.cpp
    base/module.cpp
    base/sampler.cpp
    base/tokenizer.cpp
    config/config.cpp
    models/qwen3/model.cpp
    models/qwen3/module.cpp
    models/qwen3/tokenizer.cpp
    ops/cpu_ops.cpp
    parsing/gguf.cpp
)

# Auto-detection of Metal and CUDA Backends
find_package(CUDAToolkit QUIET)
find_library(METAL_FRAMEWORK Metal QUIET)

if(APPLE AND METAL_FRAMEWORK)
    target_compile_definitions(minfer PRIVATE USE_METAL)

    target_sources(minfer PRIVATE config/metal_config.mm)

    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
    target_link_libraries(minfer PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${QUARTZCORE_FRAMEWORK}
    )

    message(STATUS "Using Metal backend")
elseif(CUDAToolkit_FOUND)
    target_compile_definitions(minfer PRIVATE USE_CUDA)
    # addn. set-up likely required here
    message(STATUS "Using CUDA backend")
else()
    message(STATUS "No GPU backend found, using CPU only")
endif()

target_include_directories(minfer PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/extern
)

target_link_libraries(minfer 
    PUBLIC pcre2-8
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(minfer PRIVATE OpenMP::OpenMP_CXX)
endif()

apply_arch_flags(minfer)