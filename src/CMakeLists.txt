add_library(minfer
    base/model.cpp
    base/module.cpp
    base/sampler.cpp
    base/tokenizer.cpp
    config/config.cpp
    models/qwen3/model.cpp
    models/qwen3/module.cpp
    models/qwen3/tokenizer.cpp
    ops/kernels.cpp
    parsing/gguf.cpp
)

# handles auto-detection of Metal and CUDA Backends
find_package(CUDAToolkit QUIET)
find_library(METAL_FRAMEWORK Metal QUIET)

if(APPLE AND METAL_FRAMEWORK)
    target_compile_definitions(minfer PRIVATE USE_METAL)

    target_sources(minfer PRIVATE config/metal_config.mm)

    # to precompile the kernels written in MSL
    set(METAL_SHADER_SOURCE ${CMAKE_SOURCE_DIR}/src/ops/kernels.metal)
    set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/kernels.air)
    set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib)
    set(METAL_C ${CMAKE_CURRENT_BINARY_DIR}/kernels_metallib.c)

    add_custom_command(
        OUTPUT ${METAL_C}
        # useful flags: -v, -fmetal-enable-logging
        # useful env variables: MTL_LOG_LEVEL=MTLLogLevelDebug, MTL_LOG_TO_STDERR=1
        # include <metal_logging> and use os_log_default.log() in the shader files to catch errors
        COMMAND xcrun -sdk macosx metal -std=metal3.2 -target air64-apple-macos26.0 -c ${METAL_SHADER_SOURCE} -o ${METAL_AIR}
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
        COMMAND xxd -i kernels.metallib > ${METAL_C}
        DEPENDS ${METAL_SHADER_SOURCE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Compiling Metal shaders (kernels) to library"
    )

    add_custom_target(metal_shaders DEPENDS ${METAL_C})
    add_dependencies(minfer metal_shaders)

    target_sources(minfer PRIVATE ${METAL_C})

    target_include_directories(minfer PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_include_directories(minfer PRIVATE ${CMAKE_SOURCE_DIR}/include/extern/Metal/Metal.hpp)

    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)

    target_link_libraries(minfer PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${QUARTZCORE_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
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