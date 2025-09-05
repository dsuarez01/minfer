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

target_compile_options(minfer PRIVATE ${ARCH_FLAGS})